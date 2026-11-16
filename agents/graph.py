"""LangGraph workflow — multi-agent trading pipeline (v2.0).

v2.0 Flow:
    market_data + kafka_events + l2_order_flow
    → hmm_regime
    → quant_screener
    → strategy_candidates (enabled sleeves only; empty by default)
    → shared_news
    → [technical, fundamental, sentiment, options, macro → geopolitical_nlp, india_specific] (parallel)
    → bull_bear_debate
    → [marl_optimizer, delta_hedger] (parallel)
    → compliance_check
    → portfolio_manager
    → order_router
    → episodic_reflection
    → END
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pytz
from loguru import logger

from agents.base_agent import TradingState, create_initial_state
from agents.fundamental_agent import fundamental_agent
from agents.india_specific_agent import india_specific_agent
from agents.macro_agent import macro_agent
from agents.market_data_agent import market_data_agent
from agents.options_agent import options_agent
from agents.portfolio_manager_agent import portfolio_manager_agent
from agents.risk_manager_agent import risk_manager_agent
from agents.sentiment_agent import sentiment_agent
from agents.technical_agent import technical_agent
from broker.order_manager import place_order
from config.instruments import get_token
from config.settings import settings
from config.llm_runtime import current_context, llm_operation, mark_degraded
from strategies.candidate import (
    HORIZONS, AssetId, CandidateInvalid, TradeCandidate, horizon_to_product,
    tick_aligned_levels, to_decimal,
)

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = None

# Deferred import to avoid circular dependency
from agents.quantitative_screener import quantitative_screener_node
from agents.strategy_candidates import strategy_candidates_node

IST = pytz.timezone("Asia/Kolkata")

# Live entries stay blocked until live fills, partial fills and exit reconciliation are validated.
LIVE_ORDER_LIFECYCLE_VALIDATED = False


async def _shared_news_node(state: TradingState) -> dict[str, Any]:
    from data.news_fetcher import fetch_news_snapshot

    if state.get("research_mode"):
        if state.get("execution_enabled", True) or not state.get("news_snapshot", {}).get("as_of"):
            raise ValueError("Research replay requires frozen news and disabled execution")
        return {"agent_logs": ["[NewsSnapshot] Using captured research news."]}
    try:
        snapshot = await fetch_news_snapshot(state.get("watchlist", []))
    except Exception as exc:
        message = f"[NewsSnapshot] Collection failed ({type(exc).__name__})"
        logger.error(message)
        return {"news_snapshot": {}, "agent_logs": [message], "errors": [message]}
    errors = [f"[NewsSnapshot] {error}" for error in snapshot["errors"]]
    return {
        "news_snapshot": snapshot,
        "agent_logs": [
            f"[NewsSnapshot] Collected {len(snapshot['articles'])} unique articles "
            f"at {snapshot['as_of']}."
        ],
        "errors": errors,
    }


# ─── v2 Node Wrappers (safe imports) ────────────────────────────

def _kafka_events_node(state: TradingState) -> dict[str, Any]:
    """Consume event-driven Kafka/polling stream."""
    try:
        from data.event_driven_kafka_stream import event_stream
        critical = event_stream.get_critical_events()
        events = critical + event_stream.get_pending_events(max_events=10)
        return {
            "active_macro_event": critical[0] if critical else {},
            "agent_logs": [f"[KafkaEvents] Consumed {len(events)} events, {len(critical)} critical"],
        }
    except Exception as e:
        return {"agent_logs": [f"[KafkaEvents] Error: {e}"], "errors": [str(e)]}


def _l2_order_flow_node(state: TradingState) -> dict[str, Any]:
    """Analyze L2 order book for OFI signals."""
    try:
        from data.l2_order_flow_analyzer import l2_analyzer
        signals = {}
        for symbol in state.get("watchlist", [])[:10]:
            quote = state.get("live_quotes", {}).get(symbol, {})
            if quote.get("best_bid") and quote.get("best_ask"):
                signals[symbol] = l2_analyzer.update_and_signal(
                    symbol,
                    {"best_5_buy_data": quote["best_bid"], "best_5_sell_data": quote["best_ask"]},
                    quote,
                )
        return {
            "ofi_signals": signals,
            "agent_logs": [f"[L2OrderFlow] OFI signals for {len(signals)} symbols"],
        }
    except Exception as e:
        return {"agent_logs": [f"[L2OrderFlow] Error: {e}"], "errors": [str(e)]}


def _hmm_regime_node(state: TradingState) -> dict[str, Any]:
    """Classify current market regime using the fitted HMM; unavailable means no new entries."""
    from risk.hmm_regime_classifier import hmm_classifier

    from risk.dead_mans_switch import dead_mans_switch

    try:
        regime = hmm_classifier.classify_latest()
        classify_error = None
    except Exception as exc:
        classify_error = type(exc).__name__
        regime = hmm_classifier.unavailable(f"classification error: {type(exc).__name__}: {exc}")
    # Only load or classify *errors* count against the circuit breaker; a model that is simply
    # not fitted or is stale is a configuration state that already blocks entries.
    load_error = getattr(hmm_classifier, "load_error", None)
    if classify_error or load_error:
        dead_mans_switch.record_failure("hmm_classifier", classify_error or str(load_error))
    elif not regime.get("degraded"):
        dead_mans_switch.record_success("hmm_classifier")
    if regime.get("degraded"):
        message = f"HMM unavailable/degraded — new entries blocked: {regime.get('reason', 'unknown')}"
        logger.warning(message)
        return {"hmm_regime": regime, "agent_logs": [f"[HMMRegime] {message}"], "errors": [message]}
    return {
        "hmm_regime": regime,
        "agent_logs": [
            f"[HMMRegime] {regime['regime']} (conf={regime.get('confidence', 0):.0%}, "
            f"panic_risk={regime.get('panic_risk', 0):.1%}, as_of={regime.get('as_of')})"
        ],
    }


def _geopolitical_nlp_node(state: TradingState) -> dict[str, Any]:
    """Geopolitical NLP risk analysis."""
    try:
        from agents.geopolitical_nlp_agent import geopolitical_nlp_agent as geo_agent
        return geo_agent(state)
    except Exception as e:
        return {"agent_logs": [f"[GeopoliticalNLP] Error: {e}"], "errors": [str(e)]}


def _bull_bear_debate_node(state: TradingState) -> dict[str, Any]:
    """Bull/Bear adversarial debate synthesis."""
    try:
        from agents.bull_bear_synthesis_agent import bull_bear_synthesis_agent as debate_fn
        return debate_fn(state)
    except Exception as e:
        return {"agent_logs": [f"[BullBearDebate] Error: {e}"], "errors": [str(e)]}


def _marl_optimizer_node(state: TradingState) -> dict[str, Any]:
    """MARL + HRP portfolio optimization."""
    try:
        from risk.marl_portfolio_optimizer import MARLPortfolioOptimizer, marl_optimizer
        import pandas as pd

        prices = {}
        for sym, data in state.get("market_data", {}).items():
            if isinstance(data, pd.DataFrame) and "close" in data.columns:
                prices[sym] = data["close"].dropna().to_numpy()

        if prices:
            signals = {
                symbol: signal for symbol, signal in state.get("technical_signals", {}).items()
                if symbol in prices and len(prices[symbol]) >= 20
            }
            optimizer = MARLPortfolioOptimizer() if state.get("research_mode") else marl_optimizer
            allocations = optimizer.optimize_portfolio(
                signals=signals, prices=prices, regime=state.get("hmm_regime", {}),
                available_capital=state.get("available_capital", 0),
            )
        else:
            allocations = {}

        return {
            "optimal_allocations": allocations,
            "agent_logs": [f"[MARLOptimizer] Allocations for {len(allocations)} symbols"],
        }
    except Exception as e:
        return {"agent_logs": [f"[MARLOptimizer] Error: {e}"], "errors": [str(e)]}


def _delta_hedger_node(state: TradingState) -> dict[str, Any]:
    """AI delta hedging for options positions."""
    try:
        from risk.ai_options_delta_hedger import delta_hedger
        positions = state.get("current_positions", {})
        portfolio_greeks = delta_hedger.compute_portfolio_greeks(
            positions, state.get("live_quotes", {}),
        )
        return {
            "portfolio_greeks": portfolio_greeks,
            "agent_logs": [f"[DeltaHedger] Portfolio delta: {portfolio_greeks.get('delta', 0):.2f}"],
        }
    except Exception as e:
        return {"agent_logs": [f"[DeltaHedger] Error: {e}"], "errors": [str(e)]}


def _compliance_check_node(state: TradingState) -> dict[str, Any]:
    """FEMA/LRS compliance gate."""
    if state.get("research_mode"):
        return {"lrs_status": {}, "agent_logs": ["[Compliance] Research is limited to INR paper simulation."]}
    try:
        from config.fema_lrs_controller import fema_controller
        lrs_status = fema_controller.get_lrs_status()
        return {
            "lrs_status": lrs_status,
            "agent_logs": [f"[Compliance] LRS remaining: ${lrs_status.get('remaining_usd', 0):,.0f}"],
        }
    except Exception as e:
        return {"agent_logs": [f"[Compliance] Error: {e}"], "errors": [str(e)]}


def _product_for(horizon: Any) -> str:
    """Broker product for a horizon; unknown values fall back to DELIVERY (never an intraday product)."""
    try:
        return horizon_to_product(str(horizon))
    except CandidateInvalid:
        return "DELIVERY"


def _instrument_tick_and_lot(symbol: str) -> tuple[Decimal, int]:
    """Tick size and lot size from the instrument master; NSE equity defaults when it has no entry."""
    tick, lot = Decimal("0.05"), 1
    try:
        from config.instruments import get_instrument

        instrument = get_instrument(symbol, "NSE")
        if instrument:
            # Angel's master quotes tick sizes in paise.
            parsed = Decimal(str(float(instrument["tick_size"]) / 100))
            if parsed > 0:
                tick = parsed
            lot = max(1, int(instrument.get("lotsize") or 1))
    except Exception:
        pass
    return tick, lot


def _tick_aligned_levels(
    action: str, price: Any, stop: Any, target: Any, tick: Decimal,
) -> tuple[float, float, float]:
    return tick_aligned_levels(action, price, stop, target, tick)


def _build_trade_candidate(
    *, symbol: str, action: str, decision: dict[str, Any], technical: dict[str, Any],
    contract: Any, quantity: int, atr_value: Any, tick: Decimal, lot: int, now: datetime,
) -> TradeCandidate:
    """Assemble and validate the typed candidate; raises CandidateInvalid to block the entry."""
    horizon = str(decision["holding_period"]).upper()
    indicators = technical.get("indicators") if isinstance(technical, dict) else {}
    policy_name = (
        (technical.get("exit_policy") if isinstance(technical, dict) else None)
        or contract.default_exit_policy
    )
    from risk.exit_policy import get_atr_exit_policy

    try:
        timeframe = get_atr_exit_policy(policy_name).atr_timeframe
    except ValueError as exc:
        raise CandidateInvalid(str(exc)) from exc
    signal_time = now.astimezone(timezone.utc)
    candidate = TradeCandidate(
        asset=AssetId(venue="NSE", segment=contract.segment, currency="INR", symbol=symbol),
        candidate_id=str(decision.get("candidate_id") or f"{symbol}-{signal_time:%Y%m%d%H%M%S}"),
        strategy=contract.name,
        strategy_version=contract.version,
        side=action,
        horizon=horizon,
        product=horizon_to_product(horizon),
        signal_time=signal_time,
        expires_at=signal_time + timedelta(minutes=30),
        tick_size=tick,
        lot_size=lot,
        entry_price=to_decimal(decision["price"], "price"),
        stop_loss=to_decimal(decision["stop_loss"], "stop_loss"),
        target=to_decimal(decision["target"], "target"),
        exit_policy=policy_name,
        atr_value=to_decimal(atr_value if atr_value not in (None, 0) else (indicators or {}).get("daily_atr_14"), "atr"),
        atr_timeframe=timeframe,
        thesis=str(decision.get("reasoning") or "")[:500],
        review_contract_version=contract.review.version if contract.review else None,
    )
    return candidate.validate(quantity=quantity, now=now)


def _order_router_node(state: TradingState) -> dict[str, Any]:
    """Intelligent order routing across Angel/IBKR/Alpaca."""
    if not state.get("execution_enabled", True):
        return {
            "executed_orders": [],
            "agent_logs": ["[OrderRouter] Execution disabled; analysis mandate only."],
            "errors": [],
        }
    try:
        from broker.intelligent_order_router import order_router
    except ImportError:
        order_router = None

    logs: list[str] = []
    errors: list[str] = []
    executed: list[dict[str, Any]] = []
    review_contract_results: dict[str, Any] = {}

    decisions = state.get("trading_decisions", [])
    if not isinstance(decisions, list):
        return {
            "executed_orders": [],
            "agent_logs": logs,
            "errors": ["[OrderRouter] Trading decisions must be a list."],
        }
    now_ist = datetime.now(IST)
    entries_today: int | None = None
    from risk.position_sizer import position_sizer

    projected_entry_exposure: dict[str, float] = {}
    projected_entry_quantities: dict[str, float] = {}
    projected_entry_prices: dict[str, float] = {}
    projected_entry_sectors: dict[str, dict[str, str]] = {}
    projected_cash_committed = 0.0
    projected_exposure = 0.0
    projected_symbol_exposure: dict[str, float] = {}
    projected_positions: dict[str, dict[str, Any]] = {}
    initial_positions = {
        symbol: dict(position)
        for symbol, position in state.get("current_positions", {}).items()
        if isinstance(position, dict)
    }
    from broker.paper_ledger import position_book_identity

    try:
        expected_paper_book = position_book_identity(state.get("current_positions", {}))
    except (TypeError, ValueError, OverflowError):
        expected_paper_book = None
    projected_stop_risk = 0.0
    risk_history: dict[str, Any] = {}

    def risk_returns(symbol: str):
        from risk.daily_portfolio_risk import daily_returns

        if symbol not in risk_history:
            as_of = state.get("research_as_of") if state.get("research_mode") else None
            risk_history[symbol] = daily_returns(
                state.get("market_data", {}).get(symbol), as_of or None,
            )
        return risk_history[symbol]

    def track_projected_stop_risk(
        signal: dict[str, Any], quantity: int, fill_price: float,
    ) -> None:
        nonlocal projected_stop_risk
        from risk.portfolio_stop_risk import entry_stop_risk

        try:
            requested_price = float(signal["price"])
            conservative_price = (
                max(requested_price, fill_price) if signal["action"] == "BUY"
                else min(requested_price, fill_price)
            )
            projected_stop_risk += entry_stop_risk(
                {**signal, "price": conservative_price}, quantity,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            projected_stop_risk = math.inf
            errors.append(f"[OrderRouter] Filled entry stop risk is unavailable: {exc}")

    for decision in decisions:
        if not isinstance(decision, dict):
            errors.append("[OrderRouter] Invalid order decision: expected an object.")
            continue
        symbol = decision.get("symbol", "")
        if not isinstance(symbol, str):
            errors.append("[OrderRouter] Invalid order decision: symbol must be text.")
            continue
        symbol = symbol.strip().upper()
        action = str(decision.get("action", "")).upper()
        technical_signals = state.get("technical_signals")
        technical = technical_signals.get(symbol, {}) if isinstance(technical_signals, dict) else {}
        indicators = technical.get("indicators", {}) if isinstance(technical, dict) else {}
        atr = decision.get("atr")
        if atr is None and isinstance(indicators, dict):
            atr = (
                indicators.get("policy_atr") or indicators.get("daily_atr_14")
                or indicators.get("atr_14")
            )
        raw_quantity = decision.get("quantity", 0)
        if isinstance(raw_quantity, bool) or not isinstance(raw_quantity, (int, float)):
            errors.append(f"[OrderRouter] Invalid quantity for {symbol or 'unknown symbol'}")
            continue
        try:
            quantity_value = float(raw_quantity)
        except (OverflowError, ValueError):
            quantity_value = math.nan
        if not math.isfinite(quantity_value) or not quantity_value.is_integer():
            errors.append(f"[OrderRouter] Invalid quantity for {symbol or 'unknown symbol'}")
            continue
        quantity = int(quantity_value)
        price = decision.get("price", 0)
        currency = decision.get("currency", "INR")

        if not symbol or action not in {"BUY", "SELL"} or quantity <= 0:
            errors.append(f"[OrderRouter] Invalid order decision for {symbol or 'unknown symbol'}")
            continue
        position = state.get("current_positions", {}).get(symbol, {})
        try:
            position_quantity = int(position.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            errors.append(f"[OrderRouter] Invalid position quantity for {symbol}")
            continue
        is_exit = (
            (action == "SELL" and position_quantity > 0)
            or (action == "BUY" and position_quantity < 0)
        )
        holding_period = str(decision.get("holding_period", "")).upper()
        if action == "SELL" and position_quantity <= 0:
            # Paper mode may open intraday (MIS) shorts; they are squared off at 15:15.
            if not (settings.paper_trading and holding_period == "INTRADAY"):
                errors.append(f"[OrderRouter] Naked short sale blocked for {symbol}")
                continue
            from risk.stop_loss_manager import stop_on_protective_side

            try:
                short_price = float(price or 0)
                short_stop = float(decision.get("stop_loss") or 0)
            except (TypeError, ValueError):
                short_price, short_stop = 0.0, 0.0
            if not stop_on_protective_side("SELL", short_price, short_stop):
                errors.append(f"[OrderRouter] Intraday short blocked for {symbol}: stop-loss must be above entry")
                continue
        if is_exit:
            from operations.corporate_actions import warn_if_held

            warn_if_held(symbol)
        if is_exit and quantity > abs(position_quantity):
            errors.append(f"[OrderRouter] Exit quantity exceeds held position for {symbol}")
            continue
        if is_exit:
            from risk.stop_loss_manager import stop_loss_manager

            if symbol in stop_loss_manager.pending_exits():
                errors.append(
                    f"[OrderRouter] Strategy exit for {symbol} deferred: a protective exit awaits reconciliation"
                )
                continue
        elif not settings.paper_trading and not LIVE_ORDER_LIFECYCLE_VALIDATED:
            errors.append(
                f"[OrderRouter] Live entry blocked for {symbol}: the live order lifecycle "
                "(fill confirmation, partial fills, exit reconciliation) is not validated."
            )
            continue
        if not is_exit:
            started = state.get("scan_started_monotonic")
            if isinstance(started, (int, float)) and not isinstance(started, bool):
                if time.monotonic() - float(started) > settings.scan_entry_deadline_seconds:
                    errors.append(
                        f"[OrderRouter] Entry blocked for {symbol}: scan exceeded entry deadline"
                    )
                    continue
            from agents.review_context import strategy_review_contract_report

            review_contract_results[symbol] = strategy_review_contract_report(
                state, symbol, decision,
            )
            from agents.sentiment_agent import validate_candidate_sentiment
            from agents.sentiment_evidence import sentiment_digest

            try:
                candidate_digest = sentiment_digest(validate_candidate_sentiment(state, symbol))
            except (TypeError, ValueError):
                errors.append(f"[OrderRouter] Required sentiment unavailable or invalid: {symbol}")
                continue
            screening = state.get("screener_results")
            qualified = screening.get("qualified") if isinstance(screening, dict) else None
            rejected = screening.get("rejected") if isinstance(screening, dict) else None
            watchlist = state.get("watchlist")
            if (
                not isinstance(qualified, list)
                or not isinstance(rejected, list)
                or not isinstance(watchlist, list)
                or symbol not in qualified
                or symbol in rejected
                or symbol not in watchlist
            ):
                errors.append(f"[OrderRouter] New entry lacks valid screener eligibility: {symbol}")
                continue
            if decision.get("ai_review_passed") is not True:
                errors.append(f"[OrderRouter] New entry lacks a successful portfolio-manager review: {symbol}")
                continue
            from agents.review_context import (
                HistoryEvidenceError, ReviewContextError, candidate_review_identity, debate_review_digest,
                technical_review_digest, valid_entry_debate,
            )

            debate = state.get("debate_results", {}).get(symbol, {})
            try:
                debate_digest = debate_review_digest(debate)
            except ReviewContextError:
                errors.append(f"[OrderRouter] New entry has invalid debate evidence: {symbol}")
                continue
            try:
                technical_digest = technical_review_digest(technical)
                candidate_id = candidate_review_identity(state, symbol)
            except HistoryEvidenceError:
                errors.append(f"[OrderRouter] Historical data evidence missing, expired, or changed: {symbol}")
                continue
            except ReviewContextError:
                errors.append(f"[OrderRouter] New entry has invalid technical evidence: {symbol}")
                continue
            if (
                not valid_entry_debate(debate, action)
                or debate.get("sentiment_review_digest") != candidate_digest
                or decision.get("sentiment_review_digest") != candidate_digest
                or decision.get("debate_review_digest") != debate_digest
                or decision.get("debate_context_json") != debate.get("debate_context_json")
            ):
                errors.append(f"[OrderRouter] New entry lacks a complete aligned debate review: {symbol}")
                continue
            if (
                debate.get("technical_review_digest") != technical_digest
                or decision.get("technical_review_digest") != technical_digest
            ):
                errors.append(f"[OrderRouter] New entry has changed technical evidence: {symbol}")
                continue
            if debate.get("candidate_id") != candidate_id or decision.get("candidate_id") != candidate_id:
                errors.append(f"[OrderRouter] New entry has mismatched candidate identity: {symbol}")
                continue
            from operations.candidate_reviews import CandidateReviewUnavailable, verify_candidate_review
            from sqlalchemy.exc import SQLAlchemyError

            try:
                verify_candidate_review(state, decision)
            except (CandidateReviewUnavailable, ReviewContextError) as exc:
                errors.append(f"[OrderRouter] Stored candidate review unavailable for {symbol}: {exc}")
                continue
            except (SQLAlchemyError, OSError) as exc:
                errors.append(f"[OrderRouter] Stored candidate review unavailable for {symbol}: {type(exc).__name__}")
                continue
            from operations.control_state import control_state
            from risk.dead_mans_switch import dead_mans_switch

            if control_state.paused() or dead_mans_switch.is_halted:
                errors.append(f"[OrderRouter] New entry blocked by operational safety controls: {symbol}")
                continue
            from broker.portfolio_tracker import portfolio_tracker

            if state.get("portfolio_snapshot_valid") is not True:
                errors.append(f"[OrderRouter] Portfolio snapshot unavailable; new entry blocked: {symbol}")
                continue
            try:
                marks_fresh = portfolio_tracker.update_unrealised_pnl()
            except Exception as exc:
                errors.append(f"[OrderRouter] Portfolio mark refresh failed for {symbol}: {exc}")
                continue
            if marks_fresh is not True:
                errors.append(f"[OrderRouter] Fresh portfolio marks unavailable; new entry blocked: {symbol}")
                continue
            try:
                portfolio_value = float(portfolio_tracker.total_value)
                fresh_available_capital = float(portfolio_tracker.available_capital)
                if (
                    not math.isfinite(portfolio_value) or portfolio_value <= 0
                    or not math.isfinite(fresh_available_capital) or fresh_available_capital < 0
                ):
                    raise ValueError("fresh portfolio equity or cash is invalid")
                current_book = portfolio_tracker.positions
                if set(current_book) != set(initial_positions) or any(
                    current_book[held].get("quantity") != initial_positions[held].get("quantity")
                    or current_book[held].get("entry_trade_ids") != initial_positions[held].get("entry_trade_ids")
                    or current_book[held].get("holding_period") != initial_positions[held].get("holding_period")
                    for held in initial_positions
                ):
                    raise ValueError("position identities changed since the scan snapshot")
                projected_symbol_exposure = position_sizer.exposure_by_symbol(current_book)
                for projected_symbol, notional in projected_entry_exposure.items():
                    projected_symbol_exposure[projected_symbol] = (
                        projected_symbol_exposure.get(projected_symbol, 0.0) + notional
                    )
                projected_exposure = sum(projected_symbol_exposure.values())
                projected_positions = {
                    held_symbol: dict(position)
                    for held_symbol, position in current_book.items()
                }
                for projected_symbol, quantity in projected_entry_quantities.items():
                    position = projected_positions.setdefault(projected_symbol, {
                        "quantity": 0,
                        "ltp": projected_entry_prices.get(projected_symbol),
                    })
                    position["quantity"] = float(position.get("quantity") or 0) + quantity
                    if not position.get("ltp"):
                        position["ltp"] = projected_entry_prices[projected_symbol]
                    position.update(projected_entry_sectors[projected_symbol])
                available_capital = max(
                    0.0, fresh_available_capital - projected_cash_committed,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(f"[OrderRouter] Fresh portfolio valuation unavailable for {symbol}: {exc}")
                continue
            if portfolio_tracker.has_hit_daily_loss_limit():
                errors.append(f"[OrderRouter] Daily loss safety halt blocks new entries: {symbol}")
                continue
            if portfolio_tracker.has_hit_portfolio_drawdown_limit():
                errors.append(f"[OrderRouter] Portfolio drawdown safety halt blocks new entries: {symbol}")
                continue
            from operations.corporate_actions import blocking_action

            try:
                pending_action = blocking_action(symbol, now_ist.date())
            except Exception as exc:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: corporate action check unavailable "
                    f"({type(exc).__name__})"
                )
                continue
            if pending_action:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: unprocessed corporate action "
                    f"{pending_action['action_type']} (ex {pending_action['ex_date']})"
                )
                continue
            held_quantity = projected_positions.get(symbol, {}).get("quantity") or 0
            if abs(float(held_quantity)) > 0:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: an open position already exists"
                )
                continue
            from risk.hmm_regime_classifier import entry_regime_block_reason, entry_regime_position_cap

            regime_block = entry_regime_block_reason(state.get("hmm_regime"))
            if regime_block:
                errors.append(f"[OrderRouter] New entry blocked for {symbol}: {regime_block}")
                continue
            try:
                regime_position_cap = entry_regime_position_cap(state.get("hmm_regime"))
            except ValueError as exc:
                errors.append(f"[OrderRouter] New entry blocked for {symbol}: {exc}")
                continue

            if currency != "INR":
                errors.append(f"[OrderRouter] New entries require a configured segment-specific risk model: {symbol}")
                continue
            risk_data = state.get("risk_assessment", {}).get(symbol, {})
            if not isinstance(risk_data, dict) or risk_data.get("approved") is not True:
                errors.append(f"[OrderRouter] New entry is not risk-approved: {symbol}")
                continue
            technical = state.get("technical_signals", {}).get(symbol, {})
            from risk.hmm_regime_classifier import entry_regime_strategy_block_reason

            strategy_reason = entry_regime_strategy_block_reason(
                state.get("hmm_regime"), technical.get("strategy") if isinstance(technical, dict) else None,
            )
            if strategy_reason:
                errors.append(f"[OrderRouter] New entry blocked for {symbol}: {strategy_reason}")
                continue
            technical_action = str(technical.get("signal", "")).upper()
            expected_action = (
                "BUY" if technical_action in {"BUY", "STRONG_BUY"}
                else "SELL" if technical_action in {"SELL", "STRONG_SELL"}
                else ""
            )
            if expected_action != action or risk_data.get("deterministic_action") != action:
                errors.append(f"[OrderRouter] Entry does not match the deterministic signal: {symbol}")
                continue
            if len(projected_symbol_exposure) >= settings.max_positions:
                errors.append(
                    f"[OrderRouter] Max positions ({settings.max_positions}) reached: {symbol}"
                )
                continue
            from risk.daily_portfolio_risk import RiskHistoryUnavailable, correlation_block_reason

            try:
                correlation_reason = correlation_block_reason(
                    symbol, action, projected_positions or {}, risk_returns,
                )
            except RiskHistoryUnavailable as exc:
                errors.append(f"[OrderRouter] Daily risk history unavailable for {symbol}: {exc}")
                continue
            if correlation_reason:
                errors.append(f"[OrderRouter] Entry blocked for {symbol}: {correlation_reason}")
                continue
            from risk.portfolio_stop_risk import (
                StopRiskUnavailable, open_stop_risk, remaining_stop_risk,
            )
            from risk.stop_loss_manager import stop_loss_manager

            try:
                stop_budget = remaining_stop_risk(
                    portfolio_value,
                    open_stop_risk(current_book, stop_loss_manager.snapshot()),
                    projected_stop_risk,
                )
            except (StopRiskUnavailable, TypeError, ValueError, OverflowError) as exc:
                errors.append(f"[OrderRouter] Portfolio stop risk unavailable for {symbol}: {exc}")
                continue

            indicators = technical.get("indicators", {})
            quote = state.get("live_quotes", {}).get(symbol, {})
            deterministic_price = quote.get("ltp") or indicators.get("close")
            deterministic_stop = risk_data.get("stop_loss") or technical.get("stop_loss")
            deterministic_target = risk_data.get("target") or technical.get("target")
            holding_period = str(
                risk_data.get("holding_period") or technical.get("holding_period") or "INTRADAY"
            ).upper()
            from strategies.contracts import require_strategy_contract

            try:
                entry_contract = require_strategy_contract(
                    technical.get("strategy") if isinstance(technical, dict) else None,
                )
            except ValueError as exc:
                errors.append(f"[OrderRouter] Candidate contract invalid for {symbol}: {exc}")
                continue
            if holding_period not in entry_contract.horizons:
                errors.append(
                    f"[OrderRouter] Unsupported horizon {holding_period} for {symbol}: "
                    f"{entry_contract.name} allows {', '.join(entry_contract.horizons)}"
                )
                continue
            tick_size, lot_size = _instrument_tick_and_lot(symbol)
            try:
                # Levels move at most one tick, toward the entry, so risk is never understated.
                deterministic_price, deterministic_stop, deterministic_target = _tick_aligned_levels(
                    action, deterministic_price, deterministic_stop, deterministic_target, tick_size,
                )
            except CandidateInvalid:
                pass  # left as-is; _validate_entry_decision reports malformed numbers
            decision = {
                **decision,
                "price": deterministic_price,
                "stop_loss": deterministic_stop,
                "target": deterministic_target,
                "holding_period": holding_period,
            }
            validation_error = _validate_entry_decision(
                decision, action, allowed_horizons=entry_contract.horizons,
            )
            if validation_error:
                errors.append(f"[OrderRouter] Invalid entry for {symbol}: {validation_error}")
                continue
            from risk.sector_exposure import (
                SectorMetadataUnavailable, validate_candidate_sector,
            )

            try:
                sector_classification = validate_candidate_sector(symbol, risk_data)
            except (SectorMetadataUnavailable, TypeError, ValueError) as exc:
                errors.append(
                    f"[OrderRouter] Sector classification unavailable for {symbol}: {exc}"
                )
                continue
            held_sector = projected_positions.get(symbol, {}).get("sector")
            if held_sector and held_sector != sector_classification["sector"]:
                errors.append(
                    f"[OrderRouter] Sector classification changed for held position {symbol}"
                )
                continue
            price = float(decision["price"])
            stop_loss = float(decision["stop_loss"])
            target = float(decision["target"])
            if entries_today is None:
                from broker.portfolio_tracker import portfolio_tracker

                entries_today = portfolio_tracker.daily_entry_count()
            if entries_today >= settings.max_daily_entries:
                errors.append(
                    f"[OrderRouter] Daily entry cap reached ({settings.max_daily_entries}): {symbol}"
                )
                continue
            sizing_signal = {
                "symbol": symbol,
                "action": action,
                "price": price,
                "stop_loss": stop_loss,
                "target": target,
                "holding_period": holding_period,
            }
            kelly_inputs: dict[str, float] = {}
            if settings.signal_gate_enabled:
                from models.signal_gate.gate import signal_gate

                gate_signal = {
                    **sizing_signal, "confidence": decision.get("confidence"), "atr": atr,
                }
                gate_ok, gate_reason = signal_gate.should_execute(gate_signal, now_ist)
                if not gate_ok:
                    errors.append(f"[OrderRouter] Signal gate blocked {symbol}: {gate_reason}")
                    continue
                if signal_gate.calibration_passed:
                    payoff = signal_gate.payoff_inputs(gate_signal, now_ist)
                    if payoff is None:
                        errors.append(f"[OrderRouter] Signal gate blocked {symbol}: payoff inputs unavailable")
                        continue
                    kelly_inputs = {"calibrated_p": payoff[0], "payoff_b": payoff[1]}
            max_quantity, sizing_reason = position_sizer.size(
                sizing_signal,
                **kelly_inputs,
                portfolio_value=portfolio_value,
                available_capital=available_capital,
                current_exposure=projected_exposure,
                current_symbol_exposure=projected_symbol_exposure.get(symbol, 0.0),
                risk_multiplier=float(risk_data.get("vix_factor", 1.0)),
                max_position_pct=regime_position_cap,
                remaining_stop_risk=stop_budget,
            )
            if max_quantity <= 0:
                errors.append(
                    f"[OrderRouter] Entry blocked by risk sizing for {symbol}: "
                    f"{sizing_reason or 'no safe quantity'}"
                )
                continue
            quantity = max_quantity
            sizing_error = position_sizer.validate_quantity(
                sizing_signal,
                quantity,
                portfolio_value=portfolio_value,
                available_capital=available_capital,
                current_exposure=projected_exposure,
                current_symbol_exposure=projected_symbol_exposure.get(symbol, 0.0),
                risk_multiplier=float(risk_data.get("vix_factor", 1.0)),
                max_position_pct=regime_position_cap,
                remaining_stop_risk=stop_budget,
            )
            if sizing_error:
                errors.append(
                    f"[OrderRouter] Entry blocked by risk sizing for {symbol}: {sizing_error}"
                )
                continue
            candidate_positions = {
                held_symbol: dict(position)
                for held_symbol, position in projected_positions.items()
            }
            candidate_position = candidate_positions.setdefault(symbol, {"quantity": 0})
            candidate_position["quantity"] = (
                float(candidate_position.get("quantity") or 0)
                + (quantity if action == "BUY" else -quantity)
            )
            candidate_position["ltp"] = price
            candidate_position.update(sector_classification)
            from risk.portfolio_var import historical_portfolio_var_pct
            from risk.sector_exposure import sector_limit_block_reason
            from risk.daily_portfolio_risk import RiskHistoryUnavailable

            try:
                portfolio_var_pct = historical_portfolio_var_pct(
                    candidate_positions, risk_returns, portfolio_value,
                )
                sector_reason = sector_limit_block_reason(
                    candidate_positions, portfolio_value,
                )
            except (RiskHistoryUnavailable, SectorMetadataUnavailable, TypeError, ValueError, OverflowError) as exc:
                errors.append(
                    f"[OrderRouter] Projected portfolio risk unavailable for {symbol}: {exc}"
                )
                continue
            if portfolio_var_pct > settings.max_portfolio_var_pct:
                errors.append(
                    f"[OrderRouter] Projected portfolio VaR {portfolio_var_pct:.2%} exceeds "
                    f"{settings.max_portfolio_var_pct:.2%}: {symbol}"
                )
                continue
            if sector_reason:
                errors.append(f"[OrderRouter] Entry blocked for {symbol}: {sector_reason}")
                continue
            market_bars = state.get("market_data", {}).get(symbol)
            if market_bars is None:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: prior-session liquidity data is unavailable"
                )
                continue
            from data.indicators import average_daily_turnover_inr

            try:
                average_turnover = average_daily_turnover_inr(
                    market_bars,
                    state.get("research_as_of") or None,
                    minimum_sessions=settings.screener_min_liquidity_sessions,
                )
            except (KeyError, TypeError, ValueError):
                average_turnover = None
            if average_turnover is None or average_turnover <= 0:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: prior-session liquidity data is invalid"
                )
                continue
            participation = quantity * price / average_turnover
            if participation > settings.max_order_participation_pct:
                errors.append(
                    f"[OrderRouter] Entry blocked for {symbol}: planned order exceeds "
                    f"{settings.max_order_participation_pct:.2%} of average daily turnover"
                )
                continue
            contract_report = review_contract_results.get(symbol)
            if isinstance(contract_report, dict) and contract_report.get("passed") is not True:
                failed = contract_report.get("failed_requirements")
                failed_text = ", ".join(failed) if isinstance(failed, list) else "unknown requirement"
                errors.append(
                    f"[OrderRouter] Required strategy review contract failed for {symbol}: {failed_text}"
                )
                continue
            decision = {
                **decision,
                "price": price,
                "stop_loss": stop_loss,
                "target": target,
                "quantity": quantity,
                "holding_period": str(decision["holding_period"]).upper(),
                "order_type": str(decision.get("order_type", "LIMIT")).upper(),
            }
        if not settings.paper_trading and not is_exit:
            time_machine = decision.get("time_machine_context") or {}
            bars = time_machine.get("bars")
            if bars is None:
                bars = state.get("five_minute_bars", {}).get(symbol)
            if bars is None:
                errors.append(
                    f"[OrderRouter] Live entry blocked for {symbol}: "
                    "60-session five-minute Time Machine evidence is missing."
                )
                continue
            try:
                from database.hash_ledger import hash_ledger
                from risk.timemachine_validator import timemachine_validator

                simulation = timemachine_validator.validate(
                    bars,
                    entry_price=float(decision.get("price", 0)),
                    target_price=float(decision.get("target", 0)),
                    stop_price=float(decision.get("stop_loss", 0)),
                    quantity=int(quantity),
                    is_intraday=holding_period == "INTRADAY",
                    historical_setup_returns=time_machine.get("historical_setup_returns"),
                    accuracy_score=float(time_machine.get("accuracy_score", 0)),
                    remaining_time_buckets=time_machine.get("remaining_time_buckets"),
                )
                hash_ledger.append("timemachine_simulation", {
                    "symbol": symbol, "approved": simulation.approved,
                    "ev_net": simulation.ev_net,
                    "prob_target_first": simulation.prob_target_first,
                    "prob_stop_first": simulation.prob_stop_first,
                    "prob_open_at_close": simulation.prob_open_at_close,
                    "quality_score": simulation.quality_score,
                })
                if not simulation.approved:
                    errors.append(
                        f"[OrderRouter] Live entry blocked for {symbol}: {simulation.reason}."
                    )
                    continue
            except Exception as exc:
                errors.append(
                    f"[OrderRouter] Live entry blocked for {symbol}: "
                    f"Time Machine validation failed ({type(exc).__name__})."
                )
                continue
        if not is_exit and holding_period == "INTRADAY" and not (
            now_ist.hour > 9 or (now_ist.hour == 9 and now_ist.minute >= 30)
        ):
            errors.append(f"[OrderRouter] Intraday entry blocked before 09:30 IST: {symbol}")
            continue
        if not is_exit and holding_period == "INTRADAY" and (
            now_ist.hour > 14 or (now_ist.hour == 14 and now_ist.minute >= 45)
        ):
            errors.append(f"[OrderRouter] Intraday entry cutoff reached: {symbol}")
            continue

        candidate_metadata: dict[str, Any] = {}
        if not is_exit:
            try:
                trade_candidate = _build_trade_candidate(
                    symbol=symbol, action=action, decision=decision, technical=technical,
                    contract=entry_contract, quantity=quantity, atr_value=atr,
                    tick=tick_size, lot=lot_size, now=now_ist,
                )
            except CandidateInvalid as exc:
                errors.append(f"[OrderRouter] Candidate contract invalid for {symbol}: {exc}")
                continue
            candidate_metadata = {
                "strategy_version": trade_candidate.strategy_version,
                "exit_policy": trade_candidate.exit_policy,
                "atr_timeframe": trade_candidate.atr_timeframe,
                "book": entry_contract.book or "core",
            }

        use_intelligent_router = (
            currency != "INR"
            or (not settings.paper_trading and settings.trademax_broker_mode != "angel_only")
        )
        if use_intelligent_router:
            if order_router is None:
                errors.append(f"[OrderRouter] No configured router for {symbol}")
                continue
            try:
                report = order_router.execute_order(
                    symbol=symbol, action=action, quantity=quantity,
                    price=price, currency=currency,
                    exchange=decision.get("exchange", "NSE" if currency == "INR" else ""),
                    product_type=_product_for(holding_period),
                )
                if report.status in {"ERROR", "SUBMISSION_UNCERTAIN", "REJECTED", "PAPER_UNSUPPORTED"}:
                    errors.append(
                        f"[OrderRouter] {symbol}: {report.status}; order id={report.order_id or 'unknown'}"
                    )
                elif report.status not in {"EXECUTED", "COMPLETE", "FILLED", "PARTIAL"}:
                    errors.append(
                        f"[OrderRouter] {symbol}: order submitted but not yet confirmed filled "
                        f"(status={report.status}, order_id={report.order_id or 'unknown'})"
                    )
                else:
                    routed_order = {
                        "symbol": symbol, "action": action,
                        "quantity": report.filled_qty or quantity,
                        "fill_price": report.avg_price or price, "broker": report.broker,
                        "algo": report.algo, "slippage_bps": report.slippage_bps,
                        "order_id": report.order_id, "status": report.status,
                        "is_exit": is_exit, "atr": atr,
                        "horizon": holding_period, **candidate_metadata,
                    }
                    executed.append(routed_order)
                    if not is_exit:
                        filled_quantity = abs(int(report.filled_qty or quantity))
                        fill_price = float(report.avg_price or price)
                        track_projected_stop_risk(sizing_signal, filled_quantity, fill_price)
                        entries_today = (entries_today or 0) + 1
                        notional = filled_quantity * fill_price
                        projected_entry_exposure[symbol] = (
                            projected_entry_exposure.get(symbol, 0.0) + notional
                        )
                        projected_entry_quantities[symbol] = (
                            projected_entry_quantities.get(symbol, 0.0)
                            + (filled_quantity if action == "BUY" else -filled_quantity)
                        )
                        projected_entry_prices[symbol] = fill_price
                        projected_entry_sectors[symbol] = sector_classification
                        projected_cash_committed += notional
                    logs.append(
                        f"[OrderRouter] {action} {quantity}x {symbol} via {report.broker} "
                        f"(status={report.status})"
                    )
            except Exception as exc:
                errors.append(
                    f"[OrderRouter] {symbol} execution outcome requires reconciliation: "
                    f"{type(exc).__name__}"
                )
            continue

        if currency != "INR":
            errors.append(f"[OrderRouter] No safe broker route available for {symbol}")
            continue

        # Default Angel One path preserves existing paper/live order accounting.
        token = get_token(symbol)
        if not token:
            errors.append(f"[OrderRouter] Token not found for {symbol}")
            continue
        if settings.paper_trading and not is_exit and expected_paper_book is None:
            errors.append(f"[OrderRouter] Paper position identity unavailable for {symbol}")
            continue

        try:
            result = place_order(
                symbol=symbol, token=token, action=action,
                quantity=quantity,
                order_type=decision.get("order_type", "LIMIT"),
                price=price,
                stop_loss=decision.get("stop_loss"),
                target=decision.get("target"),
                product_type=_product_for(
                    (position.get("holding_period") if is_exit else None) or decision.get("holding_period"),
                ),
                horizon=str(decision["holding_period"]).upper() if not is_exit else None,
                **candidate_metadata,
                strategy=decision.get("strategy", "agent_pipeline"),
                confidence=decision.get("confidence", 0),
                reasoning=decision.get("reasoning", ""),
                allow_requested_price_reference=False,
                atr=atr,
                is_exit=is_exit,
                expected_entry_trade_ids=position.get("entry_trade_ids") if is_exit else None,
                expected_paper_book=(
                    position_book_identity(expected_paper_book)
                    if not is_exit and expected_paper_book is not None else None
                ),
                sector=sector_classification["sector"] if not is_exit else None,
                sector_source=sector_classification["sector_source"] if not is_exit else None,
                sector_observed_at=sector_classification["sector_observed_at"] if not is_exit else None,
            )
            if result:
                result["is_exit"] = is_exit
                executed.append(result)
                if not is_exit:
                    filled_quantity = abs(int(result.get("quantity") or quantity))
                    fill_price = float(result.get("fill_price") or price)
                    if settings.paper_trading and expected_paper_book is not None:
                        trade_id = result.get("trade_id")
                        if type(trade_id) is int and trade_id > 0:
                            book_position = expected_paper_book.setdefault(symbol, {
                                "quantity": 0, "entry_trade_ids": [], "holding_period": holding_period,
                            })
                            book_position["quantity"] += filled_quantity if action == "BUY" else -filled_quantity
                            book_position["entry_trade_ids"].append(trade_id)
                        else:
                            expected_paper_book = None
                            errors.append(f"[OrderRouter] Paper fill identity unavailable after {symbol}")
                    track_projected_stop_risk(sizing_signal, filled_quantity, fill_price)
                    entries_today = (entries_today or 0) + 1
                    notional = filled_quantity * fill_price
                    projected_entry_exposure[symbol] = (
                        projected_entry_exposure.get(symbol, 0.0) + notional
                    )
                    projected_entry_quantities[symbol] = (
                        projected_entry_quantities.get(symbol, 0.0)
                        + (filled_quantity if action == "BUY" else -filled_quantity)
                    )
                    projected_entry_prices[symbol] = fill_price
                    projected_entry_sectors[symbol] = sector_classification
                    projected_cash_committed += notional
                logs.append(
                    f"[OrderRouter] {'PAPER' if result.get('is_paper') else 'LIVE'} "
                    f"{action} {quantity}x {symbol} @ ₹{result.get('fill_price', price)}"
                )
            else:
                errors.append(f"[OrderRouter] Order not filled for {symbol} (see paper fill audit / logs)")
        except Exception as exc:
            errors.append(f"[OrderRouter] {symbol} execution error: {exc}")

    return {
        "executed_orders": executed, "agent_logs": logs, "errors": errors,
        "review_contract_results": review_contract_results,
    }


def _validate_entry_decision(
    decision: dict[str, Any], action: str, *, require_limit_order: bool = True,
    allowed_horizons: tuple[str, ...] = HORIZONS,
) -> str | None:
    """Validate all model-authored entry fields before they reach a broker."""
    holding_period = str(decision.get("holding_period", "")).upper()
    if holding_period not in HORIZONS or holding_period not in allowed_horizons:
        return f"holding_period must be one of {', '.join(allowed_horizons)}"
    order_type = str(decision.get("order_type", "LIMIT")).upper()
    if order_type not in {"LIMIT", "MARKET"}:
        return "order_type must be LIMIT or MARKET"
    if require_limit_order and order_type != "LIMIT":
        return "new entries require LIMIT orders"
    values: dict[str, float] = {}
    for field in ("price", "stop_loss", "target"):
        raw_value = decision.get(field)
        if isinstance(raw_value, bool):
            return f"{field} must be a positive finite number"
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return f"{field} must be a positive finite number"
        if not math.isfinite(value) or value <= 0:
            return f"{field} must be a positive finite number"
        values[field] = value

    price = values["price"]
    stop_loss = values["stop_loss"]
    target = values["target"]
    if action == "BUY" and not stop_loss < price < target:
        return "BUY requires stop_loss < price < target"
    if action == "SELL" and not target < price < stop_loss:
        return "SELL requires target < price < stop_loss"
    return None


def _episodic_reflection_node(state: TradingState) -> dict[str, Any]:
    """Post-execution episodic reflection."""
    if state.get("research_mode"):
        return {"agent_logs": ["[EpisodicReflection] Skipped for isolated research."]}
    try:
        from agents.episodic_reflection_agent import episodic_reflection_agent as reflect_fn
        return reflect_fn(state)
    except Exception as e:
        return {"agent_logs": [f"[EpisodicReflection] Error: {e}"], "errors": [str(e)]}


def build_trading_graph(checkpointer: Any = None, *, research: bool = False) -> Any:
    """Build and compile the LangGraph trading pipeline (v2.0).

    Flow:
        market_data → [kafka_events, l2_order_flow] (parallel data ingestion)
        → hmm_regime (regime classification)
        → shared_news → [technical, fundamental, sentiment, options, macro → geopolitical_nlp, india_specific]
          (parallel analysis; geopolitical runs after macro so it sees this scan's macro data)
        → bull_bear_debate (adversarial synthesis)
        → risk_manager
        → [marl_optimizer, delta_hedger] (parallel portfolio optimization)
        → compliance_check (FEMA/LRS gate)
        → portfolio_manager
        → order_router (intelligent multi-broker execution)
        → episodic_reflection (post-trade learning)
        → END
    """
    if StateGraph is None:
        logger.error("langgraph not installed. Cannot build graph.")
        return None

    workflow = StateGraph(TradingState)

    # ── Data Ingestion Layer ─────────────────────────────
    workflow.add_node("market_data", market_data_agent)
    workflow.add_node("kafka_events", _kafka_events_node)
    workflow.add_node("l2_order_flow", _l2_order_flow_node)

    # ── Regime Classification ────────────────────────────
    workflow.add_node("hmm_regime", _hmm_regime_node)

    # ── Quantitative Screener (pre-LLM filter) ──────────
    workflow.add_node("quant_screener", quantitative_screener_node)
    workflow.add_node("strategy_candidates", strategy_candidates_node)
    workflow.add_node("shared_news", _shared_news_node)

    # ── Analyst Agents (parallel fan-out) ────────────────
    workflow.add_node("technical", technical_agent)
    workflow.add_node("fundamental", fundamental_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("options", options_agent)
    workflow.add_node("macro", macro_agent)
    workflow.add_node("india_specific", india_specific_agent)
    workflow.add_node("geopolitical_nlp", _geopolitical_nlp_node)

    # ── Synthesis Layer ──────────────────────────────────
    workflow.add_node("bull_bear_debate", _bull_bear_debate_node)

    # ── Risk + Optimization ──────────────────────────────
    workflow.add_node("risk_manager", risk_manager_agent)
    workflow.add_node("marl_optimizer", _marl_optimizer_node)
    workflow.add_node("delta_hedger", _delta_hedger_node)

    # ── Compliance ───────────────────────────────────────
    workflow.add_node("compliance_check", _compliance_check_node)

    # ── Final Decision + Execution ───────────────────────
    workflow.add_node("portfolio_manager", portfolio_manager_agent)
    workflow.add_node("order_router", _order_router_node)
    workflow.add_node("episodic_reflection", _episodic_reflection_node)

    # ── Entry ────────────────────────────────────────────
    workflow.set_entry_point("quant_screener" if research else "market_data")

    # ── Edges ────────────────────────────────────────────

    # market_data → parallel data enrichment
    workflow.add_edge("market_data", "kafka_events")
    workflow.add_edge("market_data", "l2_order_flow")

    # Data enrichment → HMM regime
    workflow.add_edge("kafka_events", "hmm_regime")
    workflow.add_edge("l2_order_flow", "hmm_regime")

    # HMM regime → quant screener (pre-LLM filter)
    workflow.add_edge("hmm_regime", "quant_screener")

    # All analysts consume the same news snapshot.
    workflow.add_edge("quant_screener", "strategy_candidates")
    workflow.add_edge("strategy_candidates", "shared_news")
    workflow.add_edge("shared_news", "technical")
    workflow.add_edge("shared_news", "fundamental")
    workflow.add_edge("shared_news", "sentiment")
    workflow.add_edge("shared_news", "options")
    workflow.add_edge("shared_news", "macro")
    workflow.add_edge("shared_news", "india_specific")
    # Geopolitical analysis reads this scan's macro data, so it runs after macro.
    workflow.add_edge("macro", "geopolitical_nlp")

    # Fan-in: the debate waits for every analyst, including the chained geopolitical node.
    workflow.add_edge(
        ["technical", "fundamental", "sentiment", "options", "india_specific", "geopolitical_nlp"],
        "bull_bear_debate",
    )

    # Debate → risk_manager → parallel optimization
    workflow.add_edge("bull_bear_debate", "risk_manager")
    workflow.add_edge("risk_manager", "marl_optimizer")
    workflow.add_edge("risk_manager", "delta_hedger")

    # Optimization → compliance → portfolio_manager → execution → reflection → END
    workflow.add_edge("marl_optimizer", "compliance_check")
    workflow.add_edge("delta_hedger", "compliance_check")
    workflow.add_edge("compliance_check", "portfolio_manager")
    workflow.add_edge("portfolio_manager", "order_router")
    workflow.add_edge("order_router", "episodic_reflection")
    workflow.add_edge("episodic_reflection", END)

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Trading graph v2.1 compiled successfully ({} nodes).", 22)
    return graph


@llm_operation("scan")
async def run_trading_pipeline(
    watchlist: list[str] | None = None, *,
    execute_orders: bool = True,
    on_execution_complete: Callable[[TradingState], None] | None = None,
) -> TradingState:
    """Run the full trading pipeline once.

    Args:
        watchlist: Optional override of default watchlist.
        execute_orders: Whether approved decisions may reach the order router.
        on_execution_complete: Track completed orders before research capture.

    Returns:
        Final TradingState after all agents have run.
    """
    if watchlist is None:
        from config.universe import get_active_watchlist

        watchlist = get_active_watchlist()
    initial_state = create_initial_state(watchlist)
    initial_state["execution_enabled"] = execute_orders
    initial_state["scan_started_monotonic"] = time.monotonic()
    context = current_context()
    initial_state["llm_profile"] = {
        "mode": context.profile.mode, "revision": context.profile.revision,
        "run_id": context.run_id, "fast_model": context.profile.fast_model,
        "decision_model": context.profile.decision_model,
    }

    logger.info("=" * 60)
    logger.info("TRADING PIPELINE START — {} symbols", len(initial_state["watchlist"]))
    logger.info("=" * 60)

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from agents.checkpoint_serializer import TradingCheckpointSerializer

        checkpoint_path = Path(__file__).resolve().parents[1] / "database" / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(checkpoint_path / "trading.sqlite")) as connection:
            checkpointer = AsyncSqliteSaver(connection, serde=TradingCheckpointSerializer())
            graph = build_trading_graph(checkpointer)
            if graph is None:
                raise RuntimeError("LangGraph runtime is unavailable")
            checkpoint_id = f"{initial_state['date']}:{context.run_id}"
            final_state = await graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": checkpoint_id}},
            )

        try:
            if on_execution_complete is not None:
                on_execution_complete(final_state)
        finally:
            if settings.paper_trading:
                try:
                    from research.tracking import record_scan
                    capture = await asyncio.to_thread(record_scan, final_state, context.run_id)
                    final_state["research_capture"] = capture
                    if capture["snapshot_status"] != "complete":
                        logger.warning("[Research] Snapshot incomplete: {}", capture.get("problems", []))
                except Exception as exc:
                    message = f"Research capture failed ({type(exc).__name__})"
                    logger.error("[Research] {}", message)
                    final_state["research_capture"] = {"snapshot_status": "error", "error": message}

        # Log summary
        decisions = final_state.get("trading_decisions", [])
        executed = final_state.get("executed_orders", [])
        errs = final_state.get("errors", [])

        logger.info("Pipeline complete: {} decisions, {} executed, {} errors", len(decisions), len(executed), len(errs))

        if errs:
            mark_degraded()
            for e in errs:
                logger.warning("Pipeline error: {}", e)

        return final_state

    except Exception as exc:
        logger.exception("Trading pipeline failed: {}", exc)
        mark_degraded()
        raise
