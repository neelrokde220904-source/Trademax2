"""Risk Manager Agent — Kelly Criterion, VaR, drawdown, position limits, correlation check."""

from __future__ import annotations

import math
from typing import Any

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings
from risk.daily_portfolio_risk import (
    RiskHistoryUnavailable, correlation_block_reason, daily_returns, historical_var,
)
from risk.position_sizer import position_sizer


class RiskManagerAgent(BaseAgent):
    """Evaluates risk for proposed trades and enforces hard limits."""

    def __init__(self) -> None:
        super().__init__("RiskManagerAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        # Gather all signals
        tech = state.get("technical_signals", {})
        fund = state.get("fundamental_signals", {})
        sent = state.get("sentiment_signals", {})
        opts = state.get("options_signals", {})
        macro = state.get("macro_signals", {})
        india = state.get("india_signals", {})

        india_vix = state.get("india_vix", 0)
        portfolio_value = state.get("portfolio_value", settings.initial_capital)
        available_capital = state.get("available_capital", settings.initial_capital)
        current_positions = state.get("current_positions", {})
        market_data = state.get("market_data", {})

        # --- Global Risk Checks ---
        # Daily loss limit
        if state.get("research_mode"):
            daily_loss_hit = state["research_daily_loss_hit"]
        else:
            from broker.portfolio_tracker import portfolio_tracker
            daily_loss_hit = portfolio_tracker.has_hit_daily_loss_limit()
        if daily_loss_hit:
            logs.append(self.log("⛔ DAILY LOSS LIMIT HIT — blocking all new trades."))
            return {
                "risk_assessment": {"blocked": True, "reason": "Daily loss limit exceeded"},
                "agent_logs": logs,
                "errors": errors,
            }
        if not state.get("research_mode"):
            from broker.portfolio_tracker import portfolio_tracker
            if portfolio_tracker.has_hit_portfolio_drawdown_limit():
                reason = "Persistent portfolio drawdown limit reached"
                logs.append(self.log("⛔ {} — blocking all new trades.", reason))
                return {
                    "risk_assessment": {"blocked": True, "reason": reason},
                    "agent_logs": logs,
                    "errors": errors,
                }

        # Max positions
        if len(current_positions) >= settings.max_positions:
            logs.append(self.log("⛔ MAX POSITIONS ({}) reached — no new entries.", settings.max_positions))
            return {
                "risk_assessment": {"blocked": True, "reason": f"Max positions ({settings.max_positions}) reached"},
                "agent_logs": logs,
                "errors": errors,
            }

        # VIX check
        if not isinstance(india_vix, (int, float)) or not math.isfinite(india_vix) or india_vix <= 0:
            message = "India VIX is unavailable or invalid; new entries blocked"
            logs.append(self.log(message))
            return {
                "risk_assessment": {"blocked": True, "reason": message},
                "agent_logs": logs, "errors": [message],
            }
        vix_factor = 1.0
        if india_vix > settings.vix_halt_threshold:
            logs.append(self.log("⛔ VIX ({:.1f}) > halt threshold ({}) — halting trades.", india_vix, settings.vix_halt_threshold))
            return {
                "risk_assessment": {"blocked": True, "reason": f"VIX too high: {india_vix}"},
                "agent_logs": logs,
                "errors": errors,
            }
        elif india_vix > settings.max_nifty_vix:
            vix_factor = 0.5
            logs.append(self.log("⚠️ VIX ({:.1f}) elevated — reducing position sizes by 50%.", india_vix))

        if not state.get("research_mode") and state.get("portfolio_snapshot_valid") is not True:
            message = "Portfolio synchronization or fresh position marks unavailable; new entries blocked"
            logs.append(self.log(message))
            return {
                "risk_assessment": {"blocked": True, "reason": message},
                "agent_logs": logs, "errors": [message],
            }

        regime_position_cap = None
        portfolio_stop_budget = None
        if not state.get("research_mode"):
            from risk.hmm_regime_classifier import entry_regime_position_cap
            from risk.portfolio_stop_risk import (
                StopRiskUnavailable, open_stop_risk, remaining_stop_risk,
            )
            from risk.stop_loss_manager import stop_loss_manager
            try:
                regime_position_cap = entry_regime_position_cap(state.get("hmm_regime"))
            except ValueError as exc:
                message = f"HMM position limit unavailable; new entries blocked: {exc}"
                return {
                    "risk_assessment": {"blocked": True, "reason": message},
                    "agent_logs": [self.log(message)], "errors": [message],
                }
            try:
                portfolio_stop_budget = remaining_stop_risk(
                    portfolio_value,
                    open_stop_risk(current_positions, stop_loss_manager.snapshot()),
                )
            except StopRiskUnavailable as exc:
                message = f"Portfolio stop risk unavailable; new entries blocked: {exc}"
                return {
                    "risk_assessment": {"blocked": True, "reason": message},
                    "agent_logs": [self.log(message)], "errors": [message],
                }

        # --- Per-Symbol Risk Assessment ---
        risk_assessment: dict[str, Any] = {"blocked": False}

        # Only assess symbols that passed the upstream quantitative screener.
        eligible_symbols = set(state.get("watchlist", []))
        all_symbols = eligible_symbols.intersection(set(tech) | set(fund) | set(sent))
        history_cache: dict[str, Any] = {}
        as_of = state.get("research_as_of") if state.get("research_mode") else None

        def returns_for(symbol: str):
            if symbol not in history_cache:
                history_cache[symbol] = daily_returns(market_data.get(symbol), as_of or None)
            return history_cache[symbol]

        for symbol in sorted(all_symbols):
            try:
                # Collect signals
                t_sig = tech.get(symbol, {})

                # Skip if India-specific says AVOID
                i_sig = india.get(symbol, {})
                if i_sig.get("signal") == "AVOID":
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": i_sig.get("reasoning", "India-specific block"),
                    }
                    continue

                # The deterministic technical signal is the entry gate; other agents
                # may corroborate it but cannot create a trade direction.
                technical_signal = str(t_sig.get("signal", "")).upper()
                if technical_signal in {"BUY", "STRONG_BUY"}:
                    action = "BUY"
                elif technical_signal in {"SELL", "STRONG_SELL"}:
                    action = "SELL"
                else:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": "No deterministic technical entry signal",
                    }
                    continue

                if not state.get("research_mode"):
                    from risk.hmm_regime_classifier import entry_regime_strategy_block_reason
                    strategy_reason = entry_regime_strategy_block_reason(
                        state.get("hmm_regime"), t_sig.get("strategy"),
                    )
                    if strategy_reason:
                        risk_assessment[symbol] = {
                            "approved": False, "rejection_reason": strategy_reason,
                        }
                        logs.append(self.log("{}: REJECTED — {}", symbol, strategy_reason))
                        continue

                try:
                    candidate_returns = returns_for(symbol)
                    var_1day = historical_var(candidate_returns, action)
                    correlation_reason = correlation_block_reason(
                        symbol, action, current_positions, returns_for,
                    )
                except RiskHistoryUnavailable as exc:
                    reason = f"Daily risk history unavailable: {exc}"
                    risk_assessment[symbol] = {"approved": False, "rejection_reason": reason}
                    logs.append(self.log("{}: REJECTED — {}", symbol, reason))
                    continue

                sector_classification: dict[str, str] | None = None
                if not state.get("research_mode"):
                    from risk.sector_exposure import (
                        SectorMetadataUnavailable, validate_candidate_sector,
                    )

                    try:
                        sector_inputs = state.get("research_fundamental_inputs")
                        sector_classification = validate_candidate_sector(
                            symbol,
                            sector_inputs.get(symbol) if isinstance(sector_inputs, dict) else None,
                        )
                    except (SectorMetadataUnavailable, TypeError, ValueError) as exc:
                        reason = f"Sector classification unavailable: {exc}"
                        risk_assessment[symbol] = {
                            "approved": False, "rejection_reason": reason,
                        }
                        logs.append(self.log("{}: REJECTED — {}", symbol, reason))
                        continue

                # Check VaR limit
                if var_1day > settings.max_portfolio_var_pct:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": (
                            f"VaR too high: {var_1day:.4f} "
                            f"(limit: {settings.max_portfolio_var_pct:.2%})"
                        ),
                        "var_1day": var_1day,
                    }
                    logs.append(self.log(
                        "{}: REJECTED — VaR {:.2%} > {:.2%}",
                        symbol, var_1day, settings.max_portfolio_var_pct,
                    ))
                    continue

                if correlation_reason:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": correlation_reason,
                    }
                    logs.append(self.log("{}: REJECTED — high correlation", symbol))
                    continue

                # Use deterministic technical levels; missing ATR-derived levels fail closed.
                stop_loss = t_sig.get("stop_loss", 0)
                target = t_sig.get("target", 0)
                close = t_sig.get("indicators", {}).get("close", 0)
                quote_price = (state.get("live_quotes", {}).get(symbol) or {}).get("ltp", 0)
                if not close and quote_price:
                    close = quote_price

                if not stop_loss or not target:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": "Daily-ATR stop/target unavailable",
                    }
                    logs.append(self.log("{}: REJECTED — daily-ATR stop/target unavailable", symbol))
                    continue

                exposure_by_symbol = position_sizer.exposure_by_symbol(current_positions)
                current_exposure = sum(exposure_by_symbol.values())
                current_symbol_exposure = exposure_by_symbol.get(symbol, 0.0)
                max_qty, sizing_reason = position_sizer.size(
                    {
                        "symbol": symbol,
                        "action": action,
                        "price": close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "holding_period": "INTRADAY",
                    },
                    portfolio_value=portfolio_value,
                    available_capital=available_capital,
                    current_exposure=current_exposure,
                    current_symbol_exposure=current_symbol_exposure,
                    risk_multiplier=vix_factor,
                    max_position_pct=regime_position_cap,
                    remaining_stop_risk=portfolio_stop_budget,
                )
                if max_qty <= 0:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": sizing_reason or "No safe position size",
                    }
                    logs.append(self.log("{}: REJECTED — {}", symbol, sizing_reason))
                    continue
                max_position_value = max_qty * float(close)

                # Risk score (0-100, lower is better)
                risk_score = self._compute_risk_score(var_1day, india_vix, len(current_positions))

                risk_assessment[symbol] = {
                    "approved": True,
                    "max_quantity": max_qty,
                    "max_position_value": round(max_position_value, 2),
                    "stop_loss": stop_loss,
                    "target": target,
                    "deterministic_action": action,
                    "holding_period": str(t_sig.get("holding_period", "INTRADAY")).upper(),
                    "risk_score": risk_score,
                    "var_1day": round(var_1day, 4),
                    "vix_factor": vix_factor,
                    "rejection_reason": None,
                    **(sector_classification or {}),
                }

                logs.append(
                    self.log(
                        "{}: APPROVED — max qty {}, SL ₹{}, risk score {}/100",
                        symbol, max_qty, stop_loss, risk_score,
                    )
                )

            except Exception as exc:
                errors.append(self.log("{}: Risk assessment error — {}", symbol, exc))

        return {
            "risk_assessment": risk_assessment,
            "agent_logs": logs,
            "errors": errors,
        }

    def _compute_risk_score(self, var: float, vix: float, n_positions: int) -> int:
        """Compute a 0–100 risk score (lower = safer)."""
        score = 0
        score += int(var * 1000)  # VaR contribution
        score += int(vix * 1.5)  # VIX contribution
        score += n_positions * 5  # Position count contribution
        return min(100, max(0, score))


def risk_manager_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return RiskManagerAgent().run(state)
