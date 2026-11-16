"""Explicit, isolated paper comparisons on previously captured scan inputs."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from agents.base_agent import create_initial_state
from agents.graph import _validate_entry_decision, build_trading_graph
from backtester.engine import BacktestEngine
from config.llm_runtime import Profile, configured, llm_run, profile_for
from config.settings import settings
from data.nse_calendar import MUHURAT_SESSION, REGULAR_WEEKEND_SESSIONS, is_nse_holiday
from database.db import get_db
from database.models import LLMRun, LLMUsage, ResearchComparison
from research.tracking import load_snapshot
from risk.hmm_regime_classifier import (
    entry_regime_block_reason, entry_regime_position_cap, entry_regime_strategy_block_reason,
)

IST = ZoneInfo("Asia/Kolkata")
BAR_LENGTH = timedelta(minutes=15)
MODES = ("claude", "economy", "rules")


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _next_bar(observed: datetime) -> pd.Timestamp:
    local = observed.astimezone(IST)
    opening = local.replace(hour=9, minute=15, second=0, microsecond=0)
    intervals = max(0, int((local - opening).total_seconds() // 900) + 1)
    return pd.Timestamp(opening + timedelta(minutes=15 * intervals))


def _session_bars(after: datetime, through: datetime) -> list[pd.Timestamp]:
    bars = []
    day = after.astimezone(IST).date()
    last_day = through.astimezone(IST).date()
    while day <= last_day:
        if day.year != 2026:
            raise ValueError("NSE replay calendar is verified for 2026 only")
        if day == MUHURAT_SESSION:
            session_start = datetime.combine(day, time(14, 20), IST)
            session_end = datetime.combine(day, time(15, 35), IST)
            if after < session_end and through > session_start:
                raise ValueError("Muhurat replay requires a verified special-session exit policy")
        if (day.weekday() < 5 and not is_nse_holiday(day)) or day in REGULAR_WEEKEND_SESSIONS:
            opening = datetime.combine(day, time(9, 15), IST)
            for interval in range(25):
                stamp = pd.Timestamp(opening + interval * BAR_LENGTH)
                if pd.Timestamp(after) < stamp + BAR_LENGTH <= pd.Timestamp(through):
                    bars.append(stamp)
        day += timedelta(days=1)
    return bars


def _run_research_pipeline(state: dict[str, Any], profile: Profile) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        with llm_run("research_comparison", profile=profile) as context:
            graph = build_trading_graph(research=True)
            if graph is None:
                raise RuntimeError("Research graph is unavailable")
            result = await graph.ainvoke(state)
            with get_db() as db:
                record = db.get(LLMRun, context.run_id)
                if record is None or record.degraded:
                    raise RuntimeError(f"{profile.mode} research model run degraded")
                if profile.mode != "rules" and not db.query(LLMUsage).filter(
                    LLMUsage.run_id == context.run_id, LLMUsage.status == "success",
                ).count():
                    raise RuntimeError(f"{profile.mode} made no successful model calls for this snapshot")
            if result.get("errors"):
                raise RuntimeError(f"{profile.mode} research pipeline reported errors: {result['errors']}")
            return result
    return asyncio.run(run())


class _PaperBook:
    """Uses the existing cost and risk sizer without accessing an order router."""

    def __init__(self) -> None:
        self.engine = BacktestEngine()
        self.pending: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
        self.seen: set[tuple[str, pd.Timestamp]] = set()
        self.first_observed: datetime | None = None
        self.day: Any = None
        self.day_start_equity = self.engine.initial_capital
        self.loss_halted = False
        self.audit: list[dict[str, Any]] = []
        self.last_data: dict[str, pd.DataFrame] = {}
        self.last_as_of: datetime | None = None

    def _equity(self, data: dict[str, pd.DataFrame], at: datetime) -> float:
        equity = self.engine._capital
        for symbol, trade in self.engine._positions.items():
            mark = self._mark(symbol, trade, data, at)
            if trade.action == "BUY":
                equity += mark * trade.quantity
            else:
                equity += trade.entry_price * trade.quantity + (trade.entry_price - mark) * trade.quantity
        return equity

    @staticmethod
    def _mark(symbol: str, trade: Any, data: dict[str, pd.DataFrame], at: datetime) -> float:
        frame = data.get(symbol)
        if frame is None:
            raise ValueError(f"Captured market data missing for open position {symbol}")
        complete = frame[frame.index + BAR_LENGTH <= pd.Timestamp(at)]
        complete = complete[complete.index >= pd.Timestamp(trade.entry_date)]
        mark = _positive(complete.iloc[-1].get("close")) if not complete.empty else trade.entry_price
        if mark is None:
            raise ValueError(f"Invalid captured mark for {symbol}")
        return mark

    def _roll_day(self, day: Any, data: dict[str, pd.DataFrame], at: datetime) -> None:
        if self.day == day:
            return
        if any(trade.holding_period == "INTRADAY" for trade in self.engine._positions.values()):
            raise ValueError("A required 15:15 square-off bar was not captured")
        self.day = day
        self.day_start_equity = self._equity(data, at)
        self.loss_halted = False
        self.engine._entries_today = 0

    def advance(self, snapshot: dict[str, Any]) -> None:
        observed = datetime.fromisoformat(snapshot["research_as_of"]).astimezone(timezone.utc)
        if self.last_as_of is not None and observed <= self.last_as_of:
            raise ValueError("Research snapshots must have strictly increasing observation times")
        local_observed = observed.astimezone(IST)
        if local_observed.date() == MUHURAT_SESSION and time(14, 20) <= local_observed.time() < time(15, 35):
            raise ValueError("Muhurat replay requires a verified special-session exit policy")
        if local_observed.year != 2026:
            raise ValueError("NSE replay calendar is verified for 2026 only")
        data = BacktestEngine._normalize_data(snapshot["market_data"])
        if self.first_observed is None:
            self.first_observed = observed
            self.day = observed.astimezone(IST).date()
            self.seen = {
                (symbol, stamp) for symbol, frame in data.items() for stamp in frame.index
                if stamp + BAR_LENGTH <= pd.Timestamp(observed)
            }
            self.last_data, self.last_as_of = data, observed
            return

        events: dict[pd.Timestamp, set[str]] = {}
        for symbol, frame in data.items():
            for stamp in frame.index:
                key = (symbol, stamp)
                completed = stamp + BAR_LENGTH
                if key in self.seen or completed > pd.Timestamp(observed):
                    continue
                if completed <= pd.Timestamp(self.last_as_of):
                    self.seen.add(key)
                    local = stamp.tz_convert(IST)
                    if completed > pd.Timestamp(self.first_observed) and (
                        time(9, 15) <= local.time() <= time(15, 15)
                    ):
                        self.audit.append({"symbol": symbol, "status": "late_bar_ignored", "bar": stamp.isoformat()})
                    continue
                events.setdefault(stamp, set()).add(symbol)
        expected_bars = set(_session_bars(self.last_as_of, observed))
        for stamp in events:
            if time(9, 15) <= stamp.tz_convert(IST).time() <= time(15, 15) and stamp not in expected_bars:
                raise ValueError(f"Unexpected NSE session bar at {stamp.isoformat()}")
        for stamp in expected_bars:
            events.setdefault(stamp, set())
        for stamp in sorted(events):
            local = stamp.tz_convert(IST)
            if not (time(9, 15) <= local.time() <= time(15, 15)):
                for symbol in sorted(events[stamp]):
                    self.seen.add((symbol, stamp))
                continue
            self._roll_day(local.date(), data, stamp.to_pydatetime())
            for symbol, trade in self.engine._positions.items():
                if symbol not in events[stamp]:
                    if local.time() == time(15, 15) and trade.holding_period == "INTRADAY":
                        raise ValueError(f"Missing 15:15 square-off bar for {symbol}")
                    raise ValueError(f"Missing 15-minute bar for open position {symbol} at {stamp.isoformat()}")
            for symbol in sorted(events[stamp]):
                row = data[symbol].loc[stamp]
                opening, high, low, close = (_positive(row.get(field)) for field in ("open", "high", "low", "close"))
                if opening is None or high is None or low is None or close is None:
                    raise ValueError(f"Invalid captured OHLC bar for {symbol} at {stamp.isoformat()}")
                if low > min(opening, close) or high < max(opening, close) or low > high:
                    raise ValueError(f"Inconsistent captured OHLC bar for {symbol} at {stamp.isoformat()}")

            if local.time() == time(15, 15):
                for symbol, trade in list(self.engine._positions.items()):
                    if trade.holding_period != "INTRADAY":
                        continue
                    if symbol not in events[stamp]:
                        raise ValueError(f"Missing 15:15 square-off bar for {symbol}")
                    self.engine._close_trade(symbol, float(data[symbol].loc[stamp, "open"]),
                                             stamp.to_pydatetime(), "INTRADAY_SQUARE_OFF")

            for symbol in sorted(events[stamp]):
                pending = self.pending.pop((symbol, stamp), None)
                if pending is None:
                    continue
                opening = float(data[symbol].loc[stamp, "open"])
                limit = pending.get("limit_price")
                if limit is not None and (
                    (pending["action"] == "BUY" and opening > limit)
                    or (pending["action"] == "SELL" and opening < limit)
                ):
                    self.audit.append({"symbol": symbol, "status": "limit_not_marketable", "bar": stamp.isoformat()})
                    continue
                if pending["kind"] == "exit":
                    if symbol in self.engine._positions:
                        self.engine._close_trade(
                            symbol, opening, stamp.to_pydatetime(), "MODEL_EXIT", limit_price=limit,
                        )
                        self.audit.append({"symbol": symbol, "status": "filled_exit", "bar": stamp.isoformat()})
                    continue
                if self.loss_halted:
                    self.audit.append({"symbol": symbol, "status": "daily_loss_blocked", "bar": stamp.isoformat()})
                    continue
                before = symbol in self.engine._positions
                equity = self.engine._calculate_equity(data, stamp.to_pydatetime(), price_field="open")
                exposure = self.engine._current_exposure(data, stamp.to_pydatetime(), price_field="open")
                self.engine._process_signal(
                    {**pending["signal"], "price": opening}, stamp.to_pydatetime(), equity, exposure,
                )
                filled = not before and symbol in self.engine._positions
                self.audit.append({
                    "symbol": symbol, "status": "filled" if filled else "risk_or_fill_rejected",
                    "bar": stamp.isoformat(),
                })
            self.engine._check_exits(data, stamp.to_pydatetime())
            equity = self._equity(data, (stamp + BAR_LENGTH).to_pydatetime())
            if equity <= self.day_start_equity * (1 - settings.max_daily_loss_pct):
                self.loss_halted = True
            self.seen.update((symbol, stamp) for symbol in events[stamp])

        for key in list(self.pending):
            symbol, expected = key
            if expected + BAR_LENGTH <= pd.Timestamp(observed):
                self.pending.pop(key)
                self.audit.append({
                    "symbol": symbol, "status": "missing_next_bar", "bar": expected.isoformat(),
                })
        if self.day != observed.astimezone(IST).date():
            self._roll_day(observed.astimezone(IST).date(), data, observed)
        if observed.astimezone(IST).time() >= time(15, 30) and any(
            trade.holding_period == "INTRADAY" for trade in self.engine._positions.values()
        ):
            raise ValueError("A required 15:15 square-off bar was not captured")
        self.last_data, self.last_as_of = data, observed

    def state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if self.last_as_of is None:
            raise RuntimeError("Paper portfolio has not been initialized")
        state = create_initial_state(snapshot["watchlist"])
        state.update(snapshot)
        positions = {}
        for symbol, trade in self.engine._positions.items():
            mark = self._mark(symbol, trade, self.last_data, self.last_as_of)
            positions[symbol] = {
                "quantity": trade.quantity if trade.action == "BUY" else -trade.quantity,
                "avg_price": trade.entry_price, "price": trade.entry_price,
                "ltp": mark, "holding_period": trade.holding_period,
            }
        state.update(
            research_mode=True, execution_enabled=False, research_daily_loss_hit=self.loss_halted,
            portfolio_value=self._equity(self.last_data, self.last_as_of),
            available_capital=self.engine._capital, current_positions=positions,
            broker_balances={}, global_quotes={}, lrs_status={},
            reflection_heuristics=[], similar_past_trades={}, errors=[], agent_logs=[],
        )
        return state

    def submit(self, state: dict[str, Any]) -> None:
        observed = datetime.fromisoformat(state["research_as_of"]).astimezone(IST)
        next_bar = _next_bar(observed)
        risk = state.get("risk_assessment", {})
        risk_blocked = bool(risk.get("blocked"))
        for decision in state.get("trading_decisions", []):
            if not isinstance(decision, dict):
                self.audit.append({"status": "invalid_decision"})
                continue
            raw_symbol = decision.get("symbol")
            symbol = raw_symbol.strip().upper() if isinstance(raw_symbol, str) else ""
            action = str(decision.get("action") or "").upper()
            quantity = _positive(decision.get("quantity"))
            if not symbol or action not in {"BUY", "SELL"} or quantity is None or not quantity.is_integer():
                self.audit.append({"symbol": symbol, "status": "invalid_decision"})
                continue
            if (symbol, next_bar) in self.pending:
                self.audit.append({"symbol": symbol, "status": "duplicate_decision"})
                continue
            trade = self.engine._positions.get(symbol)
            if trade:
                is_exit = (trade.action == "BUY" and action == "SELL") or (trade.action == "SELL" and action == "BUY")
                if not is_exit or quantity != trade.quantity:
                    self.audit.append({"symbol": symbol, "status": "unsupported_or_partial_exit"})
                    continue
                order_type = str(decision.get("order_type", "LIMIT")).upper()
                limit = _positive(decision.get("price")) if order_type == "LIMIT" else None
                if order_type not in {"LIMIT", "MARKET"} or (order_type == "LIMIT" and limit is None):
                    self.audit.append({"symbol": symbol, "status": "invalid_exit"})
                    continue
                self.pending[(symbol, next_bar)] = {"kind": "exit", "action": action, "limit_price": limit}
                continue
            if risk_blocked:
                self.audit.append({"symbol": symbol, "status": "risk_or_signal_blocked"})
                continue
            if not (time(9, 30) <= observed.time() < time(14, 45)) or next_bar.time() >= time(14, 45):
                self.audit.append({"symbol": symbol, "status": "outside_entry_window"})
                continue
            if str(decision.get("currency", "INR")).upper() != "INR":
                self.audit.append({"symbol": symbol, "status": "unsupported_currency"})
                continue
            if entry_regime_block_reason(state.get("hmm_regime")):
                self.audit.append({"symbol": symbol, "status": "regime_blocked"})
                continue
            if symbol in state.get("india_signals", {}).get("fo_ban_list", []):
                self.audit.append({"symbol": symbol, "status": "fo_ban"})
                continue
            assessment = risk.get(symbol, {})
            technical = state.get("technical_signals", {}).get(symbol, {})
            if not isinstance(technical, dict):
                self.audit.append({"symbol": symbol, "status": "risk_or_signal_blocked"})
                continue
            regime_reason = entry_regime_strategy_block_reason(
                state.get("hmm_regime"), technical.get("strategy"),
            )
            if regime_reason:
                self.audit.append({"symbol": symbol, "status": "regime_blocked", "reason": regime_reason})
                continue
            try:
                regime_position_cap = entry_regime_position_cap(state.get("hmm_regime"))
            except ValueError as exc:
                self.audit.append({"symbol": symbol, "status": "regime_blocked", "reason": str(exc)})
                continue
            tech_action = str(technical.get("signal") or "").upper()
            expected = "BUY" if tech_action in {"BUY", "STRONG_BUY"} else (
                "SELL" if tech_action in {"SELL", "STRONG_SELL"} else ""
            )
            if not isinstance(assessment, dict) or assessment.get("approved") is not True or (
                assessment.get("deterministic_action") != action or expected != action
            ):
                self.audit.append({"symbol": symbol, "status": "risk_or_signal_blocked"})
                continue
            period = str(assessment.get("holding_period") or technical.get("holding_period") or "INTRADAY").upper()
            if action == "SELL" and period != "INTRADAY":
                self.audit.append({"symbol": symbol, "status": "naked_short_blocked"})
                continue
            quote = state.get("live_quotes", {}).get(symbol, {})
            pinned = {
                **decision,
                "price": quote.get("ltp") or technical.get("indicators", {}).get("close"),
                "stop_loss": assessment.get("stop_loss") or technical.get("stop_loss"),
                "target": assessment.get("target") or technical.get("target"),
                "holding_period": period,
                "atr": technical.get("indicators", {}).get("daily_atr_14"),
                "atr_exit_policy": technical.get("atr_exit_policy", ""),
            }
            error = _validate_entry_decision(pinned, action, require_limit_order=False)
            if error:
                self.audit.append({"symbol": symbol, "status": "invalid_entry", "reason": error})
                continue
            pinned["risk_multiplier"] = assessment.get("vix_factor", 1.0)
            pinned["hmm_position_cap"] = regime_position_cap
            limit = float(pinned["price"]) if str(pinned.get("order_type", "LIMIT")).upper() == "LIMIT" else None
            pinned["limit_price"] = limit
            self.pending[(symbol, next_bar)] = {
                "kind": "entry", "action": action, "signal": pinned, "limit_price": limit,
            }

    def report(self, mode: str) -> dict[str, Any]:
        if self.last_as_of is None:
            raise RuntimeError("Paper portfolio has not been initialized")
        final = self._equity(self.last_data, self.last_as_of)
        positions = [{
            "symbol": symbol, "action": trade.action, "quantity": trade.quantity,
            "entry_time": trade.entry_date.isoformat(), "entry_price": trade.entry_price,
            "holding_period": trade.holding_period,
            "stop_loss": trade.stop_loss, "target": trade.target,
            "atr": trade.atr, "atr_exit_policy": trade.atr_exit_policy,
        } for symbol, trade in self.engine._positions.items()]
        return {
            "mode": mode, "initial_capital": self.engine.initial_capital,
            "final_equity": round(final, 2),
            "return_pct": round((final / self.engine.initial_capital - 1) * 100, 4),
            "cash": round(self.engine._capital, 2), "open_positions": positions,
            "closed_trades": [asdict(trade) for trade in self.engine._closed_trades],
            "pending_orders": len(self.pending), "audit": self.audit,
        }


def validate_comparison(snapshot_ids: list[str]) -> list[dict[str, Any]]:
    if not settings.paper_trading:
        raise ValueError("Research comparison requires paper-trading mode")
    if len(snapshot_ids) < 2 or len(snapshot_ids) > 10:
        raise ValueError("Select between 2 and 10 captured snapshots")
    snapshots = [load_snapshot(identifier) for identifier in snapshot_ids]
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("Research snapshot IDs must be unique")
    snapshots.sort(key=lambda item: datetime.fromisoformat(item["research_as_of"]))
    profiles = [profile_for(mode) for mode in MODES]
    missing = [profile.mode for profile in profiles if profile.mode != "rules" and not configured(profile)]
    if missing:
        raise ValueError(f"Research providers are not configured: {', '.join(missing)}")
    return snapshots


def run_comparison(snapshot_ids: list[str]) -> dict[str, Any]:
    """Replay the complete analyst pipeline into independent virtual books."""
    snapshots = validate_comparison(snapshot_ids)
    profiles = [profile_for(mode) for mode in MODES]
    books = {profile.mode: _PaperBook() for profile in profiles}
    identifier = str(uuid4())
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "id": identifier, "created_at": started.isoformat(), "snapshot_ids": snapshot_ids,
        "status": "running", "variants": [],
        "limitations": [
            "Paper simulation only; no broker orders or real account positions.",
            "Historical model training may contain later knowledge.",
            "Next-bar fills assume zero model latency and use captured 15-minute bars.",
            "Replay is limited to the verified 2026 calendar and rejects Muhurat windows until special-session exits are implemented.",
            "Captured HMM entry directives and selected configured risk controls are applied; aggregate stop risk and full AI evidence parity are not.",
            "Returns are not live-trading parity; earlier comparison runs may use different rules.",
        ],
    }
    try:
        for snapshot in snapshots:
            for profile in profiles:
                book = books[profile.mode]
                book.advance(snapshot)
                state = book.state(snapshot)
                decision = _run_research_pipeline(state, profile)
                if decision.get("errors"):
                    raise RuntimeError(f"{profile.mode} research pipeline reported errors: {decision['errors']}")
                book.submit(decision)
        result["variants"] = [books[profile.mode].report(profile.mode) for profile in profiles]
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    with get_db() as db:
        db.add(ResearchComparison(
            id=identifier, created_at=started.replace(tzinfo=None),
            status=result["status"], results_json=json.dumps(result, default=str),
        ))
    return result


def list_comparisons(limit: int = 20) -> list[dict[str, Any]]:
    with get_db() as db:
        rows = db.query(ResearchComparison).order_by(ResearchComparison.created_at.desc()).limit(
            min(max(limit, 1), 100)
        ).all()
        return [json.loads(row.results_json) for row in rows]
