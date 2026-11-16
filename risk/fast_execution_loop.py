"""Deterministic pre-trade gate for tick/bar-triggered execution."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
import math
from typing import Any, Callable

import pandas as pd

from database.hash_ledger import HashChainedLedger, hash_ledger
from risk.timemachine_validator import TimeMachineResult, TimeMachineValidator, timemachine_validator


@dataclass(frozen=True)
class FastExecutionResult:
    """Outcome of deterministic simulation, risk check, and broker submission."""

    approved: bool
    simulation: TimeMachineResult
    risk_reason: str
    broker_result: dict[str, Any] | None


class FiveMinuteBarAggregator:
    """Build OHLCV bars from normalized tick events with bounded history."""

    def __init__(self, token_to_symbol: dict[str, str], max_bars: int = 4_500) -> None:
        if max_bars < 2:
            raise ValueError("max_bars must be at least two")
        self._token_to_symbol = token_to_symbol
        self._max_bars = max_bars
        self._current: dict[str, dict[str, Any]] = {}
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_bars)
        )
        self._last_cumulative_volume: dict[str, float] = {}
        self._last_tick_time: dict[str, pd.Timestamp] = {}

    def update(self, tick: dict[str, Any]) -> tuple[str, pd.DataFrame] | None:
        """Update a symbol bar and return its history when a 5-minute bar closes."""
        token = str(tick.get("token", ""))
        symbol = self._token_to_symbol.get(token)
        price = tick.get("ltp", tick.get("last_price"))
        timestamp = tick.get("timestamp") or tick.get("received_at")
        if not symbol or price is None or timestamp is None:
            return None
        try:
            price = float(price)
            volume = float(tick.get("volume", 0) or 0)
            moment = pd.Timestamp(timestamp)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(price) or price <= 0 or not math.isfinite(volume) or volume < 0 or pd.isna(moment):
            return None
        if moment.tzinfo is None:
            moment = moment.tz_localize("Asia/Kolkata")
        else:
            moment = moment.tz_convert("Asia/Kolkata")
        previous_time = self._last_tick_time.get(symbol)
        if previous_time is not None and moment < previous_time:
            return None
        bucket = moment.floor("5min")
        previous_volume = (
            volume if previous_time is None or moment.date() != previous_time.date()
            else self._last_cumulative_volume[symbol]
        )
        incremental_volume = max(0.0, volume - previous_volume)
        self._last_cumulative_volume[symbol] = max(volume, previous_volume)
        self._last_tick_time[symbol] = moment
        current = self._current.get(symbol)
        completed = None
        if current and bucket > current["bucket"]:
            self._history[symbol].append({
                "datetime": current["bucket"], "open": current["open"],
                "high": current["high"], "low": current["low"],
                "close": current["close"], "volume": current["volume"],
            })
            completed = pd.DataFrame(self._history[symbol]).set_index("datetime")
            current = None
        if current is None:
            self._current[symbol] = {
                "bucket": bucket, "open": price, "high": price, "low": price,
                "close": price, "volume": incremental_volume,
            }
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += incremental_volume
        return (symbol, completed) if completed is not None else None


class FastExecutionLoop:
    """Run no-LLM execution checks on a candidate produced by a local strategy."""

    def __init__(
        self,
        submit_order: Callable[[dict[str, Any]], dict[str, Any]],
        risk_check: Callable[[dict[str, Any]], tuple[bool, str]],
        *,
        validator: TimeMachineValidator = timemachine_validator,
        ledger: HashChainedLedger = hash_ledger,
        max_bars: int = 4_500,
    ) -> None:
        self._submit_order = submit_order
        self._risk_check = risk_check
        self._validator = validator
        self._ledger = ledger
        self._max_bars = max_bars
        self._tick_unsubscribe: Callable[[], None] | None = None

    def subscribe_to_ticks(
        self,
        token_to_symbol: dict[str, str],
        candidate_provider: Callable[[str, pd.DataFrame], dict[str, Any] | None],
    ) -> Callable[[], None]:
        """Connect the deterministic strategy callback to the shared tick bus."""
        from broker.websocket_feed import tick_event_bus

        aggregator = FiveMinuteBarAggregator(token_to_symbol, self._max_bars)

        def on_tick(tick: dict[str, Any]) -> None:
            completed = aggregator.update(tick)
            if completed is None:
                return
            symbol, bars = completed
            candidate = candidate_provider(symbol, bars)
            if candidate is not None:
                self.process_candidate(candidate, bars)

        if self._tick_unsubscribe:
            self._tick_unsubscribe()
        self._tick_unsubscribe = tick_event_bus.subscribe(on_tick)
        return self._tick_unsubscribe

    def process_candidate(self, candidate: dict[str, Any], five_minute_bars: pd.DataFrame) -> FastExecutionResult:
        """Validate one strategy candidate before it can reach a broker adapter."""
        required = ("symbol", "entry_price", "target_price", "stop_price", "quantity")
        missing = [key for key in required if key not in candidate]
        if missing:
            raise ValueError(f"Candidate is missing required fields: {', '.join(missing)}")
        decision = {
            "symbol": candidate["symbol"], "action": candidate.get("action", "BUY"),
            "entry_price": candidate["entry_price"], "target_price": candidate["target_price"],
            "stop_price": candidate["stop_price"], "quantity": candidate["quantity"],
            "strategy": candidate.get("strategy", "unknown"),
        }
        self._ledger.append("trade_decision", decision)
        simulation = self._validator.validate(
            five_minute_bars,
            entry_price=float(candidate["entry_price"]),
            target_price=float(candidate["target_price"]),
            stop_price=float(candidate["stop_price"]),
            quantity=int(candidate["quantity"]),
            is_intraday=candidate.get("holding_period", "INTRADAY") == "INTRADAY",
            historical_setup_returns=candidate.get("historical_setup_returns"),
            accuracy_score=float(candidate.get("accuracy_score", 0.0)),
            remaining_time_buckets=candidate.get("remaining_time_buckets"),
        )
        self._ledger.append("timemachine_simulation", {
            "symbol": candidate["symbol"], "approved": simulation.approved,
            "ev_net": simulation.ev_net, "prob_target_first": simulation.prob_target_first,
            "prob_stop_first": simulation.prob_stop_first,
            "prob_open_at_close": simulation.prob_open_at_close,
            "quality_score": simulation.quality_score,
        })
        if not simulation.approved:
            return FastExecutionResult(False, simulation, simulation.reason, None)

        risk_allowed, risk_reason = self._risk_check(candidate)
        self._ledger.append("risk_check", {
            "symbol": candidate["symbol"], "approved": bool(risk_allowed),
            "reason": risk_reason,
        })
        if not risk_allowed:
            return FastExecutionResult(False, simulation, risk_reason, None)

        order_result = self._submit_order(candidate)
        self._ledger.append("broker_submission", {
            "symbol": candidate["symbol"], "broker_order_id": order_result.get("order_id"),
            "status": order_result.get("status", "UNKNOWN"),
            "result_hash": self._ledger.payload_hash(order_result),
        })
        return FastExecutionResult(True, simulation, risk_reason, order_result)
