"""Stop-Loss Manager — ATR trailing stops and bracket target management."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytz
from loguru import logger

from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings
from operations.control_state import ControlStateStore, control_state
from risk.exit_policy import (
    chandelier_stop, get_atr_exit_policy, ratcheted_stop, stop_target_hit, time_exit_due,
)

LEGACY_ATR_TIMEFRAME = "UNSPECIFIED_LEGACY"

IST = pytz.timezone("Asia/Kolkata")

# Stop/target triggers need a recent market mark; the exit itself re-prices from a fresh tick.
PROTECTIVE_MARK_MAX_AGE_SECONDS = 60.0
MARK_FUTURE_SKEW_SECONDS = 5.0
SESSION_OPEN = (9, 15)
INTRADAY_EXIT_TIME = (15, 15)


def as_ist(value: Any) -> datetime | None:
    """Parse a datetime or ISO string; naive values are IST wall-clock (ledger convention)."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not hasattr(value, "astimezone") or not hasattr(value, "hour"):
        return None
    return IST.localize(value) if value.tzinfo is None else value.astimezone(IST)


def mark_age_seconds(position: dict[str, Any], now: datetime | None = None) -> float | None:
    marked = as_ist(position.get("mark_at"))
    if marked is None:
        return None
    return ((now or datetime.now(IST)) - marked).total_seconds()


def mark_is_fresh(position: dict[str, Any], now: datetime | None = None) -> bool:
    try:
        price = float(position.get("ltp") or 0)
    except (TypeError, ValueError):
        return False
    age = mark_age_seconds(position, now)
    return (
        math.isfinite(price) and price > 0 and age is not None
        and -MARK_FUTURE_SKEW_SECONDS <= age <= PROTECTIVE_MARK_MAX_AGE_SECONDS
    )


def stop_on_protective_side(action: str, entry_price: float, stop_loss: float | None) -> bool:
    if not stop_loss or stop_loss <= 0 or entry_price <= 0:
        return False
    return stop_loss < entry_price if action.upper() == "BUY" else stop_loss > entry_price


class StopLossManager:
    """Manages stop-losses and targets for open positions.

    Features:
    - Initial SL from signal (ATR-based)
    - Trailing stop at 2× ATR from highest price
    - Time-based exit: close intraday positions by 3:15 PM IST
    - Target exit at specified target price
    """

    def __init__(self, store: ControlStateStore | None = None) -> None:
        self._tracking: dict[str, dict[str, Any]] = {}
        self._store = store
        self._last_check: dict[str, Any] | None = None

    def _persist(self) -> None:
        if self._store is not None:
            self._store.write("protective_stops", {"positions": self.snapshot()})

    def register_trade(
        self, trade: dict[str, Any], *, persist: bool = True, restoring: bool = False,
    ) -> None:
        """Register a trade for stop-loss tracking.

        New trades must name their exit policy and ATR timeframe so a ratchet can never
        silently mix timeframes; positions restored from the ledger may predate that.
        """
        symbol = trade["symbol"]
        price = float(trade.get("price") or 0)
        exit_policy = trade.get("exit_policy")
        atr_timeframe = trade.get("atr_timeframe")
        if not restoring:
            if not exit_policy or not atr_timeframe:
                raise ValueError(
                    f"Cannot register {symbol}: exit_policy and atr_timeframe are required for new trades"
                )
        if exit_policy:
            policy = get_atr_exit_policy(exit_policy)  # unknown names raise
            if atr_timeframe and atr_timeframe != policy.atr_timeframe:
                raise ValueError(
                    f"ATR timeframe {atr_timeframe} does not match policy {exit_policy} ({policy.atr_timeframe})"
                )
            atr_timeframe = policy.atr_timeframe
        self._tracking[symbol] = {
            "stop_loss": float(trade.get("stop_loss") or 0),
            "target": float(trade.get("target") or 0),
            "highest_since_entry": price,
            "lowest_since_entry": price,
            "trailing_atr": float(trade.get("atr") or 0),
            "entry_time": trade.get("entry_time") or datetime.now(IST),
            "holding_period": trade.get("holding_period", "INTRADAY"),
            "action": str(trade.get("action", "BUY")).upper(),
            "exit_pending_order_id": None,
            "entry_trade_ids": list(trade.get("entry_trade_ids") or []),
            "mixed_lots": bool(trade.get("mixed_lots")),
            "exit_policy": exit_policy,
            "atr_timeframe": atr_timeframe or LEGACY_ATR_TIMEFRAME,
        }
        if persist:
            self._persist()
        logger.info(
            "[SL] Registered {} — SL:₹{:.2f} Target:₹{:.2f}",
            symbol, self._tracking[symbol]["stop_loss"], self._tracking[symbol]["target"],
        )

    def restore_from_positions(self, positions: dict[str, dict[str, Any]]) -> list[str]:
        """Restore ledger protection and any persisted, monotonic trailing ratchet."""
        saved = self._store.read("protective_stops") if self._store is not None else None
        saved_positions = saved.get("positions", {}) if isinstance(saved, dict) else {}
        self._tracking = {}
        invalid: list[str] = []
        for symbol, position in positions.items():
            quantity = int(position.get("quantity") or 0)
            if quantity == 0:
                continue
            action = "BUY" if quantity > 0 else "SELL"
            avg_price = float(position.get("avg_price") or 0)
            stop = float(position.get("stop_loss") or 0)
            self.register_trade({
                "symbol": symbol, "action": action, "price": avg_price,
                "stop_loss": stop, "target": position.get("target"),
                "atr": position.get("atr"),
                "holding_period": position.get("holding_period", "INTRADAY"),
                "entry_time": position.get("entry_time"),
                "entry_trade_ids": position.get("entry_trade_ids"),
                "mixed_lots": position.get("mixed_lots"),
                "exit_policy": self._known_policy_name(position.get("exit_policy")),
                "atr_timeframe": position.get("atr_timeframe"),
            }, persist=False, restoring=True)
            restored = saved_positions.get(symbol) if isinstance(saved_positions, dict) else None
            valid_stop = stop_on_protective_side(action, avg_price, stop)
            if isinstance(restored, dict) and (
                restored.get("entry_trade_ids") and position.get("entry_trade_ids")
                and restored["entry_trade_ids"] != position["entry_trade_ids"]
            ):
                logger.warning("[SL] Ignoring previous lot's ratchet for {}.", symbol)
                self._tracking[symbol]["exit_pending_order_id"] = restored.get("exit_pending_order_id")
                restored = None
            if isinstance(restored, dict):
                data = self._tracking[symbol]
                persisted_stop = float(restored.get("stop_loss") or 0)
                persisted_high = float(restored.get("highest_since_entry") or avg_price)
                persisted_low = float(restored.get("lowest_since_entry") or avg_price)
                if action == "BUY":
                    valid_stop = (
                        persisted_stop > 0
                        and persisted_stop >= stop
                        and persisted_stop <= persisted_high
                    )
                    if valid_stop:
                        data["stop_loss"] = max(data["stop_loss"], persisted_stop)
                else:
                    valid_stop = (
                        persisted_stop > 0
                        and persisted_stop <= stop
                        and persisted_stop >= persisted_low
                    )
                    if valid_stop:
                        data["stop_loss"] = (
                            min(data["stop_loss"], persisted_stop)
                            if data["stop_loss"] > 0 else persisted_stop
                        )
                data["target"] = float(restored.get("target") or data["target"])
                data["trailing_atr"] = float(
                    position.get("atr") or restored.get("trailing_atr") or 0
                )
                data["highest_since_entry"] = max(
                    avg_price, float(restored.get("highest_since_entry") or avg_price)
                )
                data["lowest_since_entry"] = min(
                    avg_price, float(restored.get("lowest_since_entry") or avg_price)
                )
                data["exit_pending_order_id"] = restored.get("exit_pending_order_id")
            if not valid_stop:
                invalid.append(symbol)
        self._persist()
        if invalid:
            logger.error("[SL] Restored positions without a valid protective stop: {}", invalid)
        return invalid

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            symbol: {
                "action": data["action"],
                "stop_loss": data["stop_loss"],
                "target": data["target"],
                "holding_period": data["holding_period"],
                "entry_time": data["entry_time"].isoformat() if hasattr(data["entry_time"], "isoformat") else data["entry_time"],
                "highest_since_entry": data["highest_since_entry"],
                "lowest_since_entry": data["lowest_since_entry"],
                "trailing_atr": data["trailing_atr"],
                "exit_pending_order_id": data["exit_pending_order_id"],
                "entry_trade_ids": data["entry_trade_ids"],
                "mixed_lots": data.get("mixed_lots", False),
                "exit_policy": data.get("exit_policy"),
                "atr_timeframe": data.get("atr_timeframe"),
            }
            for symbol, data in self._tracking.items()
        }

    def unregister_trade(self, symbol: str) -> None:
        """Remove tracking for a closed position."""
        self._tracking.pop(symbol, None)
        self._persist()

    def bind_entry_identity(self, symbol: str, position: dict[str, Any]) -> None:
        if symbol not in self._tracking:
            logger.error("[SL] Cannot bind entry identity for untracked position {}.", symbol)
            return
        self._tracking[symbol]["entry_trade_ids"] = list(position.get("entry_trade_ids") or [])
        if position.get("entry_time") is not None:
            self._tracking[symbol]["entry_time"] = position["entry_time"]
        self._persist()

    def mark_exit_pending(self, symbol: str, order_id: str) -> None:
        """Remember a submitted exit so a retry cannot duplicate an uncertain order."""
        if symbol in self._tracking:
            self._tracking[symbol]["exit_pending_order_id"] = order_id
            self._persist()

    def begin_exit(self, symbol: str) -> None:
        """Persist intent before sending: a crash must not cause an automatic duplicate."""
        if symbol not in self._tracking:
            raise ValueError(f"Protection must be registered before exiting {symbol}")
        if self._tracking[symbol].get("exit_pending_order_id"):
            raise ValueError(f"An unresolved exit already exists for {symbol}")
        self.mark_exit_pending(symbol, f"UNRESOLVED:{uuid4()}")

    def clear_exit_pending(self, symbol: str) -> None:
        if symbol in self._tracking:
            self._tracking[symbol]["exit_pending_order_id"] = None
            self._persist()

    def pending_exits(self) -> dict[str, str]:
        return {
            symbol: str(data["exit_pending_order_id"])
            for symbol, data in self._tracking.items()
            if data.get("exit_pending_order_id")
        }

    def tracked_symbols(self) -> tuple[str, ...]:
        return tuple(self._tracking)

    def last_check(self) -> dict[str, Any] | None:
        return dict(self._last_check) if self._last_check else None

    def price_trigger(self, symbol: str, price: Any) -> dict[str, Any] | None:
        """Stop/target crossing for one fresh tick; ratchets and time exits stay on the scheduled check."""
        data = self._tracking.get(symbol)
        if not data or data.get("exit_pending_order_id"):
            return None
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 0:
            return None
        return self._price_exit(symbol, price, data)

    def check_all_positions(self) -> list[dict[str, Any]]:
        """Check all tracked positions for exit conditions.

        Stop/target triggers and trailing ratchets require a fresh mark; time exits do not.

        Returns:
            List of exit signals: [{symbol, reason, exit_type}]
        """
        exits: list[dict[str, Any]] = []
        now = datetime.now(IST)
        unverified: list[str] = []

        for symbol, data in list(self._tracking.items()):
            if data.get("exit_pending_order_id"):
                continue
            pos = portfolio_tracker.positions.get(symbol)
            if not pos:
                self.unregister_trade(symbol)
                continue

            ltp = float(pos.get("ltp") or 0)
            if not mark_is_fresh(pos, now):
                unverified.append(symbol)
                exit_info = self._time_exit(symbol, ltp, data, now)
                if exit_info:
                    exits.append(exit_info)
                continue

            action = data["action"]

            # Update trailing data
            if action == "BUY":
                data["highest_since_entry"] = max(data["highest_since_entry"], ltp)
            else:
                data["lowest_since_entry"] = min(data["lowest_since_entry"], ltp)

            # Check exit conditions
            exit_info = self._check_exit(symbol, ltp, data, now)
            if exit_info:
                exits.append(exit_info)

        self._last_check = {"checked_at": now.isoformat(), "unverified_marks": unverified}
        if unverified:
            logger.warning("[SL] Stop/target unverified (no fresh mark) for {}.", unverified)
        self._persist()
        return exits

    def _check_exit(
        self, symbol: str, ltp: float, data: dict, now: datetime | None = None,
    ) -> dict[str, Any] | None:
        action = data["action"]
        stop_loss = data["stop_loss"]
        atr = data.get("trailing_atr", 0)

        # --- TRAILING STOP (for longs) ---
        if atr > 0 and stop_loss > 0:
            policy_name = data.get("exit_policy")
            multiplier = (
                get_atr_exit_policy(policy_name).trailing_multiplier
                if policy_name else settings.trailing_stop_atr_multiplier
            )
            data["stop_loss"] = ratcheted_stop(
                action, stop_loss, data["highest_since_entry"], data["lowest_since_entry"],
                atr, multiplier,
            )

        return self._price_exit(symbol, ltp, data) or self._time_exit(symbol, ltp, data, now)

    @staticmethod
    def _known_policy_name(name: Any) -> str | None:
        """Policy name when it is a registered one; unknown names fall back to legacy handling."""
        try:
            return get_atr_exit_policy(name).name if name else None
        except ValueError:
            logger.error("[SL] Unknown exit policy {!r}; using legacy trailing rules.", name)
            return None

    @staticmethod
    def _price_exit(symbol: str, ltp: float, data: dict) -> dict[str, Any] | None:
        action = data["action"]
        stop_loss = data["stop_loss"]
        target = data["target"]
        exit_action = "SELL" if action == "BUY" else "BUY"
        # A missing stop is reported by reconciliation, never auto-triggered.
        hit = stop_target_hit(action, stop_loss, target, high=ltp, low=ltp)
        if hit == "STOP_LOSS":
            reason = f"Stop-loss hit at ₹{ltp:.2f} (SL: ₹{stop_loss:.2f})"
        elif hit == "TARGET":
            reason = f"Target hit at ₹{ltp:.2f} (Target: ₹{target:.2f})"
        else:
            return None
        return {"symbol": symbol, "reason": reason, "exit_type": hit, "exit_action": exit_action}

    @staticmethod
    def _time_exit(
        symbol: str, ltp: float, data: dict, now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Time-based exits by horizon.

        INTRADAY is due from 15:15, or from the next session open if carried overnight.
        BTST, SWING and POSITIONAL follow ``risk.exit_policy.time_exit_due``.
        """
        horizon = str(data.get("holding_period") or "").upper()
        now = now or datetime.now(IST)
        exit_action = "SELL" if data["action"] == "BUY" else "BUY"
        if horizon != "INTRADAY":
            if horizon not in {"BTST", "SWING", "POSITIONAL"}:
                return None
            entry = as_ist(data.get("entry_time"))
            if entry is None:
                return None
            due = time_exit_due(horizon, entry, as_ist(now))
            if due is None:
                return None
            return {
                "symbol": symbol,
                "reason": f"{horizon} time exit ({due}; LTP: ₹{ltp:.2f})",
                "exit_type": "TIME_EXIT",
                "exit_action": exit_action,
            }
        clock = (now.hour, now.minute)
        if clock >= INTRADAY_EXIT_TIME:
            reason = f"Intraday time exit at 3:15 PM (LTP: ₹{ltp:.2f})"
        else:
            entry = as_ist(data.get("entry_time"))
            if entry is None or entry.date() >= now.date() or clock < SESSION_OPEN:
                return None
            reason = (
                f"Overdue intraday time exit: position carried from {entry.date().isoformat()} "
                f"(LTP: ₹{ltp:.2f})"
            )
        return {
            "symbol": symbol,
            "reason": reason,
            "exit_type": "TIME_EXIT",
            "exit_action": exit_action,
        }


# Singleton
stop_loss_manager = StopLossManager(control_state)
