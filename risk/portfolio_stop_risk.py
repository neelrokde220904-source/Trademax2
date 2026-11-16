"""Conservative, mark-to-stop risk estimates for protected Indian equity positions."""

from __future__ import annotations

import math
from typing import Any

from backtester.costs import calculate_indian_transaction_cost, round_trip_cost
from broker.paper_ledger import KNOWN_HOLDING_PERIODS
from config.settings import settings


class StopRiskUnavailable(ValueError):
    """The open book cannot be valued safely against active stop protection."""


def _positive_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("expected a positive finite number")
    return result


def entry_stop_risk(signal: dict[str, Any], quantity: int) -> float:
    """Estimate a new entry's loss through its stop, including both sides' costs."""
    price = _positive_number(signal.get("price"))
    stop = _positive_number(signal.get("stop_loss"))
    action = str(signal.get("action", "")).upper()
    if action not in {"BUY", "SELL"} or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("invalid entry action or quantity")
    if (action == "BUY" and stop >= price) or (action == "SELL" and stop <= price):
        raise ValueError("entry stop is on the wrong side")
    slippage = settings.slippage_pct
    if not math.isfinite(slippage) or not 0 <= slippage < 1:
        raise ValueError("invalid slippage")
    entry_fill = price * (1 + slippage if action == "BUY" else 1 - slippage)
    stop_fill = stop * (1 - slippage if action == "BUY" else 1 + slippage)
    fees = round_trip_cost(
        entry_fill, stop_fill, quantity,
        is_intraday=str(signal.get("holding_period", "INTRADAY")).upper() == "INTRADAY",
        is_entry_buy=action == "BUY",
    )
    return abs(entry_fill - stop_fill) * quantity + fees


def open_stop_risk(
    positions: dict[str, dict[str, Any]], protections: dict[str, dict[str, Any]],
) -> float:
    """Estimate additional loss from current marks to active, identity-bound stops.

    A single lot per symbol is required because the current protection manager
    and position summary do not establish safe mixed-lot protection.
    """
    total = 0.0
    for symbol, position in positions.items():
        try:
            if not isinstance(position, dict):
                raise ValueError("position is not an object")
            raw_quantity = position.get("quantity")
            if isinstance(raw_quantity, bool):
                raise ValueError("invalid quantity")
            quantity = float(raw_quantity)
            if not math.isfinite(quantity) or not quantity.is_integer() or quantity == 0:
                raise ValueError("invalid quantity")
            ids = position.get("entry_trade_ids")
            if not isinstance(ids, list) or len(ids) != 1 or isinstance(ids[0], bool) or not isinstance(ids[0], int) or ids[0] <= 0:
                raise ValueError("single-lot identity unavailable")
            holding_period = position.get("holding_period")
            if holding_period not in KNOWN_HOLDING_PERIODS:
                raise ValueError("invalid holding period")
            mark = _positive_number(position.get("ltp"))
            entry = _positive_number(position.get("avg_price"))
            initial_stop = _positive_number(position.get("stop_loss"))
            action = "BUY" if quantity > 0 else "SELL"
            if (action == "BUY" and initial_stop >= entry) or (action == "SELL" and initial_stop <= entry):
                raise ValueError("invalid initial stop")
            protection = protections.get(symbol)
            if not isinstance(protection, dict):
                raise ValueError("active stop is missing")
            if (
                protection.get("mixed_lots")
                or protection.get("entry_trade_ids") != ids
                or protection.get("action") != action
                or protection.get("holding_period") != holding_period
                or protection.get("exit_pending_order_id")
            ):
                raise ValueError("active stop identity or exit state is ambiguous")
            stop = _positive_number(protection.get("stop_loss"))
            if action == "BUY" and not initial_stop <= stop < mark:
                raise ValueError("long stop was widened or is already breached")
            if action == "SELL" and not mark < stop <= initial_stop:
                raise ValueError("short stop was widened or is already breached")
            slippage = settings.slippage_pct
            if not math.isfinite(slippage) or not 0 <= slippage < 1:
                raise ValueError("invalid slippage")
            exit_fill = stop * (1 - slippage if action == "BUY" else 1 + slippage)
            shares = abs(int(quantity))
            fee = calculate_indian_transaction_cost(
                exit_fill, shares, is_buy=action == "SELL",
                is_intraday=holding_period == "INTRADAY",
            )
            total += abs(mark - exit_fill) * shares + fee
        except (TypeError, ValueError, OverflowError) as exc:
            raise StopRiskUnavailable(f"{symbol}: {exc}") from exc
    return total


def remaining_stop_risk(equity: float, held_risk: float, projected_risk: float = 0.0) -> float:
    """Return remaining modelled risk without confusing it with the daily loss halt."""
    try:
        equity = _positive_number(equity)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (held_risk, projected_risk)):
            raise ValueError("invalid modelled risk")
        return max(0.0, equity * settings.max_open_stop_risk_pct - held_risk - projected_risk)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StopRiskUnavailable(f"portfolio stop-risk budget unavailable: {exc}") from exc
