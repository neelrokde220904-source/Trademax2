"""FIFO replay of executed paper orders.

One replay is shared by closed-lot recording, position sync after restarts, and
reconciliation, so all three agree on what is open and what has been realised.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Iterable

# Horizons plus the legacy broker-product value stored before horizons were modelled.
KNOWN_HOLDING_PERIODS = frozenset({"INTRADAY", "SWING", "BTST", "POSITIONAL", "DELIVERY"})


@dataclass
class OpenLot:
    trade_id: int
    symbol: str
    side: str  # BUY = long lot, SELL = short lot
    quantity: int
    price: float
    timestamp: datetime
    holding_period: str
    stop_loss: float | None
    target: float | None
    atr: float | None = None
    sector: str | None = None
    sector_source: str | None = None
    sector_observed_at: str | None = None
    horizon: str | None = None
    strategy_version: str | None = None
    exit_policy: str | None = None
    atr_timeframe: str | None = None
    book: str | None = None


@dataclass(frozen=True)
class ReplayedClose:
    entry_id: int
    exit_id: int
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float


@dataclass
class LedgerReplay:
    open_lots: dict[str, deque[OpenLot]] = field(default_factory=dict)
    closed: list[ReplayedClose] = field(default_factory=list)


def _ordered(trades: Iterable[Any]) -> list[Any]:
    return sorted(trades, key=lambda trade: (trade.timestamp, trade.id))


def apply_order(replay: LedgerReplay, trade: Any) -> list[ReplayedClose]:
    """Net one executed order against opposite open lots, then open any remainder."""
    action = str(trade.action).upper()
    lots = replay.open_lots.setdefault(trade.symbol, deque())
    remaining = int(trade.quantity)
    closes: list[ReplayedClose] = []
    while remaining > 0 and lots and lots[0].side != action:
        lot = lots[0]
        quantity = min(remaining, lot.quantity)
        per_share = trade.price - lot.price if lot.side == "BUY" else lot.price - trade.price
        closes.append(ReplayedClose(
            entry_id=lot.trade_id, exit_id=trade.id, symbol=trade.symbol, side=lot.side,
            quantity=quantity, entry_price=lot.price, exit_price=trade.price,
            gross_pnl=per_share * quantity,
        ))
        lot.quantity -= quantity
        remaining -= quantity
        if lot.quantity == 0:
            lots.popleft()
    if remaining > 0:
        lots.append(OpenLot(
            trade_id=trade.id, symbol=trade.symbol, side=action, quantity=remaining,
            price=float(trade.price), timestamp=trade.timestamp,
            holding_period=str(trade.holding_period or "INTRADAY").upper(),
            stop_loss=trade.stop_loss, target=trade.target, atr=getattr(trade, "atr", None),
            sector=getattr(trade, "sector", None),
            sector_source=getattr(trade, "sector_source", None),
            sector_observed_at=getattr(trade, "sector_observed_at", None),
            horizon=getattr(trade, "horizon", None),
            strategy_version=getattr(trade, "strategy_version", None),
            exit_policy=getattr(trade, "exit_policy", None),
            atr_timeframe=getattr(trade, "atr_timeframe", None),
            book=getattr(trade, "book", None),
        ))
    if not lots:
        replay.open_lots.pop(trade.symbol, None)
    replay.closed.extend(closes)
    return closes


def replay_orders(trades: Iterable[Any]) -> LedgerReplay:
    replay = LedgerReplay()
    for trade in _ordered(trades):
        apply_order(replay, trade)
    return replay


def summarize_positions(replay: LedgerReplay) -> dict[str, dict[str, Any]]:
    """Signed quantity, average cost and risk levels for each open symbol."""
    positions: dict[str, dict[str, Any]] = {}
    for symbol, lots in replay.open_lots.items():
        if not lots:
            continue
        quantity = sum(lot.quantity for lot in lots)
        sign = 1 if lots[0].side == "BUY" else -1
        latest = lots[-1]
        sector_metadata = {
            (lot.sector, lot.sector_source, lot.sector_observed_at)
            for lot in lots
        }
        sector, sector_source, sector_observed_at = (
            next(iter(sector_metadata)) if len(sector_metadata) == 1 else (None, None, None)
        )
        if not sector or not sector_source or not sector_observed_at:
            sector = sector_source = sector_observed_at = None
        holding_period, stop_loss, target = latest.holding_period, latest.stop_loss, latest.target
        mixed_lots = len({lot.holding_period for lot in lots}) > 1
        if mixed_lots:
            # Never let the latest lot's levels silently stand in for older lots:
            # protect the whole symbol with the tightest stop and earliest time exit.
            stops = [lot.stop_loss for lot in lots if lot.stop_loss]
            targets = [lot.target for lot in lots if lot.target]
            pick_stop, pick_target = (max, min) if sign > 0 else (min, max)
            stop_loss = pick_stop(stops) if stops else None
            target = pick_target(targets) if targets else None
            if any(lot.holding_period == "INTRADAY" for lot in lots):
                holding_period = "INTRADAY"
        positions[symbol] = {
            "symbol": symbol,
            "quantity": sign * quantity,
            "avg_price": sum(lot.price * lot.quantity for lot in lots) / quantity,
            "holding_period": holding_period,
            "stop_loss": stop_loss,
            "target": target,
            "mixed_lots": mixed_lots,
            "exit_policy": latest.exit_policy,
            "atr_timeframe": latest.atr_timeframe,
            "strategy_version": latest.strategy_version,
            "book": latest.book,
            "atr": latest.atr,
            "entry_time": lots[0].timestamp,
            "entry_trade_ids": [lot.trade_id for lot in lots],
            "sector": sector,
            "sector_source": sector_source,
            "sector_observed_at": sector_observed_at,
        }
    return positions


def position_book_identity(positions: Any) -> dict[str, dict[str, Any]]:
    """Create a comparable identity for every open paper position."""
    if not isinstance(positions, dict):
        raise ValueError("paper position snapshot is unavailable")
    identity: dict[str, dict[str, Any]] = {}
    for symbol, position in positions.items():
        if not isinstance(symbol, str) or not symbol or not isinstance(position, dict):
            raise ValueError("invalid paper position")
        quantity = position.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise ValueError(f"{symbol}: invalid paper quantity")
        if not math.isfinite(quantity) or not float(quantity).is_integer() or quantity == 0:
            raise ValueError(f"{symbol}: invalid paper quantity")
        ids = position.get("entry_trade_ids")
        holding_period = position.get("holding_period")
        if (
            not isinstance(ids, list) or not ids
            or any(type(entry_id) is not int or entry_id <= 0 for entry_id in ids)
            or len(ids) != len(set(ids))
            or holding_period not in KNOWN_HOLDING_PERIODS
        ):
            raise ValueError(f"{symbol}: invalid paper lot identity")
        identity[symbol] = {
            "quantity": int(quantity),
            "entry_trade_ids": list(ids),
            "holding_period": holding_period,
        }
    return identity
