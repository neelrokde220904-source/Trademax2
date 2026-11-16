"""Paper cash ledger: delivery sale proceeds are unsettled until the next NSE session (T+1).

INTRADAY trades net out the same day and are settled at once. Sizing may only
use settled cash, so proceeds of a swing/BTST exit cannot fund an entry on the
day of the sale.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from database.models import CashLedgerEntry, Trade
from strategies.candidate import is_intraday_horizon


def settlement_date(trade_day: date) -> str:
    """T+1 settlement day: the next NSE session, or the next weekday if the calendar is unknown."""
    from risk.exit_policy import next_session_date

    try:
        return next_session_date(trade_day).isoformat()
    except ValueError:
        day = trade_day + timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        logger.warning("NSE calendar unknown for {}; settling on the next weekday.", day)
        return day.isoformat()


def _trade_day(trade: Trade) -> date:
    stamp = trade.timestamp
    return stamp.date() if isinstance(stamp, datetime) else date.fromisoformat(str(stamp)[:10])


def record_trade_cash(db: Session, trade: Trade, fee: float) -> None:
    """Write the TRADE and FEE rows for one executed paper trade (idempotent per trade and kind)."""
    existing = {
        row.kind for row in db.query(CashLedgerEntry.kind).filter(CashLedgerEntry.trade_id == trade.id)
    }
    side_sign = 1.0 if str(trade.action).upper() == "SELL" else -1.0
    amount = side_sign * float(trade.price) * int(trade.quantity)
    delivery_sale = side_sign > 0 and not is_intraday_horizon(trade.holding_period)
    if "TRADE" not in existing:
        settles_on = settlement_date(_trade_day(trade)) if delivery_sale else None
        db.add(CashLedgerEntry(
            timestamp=trade.timestamp, currency="INR", amount=amount, kind="TRADE",
            settles_on=settles_on, settled=settles_on is None, trade_id=trade.id,
        ))
    if "FEE" not in existing:
        db.add(CashLedgerEntry(
            timestamp=trade.timestamp, currency="INR", amount=-abs(float(fee)), kind="FEE",
            settles_on=None, settled=True, trade_id=trade.id,
        ))


def backfill_missing(db: Session, fee_for: Any) -> int:
    """Create ledger rows for executed paper trades that predate the ledger."""
    known = {
        row.trade_id for row in db.query(CashLedgerEntry.trade_id).filter(CashLedgerEntry.kind == "TRADE")
    }
    created = 0
    for trade in db.query(Trade).filter(Trade.is_paper == True, Trade.status == "EXECUTED").all():
        if trade.id in known:
            continue
        record_trade_cash(db, trade, fee_for(trade))
        created += 1
    if created:
        db.flush()
    return created


def settle_due(db: Session, as_of: date) -> int:
    """Mark proceeds whose settlement day has arrived as settled."""
    rows = (
        db.query(CashLedgerEntry)
        .filter(CashLedgerEntry.settled == False, CashLedgerEntry.settles_on <= as_of.isoformat())
        .all()
    )
    for row in rows:
        row.settled = True
    return len(rows)


def unsettled_cash(db: Session) -> float:
    """Sale proceeds not yet settled; these cannot size new entries."""
    rows = (
        db.query(CashLedgerEntry.amount)
        .filter(CashLedgerEntry.settled == False, CashLedgerEntry.kind == "TRADE", CashLedgerEntry.amount > 0)
        .all()
    )
    return float(sum(amount for (amount,) in rows))
