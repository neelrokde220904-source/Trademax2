"""Corporate actions as a blocking check only.

A split, bonus, rights issue or merger can change a symbol's price series and
quantity without any trade. Until an operator marks an action processed, new
entries in that symbol are blocked. Exits are never blocked; they log a warning.

There is no automatic official ingestion: ``import_actions`` accepts rows an
operator copied from an exchange announcement (the ``source`` field is required).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from loguru import logger

from database.db import get_db
from database.models import CorporateAction

ACTION_TYPES = {"SPLIT", "BONUS", "DIVIDEND", "RIGHTS", "MERGER", "OTHER"}


def import_actions(rows: Iterable[dict[str, Any]]) -> int:
    """Insert new actions (symbol, action_type, ex_date, source required); duplicates are skipped."""
    added = 0
    with get_db() as db:
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            action_type = str(row.get("action_type") or "").strip().upper()
            ex_date = date.fromisoformat(str(row.get("ex_date") or "")).isoformat()
            source = str(row.get("source") or "").strip()
            if not symbol or action_type not in ACTION_TYPES or not source:
                raise ValueError("corporate action needs symbol, a known action_type, ex_date and source")
            exists = db.query(CorporateAction.id).filter(
                CorporateAction.symbol == symbol, CorporateAction.action_type == action_type,
                CorporateAction.ex_date == ex_date,
            ).first()
            if exists:
                continue
            db.add(CorporateAction(
                symbol=symbol, action_type=action_type, ex_date=ex_date,
                details=str(row.get("details") or "")[:200] or None, source=source[:200],
            ))
            added += 1
    return added


def mark_processed(symbol: str, action_type: str, ex_date: str) -> bool:
    with get_db() as db:
        row = db.query(CorporateAction).filter(
            CorporateAction.symbol == symbol.upper(), CorporateAction.action_type == action_type.upper(),
            CorporateAction.ex_date == ex_date,
        ).first()
        if row is None:
            return False
        row.processed = True
        return True


def blocking_action(symbol: str, today: date | None = None) -> dict[str, Any] | None:
    """The oldest unprocessed action for ``symbol`` with an ex-date on or before today, if any."""
    day = (today or date.today()).isoformat()
    with get_db() as db:
        row = (
            db.query(CorporateAction)
            .filter(
                CorporateAction.symbol == symbol.upper(), CorporateAction.processed == False,
                CorporateAction.ex_date <= day,
            )
            .order_by(CorporateAction.ex_date).first()
        )
        if row is None:
            return None
        return {"symbol": row.symbol, "action_type": row.action_type, "ex_date": row.ex_date}


def warn_if_held(symbol: str) -> None:
    """Exits proceed; the operator is told that a corporate action is unprocessed."""
    try:
        action = blocking_action(symbol)
    except Exception as exc:
        logger.error("Corporate action check failed for {}: {}", symbol, type(exc).__name__)
        return
    if action:
        logger.warning(
            "Exit proceeds for {} despite an unprocessed {} (ex {}).",
            symbol, action["action_type"], action["ex_date"],
        )
