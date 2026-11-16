"""Durable alert outbox: store first, deliver second, retry with backoff.

Outbox problems never raise into trading code; an alert that cannot even be
stored is logged and the caller carries on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from database.db import get_db
from database.models import AlertOutbox

DEDUP_WINDOW = timedelta(minutes=30)
MAX_ATTEMPTS = 10
BACKOFF_BASE_SECONDS = 60
BACKOFF_CAP_SECONDS = 3600
RECOVERED_SUFFIX = ":recovered"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def enqueue(
    text: str, *, parse_mode: str | None, dedup_key: str | None, severity: str,
    deliverable: bool = True, now: datetime | None = None,
) -> int | None:
    """Store an alert; return its id, or None when it is a duplicate or storage failed.

    A ``<key>:recovered`` alert pairs with ``<key>``: once recovery was sent,
    a later outage alert is a new event and is not suppressed.
    """
    now = now or _now()
    try:
        with get_db() as db:
            if dedup_key:
                latest = (
                    db.query(AlertOutbox).filter(AlertOutbox.dedup_key == dedup_key)
                    .order_by(AlertOutbox.created_at.desc(), AlertOutbox.id.desc()).first()
                )
                if latest is not None and now - latest.created_at < DEDUP_WINDOW:
                    paired = (
                        db.query(AlertOutbox)
                        .filter(AlertOutbox.dedup_key == dedup_key + RECOVERED_SUFFIX)
                        .order_by(AlertOutbox.created_at.desc(), AlertOutbox.id.desc()).first()
                    )
                    if paired is None or paired.created_at < latest.created_at:
                        return None
            row = AlertOutbox(
                created_at=now, dedup_key=dedup_key, severity=severity, text=text,
                parse_mode=parse_mode, attempts=0 if deliverable else MAX_ATTEMPTS,
                last_error=None if deliverable else "not_configured",
            )
            db.add(row)
            db.flush()
            return int(row.id)
    except Exception as exc:
        logger.error("[Outbox] Could not store alert ({})", type(exc).__name__)
        return None


def record_attempt(
    row_id: int | None, *, delivered: bool, error: str | None, now: datetime | None = None,
) -> None:
    if row_id is None:
        return
    now = now or _now()
    try:
        with get_db() as db:
            row = db.get(AlertOutbox, row_id)
            if row is None:
                return
            row.attempts = (row.attempts or 0) + 1
            row.last_attempt_at = now
            row.last_error = None if delivered else (error or "delivery_failed")[:120]
            if delivered:
                row.delivered_at = now
    except Exception as exc:
        logger.error("[Outbox] Could not record delivery attempt ({})", type(exc).__name__)


def backoff_seconds(attempts: int) -> int:
    return min(BACKOFF_BASE_SECONDS * 2 ** max(0, attempts - 1), BACKOFF_CAP_SECONDS)


def due_for_retry(now: datetime | None = None, limit: int = 20) -> list[dict[str, Any]]:
    now = now or _now()
    try:
        with get_db() as db:
            rows = (
                db.query(AlertOutbox)
                .filter(AlertOutbox.delivered_at.is_(None), AlertOutbox.attempts < MAX_ATTEMPTS)
                .order_by(AlertOutbox.id).limit(limit * 3).all()
            )
            due = []
            for row in rows:
                if row.attempts and row.last_attempt_at is not None and (
                    now - row.last_attempt_at < timedelta(seconds=backoff_seconds(row.attempts))
                ):
                    continue
                due.append({"id": row.id, "text": row.text, "parse_mode": row.parse_mode})
                if len(due) >= limit:
                    break
            return due
    except Exception as exc:
        logger.error("[Outbox] Could not read retry queue ({})", type(exc).__name__)
        return []


def undelivered_critical(limit: int = 10) -> list[dict[str, Any]]:
    """Critical alerts that never reached the operator, for the dashboard."""
    try:
        with get_db() as db:
            rows = (
                db.query(AlertOutbox)
                .filter(AlertOutbox.delivered_at.is_(None), AlertOutbox.severity == "critical")
                .order_by(AlertOutbox.id.desc()).limit(limit).all()
            )
            return [
                {
                    "id": row.id, "created_at": row.created_at.isoformat() + "Z",
                    "dedup_key": row.dedup_key, "text": row.text[:300],
                    "attempts": row.attempts, "last_error": row.last_error,
                }
                for row in rows
            ]
    except Exception:
        return []
