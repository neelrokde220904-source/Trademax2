"""Durable entry controls. Missing state starts paused; invalid state stops startup."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT


class ControlStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, key: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as db:
            table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'controls'"
            ).fetchone()
            if table is None:
                return None
            row = db.execute("SELECT payload FROM controls WHERE name = ?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError(f"Invalid persisted {key} control state")
        return value

    def write(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, allow_nan=False)
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS controls (name TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                db.execute(
                    "INSERT INTO controls VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                    (key, payload),
                )

    def paused(self) -> bool:
        value = self.read("entries")
        if value is None:
            return True
        if type(value.get("paused")) is not bool:
            raise ValueError("Invalid persisted entry pause")
        return value["paused"]

    def set_paused(self, paused: bool) -> None:
        self.write("entries", {"paused": paused})

    def record_external_cash_flow(
        self, reference: str, amount: float, source: str, occurred_at: str | None = None,
    ) -> int:
        """Append one idempotent signed external cash flow and require a fresh sync."""
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("cash-flow reference is required")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("cash-flow amount must be numeric")
        if not math.isfinite(amount) or amount == 0:
            raise ValueError("cash-flow amount must be finite and non-zero")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("cash-flow source is required")
        timestamp = occurred_at or datetime.now(timezone.utc).isoformat()
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("cash-flow timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise ValueError("cash-flow timestamp must include a timezone")
        timestamp = parsed.astimezone(timezone.utc).isoformat()

        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS capital_flows ("
                    "reference TEXT PRIMARY KEY, amount REAL NOT NULL, source TEXT NOT NULL, "
                    "occurred_at TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS capital_flow_sync ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), reconciled_count INTEGER NOT NULL)"
                )
                db.execute(
                    "INSERT OR IGNORE INTO capital_flow_sync (id, reconciled_count) VALUES (1, 0)"
                )
                existing = db.execute(
                    "SELECT amount, source FROM capital_flows WHERE reference = ?",
                    (reference.strip(),),
                ).fetchone()
                if existing is not None:
                    if float(existing[0]) != float(amount) or existing[1] != source.strip():
                        raise ValueError("cash-flow reference conflicts with a prior event")
                else:
                    db.execute(
                        "INSERT INTO capital_flows (reference, amount, source, occurred_at) "
                        "VALUES (?, ?, ?, ?)",
                        (reference.strip(), float(amount), source.strip(), timestamp),
                    )
                return int(db.execute("SELECT COUNT(*) FROM capital_flows").fetchone()[0])

    def import_angelone_cash_flow_statement(
        self, digest: str, entries: list[tuple[str, float, str]], minimum_timestamp: str,
    ) -> dict[str, int | str]:
        """Atomically append normalized Angel One statement rows with digest provenance."""
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("cash-flow statement digest is invalid")
        if not entries or len(entries) > 10_000:
            raise ValueError("cash-flow statement row count is invalid")
        try:
            baseline = datetime.fromisoformat(minimum_timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("cash-flow baseline timestamp is invalid") from exc
        if baseline.tzinfo is None:
            raise ValueError("cash-flow baseline timestamp must include a timezone")
        baseline = baseline.astimezone(timezone.utc)

        source = "angelone_statement"
        normalized: list[tuple[str, float, str]] = []
        seen: set[str] = set()
        for reference, amount, occurred_at in entries:
            if (
                not isinstance(reference, str)
                or not reference.strip()
                or len(reference.strip()) > 200
                or not reference.strip().isprintable()
            ):
                raise ValueError("cash-flow transaction reference is invalid")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ValueError("cash-flow amount must be numeric")
            if not math.isfinite(amount) or amount == 0:
                raise ValueError("cash-flow amount must be finite and non-zero")
            if not isinstance(occurred_at, str):
                raise ValueError("cash-flow timestamp is invalid")
            try:
                parsed = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("cash-flow timestamp is invalid") from exc
            if parsed.tzinfo is None:
                raise ValueError("cash-flow timestamp must include a timezone")
            parsed = parsed.astimezone(timezone.utc)
            if parsed < baseline:
                raise ValueError(
                    "statement contains transfers before the paper cash-flow baseline; "
                    "exclude historical transfers"
                )
            key = f"angelone:{reference.strip()}"
            if key in seen:
                raise ValueError("cash-flow statement contains duplicate transaction references")
            seen.add(key)
            normalized.append((key, float(amount), parsed.isoformat()))

        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS capital_flows ("
                    "reference TEXT PRIMARY KEY, amount REAL NOT NULL, source TEXT NOT NULL, "
                    "occurred_at TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS capital_flow_sync ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), reconciled_count INTEGER NOT NULL)"
                )
                db.execute(
                    "INSERT OR IGNORE INTO capital_flow_sync (id, reconciled_count) VALUES (1, 0)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS cash_flow_statements ("
                    "digest TEXT PRIMARY KEY, source TEXT NOT NULL, imported_at TEXT NOT NULL, "
                    "row_count INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS cash_flow_statement_rows ("
                    "digest TEXT NOT NULL, reference TEXT NOT NULL, "
                    "PRIMARY KEY (digest, reference))"
                )
                imported_statement = db.execute(
                    "SELECT source, row_count FROM cash_flow_statements WHERE digest = ?",
                    (digest,),
                ).fetchone()
                if imported_statement is not None:
                    if imported_statement != (source, len(normalized)):
                        raise ValueError("cash-flow statement digest conflicts with prior import")
                    count = db.execute("SELECT COUNT(*) FROM capital_flows").fetchone()[0]
                    reconciled = db.execute(
                        "SELECT reconciled_count FROM capital_flow_sync WHERE id = 1"
                    ).fetchone()[0]
                    return {
                        "statement_digest": digest,
                        "imported": 0,
                        "duplicates": len(normalized),
                        "event_count": int(count),
                        "reconciled_count": int(reconciled),
                    }

                inserted = 0
                duplicates = 0
                for reference, amount, timestamp in normalized:
                    existing = db.execute(
                        "SELECT amount, source, occurred_at FROM capital_flows WHERE reference = ?",
                        (reference,),
                    ).fetchone()
                    if existing is not None:
                        if (
                            float(existing[0]) != amount
                            or existing[1] != source
                            or existing[2] != timestamp
                        ):
                            raise ValueError("cash-flow transaction conflicts with a prior event")
                        duplicates += 1
                    else:
                        possible_legacy_duplicate = db.execute(
                            "SELECT 1 FROM capital_flows "
                            "WHERE amount = ? AND occurred_at = ? AND source != ? LIMIT 1",
                            (amount, timestamp, source),
                        ).fetchone()
                        if possible_legacy_duplicate is not None:
                            raise ValueError(
                                "cash-flow transaction may duplicate a prior manual event"
                            )
                        db.execute(
                            "INSERT INTO capital_flows (reference, amount, source, occurred_at) "
                            "VALUES (?, ?, ?, ?)",
                            (reference, amount, source, timestamp),
                        )
                        inserted += 1
                    db.execute(
                        "INSERT INTO cash_flow_statement_rows (digest, reference) VALUES (?, ?)",
                        (digest, reference),
                    )

                db.execute(
                    "INSERT INTO cash_flow_statements (digest, source, imported_at, row_count) "
                    "VALUES (?, ?, ?, ?)",
                    (digest, source, datetime.now(timezone.utc).isoformat(), len(normalized)),
                )
                total, count = db.execute(
                    "SELECT COALESCE(SUM(amount), 0), COUNT(*) FROM capital_flows"
                ).fetchone()
                reconciled = db.execute(
                    "SELECT reconciled_count FROM capital_flow_sync WHERE id = 1"
                ).fetchone()[0]
                if not math.isfinite(float(total)) or int(reconciled) > int(count):
                    raise ValueError("Invalid persisted external cash-flow ledger")
                return {
                    "statement_digest": digest,
                    "imported": inserted,
                    "duplicates": duplicates,
                    "event_count": int(count),
                    "reconciled_count": int(reconciled),
                }

    def external_cash_flow_snapshot(self) -> tuple[float, int, int]:
        """Return net flows, event count and reconciled count from the durable ledger."""
        if not self.path.exists():
            return 0.0, 0, 0
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as db:
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('capital_flows', 'capital_flow_sync')"
                )
            }
            if "capital_flows" not in tables:
                return 0.0, 0, 0
            total, count = db.execute(
                "SELECT COALESCE(SUM(amount), 0), COUNT(*) FROM capital_flows"
            ).fetchone()
            sync_row = (
                db.execute(
                    "SELECT reconciled_count FROM capital_flow_sync WHERE id = 1"
                ).fetchone()
                if "capital_flow_sync" in tables else None
            )
        total = float(total)
        count = int(count)
        reconciled = int(sync_row[0]) if sync_row is not None else 0
        if not math.isfinite(total) or count < 0 or reconciled < 0 or reconciled > count:
            raise ValueError("Invalid persisted external cash-flow ledger")
        return total, count, reconciled

    def external_cash_flows_since(self, since: str) -> float:
        """Return signed external flows on or after a timezone-aware ISO timestamp."""
        try:
            start = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("cash-flow query timestamp is invalid") from exc
        if start.tzinfo is None:
            raise ValueError("cash-flow query timestamp must include a timezone")
        if not self.path.exists():
            return 0.0
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as db:
            table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'capital_flows'"
            ).fetchone()
            if table is None:
                return 0.0
            row = db.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM capital_flows WHERE occurred_at >= ?",
                (start.astimezone(timezone.utc).isoformat(),),
            ).fetchone()
        total = float(row[0])
        if not math.isfinite(total):
            raise ValueError("Invalid persisted daily external cash flows")
        return total

    def acknowledge_external_cash_flows(self, expected_count: int) -> bool:
        """Mark exactly the synced ledger generation reconciled, preserving concurrent events."""
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("cash-flow reconciliation count is invalid")
        if not self.path.exists():
            return expected_count == 0
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                table = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'capital_flows'"
                ).fetchone()
                count = (
                    int(db.execute("SELECT COUNT(*) FROM capital_flows").fetchone()[0])
                    if table is not None else 0
                )
                if count != expected_count:
                    return False
                db.execute(
                    "CREATE TABLE IF NOT EXISTS capital_flow_sync ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), reconciled_count INTEGER NOT NULL)"
                )
                db.execute(
                    "INSERT INTO capital_flow_sync (id, reconciled_count) VALUES (1, ?) "
                    "ON CONFLICT(id) DO UPDATE SET reconciled_count = excluded.reconciled_count",
                    (expected_count,),
                )
        return True

    def external_cash_flow_reconciliation_pending(self) -> bool:
        _, count, reconciled = self.external_cash_flow_snapshot()
        return count != reconciled


# The path can be redirected (the test suite does) so nothing but the running service touches it.
control_state = ControlStateStore(Path(
    os.environ.get("TRADING_CONTROL_STATE_PATH")
    or PROJECT_ROOT / "database" / "operational_state.sqlite3"
))
