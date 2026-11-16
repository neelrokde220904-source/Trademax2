"""SQLite-backed, tamper-evident SHA-256 event ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url

from config.settings import settings

_GENESIS_HASH = "0" * 64


class HashChainedLedger:
    """Append-only ledger for decisions, simulations, risk checks, and fills."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            url = make_url(settings.database_url)
            if not url.drivername.startswith("sqlite"):
                raise RuntimeError("HashChainedLedger currently requires a SQLite database URL")
            database = url.database or ":memory:"
            path = ":memory:" if database == ":memory:" else Path(database)
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS hash_ledger (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL UNIQUE
            )"""
        )
        self._connection.commit()

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ledger payload must be finite JSON-serializable data") from exc

    @classmethod
    def _digest(cls, entry: dict[str, Any]) -> str:
        serialized = cls._canonical_json(entry).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @classmethod
    def payload_hash(cls, payload: Any) -> str:
        """Return the deterministic SHA-256 digest of JSON-serializable data."""
        return cls._digest({"payload": payload})

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Atomically append an event linked to the preceding ledger hash."""
        if not event_type.strip():
            raise ValueError("event_type cannot be empty")
        payload_json = self._canonical_json(payload)
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                previous = self._connection.execute(
                    "SELECT sequence, entry_hash FROM hash_ledger ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = int(previous["sequence"]) + 1 if previous else 1
                previous_hash = previous["entry_hash"] if previous else _GENESIS_HASH
                body = {
                    "sequence": sequence,
                    "created_at": created_at,
                    "event_type": event_type,
                    "payload_json": payload_json,
                    "previous_hash": previous_hash,
                }
                entry_hash = self._digest(body)
                self._connection.execute(
                    """INSERT INTO hash_ledger
                    (sequence, created_at, event_type, payload_json, previous_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (sequence, created_at, event_type, payload_json, previous_hash, entry_hash),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {**body, "entry_hash": entry_hash, "payload": json.loads(payload_json)}

    def verify(self) -> bool:
        """Validate every row's digest, sequence, and link to its predecessor."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, created_at, event_type, payload_json, previous_hash, entry_hash "
                "FROM hash_ledger ORDER BY sequence"
            ).fetchall()
        previous_hash = _GENESIS_HASH
        for expected_sequence, row in enumerate(rows, start=1):
            body = {
                "sequence": row["sequence"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "payload_json": row["payload_json"],
                "previous_hash": row["previous_hash"],
            }
            if row["sequence"] != expected_sequence:
                return False
            if row["previous_hash"] != previous_hash or self._digest(body) != row["entry_hash"]:
                return False
            previous_hash = row["entry_hash"]
        return True

    def entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent entries in chronological order."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, created_at, event_type, payload_json, previous_hash, entry_hash "
                "FROM hash_ledger ORDER BY sequence DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload_json"])}
            for row in reversed(rows)
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()


hash_ledger = HashChainedLedger()
