"""Dated Nifty 50 membership snapshots and the active trading universe."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytz

from config.settings import PROJECT_ROOT, settings

NIFTY50_SOURCE_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
MAX_MEMBERSHIP_AGE_DAYS = 31
NIFTY50_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "nifty50"
IST = pytz.timezone("Asia/Kolkata")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9&-]{1,30}$")


class UniverseUnavailableError(RuntimeError):
    """Raised when a current or point-in-time universe cannot be verified."""


@dataclass(frozen=True)
class MembershipSnapshot:
    observed_on: date
    source_url: str
    symbols: tuple[str, ...]


def _validate_symbols(symbols: Any) -> tuple[str, ...]:
    if not isinstance(symbols, list) or len(symbols) != 50:
        raise ValueError("Nifty 50 membership must contain exactly 50 symbols")
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols)
    if (
        len(set(normalized)) != 50
        or any(not _SYMBOL_PATTERN.fullmatch(symbol) for symbol in normalized)
    ):
        raise ValueError("Nifty 50 membership contains invalid or duplicate symbols")
    return normalized


def _read_snapshot(path: Path) -> MembershipSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed_on = date.fromisoformat(payload["observed_on"])
        source_url = payload["source_url"]
        symbols = _validate_symbols(payload["symbols"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UniverseUnavailableError(
            f"Nifty 50 membership cache is invalid: {path.name}"
        ) from exc
    if path.stem != observed_on.isoformat() or source_url != NIFTY50_SOURCE_URL:
        raise UniverseUnavailableError(
            f"Nifty 50 membership cache metadata does not match: {path.name}"
        )
    return MembershipSnapshot(observed_on, source_url, symbols)


def load_nifty50_membership(
    as_of: date | None = None,
    *,
    max_age_days: int = MAX_MEMBERSHIP_AGE_DAYS,
) -> MembershipSnapshot:
    """Load the newest membership observed on or before `as_of`; never use future data."""
    cutoff = as_of or datetime.now(IST).date()
    if not NIFTY50_CACHE_DIR.exists():
        raise UniverseUnavailableError("No dated Nifty 50 membership cache is available")
    eligible = sorted(
        (
            path for path in NIFTY50_CACHE_DIR.glob("*.json")
            if path.stem <= cutoff.isoformat()
        ),
        key=lambda path: path.stem,
        reverse=True,
    )
    if not eligible:
        raise UniverseUnavailableError(
            f"No Nifty 50 membership snapshot is available on or before {cutoff.isoformat()}"
        )
    snapshot = _read_snapshot(eligible[0])
    age = (cutoff - snapshot.observed_on).days
    if age < 0 or age > max_age_days:
        raise UniverseUnavailableError(
            f"Nifty 50 membership snapshot is stale ({age} days old)"
        )
    return snapshot


def refresh_nifty50_membership(
    observed_on: date | None = None,
) -> MembershipSnapshot:
    """Fetch and atomically cache today's official constituent list."""
    observed_on = observed_on or datetime.now(IST).date()
    NIFTY50_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = NIFTY50_CACHE_DIR / f"{observed_on.isoformat()}.json"
    if path.exists():
        return _read_snapshot(path)

    response = httpx.get(
        NIFTY50_SOURCE_URL, timeout=20.0, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    symbol_column = next(
        (column for column in (reader.fieldnames or []) if column.strip().lower() == "symbol"),
        None,
    )
    if symbol_column is None:
        raise UniverseUnavailableError("Official Nifty 50 CSV has no Symbol column")
    symbols = _validate_symbols([
        row.get(symbol_column, "") for row in reader
    ])
    payload = {
        "observed_on": observed_on.isoformat(),
        "source_url": NIFTY50_SOURCE_URL,
        "symbols": symbols,
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    return MembershipSnapshot(observed_on, NIFTY50_SOURCE_URL, symbols)


def get_active_watchlist(as_of: date | None = None) -> list[str]:
    """Return the configured universe, failing closed when Nifty membership is stale."""
    if settings.trading_universe_mode == "custom":
        return list(settings.default_watchlist)
    return list(load_nifty50_membership(as_of).symbols)


def active_universe_status() -> dict[str, Any]:
    """Return UI-safe universe readiness without silently substituting another list."""
    if settings.trading_universe_mode == "custom":
        return {
            "mode": "custom",
            "status": "ready",
            "observed_on": None,
            "age_days": None,
            "count": len(settings.default_watchlist),
        }
    try:
        snapshot = load_nifty50_membership()
    except UniverseUnavailableError as exc:
        return {
            "mode": "nifty50",
            "status": "unavailable",
            "observed_on": None,
            "age_days": None,
            "count": 0,
            "reason": str(exc),
        }
    return {
        "mode": "nifty50",
        "status": "ready",
        "observed_on": snapshot.observed_on.isoformat(),
        "age_days": (datetime.now(IST).date() - snapshot.observed_on).days,
        "count": len(snapshot.symbols),
    }
