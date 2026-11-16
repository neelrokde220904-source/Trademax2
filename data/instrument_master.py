"""Instrument Master — downloads and caches Angel One instrument master JSON."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class InstrumentMasterUnavailableError(RuntimeError):
    """Instrument master cannot safely supply current broker contracts."""


def _validate_master(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or len(data) < 1000:
        raise InstrumentMasterUnavailableError("Instrument master is empty or unexpectedly incomplete")
    if any(
        not isinstance(item, dict)
        or not all(isinstance(item.get(key), str) and item[key].strip() for key in ("symbol", "token", "exch_seg"))
        for item in data
    ):
        raise InstrumentMasterUnavailableError("Instrument master contains invalid contract identities")
    if not any(item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ") for item in data):
        raise InstrumentMasterUnavailableError("Instrument master lacks NSE cash equities")
    try:
        for item in data:
            int(item.get("lotsize", 1))
            tick = float(item.get("tick_size", 0.05))
            if not math.isfinite(tick) or tick < 0:
                raise ValueError("invalid tick size")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InstrumentMasterUnavailableError("Instrument master contains invalid precision metadata") from exc
    return data


def validate_cache_age(cache_path: Path) -> None:
    try:
        age_seconds = time.time() - cache_path.stat().st_mtime
        if not math.isfinite(age_seconds) or age_seconds < -300 or age_seconds > settings.instrument_cache_max_age_hours * 3600:
            raise InstrumentMasterUnavailableError("Instrument master cache is stale or has a future timestamp")
    except OSError as exc:
        raise InstrumentMasterUnavailableError(f"Instrument master cache unavailable: {type(exc).__name__}") from exc


def _load_valid_cache(cache_path: Path) -> list[dict[str, Any]]:
    validate_cache_age(cache_path)
    try:
        with cache_path.open("r") as fh:
            return _validate_master(json.load(fh))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentMasterUnavailableError(f"Instrument master cache unavailable: {type(exc).__name__}") from exc


async def download_instrument_master() -> list[dict[str, Any]]:
    """Download the full instrument master from Angel One and cache locally.

    Returns:
        List of instrument dicts.
    """
    cache_path = Path(settings.instrument_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading instrument master from Angel One...")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(settings.instrument_master_url)
            resp.raise_for_status()
            data = _validate_master(resp.json())

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=cache_path.parent, delete=False) as fh:
                temp_path = fh.name
                json.dump(data, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_path, cache_path)
            temp_path = None
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)
        from config.instruments import invalidate_cache
        invalidate_cache()

        logger.info("Instrument master cached: {} instruments at {}", len(data), cache_path)
        return data

    except Exception as exc:
        logger.error("Instrument master refresh failed ({}): {}", type(exc).__name__, exc)
        try:
            cached = _load_valid_cache(cache_path)
        except InstrumentMasterUnavailableError as cache_exc:
            raise InstrumentMasterUnavailableError(
                f"Instrument refresh failed and no valid fallback exists: {cache_exc}"
            ) from exc
        logger.warning("Using validated cached instrument master after refresh failure.")
        return cached


def load_instrument_master() -> list[dict[str, Any]]:
    """Load cached instrument master (synchronous)."""
    cache_path = Path(settings.instrument_cache_path)
    return _load_valid_cache(cache_path)


def build_symbol_map(instruments: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Build a lookup dict: symbol → instrument details.

    Supports lookup by 'NSE:RELIANCE' or 'NFO:NIFTY28APR...' format.
    """
    if instruments is None:
        instruments = load_instrument_master()

    symbol_map: dict[str, dict[str, Any]] = {}
    for item in instruments:
        sym = item.get("symbol", "")
        exch = item.get("exch_seg", "")
        key = f"{exch}:{sym}"
        symbol_map[key] = {
            "token": item.get("token", ""),
            "symbol": sym,
            "name": item.get("name", ""),
            "exchange": exch,
            "lot_size": int(item.get("lotsize", 1)),
            "tick_size": float(item.get("tick_size", 0.05)),
            "instrument_type": item.get("instrumenttype", ""),
            "expiry": item.get("expiry", ""),
            "strike": item.get("strike", ""),
            "option_type": item.get("optiontype", ""),
        }

    return symbol_map


def find_nfo_options(
    underlying: str,
    expiry_contains: str = "",
    option_type: str = "",
    instruments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Find NFO options contracts matching criteria.

    Args:
        underlying: e.g. 'NIFTY', 'BANKNIFTY'
        expiry_contains: partial match on expiry string
        option_type: 'CE' or 'PE'
    """
    if instruments is None:
        instruments = load_instrument_master()

    results = []
    for item in instruments:
        if item.get("exch_seg") != "NFO":
            continue
        if item.get("name", "").upper() != underlying.upper():
            continue
        if expiry_contains and expiry_contains.upper() not in str(item.get("expiry") or "").upper():
            continue
        symbol = str(item.get("symbol") or "").upper()
        inferred_side = symbol[-2:] if symbol.endswith(("CE", "PE")) else ""
        explicit_side = str(item.get("optiontype") or "").upper()
        if explicit_side and explicit_side != inferred_side:
            logger.warning("Ignoring inconsistent option type for {}", symbol)
            continue
        if not inferred_side or item.get("instrumenttype") not in {"OPTIDX", "OPTSTK"}:
            continue
        if option_type and inferred_side != option_type.upper():
            continue
        try:
            strike = float(item.get("strike") or 0) / 100.0
            lot_size = int(item.get("lotsize") or 0)
        except (TypeError, ValueError, OverflowError):
            logger.warning("Ignoring malformed option contract {}", symbol)
            continue
        if not math.isfinite(strike) or strike <= 0 or lot_size <= 0 or not item.get("token") or not item.get("expiry"):
            logger.warning("Ignoring incomplete option contract {}", symbol)
            continue
        results.append({
            "token": item.get("token", ""),
            "symbol": symbol,
            "name": item.get("name", ""),
            "expiry": item.get("expiry", ""),
            "strike": strike,
            "option_type": inferred_side,
            "lot_size": lot_size,
        })

    return sorted(results, key=lambda x: x["strike"])


def is_fno_expiry_day(
    underlying: str = "NIFTY",
    today: date | None = None,
    instruments: list[dict[str, Any]] | None = None,
) -> bool | None:
    """True when a listed index-option contract on ``underlying`` expires today.

    Expiry weekdays change by exchange decision and holidays shift them, so this
    reads the contract master instead of assuming a weekday. Returns None when
    the master is unavailable or has no parseable expiries (unknown, not False).
    """
    from datetime import datetime as _datetime

    today = today or _datetime.now(IST).date()
    try:
        contracts = find_nfo_options(underlying, instruments=instruments)
    except (InstrumentMasterUnavailableError, OSError, ValueError):
        return None
    parsed = 0
    for contract in contracts:
        try:
            expiry = _datetime.strptime(str(contract["expiry"]).title(), "%d%b%Y").date()
        except ValueError:
            continue
        parsed += 1
        if expiry == today:
            return True
    return False if parsed else None
