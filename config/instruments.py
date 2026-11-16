"""NSE/BSE symbol → Angel One token mapping cache utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_IDENTITY: tuple[str, int, int, int, int] | None = None


def _cache_identity(path: Path) -> tuple[str, int, int, int, int]:
    from data.instrument_master import InstrumentMasterUnavailableError

    try:
        stat = path.stat()
    except OSError as exc:
        raise InstrumentMasterUnavailableError(
            f"Instrument master cache unavailable: {type(exc).__name__}"
        ) from exc
    return str(path), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


def load_instrument_cache() -> dict[str, dict[str, Any]]:
    """Load the cached instrument master JSON into a symbol → details dict."""
    global _CACHE, _CACHE_IDENTITY
    if _CACHE:
        if _CACHE_IDENTITY is None:
            return _CACHE
        from data.instrument_master import validate_cache_age

        cached_path = Path(settings.instrument_cache_path)
        validate_cache_age(cached_path)
        if _cache_identity(cached_path) == _CACHE_IDENTITY:
            return _CACHE
        invalidate_cache()

    cache_path = Path(settings.instrument_cache_path)
    if not cache_path.exists():
        logger.warning("Instrument cache not found at {}. Run instrument_master.download() first.", cache_path)
        return {}

    from data.instrument_master import InstrumentMasterUnavailableError, load_instrument_master

    identity = _cache_identity(cache_path)
    raw = load_instrument_master()

    new_cache: dict[str, dict[str, Any]] = {}
    for item in raw:
        symbol = item.get("symbol", "")
        exchange = item.get("exch_seg", "")
        key = f"{exchange}:{symbol}" if exchange else symbol
        new_cache[key] = {
            "token": item.get("token", ""),
            "symbol": symbol,
            "name": item.get("name", ""),
            "exchange": exchange,
            "lot_size": int(item.get("lotsize", 1)),
            "tick_size": float(item.get("tick_size", 0.05)),
            "instrument_type": item.get("instrumenttype", ""),
            "expiry": item.get("expiry", ""),
            "strike": item.get("strike", ""),
            "option_type": item.get("optiontype", ""),
        }

    if _cache_identity(cache_path) != identity:
        raise InstrumentMasterUnavailableError("Instrument master cache changed during loading")
    _CACHE = new_cache
    _CACHE_IDENTITY = identity
    logger.info("Loaded {} instruments into cache.", len(_CACHE))
    return _CACHE


def get_token(symbol: str, exchange: str = "NSE") -> str | None:
    """Return the Angel One token for a given symbol and exchange."""
    entry = get_instrument(symbol, exchange)
    return entry["token"] if entry else None


def get_instrument(symbol: str, exchange: str = "NSE") -> dict[str, Any] | None:
    """Return full instrument details for a given symbol."""
    cache = load_instrument_cache()
    if exchange == "NSE":
        from config.symbol_changes import canonical_nse_symbol

        symbol = canonical_nse_symbol(symbol)
    key = f"{exchange}:{symbol}"
    entry = cache.get(key)
    if entry is None and exchange == "NSE" and not symbol.endswith("-EQ"):
        entry = cache.get(f"NSE:{symbol}-EQ")
    return entry


def invalidate_cache() -> None:
    """Clear the in-memory cache (e.g. after daily refresh)."""
    global _CACHE, _CACHE_IDENTITY
    _CACHE = {}
    _CACHE_IDENTITY = None
