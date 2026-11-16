"""Validated sector metadata and projected gross sector exposure."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from config.settings import settings

SECTOR_SOURCE = "yfinance.Ticker.info"


class SectorMetadataUnavailable(ValueError):
    """A sector classification cannot be trusted for entry risk."""


def validate_candidate_sector(symbol: str, data: Any) -> dict[str, str]:
    """Validate provider, symbol classification and freshness for an entry candidate."""
    if not isinstance(data, Mapping):
        raise SectorMetadataUnavailable("sector classification is missing")
    sector = data.get("sector")
    source = data.get("sector_source")
    observed_at = data.get("sector_observed_at")
    if (
        not isinstance(sector, str) or not sector.strip()
        or sector.strip().casefold() in {"unknown", "none", "n/a"}
    ):
        raise SectorMetadataUnavailable("sector classification is missing or unknown")
    if source != SECTOR_SOURCE:
        raise SectorMetadataUnavailable("sector classification source is unverified")
    if not isinstance(observed_at, str):
        raise SectorMetadataUnavailable("sector classification timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SectorMetadataUnavailable("sector classification timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise SectorMetadataUnavailable("sector classification timestamp lacks a timezone")
    age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    max_age = settings.sector_metadata_max_age_hours * 3600
    if not math.isfinite(age) or age < -60 or age > max_age:
        raise SectorMetadataUnavailable("sector classification is stale or future-dated")
    if not isinstance(symbol, str) or not symbol.strip():
        raise SectorMetadataUnavailable("sector classification symbol is invalid")
    return {
        "symbol": symbol.strip().upper(),
        "sector": sector.strip().casefold(),
        "sector_source": source,
        "sector_observed_at": timestamp.isoformat(),
    }


def sector_exposure_by_name(
    positions: Mapping[str, Any], equity: float,
) -> dict[str, float]:
    """Calculate gross notional by normalized sector as a share of equity."""
    if isinstance(equity, bool) or not isinstance(equity, (int, float)):
        raise SectorMetadataUnavailable("invalid equity for sector exposure")
    if not math.isfinite(equity) or equity <= 0:
        raise SectorMetadataUnavailable("invalid equity for sector exposure")
    notionals: dict[str, float] = {}
    for symbol in sorted(positions):
        position = positions[symbol]
        if not isinstance(position, Mapping):
            raise SectorMetadataUnavailable(f"invalid position for {symbol}")
        quantity = position.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or not math.isfinite(quantity):
            raise SectorMetadataUnavailable(f"invalid quantity for {symbol}")
        if quantity == 0:
            continue
        mark = position.get("ltp")
        if isinstance(mark, bool) or not isinstance(mark, (int, float)) or not math.isfinite(mark) or mark <= 0:
            raise SectorMetadataUnavailable(f"invalid mark for {symbol}")
        sector = position.get("sector")
        if (
            not isinstance(sector, str) or not sector.strip()
            or sector.strip().casefold() in {"unknown", "none", "n/a"}
        ):
            raise SectorMetadataUnavailable(f"sector classification unavailable for held position {symbol}")
        if position.get("sector_source") != SECTOR_SOURCE:
            raise SectorMetadataUnavailable(f"sector source is unverified for held position {symbol}")
        observed_at = position.get("sector_observed_at")
        if not isinstance(observed_at, str):
            raise SectorMetadataUnavailable(f"sector timestamp is unavailable for held position {symbol}")
        try:
            timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SectorMetadataUnavailable(
                f"sector timestamp is invalid for held position {symbol}"
            ) from exc
        if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
            raise SectorMetadataUnavailable(
                f"sector timestamp is invalid for held position {symbol}"
            )
        notionals[sector.strip().casefold()] = (
            notionals.get(sector.strip().casefold(), 0.0) + abs(float(quantity) * float(mark))
        )
    return {sector: notional / float(equity) for sector, notional in notionals.items()}


def sector_limit_block_reason(
    positions: Mapping[str, Any], equity: float,
) -> str | None:
    """Reject a projected book whose gross exposure exceeds the per-sector cap."""
    exposures = sector_exposure_by_name(positions, equity)
    for sector, exposure in sorted(exposures.items()):
        if exposure > settings.max_sector_exposure_pct:
            return (
                f"Projected {sector} sector exposure {exposure:.2%} exceeds "
                f"{settings.max_sector_exposure_pct:.2%}"
            )
    return None
