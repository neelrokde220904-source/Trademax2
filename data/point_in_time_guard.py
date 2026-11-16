"""Point-in-time filtering for historical news, filings, and fundamentals."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def filter_published_as_of(
    records: pd.DataFrame,
    as_of_timestamp: datetime | pd.Timestamp,
    *,
    published_at_column: str = "published_at",
) -> pd.DataFrame:
    """Keep only records with an explicit publication time at or before `as_of`."""
    if published_at_column not in records.columns:
        raise ValueError(f"Required point-in-time column '{published_at_column}' is missing")
    as_of = pd.Timestamp(as_of_timestamp)
    if pd.isna(as_of):
        raise ValueError("as_of_timestamp must be valid")
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    published = pd.to_datetime(records[published_at_column], errors="coerce", utc=True)
    if published.isna().any():
        raise ValueError("Every record must have a valid publication timestamp")
    return records.loc[published <= as_of].copy()


def filter_records_as_of(
    records: list[dict[str, Any]],
    as_of_timestamp: datetime | pd.Timestamp,
    *,
    published_at_key: str = "published_at",
) -> list[dict[str, Any]]:
    """List-of-records wrapper with the same fail-closed timestamp contract."""
    if not records:
        return []
    return filter_published_as_of(
        pd.DataFrame(records), as_of_timestamp, published_at_column=published_at_key
    ).to_dict(orient="records")
