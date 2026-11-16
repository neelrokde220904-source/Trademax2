"""Validated daily returns for entry VaR and directional correlation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from data.indicators import completed_intraday_bars
from data.nse_calendar import (
    MUHURAT_SESSION, REGULAR_WEEKEND_SESSIONS, calendar_year_known,
    expected_intraday_bar_starts, is_nse_holiday,
)

MIN_DAILY_RETURNS = 20
IST = "Asia/Kolkata"
class RiskHistoryUnavailable(ValueError):
    """Risk history or its session calendar cannot be verified."""


def _session_dates(as_of: str | None) -> list[date]:
    try:
        at = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz=IST)
        if pd.isna(at):
            raise ValueError("invalid timestamp")
        if at.tzinfo is not None:
            at = at.tz_convert(IST)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskHistoryUnavailable("invalid risk decision timestamp") from exc
    day = at.date() - timedelta(days=1)
    if not calendar_year_known(at.year):
        raise RiskHistoryUnavailable(f"NSE holiday calendar for {at.year} is not on file")
    dates: list[date] = []
    while len(dates) < MIN_DAILY_RETURNS + 1:
        if not calendar_year_known(day.year):
            raise RiskHistoryUnavailable(f"NSE holiday calendar for {day.year} is not on file")
        if (
            (day.weekday() < 5 and not is_nse_holiday(day))
            or day in REGULAR_WEEKEND_SESSIONS
            or day == MUHURAT_SESSION
        ):
            dates.append(day)
        day -= timedelta(days=1)
    return list(reversed(dates))


def daily_returns(df: pd.DataFrame, as_of: str | None = None) -> pd.Series:
    """Close-to-close returns from the latest 21 complete expected NSE sessions."""
    days = _session_dates(as_of)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RiskHistoryUnavailable("daily return history is missing")
    try:
        bars = completed_intraday_bars(df, as_of, bar_minutes=15)
        closes: list[float] = []
        for day in days:
            session = bars[bars["datetime"].dt.date == day]
            expected = expected_intraday_bar_starts(day, 15)
            prices = session[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = (
                pd.DatetimeIndex(session["datetime"]).equals(expected)
                and prices.map(math.isfinite).all().all()
                and (prices > 0).all().all()
                and (prices["high"] >= prices[["open", "low", "close"]].max(axis=1)).all()
                and (prices["low"] <= prices[["open", "high", "close"]].min(axis=1)).all()
            )
            if not valid:
                raise RiskHistoryUnavailable(f"missing or invalid complete NSE session {day}")
            closes.append(float(prices["close"].iloc[-1]))
    except RiskHistoryUnavailable:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RiskHistoryUnavailable("invalid intraday history for daily risk") from exc
    return pd.Series(closes, index=days, dtype=float).pct_change().iloc[1:]


def historical_var(returns: pd.Series, action: str) -> float:
    """Empirical 95% one-day loss percentile, conditional on entry direction."""
    if action not in {"BUY", "SELL"} or len(returns) != MIN_DAILY_RETURNS:
        raise RiskHistoryUnavailable("invalid daily returns or position direction")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RiskHistoryUnavailable("non-finite daily return history")
    signed = values if action == "BUY" else -values
    return max(0.0, -float(np.percentile(signed, 5)))


def adverse_correlation(
    candidate: pd.Series, action: str, held: pd.Series, held_quantity: float,
) -> float:
    """Return correlation after aligning sessions and applying position directions."""
    if (
        action not in {"BUY", "SELL"}
        or isinstance(held_quantity, bool)
        or not math.isfinite(held_quantity)
        or held_quantity == 0
    ):
        raise RiskHistoryUnavailable("invalid position direction for correlation")
    aligned = pd.concat([candidate.rename("candidate"), held.rename("held")], axis=1, join="inner")
    if (
        len(aligned) != MIN_DAILY_RETURNS
        or len(candidate) != MIN_DAILY_RETURNS
        or len(held) != MIN_DAILY_RETURNS
        or not candidate.index.is_unique
        or not held.index.is_unique
    ):
        raise RiskHistoryUnavailable("insufficient aligned daily return history")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or aligned["candidate"].std() == 0 or aligned["held"].std() == 0:
        raise RiskHistoryUnavailable("invalid return variance for correlation")
    correlation = float(aligned["candidate"].corr(aligned["held"]))
    if not math.isfinite(correlation):
        raise RiskHistoryUnavailable("invalid return variance for correlation")
    return correlation * (1 if action == "BUY" else -1) * (1 if held_quantity > 0 else -1)


def correlation_block_reason(
    symbol: str, action: str, positions: Mapping[str, Any],
    returns_for: Callable[[str], pd.Series], threshold: float = 0.8,
) -> str | None:
    """Block positively correlated signed exposure, requiring history for every held lot."""
    for held_symbol, position in positions.items():
        if not isinstance(position, dict) or isinstance(position.get("quantity"), bool):
            raise RiskHistoryUnavailable(f"invalid position for {held_symbol}")
        try:
            quantity = float(position.get("quantity"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskHistoryUnavailable(f"invalid position quantity for {held_symbol}") from exc
        if not math.isfinite(quantity):
            raise RiskHistoryUnavailable(f"invalid position quantity for {held_symbol}")
        if quantity == 0:
            continue
        correlation = adverse_correlation(
            returns_for(symbol), action, returns_for(held_symbol), quantity,
        )
        if correlation > threshold:
            return f"Adversely correlated with existing position {held_symbol} (>{threshold:.1f})"
    return None
