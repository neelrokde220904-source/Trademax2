"""Verified NSE cash-session calendar, loaded from data/calendars/nse_equity_holidays.json.

A year absent from the file is *unknown*: entries must be blocked for it (never
assumed open), while exits do not consult the calendar at all.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

CALENDAR_PATH = Path(__file__).resolve().parent / "calendars" / "nse_equity_holidays.json"


def _load_calendar(path: Path = CALENDAR_PATH) -> dict[int, dict]:
    with path.open() as handle:
        raw = json.load(handle)
    calendar: dict[int, dict] = {}
    for year, entry in raw.items():
        holidays = {str(day) for day in entry.get("holidays", [])}
        sessions = {
            date.fromisoformat(item["date"]): item["kind"]
            for item in entry.get("special_sessions", [])
        }
        calendar[int(year)] = {
            "source": entry.get("source", ""), "holidays": holidays, "special_sessions": sessions,
        }
    return calendar


_CALENDAR = _load_calendar()

# NSE/CMTR/72349 (Budget session, normal hours) and NSE/CMTR/76941 (Muhurat).
REGULAR_WEEKEND_SESSIONS = frozenset(
    day for entry in _CALENDAR.values()
    for day, kind in entry["special_sessions"].items() if kind == "regular_weekend_session"
)
MUHURAT_SESSION = next(
    day for entry in _CALENDAR.values()
    for day, kind in entry["special_sessions"].items() if kind == "muhurat"
)


def _override_holidays() -> set[str]:
    from config.settings import settings

    return set(settings.nse_holidays)


def calendar_year_known(year: int) -> bool:
    """True when the holiday list for ``year`` is on file (or supplied by an override)."""
    return year in _CALENDAR or any(day[:4] == str(year) for day in _override_holidays())


def is_nse_holiday(day: date | str) -> bool:
    """File holidays merged with the ``settings.nse_holidays`` override (weekends not included)."""
    iso = day if isinstance(day, str) else day.isoformat()
    year = int(iso[:4])
    return iso in _CALENDAR.get(year, {}).get("holidays", set()) or iso in _override_holidays()


def nse_holiday_set() -> set[str]:
    merged = set(_override_holidays())
    for entry in _CALENDAR.values():
        merged |= entry["holidays"]
    return merged


def expected_intraday_bar_starts(day: date, bar_minutes: int) -> pd.DatetimeIndex:
    if day == MUHURAT_SESSION:
        if bar_minutes != 15:
            raise ValueError("Muhurat candle boundaries are verified only for 15-minute bars")
        return pd.date_range(
            pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=20),
            periods=5, freq="15min",
        )
    if bar_minutes <= 0 or 375 % bar_minutes:
        raise ValueError("Bar interval must divide the regular 375-minute NSE session")
    return pd.date_range(
        pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15),
        periods=375 // bar_minutes, freq=f"{bar_minutes}min",
    )
