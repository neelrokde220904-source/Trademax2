"""US equity session calendar with daylight-saving handling.

Regular sessions are 09:30–16:00 America/New_York (``zoneinfo`` applies the US DST rules, so
the IST equivalent shifts twice a year); early closes and holidays come from
``data/calendars/us_equity_holidays.json`` (NYSE source, retrieved 2026-11-14). A year absent
from the file is unknown and reported as closed: nothing is assumed open.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
CALENDAR_PATH = Path(__file__).resolve().parent / "calendars" / "us_equity_holidays.json"


def _load(path: Path = CALENDAR_PATH) -> dict[int, dict]:
    with path.open() as handle:
        raw = json.load(handle)
    return {
        int(year): {
            "source": entry.get("source", ""),
            "holidays": set(entry.get("holidays", [])),
            "early_closes": dict(entry.get("early_closes", {})),
        }
        for year, entry in raw.items()
    }


_CALENDAR = _load()


def calendar_year_known(year: int) -> bool:
    return year in _CALENDAR


def is_us_holiday(day: date) -> bool:
    return day.isoformat() in _CALENDAR.get(day.year, {}).get("holidays", set())


def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """Open and close (New York time) of a regular or early-close session; None when closed or unknown."""
    if not calendar_year_known(day.year) or day.weekday() >= 5 or is_us_holiday(day):
        return None
    early = _CALENDAR[day.year]["early_closes"].get(day.isoformat())
    close_time = time.fromisoformat(early) if early else REGULAR_CLOSE
    return (
        datetime.combine(day, REGULAR_OPEN, tzinfo=NEW_YORK),
        datetime.combine(day, close_time, tzinfo=NEW_YORK),
    )


def is_open(moment: datetime) -> bool:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    local = moment.astimezone(NEW_YORK)
    bounds = session_bounds(local.date())
    return bounds is not None and bounds[0] <= local < bounds[1]


def next_open(moment: datetime) -> datetime | None:
    """The first session open strictly after ``moment`` within two weeks, or None if unknown/closed."""
    local = moment.astimezone(NEW_YORK)
    for offset in range(15):
        bounds = session_bounds(local.date() + timedelta(days=offset))
        if bounds is not None and bounds[0] > local:
            return bounds[0]
    return None
