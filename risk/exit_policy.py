"""Shared price-bounded exits and monotonic ATR protection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def collared_exit_limit(
    reference_price: float, action: str, tick_size: float, collar_pct: float,
) -> float:
    """Round inside the collar, never beyond its maximum adverse price."""
    _positive(reference_price, "reference_price")
    _positive(tick_size, "tick_size")
    _positive(collar_pct, "collar_pct")
    if collar_pct >= 1 or action not in {"BUY", "SELL"}:
        raise ValueError("Invalid exit side or collar")
    reference = Decimal(str(reference_price))
    tick = Decimal(str(tick_size))
    collar = Decimal(str(collar_pct))
    bound = reference * (1 + collar if action == "BUY" else 1 - collar)
    rounding = ROUND_FLOOR if action == "BUY" else ROUND_CEILING
    result = (bound / tick).to_integral_value(rounding=rounding) * tick
    return _positive(float(result), "limit_price")


def chandelier_stop(
    action: str, current_stop: float, favorable_price: float, atr: float, multiplier: float,
) -> float:
    """ATR changes can tighten an existing stop but never widen it."""
    for name, value in (
        ("current_stop", current_stop), ("favorable_price", favorable_price),
        ("atr", atr), ("multiplier", multiplier),
    ):
        _positive(value, name)
    if action == "BUY":
        return max(current_stop, favorable_price - multiplier * atr)
    if action == "SELL":
        return min(current_stop, favorable_price + multiplier * atr)
    raise ValueError("Chandelier side must be BUY or SELL")


@dataclass(frozen=True)
class ATRExitPolicy:
    name: str
    stop_multiplier: float
    target_multiplier: float
    trailing_multiplier: float
    atr_timeframe: str = "DAILY_14_COMPLETED_SESSIONS"
    atr_period: int = 14

    def levels(self, entry_price: float, atr: float, action: str) -> tuple[float, float]:
        _positive(entry_price, "entry_price")
        _positive(atr, "atr")
        _positive(self.stop_multiplier, "stop_multiplier")
        _positive(self.target_multiplier, "target_multiplier")
        direction = 1 if action == "BUY" else -1 if action == "SELL" else 0
        if not direction:
            raise ValueError("ATR entry side must be BUY or SELL")
        stop = entry_price - direction * self.stop_multiplier * atr
        target = entry_price + direction * self.target_multiplier * atr
        return _positive(stop, "stop"), _positive(target, "target")


# The live default stays the baseline; the others are research-only until T10 shows
# out-of-sample improvement.
BASELINE_ATR_POLICY = ATRExitPolicy("daily_atr_baseline_v1", 1.0, 2.5, 2.0)
RESEARCH_ATR_POLICY = ATRExitPolicy("daily_atr_2x_4x_v1", 2.0, 4.0, 2.0)
INTRADAY_ATR_POLICY = ATRExitPolicy(
    "intraday_atr_2x_4x_v1", 2.0, 4.0, 2.0, atr_timeframe="INTRADAY_15M_14", atr_period=14,
)
DAILY_ATR22_POLICY = ATRExitPolicy(
    "daily_atr22_2x_4x_v1", 2.0, 4.0, 2.0, atr_timeframe="DAILY_22_COMPLETED_SESSIONS", atr_period=22,
)
# Chandelier trail of k=3 ATR(22) active from entry; the 6x target is only a profit cap.
CHANDELIER_ATR22_POLICY = ATRExitPolicy(
    "chandelier_atr22_k3_v1", 3.0, 6.0, 3.0, atr_timeframe="DAILY_22_COMPLETED_SESSIONS", atr_period=22,
)
ATR_EXIT_POLICIES: dict[str, ATRExitPolicy] = {
    policy.name: policy
    for policy in (
        BASELINE_ATR_POLICY, RESEARCH_ATR_POLICY, INTRADAY_ATR_POLICY,
        DAILY_ATR22_POLICY, CHANDELIER_ATR22_POLICY,
    )
}


def get_atr_exit_policy(name: str) -> ATRExitPolicy:
    policy = ATR_EXIT_POLICIES.get(name) if isinstance(name, str) else None
    if policy is None:
        raise ValueError(f"Unknown ATR exit policy: {name}")
    return policy


def atr_before(bars: Any, period: int, before: Any, *, bar_minutes: int | None = None) -> float:
    """ATR(period) from bars completed strictly before ``before`` (no look-ahead).

    ``bars`` needs a ``high``/``low``/``close`` frame indexed by bar start time.
    With ``bar_minutes`` a bar counts only once its full interval has elapsed.
    """
    import pandas as pd

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("ATR needs a non-empty bar frame")
    cutoff = pd.Timestamp(before)
    index = pd.DatetimeIndex(bars.index)
    if index.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(index.tz)
    elif index.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    ends = index + pd.Timedelta(minutes=bar_minutes) if bar_minutes else index + pd.Timedelta(days=1)
    completed = bars.loc[ends <= cutoff].sort_index()
    if len(completed) < period + 1:
        raise ValueError(f"ATR({period}) needs {period + 1} completed bars; have {len(completed)}")
    previous_close = completed["close"].shift(1)
    true_range = pd.concat([
        completed["high"] - completed["low"],
        (completed["high"] - previous_close).abs(),
        (completed["low"] - previous_close).abs(),
    ], axis=1).max(axis=1).iloc[1:]
    # Wilder smoothing seeded with the first ``period`` true ranges.
    values = true_range.to_numpy(dtype=float)
    atr = float(values[:period].mean())
    for value in values[period:]:
        atr = (atr * (period - 1) + float(value)) / period
    return _positive(atr, "atr")


def structure_stop(
    side: str, *, entry: float, structural_level: float, atr: float, buffer_mult: float, tick: float,
) -> float | None:
    """Stop just beyond a structural level; None when its distance is outside [0.5, 3] ATR.

    A None result means the candidate is skipped, never widened or tightened to fit.
    """
    for name, value in (("entry", entry), ("structural_level", structural_level), ("atr", atr), ("tick", tick)):
        _positive(value, name)
    if buffer_mult < 0 or not math.isfinite(buffer_mult):
        raise ValueError("buffer_mult must be non-negative and finite")
    if side not in {"BUY", "SELL"}:
        raise ValueError("Structure stop side must be BUY or SELL")
    raw = Decimal(str(structural_level)) - (
        Decimal(str(buffer_mult)) * Decimal(str(atr)) if side == "BUY"
        else -Decimal(str(buffer_mult)) * Decimal(str(atr))
    )
    step = Decimal(str(tick))
    # Round away from the entry so the stop stays beyond the structure.
    stop = float((raw / step).to_integral_value(rounding=ROUND_FLOOR if side == "BUY" else ROUND_CEILING) * step)
    if stop <= 0:
        return None
    distance = entry - stop if side == "BUY" else stop - entry
    if distance <= 0 or not 0.5 * atr <= distance <= 3.0 * atr:
        return None
    return stop


def stop_target_hit(
    action: str, stop: float, target: float, high: float, low: float,
) -> str | None:
    """Which level a bar (or a single price, as high == low) crossed; the stop wins ties.

    A missing level (0 or None) is never triggered: reconciliation reports it instead.
    """
    if action == "BUY":
        if stop and stop > 0 and low <= stop:
            return "STOP_LOSS"
        if target and target > 0 and high >= target:
            return "TARGET"
    elif action == "SELL":
        if stop and stop > 0 and high >= stop:
            return "STOP_LOSS"
        if target and target > 0 and low <= target:
            return "TARGET"
    return None


def gap_fill_price(action: str, reason: str, level: float, open_price: float) -> float:
    """A bar that opens through a level fills at the open (the gap), never at the better level."""
    if action == "BUY":
        return min(level, open_price) if reason == "STOP_LOSS" else max(level, open_price)
    return max(level, open_price) if reason == "STOP_LOSS" else min(level, open_price)


def ratcheted_stop(
    action: str, current_stop: float, highest: float, lowest: float, atr: float, multiplier: float,
) -> float:
    """Trail from the favourable extreme; the result can only tighten ``current_stop``."""
    favorable = highest if action == "BUY" else lowest
    return chandelier_stop(action, current_stop, favorable, atr, multiplier)


INTRADAY_EXIT_CLOCK = (15, 15)
SESSION_OPEN_CLOCK = (9, 15)
BTST_EXIT_DELAY_MINUTES = 30
FAILED_THESIS_MIN_R = 0.5


def next_session_date(day: date) -> date:
    """Next NSE cash session after ``day``; raises when the holiday calendar is unknown."""
    from data.nse_calendar import calendar_year_known, is_nse_holiday

    candidate = day + timedelta(days=1)
    for _ in range(14):
        if not calendar_year_known(candidate.year):
            raise ValueError(f"NSE holiday calendar for {candidate.year} is not on file")
        if candidate.weekday() < 5 and not is_nse_holiday(candidate):
            return candidate
        candidate += timedelta(days=1)
    raise ValueError("No NSE session found within 14 days")


def time_exit_due(
    horizon: str, entry_time: datetime, now: datetime, failed_thesis_bars: int | None = None, *,
    bars_elapsed: int | None = None, max_favourable_r: float | None = None,
    swing_max_days: int | None = None, positional_max_days: int | None = None,
) -> str | None:
    """Reason a position must be exited by time or a failed thesis, else None.

    INTRADAY: 15:15 (or the next open if carried). BTST: 30 minutes after the next
    session opens. SWING/POSITIONAL: a maximum holding period in calendar days.
    A failed thesis (no favourable excursion of FAILED_THESIS_MIN_R after
    ``failed_thesis_bars`` completed bars) applies to every horizon when measured.
    An unknown calendar makes a BTST exit due once the entry date has passed after the
    delay: exits are never deferred because of missing reference data.
    """
    horizon = str(horizon or "INTRADAY").upper()
    clock = (now.hour, now.minute)
    if (
        failed_thesis_bars and bars_elapsed is not None and max_favourable_r is not None
        and bars_elapsed >= failed_thesis_bars and max_favourable_r < FAILED_THESIS_MIN_R
    ):
        return "FAILED_THESIS"
    if horizon == "INTRADAY":
        if clock >= INTRADAY_EXIT_CLOCK:
            return "INTRADAY_TIME_EXIT"
        if entry_time.date() < now.date() and clock >= SESSION_OPEN_CLOCK:
            return "OVERDUE_INTRADAY_EXIT"
        return None
    if horizon == "BTST":
        delay_clock = (SESSION_OPEN_CLOCK[0] * 60 + SESSION_OPEN_CLOCK[1] + BTST_EXIT_DELAY_MINUTES)
        if now.hour * 60 + now.minute < delay_clock and now.date() == entry_time.date():
            return None
        try:
            due_day = next_session_date(entry_time.date())
        except ValueError:
            due_day = entry_time.date() + timedelta(days=1)
        due = datetime.combine(due_day, datetime.min.time(), tzinfo=now.tzinfo) + timedelta(
            minutes=delay_clock,
        )
        return "BTST_TIME_EXIT" if now >= due else None
    if horizon in {"SWING", "POSITIONAL"}:  # legacy DELIVERY rows have no declared holding limit
        from config.settings import settings

        limit = (
            positional_max_days or settings.positional_max_holding_days
            if horizon == "POSITIONAL"
            else swing_max_days or settings.swing_max_holding_days
        )
        if (now.date() - entry_time.date()).days >= limit:
            return "MAX_HOLDING_PERIOD"
    return None
