"""BTST strength: late-session strength with volume confirmation, held to the next session.

Uses completed 15-minute bars only. A candidate needs the last completed bar of the
session to close in the top quarter of the day's range, above the session VWAP, on
cumulative volume above the same-time history, and is allowed only from 15:00 IST.
It exits through the BTST time exit (next session open + 30 minutes) or its stop/target.
"""

from __future__ import annotations

import math
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


class BTSTStrength:
    name = "BTSTStrength"
    earliest_entry = time(15, 0)
    latest_entry = time(15, 20)
    min_close_position = 0.75
    min_relative_volume = 1.2
    min_prior_sessions = 5

    def scan(self, symbol: str, bars: pd.DataFrame, as_of: str | None = None) -> dict[str, Any] | None:
        frame = bars.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if not {"high", "low", "close", "volume"}.issubset(frame.columns) or frame.empty:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
        frame.index = index
        frame = frame.sort_index(kind="stable")
        reference = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz=IST)
        reference = reference.tz_localize(IST) if reference.tzinfo is None else reference.tz_convert(IST)
        frame = frame[frame.index + pd.Timedelta(minutes=15) <= reference]
        if frame.empty:
            return None
        day = frame.index[-1].date()
        session = frame[frame.index.date == day]
        last_end = (session.index[-1] + pd.Timedelta(minutes=15)).time()
        if day != reference.date() or not self.earliest_entry <= last_end <= self.latest_entry:
            return None
        day_high, day_low = float(session["high"].max()), float(session["low"].min())
        close = float(session["close"].iloc[-1])
        if not all(map(math.isfinite, (day_high, day_low, close))) or day_high <= day_low:
            return None
        if (close - day_low) / (day_high - day_low) < self.min_close_position:
            return None
        typical = (session["high"] + session["low"] + session["close"]) / 3
        volume = pd.to_numeric(session["volume"], errors="coerce").clip(lower=0)
        if volume.sum() <= 0 or close <= float((typical * volume).sum() / volume.sum()):
            return None
        cutoff = session.index[-1].time()
        history = []
        for session_day, group in frame[frame.index.date < day].groupby(frame.index.date[frame.index.date < day]):
            upto = group[group.index.time <= cutoff]
            if len(upto) == len(session):
                history.append(float(pd.to_numeric(upto["volume"], errors="coerce").clip(lower=0).sum()))
        history = history[-20:]
        if len(history) < self.min_prior_sessions:
            return None
        average = sum(history) / len(history)
        if average <= 0 or float(volume.sum()) / average < self.min_relative_volume:
            return None
        return {
            "symbol": symbol, "action": "BUY", "price": close, "holding_period": "BTST",
            "strategy": self.name,
            "reasoning": (
                f"Late-session strength: close in the top {int((1 - self.min_close_position) * 100)}% of "
                f"the day's range, above VWAP, volume {float(volume.sum()) / average:.2f}x the same-time average."
            ),
        }


btst_strength = BTSTStrength()
