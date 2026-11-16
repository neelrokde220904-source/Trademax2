"""Deterministic macro-trend confirmation for bullish technical patterns."""

from __future__ import annotations

import pandas as pd


def bullish_pattern_is_aligned(data: pd.DataFrame, *, fib_tolerance: float = 0.015) -> bool:
    """Require 50/200 EMA alignment or verified support with relative volume."""
    if not 0 < fib_tolerance <= 0.05:
        raise ValueError("fib_tolerance must be in (0, 0.05]")
    frame = data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "close" not in frame or len(frame) < 200:
        return False
    close = pd.to_numeric(frame["close"], errors="coerce")
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    current = float(close.iloc[-1])
    if current > ema_50 > ema_200:
        return True
    if "volume" not in frame or not pd.notna(ema_200) or ema_200 <= 0:
        return False
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    average_volume = volume.rolling(20).mean().iloc[-1]
    relative_volume = float(volume.iloc[-1] / average_volume) if average_volume and average_volume > 0 else 0
    swing_high = float(pd.to_numeric(frame["high"], errors="coerce").iloc[-200:].max())
    swing_low = float(pd.to_numeric(frame["low"], errors="coerce").iloc[-200:].min())
    if swing_high <= swing_low:
        return False
    retracement_618 = swing_high - 0.618 * (swing_high - swing_low)
    near_support = abs(current - float(ema_200)) / float(ema_200) <= fib_tolerance
    near_fibonacci = abs(current - retracement_618) / retracement_618 <= fib_tolerance
    return relative_volume >= 1.5 and (near_support or near_fibonacci)
