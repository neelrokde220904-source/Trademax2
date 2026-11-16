"""Dependency-light chart formation and oscillator-divergence detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    missing = {"open", "high", "low", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    return frame


def detect_nr7(data: pd.DataFrame) -> bool:
    """Return whether the latest bar has the narrowest range of seven bars."""
    frame = _ohlcv(data)
    if len(frame) < 7:
        return False
    ranges = frame["high"] - frame["low"]
    return bool(ranges.iloc[-1] <= ranges.iloc[-7:].min())


def detect_vcp(data: pd.DataFrame, contractions: int = 3) -> bool:
    """Detect successively tighter price ranges, without asserting a breakout."""
    frame = _ohlcv(data)
    if contractions < 2 or len(frame) < contractions * 5:
        return False
    ranges = (frame["high"] - frame["low"]).to_numpy(dtype=float)
    segments = np.array_split(ranges[-contractions * 5 :], contractions)
    widths = [float(np.nanmedian(part)) for part in segments]
    return all(np.isfinite(width) and width > 0 for width in widths) and all(
        later < earlier for earlier, later in zip(widths, widths[1:])
    )


def detect_cup_and_handle(data: pd.DataFrame) -> bool:
    """Heuristic cup-and-handle shape test using the latest 40 bars."""
    frame = _ohlcv(data)
    if len(frame) < 40:
        return False
    highs = frame["high"].iloc[-40:-8].to_numpy(dtype=float)
    closes = frame["close"].iloc[-40:].to_numpy(dtype=float)
    left_rim, right_rim = float(np.max(highs[:8])), float(np.max(highs[-8:]))
    cup_low = float(np.min(closes[:24]))
    handle_low = float(np.min(closes[-8:]))
    rim = (left_rim + right_rim) / 2
    if rim <= 0 or cup_low >= rim or handle_low < cup_low:
        return False
    depth = (rim - cup_low) / rim
    handle_depth = (right_rim - handle_low) / right_rim if right_rim > 0 else 1
    return 0.08 <= depth <= 0.5 and handle_depth <= depth * 0.5 and right_rim >= left_rim * 0.95


def detect_high_tight_flag(data: pd.DataFrame) -> bool:
    """Detect a sharp preceding advance followed by a shallow, tight pullback."""
    frame = _ohlcv(data)
    if len(frame) < 25:
        return False
    advance = float(frame["close"].iloc[-25:-10].iloc[-1] / frame["close"].iloc[-25] - 1)
    prior_high = float(frame["high"].iloc[-10:].max())
    current = float(frame["close"].iloc[-1])
    pullback = (prior_high - current) / prior_high if prior_high > 0 else 1
    recent_ranges = (frame["high"].iloc[-10:] - frame["low"].iloc[-10:]) / frame["close"].iloc[-10:]
    return advance >= 0.8 and pullback <= 0.25 and float(recent_ranges.median()) <= 0.08


def detect_fair_value_gaps(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify unfilled three-candle gaps on the latest OHLC observations."""
    frame = _ohlcv(data)
    if len(frame) < 3:
        return []
    previous_high = float(frame["high"].iloc[-3])
    previous_low = float(frame["low"].iloc[-3])
    latest_low = float(frame["low"].iloc[-1])
    latest_high = float(frame["high"].iloc[-1])
    gaps = []
    if latest_low > previous_high:
        gaps.append({"direction": "bullish", "lower": previous_high, "upper": latest_low})
    if latest_high < previous_low:
        gaps.append({"direction": "bearish", "lower": latest_high, "upper": previous_low})
    return gaps


def detect_divergence(prices: pd.Series, oscillator: pd.Series, lookback: int = 40) -> str | None:
    """Compare the two latest local swing lows/highs for oscillator divergence."""
    if lookback < 5 or len(prices) < lookback or len(oscillator) < lookback:
        return None
    price = prices.iloc[-lookback:].astype(float).to_numpy()
    osc = oscillator.iloc[-lookback:].astype(float).to_numpy()
    low_points = [i for i in range(1, len(price) - 1) if price[i] < price[i - 1] and price[i] <= price[i + 1]]
    high_points = [i for i in range(1, len(price) - 1) if price[i] > price[i - 1] and price[i] >= price[i + 1]]
    if len(low_points) >= 2:
        first, second = low_points[-2:]
        if price[second] < price[first] and osc[second] > osc[first]:
            return "bullish"
    if len(high_points) >= 2:
        first, second = high_points[-2:]
        if price[second] > price[first] and osc[second] < osc[first]:
            return "bearish"
    return None


def analyze_chart_formations(data: pd.DataFrame) -> dict[str, Any]:
    """Compute supported formations as deterministic, non-advisory observations."""
    frame = _ohlcv(data)
    patterns = {
        "vcp": detect_vcp(frame),
        "cup_and_handle": detect_cup_and_handle(frame),
        "high_tight_flag": detect_high_tight_flag(frame),
        "nr7": detect_nr7(frame),
        "fair_value_gaps": detect_fair_value_gaps(frame),
    }
    for oscillator_name in ("rsi", "macd", "stochastic"):
        if oscillator_name in frame:
            patterns[f"{oscillator_name}_divergence"] = detect_divergence(
                frame["close"], frame[oscillator_name]
            )
    return patterns
