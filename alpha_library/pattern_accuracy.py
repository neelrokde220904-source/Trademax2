"""Rolling forward-outcome scoring for detected chart patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NEUTRAL_BASE_RATE_PRIOR = 0.5


@dataclass(frozen=True)
class AccuracyScore:
    """Forward-return and hit-rate summary for one pattern."""

    observations: int
    average_close_return: float
    end_close_return: float
    success_rate: float


class AccuracyTrials:
    """Evaluate patterns only against bars following their signal timestamps."""

    def __init__(self, candles_ahead_qty: int = 30) -> None:
        if candles_ahead_qty < 1:
            raise ValueError("candles_ahead_qty must be positive")
        self.candles_ahead_qty = candles_ahead_qty

    def score(
        self, close: pd.Series, occurrences: list[int], direction: int = 1
    ) -> AccuracyScore:
        if direction not in (-1, 1):
            raise ValueError("direction must be 1 (bullish) or -1 (bearish)")
        values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
        returns: list[float] = []
        average_returns: list[float] = []
        for index in occurrences:
            if index < 0 or index >= len(values) or index + self.candles_ahead_qty >= len(values):
                continue
            start = values[index]
            future = values[index + 1 : index + self.candles_ahead_qty + 1]
            if start <= 0 or not np.isfinite(start) or not np.all(np.isfinite(future)):
                continue
            returns.append(direction * (future[-1] / start - 1))
            average_returns.append(direction * (float(np.mean(future)) / start - 1))
        if not returns:
            return AccuracyScore(0, 0.0, 0.0, 0.0)
        return AccuracyScore(
            len(returns),
            float(np.mean(average_returns)),
            float(np.mean(returns)),
            float(np.mean(np.asarray(returns) > 0)),
        )

    def accuracy_to_average_close(
        self, close: pd.Series, occurrences: list[int], direction: int = 1
    ) -> float:
        """Average directional return to the mean close over the forward window."""
        return self.score(close, occurrences, direction).average_close_return

    def accuracy_to_end_close(
        self, close: pd.Series, occurrences: list[int], direction: int = 1
    ) -> float:
        """Average directional return to the final close in the forward window."""
        return self.score(close, occurrences, direction).end_close_return

    def get_average_percent_pattern_accuracy(
        self, close: pd.Series, occurrences: list[int], direction: int = 1
    ) -> float:
        """Return the empirical fraction of positive forward pattern outcomes."""
        return self.score(close, occurrences, direction).success_rate

    GetAverPercentPatternAccuracy = get_average_percent_pattern_accuracy
    AccuracyToAverageClose = accuracy_to_average_close
    AccuracyToEndClose = accuracy_to_end_close


def zigzag_pivots(values: pd.Series, reversal_pct: float = 0.05) -> list[tuple[int, str, float]]:
    """Return confirmed percentage-reversal pivots (confirmation is delayed)."""
    if not 0 < reversal_pct < 1:
        raise ValueError("reversal_pct must be between 0 and 1")
    prices = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(prices) < 2 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return []
    pivots: list[tuple[int, str, float]] = []
    trend = 0
    extreme_idx, extreme_price = 0, prices[0]
    for index in range(1, len(prices)):
        price = prices[index]
        if trend == 0:
            if price >= extreme_price * (1 + reversal_pct):
                pivots.append((extreme_idx, "low", float(extreme_price)))
                trend, extreme_idx, extreme_price = 1, index, price
            elif price <= extreme_price * (1 - reversal_pct):
                pivots.append((extreme_idx, "high", float(extreme_price)))
                trend, extreme_idx, extreme_price = -1, index, price
            continue
        if trend > 0 and price >= extreme_price:
            extreme_idx, extreme_price = index, price
        elif trend < 0 and price <= extreme_price:
            extreme_idx, extreme_price = index, price
        change = price / extreme_price - 1
        if trend > 0 and change <= -reversal_pct:
            pivots.append((extreme_idx, "high", float(extreme_price)))
            trend, extreme_idx, extreme_price = -1, index, price
        elif trend < 0 and change >= reversal_pct:
            pivots.append((extreme_idx, "low", float(extreme_price)))
            trend, extreme_idx, extreme_price = 1, index, price
    return pivots


def detect_fibonacci_formations(values: pd.Series, reversal_pct: float = 0.05) -> dict[str, bool]:
    """Detect coarse ABCD/Gartley and common reversal structures from confirmed pivots."""
    pivots = zigzag_pivots(values, reversal_pct)
    alternating = pivots[-5:]
    result = {"abcd": False, "gartley": False, "double_bottom": False,
              "double_top": False, "head_and_shoulders": False}
    if len(alternating) >= 4:
        a, b, c, d = [pivot[2] for pivot in alternating[-4:]]
        ab, bc, cd = abs(b - a), abs(c - b), abs(d - c)
        result["abcd"] = ab > 0 and 0.382 <= bc / ab <= 0.886 and 0.9 <= cd / ab <= 1.1
        result["gartley"] = ab > 0 and 0.55 <= bc / ab <= 0.70 and 0.72 <= cd / ab <= 0.82
    if len(alternating) >= 5:
        p0, p1, p2, p3, p4 = alternating
        tolerance = 0.03
        result["double_bottom"] = (
            p0[1] == p2[1] == "low" and abs(p0[2] / p2[2] - 1) <= tolerance
        )
        result["double_top"] = (
            p0[1] == p2[1] == "high" and abs(p0[2] / p2[2] - 1) <= tolerance
        )
        result["head_and_shoulders"] = (
            p0[1] == p2[1] == p4[1] == "high" and p1[1] == p3[1] == "low"
            and p2[2] > max(p0[2], p4[2]) and abs(p0[2] / p4[2] - 1) <= tolerance
        )
    return result
