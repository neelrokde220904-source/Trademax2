"""Vectorized, conservative bootstrap gate for candidate trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtester.costs import round_trip_cost, round_trip_cost_vectorized


@dataclass(frozen=True)
class TimeMachineResult:
    """Monte Carlo and historical evidence for one proposed trade."""

    approved: bool
    paths: int
    prob_target_first: float
    prob_stop_first: float
    prob_open_at_close: float
    ev_net: float
    historical_base_rate: float
    accuracy_score: float
    quality_score: float
    replay_count: int
    reason: str


class TimeMachineValidator:
    """Bootstrap five-minute bar profiles and conservatively score a setup."""

    def __init__(
        self, paths: int = 10_000, min_quality_score: float = 0.5,
        random_seed: int | None = None,
    ) -> None:
        if paths < 1:
            raise ValueError("paths must be positive")
        if not 0 <= min_quality_score <= 1:
            raise ValueError("min_quality_score must be between 0 and 1")
        self.paths = paths
        self.min_quality_score = min_quality_score
        self.random_seed = random_seed

    def validate(
        self,
        bars: pd.DataFrame,
        *,
        entry_price: float,
        target_price: float,
        stop_price: float,
        quantity: int,
        is_intraday: bool = True,
        historical_setup_returns: list[float] | None = None,
        accuracy_score: float = 0.5,
        remaining_time_buckets: list[int] | None = None,
    ) -> TimeMachineResult:
        """Return a fail-closed EV decision; returns are decimal fractions.

        `historical_setup_returns` should contain outcomes from exact matching
        historical setups (not all market bars). If no such sample exists,
        the setup base rate is zero and the gate blocks the trade.
        """
        if not all(np.isfinite(value) for value in (entry_price, target_price, stop_price)):
            raise ValueError("entry, target and stop prices must be finite")
        if entry_price <= 0 or quantity <= 0:
            raise ValueError("entry_price and quantity must be positive")
        if not 0 <= accuracy_score <= 1:
            raise ValueError("accuracy_score must be between 0 and 1")
        if target_price <= 0 or stop_price <= 0:
            raise ValueError("target and stop prices must be positive")
        if target_price == entry_price or stop_price == entry_price:
            raise ValueError("target and stop must differ from entry")
        if (target_price > entry_price) != (stop_price < entry_price):
            raise ValueError("target and stop must be on opposite sides of entry")
        history = self._prepare_bars(bars)
        if history.empty:
            raise ValueError("At least two valid historical bars are required")

        close = history["close"].to_numpy(dtype=float)
        open_ = history["open"].to_numpy(dtype=float)
        high = history["high"].to_numpy(dtype=float)
        low = history["low"].to_numpy(dtype=float)
        prior_close = np.roll(close, 1)
        prior_close[0] = open_[0]
        returns = np.divide(close, prior_close, out=np.ones_like(close), where=prior_close != 0) - 1
        high_factor = np.divide(high, close, out=np.ones_like(high), where=close != 0)
        low_factor = np.divide(low, close, out=np.ones_like(low), where=close != 0)
        returns, high_factor, low_factor = returns[1:], high_factor[1:], low_factor[1:]
        buckets = self._time_buckets(history.index)[1:]
        if not len(returns):
            raise ValueError("At least two valid historical bars are required")

        count_steps = max(1, len(remaining_time_buckets or [0]))
        future_buckets = remaining_time_buckets or [0] * count_steps
        rng = np.random.default_rng(self.random_seed)
        previous = np.full(self.paths, entry_price, dtype=float)
        active = np.ones(self.paths, dtype=bool)
        hit_target = np.zeros(self.paths, dtype=bool)
        hit_stop = np.zeros(self.paths, dtype=bool)
        target_up = target_price > entry_price
        target_level, stop_level = target_price, stop_price

        for bucket in future_buckets:
            eligible = np.flatnonzero(buckets == bucket)
            if not len(eligible):
                eligible = np.arange(len(returns))
            samples = rng.choice(eligible, size=self.paths)
            step_return = returns[samples]
            close_price = previous * (1.0 + step_return)
            bar_high = close_price * high_factor[samples]
            bar_low = close_price * low_factor[samples]
            # A bar touching both barriers is counted as a stop to avoid
            # optimistic assumptions about the intrabar price sequence.
            if target_up:
                touches_stop = active & (bar_low <= stop_level)
                touches_target = active & ~touches_stop & (bar_high >= target_level)
            else:
                touches_stop = active & (bar_high >= stop_level)
                touches_target = active & ~touches_stop & (bar_low <= target_level)
            hit_stop |= touches_stop
            hit_target |= touches_target
            active &= ~(touches_stop | touches_target)
            previous = close_price

        open_at_close = active
        p_target = float(hit_target.mean())
        p_stop = float(hit_stop.mean())
        p_close = float(open_at_close.mean())
        target_return = abs(target_price - entry_price) / entry_price
        stop_return = -abs(stop_price - entry_price) / entry_price
        close_returns = (previous[open_at_close] / entry_price - 1) * (1 if target_up else -1)
        gross_returns = np.concatenate((
            np.full(int(hit_target.sum()), target_return),
            np.full(int(hit_stop.sum()), stop_return),
            close_returns,
        ))
        gross_pnl = float(gross_returns.mean() * entry_price * quantity)
        # Conservative estimate: charge costs at the stop/target levels; for
        # paths still open use their simulated close as the exit price.
        exit_prices = np.concatenate((
            np.full(int(hit_target.sum()), target_price),
            np.full(int(hit_stop.sum()), stop_price),
            previous[open_at_close],
        ))
        costs = (
            float(np.mean(round_trip_cost_vectorized(
                entry_price, exit_prices, quantity, is_intraday=is_intraday
            ))) if len(exit_prices) else 0.0
        )
        ev_net = gross_pnl - float(costs)

        exact_returns = historical_setup_returns or []
        if any(not np.isfinite(value) for value in exact_returns):
            raise ValueError("historical setup returns must be finite")
        # Include only outcomes that exceeded the round-trip cost in the base rate.
        average_cost_return = (
            round_trip_cost(entry_price, entry_price, quantity, is_intraday=is_intraday)
            / (entry_price * quantity)
        )
        base_rate = (
            sum(value > average_cost_return for value in exact_returns) / len(exact_returns)
            if exact_returns else 0.0
        )
        quality = (base_rate + accuracy_score) / 2 if exact_returns else 0.0
        approved = ev_net > 0 and bool(exact_returns) and quality >= self.min_quality_score
        if not exact_returns:
            reason = "No exact-setup historical sample is available"
        elif ev_net <= 0:
            reason = "Net expected value is not positive"
        elif quality < self.min_quality_score:
            reason = "Historical setup/accuracy score is below threshold"
        else:
            reason = "Monte Carlo and historical setup gates passed"
        return TimeMachineResult(
            approved, self.paths, p_target, p_stop, p_close, ev_net, base_rate,
            accuracy_score, quality, len(exact_returns), reason,
        )

    @staticmethod
    def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        if not required.issubset({str(column).lower() for column in bars.columns}):
            raise ValueError("bars must contain open, high, low and close columns")
        frame = bars.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        frame = frame.loc[:, ["open", "high", "low", "close"]]
        frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
        return frame

    @staticmethod
    def _time_buckets(index: Any) -> np.ndarray:
        if isinstance(index, pd.DatetimeIndex):
            return (index.hour * 60 + index.minute).to_numpy()
        return np.zeros(len(index), dtype=int)


timemachine_validator = TimeMachineValidator()
