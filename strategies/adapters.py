"""One interface for every strategy sleeve: ``generate(symbol, bars_by_tf, as_of) -> TradeCandidate | None``.

Adapters normalise their inputs to *completed* bars only, run the existing (unchanged)
signal logic, then replace the strategy's own fixed stop/target with its sleeve's registered
exit policy so that exits are timeframe-explicit and identical in simulation and paper.
A sleeve never trades unless its own ``strategy_<name>_enabled`` flag is on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from risk.exit_policy import atr_before, get_atr_exit_policy, structure_stop
from strategies.candidate import (
    AssetId, CandidateInvalid, TradeCandidate, horizon_to_product, tick_aligned_levels, to_decimal,
)
from strategies.contracts import StrategyContract, require_strategy_contract

IST = ZoneInfo("Asia/Kolkata")
SESSION_CLOSE = (15, 30)
CANDIDATE_LIFETIME = timedelta(minutes=15)


def _ist(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise CandidateInvalid("as_of must be timezone-aware")
    return moment.astimezone(IST)


def _localize(index: pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(index))
    return index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)


def completed_intraday(bars: Any, minutes: int, as_of: datetime) -> pd.DataFrame | None:
    """Intraday bars whose full interval has elapsed by ``as_of`` (no in-progress bar, no future bar)."""
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return None
    frame = bars.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    frame.index = _localize(frame.index)
    frame = frame.sort_index(kind="stable")
    frame = frame[frame.index + pd.Timedelta(minutes=minutes) <= _ist(as_of)]
    return frame if not frame.empty else None


def completed_daily(bars: Any, as_of: datetime) -> pd.DataFrame | None:
    """Daily bars for finished sessions: earlier dates, or today's only once 15:30 IST has passed."""
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return None
    now = _ist(as_of)
    frame = bars.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    dates = _localize(frame.index).date
    keep = [
        day < now.date() or (day == now.date() and (now.hour, now.minute) >= SESSION_CLOSE)
        for day in dates
    ]
    frame = frame[keep].sort_index(kind="stable")
    return frame if not frame.empty else None


def resample_completed(bars: pd.DataFrame, minutes: int, as_of: datetime) -> pd.DataFrame | None:
    """Aggregate completed finer bars into ``minutes`` bars, keeping only fully elapsed buckets."""
    frame = completed_intraday(bars, 1, as_of)
    if frame is None:
        return None
    rule = f"{minutes}min"
    agg = frame.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["close"])
    agg = agg[agg.index + pd.Timedelta(minutes=minutes) <= _ist(as_of)]
    return agg if not agg.empty else None


def daily_from_intraday(bars: pd.DataFrame, as_of: datetime) -> pd.DataFrame | None:
    """Daily OHLCV from completed intraday bars, for finished sessions only."""
    frame = completed_intraday(bars, 1, as_of)
    if frame is None:
        return None
    agg = frame.groupby(frame.index.date).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    agg.index = pd.DatetimeIndex(pd.to_datetime(agg.index))
    return completed_daily(agg, as_of)


class SleeveAdapter:
    """Shared candidate construction; subclasses supply the sleeve-specific signal."""

    key: str
    contract_name: str
    timeframe: str  # the bar set the signal needs, e.g. "5m", "15m", "1d"

    def __init__(self) -> None:
        self.contract: StrategyContract | None = None

    # -- hooks -------------------------------------------------------------------------
    def signal(self, symbol: str, bars_by_tf: dict[str, Any], as_of: datetime) -> dict[str, Any] | None:
        raise NotImplementedError

    def atr_bars(self, bars_by_tf: dict[str, Any], as_of: datetime, policy: Any) -> tuple[pd.DataFrame | None, int | None]:
        """Completed bars and bar length (minutes, None = daily) matching the policy's ATR timeframe."""
        if policy.atr_timeframe.startswith("INTRADAY_15M"):
            fifteen = completed_intraday(bars_by_tf.get("15m"), 15, as_of)
            if fifteen is None and bars_by_tf.get("5m") is not None:
                fifteen = resample_completed(bars_by_tf["5m"], 15, as_of)
            return fifteen, 15
        return completed_daily(bars_by_tf.get("1d"), as_of), None

    def levels(
        self, signal: dict[str, Any], entry: float, atr: float, policy: Any, tick: Decimal,
    ) -> tuple[float, float] | None:
        """Stop and target from the registered policy; subclasses may substitute a structure stop."""
        return policy.levels(entry, atr, signal["action"])

    # -- public ------------------------------------------------------------------------
    def generate(
        self, symbol: str, bars_by_tf: dict[str, Any], as_of: datetime, *,
        tick: Decimal = Decimal("0.05"), lot_size: int = 1,
    ) -> TradeCandidate | None:
        self.contract = require_strategy_contract(self.contract_name)  # disabled sleeve -> ValueError
        as_of = _ist(as_of)
        signal = self.signal(symbol, bars_by_tf, as_of)
        if not signal or signal.get("action") not in {"BUY", "SELL"}:
            return None
        policy = get_atr_exit_policy(self.contract.default_exit_policy)
        bars, bar_minutes = self.atr_bars(bars_by_tf, as_of, policy)
        if bars is None:
            return None
        try:
            atr = atr_before(bars, policy.atr_period, as_of, bar_minutes=bar_minutes)
        except ValueError:
            return None  # not enough completed history: skip, never guess
        entry = float(signal["price"])
        levels = self.levels(signal, entry, atr, policy, tick)
        if levels is None:
            return None
        side = signal["action"]
        entry, stop, target = tick_aligned_levels(side, entry, levels[0], levels[1], tick)
        horizon = signal.get("holding_period") or self.contract.horizons[0]
        segment = self.contract.segment
        try:
            return TradeCandidate(
                asset=AssetId("NSE", segment, "INR", symbol),
                candidate_id=f"{self.contract.name}-{symbol}-{as_of:%Y%m%d%H%M}",
                strategy=self.contract.name, strategy_version=self.contract.version,
                side=side, horizon=horizon, product=horizon_to_product(horizon),
                signal_time=as_of, expires_at=as_of + CANDIDATE_LIFETIME,
                tick_size=tick, lot_size=lot_size,
                entry_price=to_decimal(entry), stop_loss=to_decimal(stop), target=to_decimal(target),
                exit_policy=policy.name, atr_value=to_decimal(atr), atr_timeframe=policy.atr_timeframe,
                thesis=str(signal.get("reasoning") or signal.get("setup") or "")[:500],
                review_contract_version=self.contract.review.version if self.contract.review else None,
            ).validate(now=as_of)
        except CandidateInvalid:
            return None


class ORBAdapter(SleeveAdapter):
    key, contract_name, timeframe = "orb_intraday", "ORBIntraday", "5m"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.orb_intraday import orb_intraday

        bars = completed_intraday(bars_by_tf.get("5m"), 5, as_of)
        return orb_intraday.scan(symbol, bars, as_of.isoformat()) if bars is not None else None

    def levels(self, signal, entry, atr, policy, tick):
        midpoint = (float(signal["opening_range_high"]) + float(signal["opening_range_low"])) / 2
        stop = structure_stop(
            signal["action"], entry=entry, structural_level=midpoint, atr=atr, buffer_mult=0.1,
            tick=float(tick),
        )
        if stop is None:
            return None  # structure too tight or too wide: skip the candidate
        risk = abs(entry - stop)
        target = entry + 2 * risk if signal["action"] == "BUY" else entry - 2 * risk
        return stop, target


class SupertrendAdapter(SleeveAdapter):
    key, contract_name, timeframe = "supertrend_adx", "SupertrendADX", "15m"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.supertrend_adx import SupertrendADX

        bars = completed_intraday(bars_by_tf.get("15m"), 15, as_of)
        return SupertrendADX()._analyse(symbol, bars) if bars is not None else None


class MeanReversionAdapter(SleeveAdapter):
    """Long-only: paper short inventory exists for INTRADAY only, so SELL signals are dropped."""

    key, contract_name, timeframe = "mean_reversion", "MeanReversion", "1d"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.mean_reversion import MeanReversion

        bars = completed_daily(bars_by_tf.get("1d"), as_of)
        result = MeanReversion()._analyse(symbol, bars) if bars is not None else None
        return result if result and result["action"] == "BUY" else None


class MomentumBreakoutAdapter(SleeveAdapter):
    key, contract_name, timeframe = "momentum_breakout", "MomentumBreakout", "1d"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.momentum_breakout import MomentumBreakout

        bars = completed_daily(bars_by_tf.get("1d"), as_of)
        return MomentumBreakout()._analyse(symbol, bars) if bars is not None else None


class VCPAdapter(SleeveAdapter):
    key, contract_name, timeframe = "vcp_nr7_swing", "VCPNR7Swing", "1d"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.vcp_nr7_swing import vcp_nr7_swing

        bars = completed_daily(bars_by_tf.get("1d"), as_of)
        return vcp_nr7_swing.scan(symbol, bars) if bars is not None else None


class BTSTAdapter(SleeveAdapter):
    key, contract_name, timeframe = "btst_strength", "BTSTStrength", "15m"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.btst_strength import btst_strength

        bars = completed_intraday(bars_by_tf.get("15m"), 15, as_of)
        return btst_strength.scan(symbol, bars, as_of.isoformat()) if bars is not None else None

    def atr_bars(self, bars_by_tf, as_of, policy):
        # BTST holds overnight, so its ATR comes from finished daily sessions.
        daily = completed_daily(bars_by_tf.get("1d"), as_of)
        if daily is None and bars_by_tf.get("15m") is not None:
            daily = daily_from_intraday(bars_by_tf["15m"], as_of)
        return daily, None


class ETFAdapter(SleeveAdapter):
    key, contract_name, timeframe = "etf_rotation", "ETFRotation", "1d"

    def signal(self, symbol, bars_by_tf, as_of):
        from strategies.etf_rotation import etf_rotation

        bars = completed_daily(bars_by_tf.get("1d"), as_of)
        peers = {
            peer: frame for peer, raw in (bars_by_tf.get("1d_peers") or {}).items()
            if (frame := completed_daily(raw, as_of)) is not None
        }
        return etf_rotation.scan(symbol, bars, peers) if bars is not None else None


ADAPTERS: dict[str, SleeveAdapter] = {
    adapter.key: adapter
    for adapter in (
        ORBAdapter(), SupertrendAdapter(), MeanReversionAdapter(), MomentumBreakoutAdapter(),
        VCPAdapter(), BTSTAdapter(), ETFAdapter(),
    )
}


def enabled_adapters() -> list[SleeveAdapter]:
    """Only sleeves whose own flag is on; with the defaults this is always empty."""
    from strategies.contracts import strategy_contract

    return [
        adapter for adapter in ADAPTERS.values()
        if (contract := strategy_contract(adapter.contract_name)) is not None and contract.is_enabled()
    ]
