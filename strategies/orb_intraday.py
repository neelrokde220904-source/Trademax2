"""Opening-range breakout signals for five-minute NSE bars."""

from __future__ import annotations

import math
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta

IST = ZoneInfo("Asia/Kolkata")


class ORBIntraday:
    """Build 09:15–09:30 opening ranges and conservative breakout plans."""

    entry_cutoff = time(14, 45)
    mandatory_flatten_time = time(15, 15)
    breakout_min_relative_volume = 1.5
    relative_volume_lookback_sessions = 20
    min_relative_volume_samples = 5
    vwap_chop_lookback_bars = 8
    vwap_chop_band_pct = 0.005

    def scan(
        self, symbol: str, bars: pd.DataFrame, as_of: str | None = None
    ) -> dict[str, Any] | None:
        """Return a volume- and momentum-confirmed plan from completed five-minute bars."""
        frame = bars.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if not {"high", "low", "close", "volume"}.issubset(frame.columns) or len(frame) < 4:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if index.tz is None:
            index = index.tz_localize(IST)
        else:
            index = index.tz_convert(IST)
        frame.index = index
        frame = frame.sort_index(kind="stable")
        reference = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz=IST)
        if reference.tzinfo is None:
            reference = reference.tz_localize(IST)
        else:
            reference = reference.tz_convert(IST)
        frame = frame[frame.index + pd.Timedelta(minutes=5) <= reference]
        if len(frame) < 4:
            return None
        session_date = frame.index[-1].date()
        session = frame[frame.index.date == session_date]
        opening = session.between_time("09:15", "09:29", inclusive="both")
        latest_time = session.index[-1].time()
        if opening.empty or latest_time < time(9, 30) or latest_time >= self.entry_cutoff:
            return None
        opening_high = float(opening["high"].max())
        opening_low = float(opening["low"].min())
        midpoint = (opening_high + opening_low) / 2
        latest = float(session["close"].iloc[-1])
        if latest > opening_high:
            action, stop = "BUY", midpoint
        elif latest < opening_low:
            action, stop = "SELL", midpoint
        else:
            return None

        same_time_history = frame[
            (frame.index.date < session_date) & (frame.index.time == latest_time)
        ]["volume"].tail(self.relative_volume_lookback_sessions)
        if len(same_time_history) < self.min_relative_volume_samples:
            return None
        average_volume = float(same_time_history.mean())
        latest_volume = float(session["volume"].iloc[-1])
        if (
            not math.isfinite(average_volume)
            or not math.isfinite(latest_volume)
            or average_volume <= 0
            or latest_volume / average_volume < self.breakout_min_relative_volume
        ):
            return None

        typical_price = (session["high"] + session["low"] + session["close"]) / 3
        volume = pd.to_numeric(session["volume"], errors="coerce").clip(lower=0)
        cumulative_volume = volume.cumsum()
        vwap = (typical_price * volume).cumsum().div(cumulative_volume.where(cumulative_volume > 0))
        latest_vwap = float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else 0
        if latest_vwap <= 0 or (action == "BUY" and latest <= latest_vwap) or (
            action == "SELL" and latest >= latest_vwap
        ):
            return None

        if len(session) >= self.vwap_chop_lookback_bars:
            recent_vwap = vwap.tail(self.vwap_chop_lookback_bars)
            recent_close = session["close"].tail(self.vwap_chop_lookback_bars)
            if (
                ((recent_close - recent_vwap).abs() / recent_vwap <= self.vwap_chop_band_pct).all()
            ):
                return None

        rsi = ta.rsi(frame["close"], length=14)
        latest_rsi = float(rsi.iloc[-1]) if rsi is not None and pd.notna(rsi.iloc[-1]) else None
        if latest_rsi is None or (
            action == "BUY" and not 40 <= latest_rsi <= 70
        ) or (action == "SELL" and not 30 <= latest_rsi <= 60):
            return None

        atr = float(session["atr"].iloc[-1]) if "atr" in session and pd.notna(session["atr"].iloc[-1]) else 0
        if atr > 0:
            stop = max(stop, latest - atr) if action == "BUY" else min(stop, latest + atr)
        risk = abs(latest - stop)
        if risk <= 0:
            return None
        return {
            "symbol": symbol,
            "action": action,
            "price": latest,
            "opening_range_high": opening_high,
            "opening_range_low": opening_low,
            "relative_volume": round(latest_volume / average_volume, 3),
            "rsi_14": round(latest_rsi, 2),
            "stop_loss": round(stop, 2),
            "target": round(latest + 1.5 * risk if action == "BUY" else latest - 1.5 * risk, 2),
            "holding_period": "INTRADAY",
            "strategy": "ORBIntraday",
            "mandatory_flatten_at": self.mandatory_flatten_time.isoformat(),
        }


orb_intraday = ORBIntraday()
