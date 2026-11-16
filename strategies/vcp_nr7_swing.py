"""Deterministic VCP/NR7 swing breakout scanner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_library.chart_formations import detect_nr7, detect_vcp
from alpha_library.trend_context_filter import bullish_pattern_is_aligned


class VCPNR7Swing:
    """Detect contraction setups that break a prior resistance level."""

    def scan(self, symbol: str, bars: pd.DataFrame) -> dict[str, Any] | None:
        frame = bars.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if len(frame) < 200 or not {"high", "low", "close"}.issubset(frame.columns):
            return None
        setup = detect_vcp(frame) or detect_nr7(frame)
        aligned = bullish_pattern_is_aligned(frame)
        resistance = float(frame["high"].iloc[-21:-1].max())
        close = float(frame["close"].iloc[-1])
        if not setup or not aligned or close <= resistance:
            return None
        atr = float(frame["atr"].iloc[-1]) if "atr" in frame and pd.notna(frame["atr"].iloc[-1]) else 0
        if atr <= 0:
            true_range = pd.concat(
                [
                    frame["high"] - frame["low"],
                    (frame["high"] - frame["close"].shift()).abs(),
                    (frame["low"] - frame["close"].shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = float(true_range.rolling(14).mean().iloc[-1])
        if atr <= 0 or not pd.notna(atr):
            return None
        stop = close - 2 * atr
        return {
            "symbol": symbol,
            "action": "BUY",
            "price": close,
            "stop_loss": round(stop, 2),
            "target": round(close + 2 * (close - stop), 2),
            "holding_period": "SWING",
            "strategy": "VCPNR7Swing",
            "setup": "VCP" if detect_vcp(frame) else "NR7",
        }


vcp_nr7_swing = VCPNR7Swing()
