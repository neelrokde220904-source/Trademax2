"""Optional TA-Lib streaming helpers with a dependency-free fallback."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

_BUFFER_SIZE = 256


class TalibStreaming:
    """Maintain a bounded bar buffer and expose latest indicator/pattern values.

    TA-Lib's stream API is used when installed. Custom indicators which are not
    provided by TA-Lib (Supertrend, VWAP, CHOP, CRSI and CKSP) remain explicit
    fallback calculations rather than being represented as TA-Lib functions.
    """

    def __init__(self, buffer_size: int = _BUFFER_SIZE) -> None:
        if buffer_size < 200:
            raise ValueError("buffer_size must be at least 200 bars")
        self._bars: deque[dict[str, float]] = deque(maxlen=buffer_size)
        self._talib: Any = None
        try:
            import talib
            self._talib = talib
        except ImportError:
            self._talib = None

    def update(self, bar: dict[str, float]) -> dict[str, Any]:
        """Append one OHLCV bar and calculate latest supported outputs."""
        required = ("open", "high", "low", "close", "volume")
        values = {key: float(bar[key]) for key in required}
        if any(not np.isfinite(value) for value in values.values()) or values["close"] <= 0:
            raise ValueError("bar must contain finite OHLCV values and a positive close")
        self._bars.append(values)
        frame = pd.DataFrame(self._bars)
        result: dict[str, Any] = {"bars_buffered": len(self._bars), "indicators": {}}
        if self._talib is not None and len(frame) >= 200:
            result["indicators"] = {
                **self._stream_indicators(frame),
                **self._custom_indicators(frame),
            }
            result["candlestick_patterns"] = self._stream_patterns(frame)
        else:
            close = frame["close"]
            result["indicators"] = {
                "sma_20": float(close.tail(20).mean()) if len(frame) >= 20 else None,
                "ema_20": float(close.ewm(span=20, adjust=False).mean().iloc[-1]),
                "rsi_14": self._rsi(close),
                "vwap": self._vwap(frame),
                **self._custom_indicators(frame),
            }
            result["candlestick_patterns"] = {}
        return result

    def _stream_indicators(self, frame: pd.DataFrame) -> dict[str, float]:
        talib = self._talib
        close = frame["close"].to_numpy(dtype=float)
        high, low = frame["high"].to_numpy(dtype=float), frame["low"].to_numpy(dtype=float)
        open_, volume = frame["open"].to_numpy(dtype=float), frame["volume"].to_numpy(dtype=float)
        stream = talib.stream
        calculated = {
            "sma_20": stream.SMA(close, timeperiod=20),
            "ema_20": stream.EMA(close, timeperiod=20),
            "rsi_14": stream.RSI(close, timeperiod=14),
            "macd": stream.MACD(close)[0],
            "adx_14": stream.ADX(high, low, close, timeperiod=14),
            "atr_14": stream.ATR(high, low, close, timeperiod=14),
            "bbands_upper": stream.BBANDS(close)[0],
            "stochrsi": stream.STOCHRSI(close)[0],
        }
        return {name: float(value) for name, value in calculated.items()}

    @classmethod
    def _custom_indicators(cls, frame: pd.DataFrame) -> dict[str, float | None]:
        high, low, close = frame["high"], frame["low"], frame["close"]
        previous_close = close.shift()
        true_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / 10, adjust=False, min_periods=10).mean()
        range_sum = true_range.rolling(14).sum()
        high_14, low_14 = high.rolling(14).max(), low.rolling(14).min()
        span = high_14 - low_14
        chop = 100 * np.log10(range_sum / span) / np.log10(14)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        smooth_tr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / smooth_tr
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / smooth_tr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        mid = (high + low) / 2
        upper = mid + 3 * atr
        lower = mid - 3 * atr
        direction = 1
        supertrend = float(lower.iloc[-1]) if len(frame) else None
        for index in range(1, len(frame)):
            if close.iloc[index] > upper.iloc[index - 1]:
                direction = 1
            elif close.iloc[index] < lower.iloc[index - 1]:
                direction = -1
            supertrend = float(lower.iloc[index] if direction > 0 else upper.iloc[index])

        volume = frame["volume"].to_numpy(dtype=float)
        typical = ((high + low + close) / 3).to_numpy(dtype=float)
        vwap = float(np.dot(typical, volume) / volume.sum()) if volume.sum() > 0 else None
        close_returns = close.pct_change()
        rank_return = close_returns.rolling(100).apply(
            lambda values: float((values <= values[-1]).mean() * 100), raw=True
        )
        rsi_three = cls._rsi(close.tail(100), period=3)
        streak = close.diff().gt(0).astype(int).groupby((close.diff().gt(0) != close.diff().gt(0).shift()).cumsum()).cumsum()
        streak_rsi = cls._rsi(streak.astype(float).tail(100), period=2)
        cksp_long = high.rolling(20).max() - 3 * atr
        cksp_short = low.rolling(20).min() + 3 * atr
        return {
            "adx_14": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
            "atr_10": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None,
            "supertrend": supertrend,
            "supertrend_direction": float(direction),
            "vwap": vwap,
            "choppiness_14": float(chop.iloc[-1]) if pd.notna(chop.iloc[-1]) else None,
            "crsi": (
                float(np.nanmean([rsi_three, streak_rsi, rank_return.iloc[-1]]))
                if rsi_three is not None and streak_rsi is not None and pd.notna(rank_return.iloc[-1])
                else None
            ),
            "cksp_long_stop": float(cksp_long.iloc[-1]) if pd.notna(cksp_long.iloc[-1]) else None,
            "cksp_short_stop": float(cksp_short.iloc[-1]) if pd.notna(cksp_short.iloc[-1]) else None,
        }

    def _stream_patterns(self, frame: pd.DataFrame) -> dict[str, int]:
        talib = self._talib
        functions = talib.get_function_groups().get("Pattern Recognition", [])
        arrays = {
            name: frame[name].to_numpy(dtype=float)
            for name in ("open", "high", "low", "close")
        }
        result = {}
        for pattern in functions:
            function = getattr(talib.stream, pattern)
            result[pattern] = int(function(
                arrays["open"], arrays["high"], arrays["low"], arrays["close"]
            ))
        return result

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float | None:
        delta = close.diff()
        gains = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        losses = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        if len(close) < period + 1:
            return None
        if losses.iloc[-1] == 0:
            return 100.0
        return float(100 - 100 / (1 + gains.iloc[-1] / losses.iloc[-1]))

    @staticmethod
    def _vwap(frame: pd.DataFrame) -> float | None:
        volume = frame["volume"].to_numpy(dtype=float)
        total_volume = float(volume.sum())
        if total_volume <= 0:
            return None
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3
        return float((typical.to_numpy(dtype=float) * volume).sum() / total_volume)
