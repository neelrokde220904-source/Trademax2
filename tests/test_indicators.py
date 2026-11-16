"""Tests for data/indicators.py."""

import pandas as pd
import numpy as np

from data.indicators import (
    _compute_composite_signal,
    compute_all_indicators,
    get_latest_signals,
    is_current_session_breakout,
    is_vwap_chop,
)


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })


def test_compute_all_indicators():
    """compute_all_indicators should add expected columns."""
    df = _make_ohlcv(200)
    result = compute_all_indicators(df)

    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    assert "atr" in result.columns

    # Should not have all NaN
    assert result["rsi"].dropna().shape[0] > 0


def test_get_latest_signals():
    """get_latest_signals should return a valid signal dict."""
    df = _make_ohlcv(200)
    df = compute_all_indicators(df)
    signals = get_latest_signals(df)

    assert "composite_signal" in signals
    assert signals["composite_signal"] in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    assert "rsi" in signals
    assert "macd" in signals


def test_vwap_uses_completed_session_cumulative_volume():
    times = list(pd.date_range("2026-11-12 09:15", periods=35, freq="15min"))
    times += list(pd.date_range("2026-11-13 09:15", periods=35, freq="15min"))
    close = np.arange(len(times), dtype=float) + 100
    volume = np.arange(len(times), dtype=float) + 10
    df = pd.DataFrame({
        "datetime": times,
        "open": close,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": volume,
    })

    result = compute_all_indicators(df)

    assert result["vwap"].notna().all()
    expected_first = (
        (close[35] * volume[35])
        / volume[35]
    )
    expected_second = (
        (close[35] * volume[35] + close[36] * volume[36])
        / (volume[35] + volume[36])
    )
    assert result.loc[35, "vwap"] == expected_first
    assert result.loc[36, "vwap"] == expected_second


def test_oversold_rsi_does_not_create_a_trend_buy_by_itself():
    signal, _confidence, _reason = _compute_composite_signal({"rsi_14": 30}, close=100)
    assert signal == "HOLD"


def test_vwap_chop_requires_eight_completed_bars_in_the_same_session():
    from tests.test_entry_levels import _sessions

    frame, today = _sessions(1, today_bars=8)
    as_of = f"{today.date()}T11:30:00+05:30"
    assert is_vwap_chop(frame, as_of, lookback_bars=8, band_pct=0.005)
    assert not is_vwap_chop(frame, f"{today.date()}T10:15:00+05:30", lookback_bars=8)


def test_breakout_detection_uses_only_closed_current_session_bars():
    from tests.test_entry_levels import _sessions

    frame, today = _sessions(5, today_bars=3)
    frame.loc[frame.index[-1], ["close", "high"]] = [103.0, 104.0]
    assert is_current_session_breakout(
        frame, f"{today.date()}T10:00:00+05:30", lookback_bars=20
    )
    assert not is_current_session_breakout(
        frame, f"{today.date()}T09:50:00+05:30", lookback_bars=20
    )
