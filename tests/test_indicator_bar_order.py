import pandas as pd
import pytest

from data.indicators import completed_intraday_bars, is_vwap_chop, session_vwap


def _bars(prices, timestamps):
    return pd.DataFrame({
        "datetime": timestamps, "open": prices, "high": prices,
        "low": prices, "close": prices, "volume": [100] * len(prices),
    })


def test_chop_filter_does_not_restart_vwap_at_lookback_boundary():
    frame = _bars(
        [90] * 8 + [100] * 8,
        pd.date_range("2026-11-13 09:15", periods=16, freq="15min"),
    )
    assert not is_vwap_chop(frame, "2026-11-13T13:15:00+05:30")


def test_session_vwap_sorts_timestamps_and_returns_original_row_alignment():
    frame = _bars([100, 90], pd.to_datetime(["2026-11-13 10:00", "2026-11-13 09:15"]))
    frame.index = [50, 20]
    result = session_vwap(frame)
    assert result.index.tolist() == [50, 20]
    assert result.tolist() == pytest.approx([95, 90])


def test_completed_bars_are_sorted_and_identical_duplicates_do_not_change_indicators():
    frame = _bars([101, 100, 100], pd.to_datetime([
        "2026-11-13 09:30", "2026-11-13 09:15", "2026-11-13 09:15",
    ]))
    result = completed_intraday_bars(frame, "2026-11-13T09:45:00+05:30")
    assert result["close"].tolist() == [100, 101]


def test_conflicting_duplicate_bars_are_rejected_not_silently_selected():
    frame = _bars([101, 100], pd.to_datetime(["2026-11-13 09:15"] * 2))
    with pytest.raises(ValueError, match="Conflicting"):
        completed_intraday_bars(frame, "2026-11-13T09:45:00+05:30")
