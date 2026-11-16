import pandas as pd

from strategies.orb_intraday import ORBIntraday


def _bars(days=6, breakout=True, volume=200):
    rows = []
    start = pd.Timestamp("2026-06-01")
    for day in range(days):
        date = start + pd.Timedelta(days=day)
        for index, minute in enumerate((15, 20, 25, 30)):
            is_breakout = day == days - 1 and breakout and index == 3
            rows.append({
                "datetime": pd.Timestamp(
                    f"{date.date()} 09:{minute:02d}:00", tz="Asia/Kolkata"
                ),
                "open": 100.0,
                "high": 103.0 if is_breakout else 101.0,
                "low": 99.0,
                "close": 102.0 if is_breakout else 100.0,
                "volume": volume if day == days - 1 and index == 3 else 100,
            })
    return pd.DataFrame(rows).set_index("datetime")


def test_orb_requires_closed_bar_and_volume_confirmed_breakout(monkeypatch):
    import strategies.orb_intraday as module

    monkeypatch.setattr(module.ta, "rsi", lambda close, length: pd.Series(50.0, index=close.index))
    bars = _bars()
    strategy = ORBIntraday()

    assert strategy.scan("TEST", bars, "2026-06-06T09:34:00+05:30") is None
    result = strategy.scan("TEST", bars, "2026-06-06T09:35:00+05:30")

    assert result is not None
    assert result["action"] == "BUY"
    assert result["relative_volume"] == 2.0
    assert result["rsi_14"] == 50.0


def test_orb_abstains_without_historical_same_time_volume(monkeypatch):
    import strategies.orb_intraday as module

    monkeypatch.setattr(module.ta, "rsi", lambda close, length: pd.Series(50.0, index=close.index))
    result = ORBIntraday().scan(
        "TEST", _bars(days=4), "2026-06-04T09:35:00+05:30"
    )
    assert result is None


def test_orb_abstains_when_breakout_volume_is_below_threshold(monkeypatch):
    import strategies.orb_intraday as module

    monkeypatch.setattr(module.ta, "rsi", lambda close, length: pd.Series(50.0, index=close.index))
    result = ORBIntraday().scan(
        "TEST", _bars(volume=149), "2026-06-06T09:35:00+05:30"
    )
    assert result is None


def test_orb_abstains_during_extended_vwap_chop(monkeypatch):
    import strategies.orb_intraday as module

    monkeypatch.setattr(module.ta, "rsi", lambda close, length: pd.Series(50.0, index=close.index))
    rows = []
    start = pd.Timestamp("2026-06-01")
    times = (15, 20, 25, 30, 35, 40, 45, 50)
    for day in range(6):
        date = start + pd.Timedelta(days=day)
        for index, minute in enumerate(times):
            is_breakout = day == 5 and index == len(times) - 1
            rows.append({
                "datetime": pd.Timestamp(
                    f"{date.date()} 09:{minute:02d}:00", tz="Asia/Kolkata"
                ),
                "open": 100.0,
                "high": 100.3 if is_breakout else 100.1,
                "low": 99.9,
                "close": 100.2 if is_breakout else 100.0,
                "volume": 200 if is_breakout else 100,
            })
    bars = pd.DataFrame(rows).set_index("datetime")

    result = ORBIntraday().scan("TEST", bars, "2026-06-06T09:55:00+05:30")
    assert result is None


def test_orb_abstains_when_rsi_is_outside_long_momentum_zone(monkeypatch):
    import strategies.orb_intraday as module

    monkeypatch.setattr(module.ta, "rsi", lambda close, length: pd.Series(75.0, index=close.index))
    result = ORBIntraday().scan("TEST", _bars(), "2026-06-06T09:35:00+05:30")
    assert result is None
