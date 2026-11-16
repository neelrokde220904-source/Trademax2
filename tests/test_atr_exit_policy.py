import pandas as pd
import pytest

from backtester.engine import BacktestEngine
from risk.exit_policy import BASELINE_ATR_POLICY, RESEARCH_ATR_POLICY, chandelier_stop


@pytest.mark.parametrize("side,levels", [
    ("BUY", (960.0, 1080.0)), ("SELL", (1040.0, 920.0)),
])
def test_research_atr_levels_are_2x_4x(side, levels):
    assert RESEARCH_ATR_POLICY.levels(1000, 20, side) == levels
    assert BASELINE_ATR_POLICY.name != RESEARCH_ATR_POLICY.name


@pytest.mark.parametrize("side,initial,favorable,expected", [
    ("BUY", 980.0, 1040.0, 1000.0),
    ("SELL", 1020.0, 960.0, 1000.0),
])
def test_atr_growth_cannot_widen_chandelier_stop(side, initial, favorable, expected):
    tightened = chandelier_stop(side, initial, favorable, 20, 2)
    assert tightened == expected
    assert chandelier_stop(side, tightened, favorable, 50, 2) == tightened


def test_research_trailing_stop_uses_completed_bar_only(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "min_net_reward_risk", 1.0)
    engine = BacktestEngine(initial_capital=100_000)
    engine._process_signal({
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 96, "target": 108, "atr": 2,
        "atr_exit_policy": RESEARCH_ATR_POLICY.name,
    }, pd.Timestamp("2026-11-12T10:00:00+05:30"), 100_000, 0)
    assert "TEST" in engine._positions
    index = pd.date_range("2026-11-12T10:00:00+05:30", periods=2, freq="15min")
    data = {"TEST": pd.DataFrame({
        "open": [100, 102], "high": [106, 103],
        "low": [97, 101], "close": [105, 102], "volume": [1000, 1000],
    }, index=index)}
    engine._check_exits(data, index[0])
    assert "TEST" in engine._positions
    assert engine._positions["TEST"].stop_loss == 102
    engine._check_exits(data, index[1])
    assert engine._closed_trades[0].exit_price == 102
    assert engine._closed_trades[0].exit_reason == "STOP_LOSS"


def test_backtest_daily_entry_cap_does_not_reset_each_intraday_bar(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "max_daily_entries", 1)
    monkeypatch.setattr(settings, "min_net_reward_risk", 1.0)
    times = pd.date_range("2026-11-12T10:00:00+05:30", periods=5, freq="15min")
    data = {"TEST": pd.DataFrame({
        "open": [100] * 5, "high": [106] * 5, "low": [99] * 5,
        "close": [100] * 5, "volume": [1000] * 5,
    }, index=times)}

    def signals(symbol, history, _index):
        return [{
            "symbol": symbol, "action": "BUY", "price": 100,
            "stop_loss": 98, "target": 105,
        }]

    result = BacktestEngine(initial_capital=100_000).run(data, signals)
    assert len(result.trades) == 1
