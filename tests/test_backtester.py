"""Tests for backtester engine and metrics."""

import pandas as pd
import numpy as np
import pytest

from backtester.engine import BacktestEngine
from backtester.metrics import calculate_metrics


def _make_data(symbols: list[str], n: int = 252) -> dict[str, pd.DataFrame]:
    """Generate synthetic daily data for backtesting."""
    np.random.seed(42)
    data = {}
    for symbol in symbols:
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 1.0)
        close = np.maximum(close, 10)  # prevent negative prices
        df = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n) * 1.0),
            "low": close - abs(np.random.randn(n) * 1.0),
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, n),
        }, index=dates)
        data[symbol] = df
    return data


def _dummy_strategy(symbol: str, df: pd.DataFrame, bar_index: int):
    """Buy every 30 bars with simple SL/target."""
    if bar_index < 30 or bar_index % 30 != 0:
        return []
    close = float(df.iloc[-1]["close"])
    return [{
        "symbol": symbol,
        "action": "BUY",
        "price": close,
        "stop_loss": close * 0.95,
        "target": close * 1.10,
        "strategy": "test",
        "holding_period": "SWING",
    }]


def test_backtest_runs():
    """Backtest should execute without errors."""
    data = _make_data(["RELIANCE", "TCS"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    assert result.initial_capital == 100_000
    assert result.final_capital > 0
    assert len(result.equity_curve) > 0


def test_backtest_metrics():
    """Metrics calculation should produce valid numbers."""
    data = _make_data(["RELIANCE"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    metrics = calculate_metrics(result)
    assert metrics.total_trades >= 0
    assert metrics.win_rate >= 0
    assert metrics.win_rate <= 1
    assert metrics.risk_free_rate == 0.065


def test_backtest_costs():
    """Trades should have nonzero transaction costs."""
    data = _make_data(["RELIANCE"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    if result.trades:
        assert any(t.costs > 0 for t in result.trades)


def test_backtest_uses_live_risk_budget_costs_and_finalized_equity():
    dates = pd.bdate_range("2026-01-05", periods=2)
    data = {
        "TEST": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 119.0],
                "low": [99.0, 99.0],
                "close": [100.0, 118.0],
                "volume": [100_000, 100_000],
            },
            index=dates,
        )
    }

    def strategy(symbol, _frame, bar_index):
        if bar_index:
            return []
        return [{
            "symbol": symbol, "action": "BUY", "price": 100.0,
            "stop_loss": 98.0, "target": 118.0,
            "holding_period": "INTRADAY", "strategy": "risk-gated-test",
        }]

    result = BacktestEngine(initial_capital=25_000).run(data, strategy)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.quantity == 24
    assert trade.costs > 0
    assert trade.net_pnl == pytest.approx(trade.pnl - trade.costs)
    assert result.final_capital == pytest.approx(25_000 + trade.net_pnl)
    assert result.equity_curve[-1] == pytest.approx(result.final_capital)


def test_walk_forward_preserves_training_bars_for_test_lookback():
    dates = pd.bdate_range("2026-01-05", periods=5)
    data = {
        "TEST": pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0, 101.0, 101.0, 101.0, 119.0],
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [100_000] * 5,
            },
            index=dates,
        )
    }

    def strategy(symbol, frame, _bar_index):
        if frame.index[-1].date() != dates[2].date() or len(frame) < 3:
            return []
        return [{
            "symbol": symbol, "action": "BUY", "price": 100.0,
            "stop_loss": 98.0, "target": 118.0,
            "holding_period": "INTRADAY", "strategy": "walk-forward-test",
        }]

    result = BacktestEngine(initial_capital=25_000).run_walk_forward(
        data, strategy, train_days=3, test_days=2, step_days=2,
    )

    assert len(result.folds) == 1
    assert len(result.combined_trades) == 1
    assert result.combined_trades[0].entry_date.date() == dates[3].date()


def test_walk_forward_includes_final_partial_test_window():
    dates = pd.bdate_range("2026-01-05", periods=8)
    data = {
        "TEST": pd.DataFrame(
            {
                "open": [100.0] * 8,
                "high": [101.0] * 8,
                "low": [99.0] * 8,
                "close": [100.0] * 8,
                "volume": [100_000] * 8,
            },
            index=dates,
        )
    }

    result = BacktestEngine(initial_capital=25_000).run_walk_forward(
        data, lambda *_args: [], train_days=3, test_days=3, step_days=3,
    )

    assert len(result.fold_ranges) == 2
    assert result.fold_ranges[-1]["test_start"] == dates[6].strftime("%Y-%m-%d")
    assert result.fold_ranges[-1]["test_end"] == dates[7].strftime("%Y-%m-%d")


def test_backtest_accepts_fetcher_datetime_column_and_fills_next_open():
    dates = pd.bdate_range("2026-01-05", periods=3)
    data = {
        "TEST": pd.DataFrame({
            "datetime": dates,
            "open": [100.0, 102.0, 110.0],
            "high": [101.0, 103.0, 123.0],
            "low": [99.0, 101.0, 109.0],
            "close": [100.0, 102.0, 122.0],
            "volume": [100_000] * 3,
        })
    }

    def strategy(symbol, frame, _bar_index):
        if frame.index[-1].date() != dates[0].date():
            return []
        return [{
            "symbol": symbol, "action": "BUY", "price": 100.0,
            "stop_loss": 95.0, "target": 122.0,
            "holding_period": "SWING", "strategy": "next-open-test",
        }]

    result = BacktestEngine(initial_capital=25_000).run(data, strategy)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date.date() == dates[1].date()
    assert trade.entry_price == pytest.approx(102.0 * (1 + 0.0005))
    assert trade.exit_reason == "TARGET"
    assert trade.exit_price == pytest.approx(122.0 * (1 - 0.0005))


def test_backtest_models_stop_gap_at_open():
    dates = pd.bdate_range("2026-01-05", periods=3)
    data = {
        "TEST": pd.DataFrame(
            {
                "open": [100.0, 100.0, 90.0],
                "high": [101.0, 101.0, 95.0],
                "low": [99.0, 99.0, 89.0],
                "close": [100.0, 100.0, 93.0],
                "volume": [100_000] * 3,
            },
            index=dates,
        )
    }

    def strategy(symbol, frame, _bar_index):
        if frame.index[-1].date() != dates[0].date():
            return []
        return [{
            "symbol": symbol, "action": "BUY", "price": 100.0,
            "stop_loss": 95.0, "target": 120.0,
            "holding_period": "SWING", "strategy": "gap-stop-test",
        }]

    result = BacktestEngine(initial_capital=25_000).run(data, strategy)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == pytest.approx(90.0 * (1 - 0.0005))
