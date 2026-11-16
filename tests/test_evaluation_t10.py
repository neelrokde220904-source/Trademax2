"""T10: walk-forward evaluation harness, bootstrap interval, append-only run table."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import main
from backtester.engine import BacktestEngine
from config.settings import settings
from database import db
from database.models import EvaluationRun
from research import evaluation as ev


def synthetic_bars(symbols=("AAA", "BBB", "CCC"), sessions=40, seed=1, drift=0.0004):
    out = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(seed + offset)
        frames, price = [], 100.0
        for day in pd.bdate_range("2026-08-03", periods=sessions):
            index = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=25, freq="15min", tz="Asia/Kolkata")
            steps = rng.normal(drift, 0.003, 25)
            close = price * np.cumprod(1 + steps)
            price = float(close[-1])
            frames.append(pd.DataFrame({
                "datetime": index, "open": close / (1 + steps), "high": close * 1.002,
                "low": close * 0.998, "close": close, "volume": rng.integers(8000, 12000, 25).astype(float),
            }))
        out[symbol] = pd.concat(frames, ignore_index=True)
    return out


# ── windows ──────────────────────────────────────────────────────────────────

def test_windows_are_consecutive_non_overlapping_and_respect_purge_and_embargo():
    sessions = list(range(1000))
    windows = ev.walk_forward_windows(sessions, folds=3, purge_bars=10, embargo_bars=5)
    assert len(windows) == 3
    block = 1000 // 4
    assert windows[0][0] == block + 10
    assert windows[0][1] == 2 * block - 1 - 5
    for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start - previous_end >= 10 + 5  # a gap of at least purge + embargo between folds
    assert windows[-1][1] == 1000 - 1 - 5
    with pytest.raises(ev.EvaluationUnavailable):
        ev.walk_forward_windows(list(range(30)), folds=3, purge_bars=10, embargo_bars=10)
    with pytest.raises(ValueError):
        ev.walk_forward_windows(sessions, folds=0, purge_bars=0, embargo_bars=0)


# ── statistics ───────────────────────────────────────────────────────────────

def test_bootstrap_interval_is_deterministic_and_brackets_the_mean():
    values = list(np.random.default_rng(0).normal(5, 20, 300))
    low, high = ev.bootstrap_expectancy_ci(values)
    assert (low, high) == ev.bootstrap_expectancy_ci(values)
    assert low < float(np.mean(values)) < high
    assert ev.bootstrap_expectancy_ci([]) == (0.0, 0.0)


@pytest.mark.parametrize("trades,low,high,expected", [
    (99, 5.0, 9.0, "inconclusive"),      # too few trades even though the interval excludes zero
    (500, -2.0, 6.0, "inconclusive"),    # interval spans zero
    (500, 1.0, 6.0, "positive"),
    (500, -6.0, -1.0, "negative"),
    (0, 0.0, 0.0, "inconclusive"),
])
def test_classification_rules(trades, low, high, expected):
    assert ev.classify({"trade_count": trades, "expectancy_ci95": [low, high]}) == expected


def test_summary_metrics_cover_the_required_report_fields():
    from backtester.engine import BacktestTrade

    def trade(pnl, reason, action="BUY"):
        return BacktestTrade(
            symbol="X", action=action, entry_date=pd.Timestamp("2026-11-02"), entry_price=100.0, quantity=10,
            stop_loss=95.0, target=110.0, strategy="S", net_pnl=pnl, exit_price=100.0 + pnl / 10,
            exit_reason=reason, initial_stop=95.0, highest_since_entry=104.0, lowest_since_entry=98.0,
        )

    trades = [trade(50, "TARGET"), trade(-20, "STOP_LOSS"), trade(30, "TIME_EXIT"), trade(-10, "TIME_EXIT")]
    metrics = ev.summarize_trades(
        trades, capital=10_000, exposure_bars=40, total_bars=100, signals=20,
        equity_curves=[[10_000, 10_050, 10_020, 9_980, 10_030]],
    )
    assert metrics["trade_count"] == 4 and metrics["net_expectancy"] == pytest.approx(12.5)
    assert metrics["profit_factor"] == pytest.approx(80 / 30)
    assert metrics["exposure"] == 0.4 and metrics["fill_rate"] == 0.2
    assert metrics["turnover"] == pytest.approx(sum(t.entry_price * t.quantity + t.exit_price * t.quantity for t in trades) / 10_000)
    assert sum(metrics["exit_mix"].values()) == pytest.approx(1.0)
    assert metrics["exit_mix"]["TIME_EXIT"] == 0.5
    assert metrics["max_drawdown_pct"] == pytest.approx((10_050 - 9_980) / 10_050 * 100)
    assert metrics["avg_mfe"] == 40.0 and metrics["avg_mae"] == 20.0
    assert metrics["net_expectancy_r"] == pytest.approx(np.mean([50 / 50, -20 / 50, 30 / 50, -10 / 50]))
    assert len(metrics["expectancy_ci95"]) == 2


# ── no look-ahead ────────────────────────────────────────────────────────────

def test_daily_atr_lookup_never_sees_its_own_or_later_sessions():
    frame = synthetic_bars(("AAA",), sessions=30)["AAA"]
    base = ev._daily_atr_lookup(frame, 14)
    poisoned = frame.copy()
    last_day = poisoned["datetime"].dt.normalize().iloc[-1]
    poisoned.loc[poisoned["datetime"].dt.normalize() == last_day, "high"] += 500
    changed = ev._daily_atr_lookup(poisoned, 14)
    assert changed.iloc[-1] == base.iloc[-1]  # the value for the last day ignores that day's own bars
    assert base.iloc[:14].isna().all()


def test_composite_signals_are_keyed_to_completed_bars_and_use_prior_session_atr():
    frame = synthetic_bars(("AAA",), sessions=40, drift=0.002)["AAA"]
    signals = ev._composite_trend_signals("AAA", frame, "daily_atr_baseline_v1")
    assert signals  # the trending fixture produces entries
    for stamp, signal in list(signals.items())[:20]:
        assert signal["atr"] > 0 and signal["atr_exit_policy"] == "daily_atr_baseline_v1"
        assert signal["holding_period"] == "INTRADAY" and signal["strategy"] == "CompositeTrend"


def test_engine_intraday_rules_keep_entries_inside_the_window_and_flatten_at_1515():
    engine = BacktestEngine(initial_capital=100_000, intraday_rules=True)
    assert engine._intraday_entry_allowed(pd.Timestamp("2026-11-03 09:30", tz="Asia/Kolkata"))
    assert not engine._intraday_entry_allowed(pd.Timestamp("2026-11-03 09:15", tz="Asia/Kolkata"))
    assert not engine._intraday_entry_allowed(pd.Timestamp("2026-11-03 14:45", tz="Asia/Kolkata"))
    bars = synthetic_bars(("AAA",), sessions=12, drift=0.0)["AAA"]
    called = []

    def fn(symbol, history, _i):
        stamp = history.index[-1]
        if (stamp.hour, stamp.minute) == (10, 0):
            called.append(stamp)
            return [{"symbol": symbol, "action": "BUY", "strategy": "S", "holding_period": "INTRADAY",
                     "atr": 1.0, "atr_exit_policy": "daily_atr_baseline_v1"}]
        return []

    settings.min_net_reward_risk, original = 0.1, settings.min_net_reward_risk
    try:
        result = engine.run({"AAA": bars}, fn)
    finally:
        settings.min_net_reward_risk = original
    assert called and result.trades
    for trade in result.trades:
        assert trade.entry_date.hour >= 9 and (trade.exit_date.hour, trade.exit_date.minute) <= (15, 15) or trade.exit_reason in {"STOP_LOSS", "TARGET"}
        assert trade.exit_date.date() == trade.entry_date.date()  # never carried overnight


# ── orchestration ────────────────────────────────────────────────────────────

def test_a_full_run_records_a_row_per_configuration_and_never_deletes(monkeypatch):
    bars = synthetic_bars(drift=0.0015)
    common = dict(
        strategy="CompositeTrend", policy="daily_atr_baseline_v1", universe=list(bars),
        start="2026-08-03", end="2026-09-25", folds=2, purge_bars=4, embargo_bars=4, bars=bars,
    )
    monkeypatch.setattr(settings, "min_net_reward_risk", 1.0)
    small = ev.run_walk_forward_evaluation(capital=10_000.0, **common)
    large = ev.run_walk_forward_evaluation(capital=100_000.0, **common)
    again = ev.run_walk_forward_evaluation(capital=100_000.0, **common)
    assert small["capital"] != large["capital"] and large["config_digest"] == again["config_digest"]
    assert small["config_digest"] != large["config_digest"]
    with db.get_db() as session:
        rows = session.query(EvaluationRun).order_by(EvaluationRun.id).all()
        assert [row.capital for row in rows] == [10_000.0, 100_000.0, 100_000.0]  # appended, not replaced
        assert all(row.strategy == "CompositeTrend" for row in rows)
        assert json.loads(rows[0].universe_json) == sorted(bars)
    for report in (small, large):
        assert report["outcome"] == "inconclusive"  # a short synthetic run is never conclusive
        assert report["folds"] == 2 and any("cannot promote" in line for line in report["limitations"])
        assert {"net_expectancy", "profit_factor", "max_drawdown_pct", "exposure", "turnover",
                "avg_mae", "avg_mfe", "exit_mix", "fill_rate", "expectancy_ci95"} <= set(report["metrics"])
    assert ev.list_runs()[0]["strategy"] == "CompositeTrend"


def test_unknown_inputs_and_missing_data_are_rejected(monkeypatch):
    bars = synthetic_bars()
    with pytest.raises(ev.EvaluationUnavailable):
        ev.run_walk_forward_evaluation("Nope", "daily_atr_baseline_v1", ["AAA"], "2026-08-03", "2026-09-25", 1e5, bars=bars)
    with pytest.raises(ValueError):
        ev.run_walk_forward_evaluation("CompositeTrend", "made_up", ["AAA"], "2026-08-03", "2026-09-25", 1e5, bars=bars)
    with pytest.raises(ValueError):
        ev.run_walk_forward_evaluation("CompositeTrend", "daily_atr_baseline_v1", ["AAA"], "2026-08-03", "2026-09-25", 0, bars=bars)
    with pytest.raises(ev.EvaluationUnavailable):
        ev.run_walk_forward_evaluation(
            "CompositeTrend", "daily_atr_baseline_v1", ["ZZZ"], "2026-08-03", "2026-09-25", 1e5, bars={},
            fetch=False,
        )


def test_cached_data_is_read_before_any_free_download(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    frame = synthetic_bars(("AAA",), sessions=5)["AAA"]
    path = tmp_path / "15m" / "AAA.csv"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    monkeypatch.setattr(ev, "_fetch_free", lambda *_a: pytest.fail("downloaded despite a cache hit"))
    loaded = ev.load_bars(["AAA"], "15m")
    assert len(loaded["AAA"]) == len(frame)
    assert ev.load_bars(["BBB"], "15m", fetch=False) == {}  # offline mode never downloads
    with pytest.raises(ValueError):
        ev.load_bars(["AAA"], "5m")


def test_the_free_download_is_cached_for_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    frame = synthetic_bars(("AAA",), sessions=5)["AAA"]
    calls = []
    monkeypatch.setattr(ev, "_fetch_free", lambda symbol, interval: calls.append(symbol) or frame)
    assert "AAA" in ev.load_bars(["AAA"], "15m")
    assert (tmp_path / "15m" / "AAA.csv").exists()
    ev.load_bars(["AAA"], "15m")
    assert calls == ["AAA"]


def test_data_digest_changes_with_the_data():
    one = synthetic_bars(("AAA",), seed=1)
    other = synthetic_bars(("AAA",), seed=2)
    assert ev.data_digest(one) == ev.data_digest(one) and ev.data_digest(one) != ev.data_digest(other)


# ── CLI ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def command(monkeypatch):
    monkeypatch.setattr(main, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main, "init_db", lambda: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["main.py", *args])
        return main.main()

    return run


def test_evaluate_requires_a_date_range(command):
    with pytest.raises(SystemExit):
        command("--evaluate", "CompositeTrend")


def test_evaluate_runs_both_capital_scenarios_and_writes_a_report(command, monkeypatch, capsys, tmp_path):
    calls = []

    def fake(strategy, policy, universe, start, end, capital, **_kw):
        calls.append(capital)
        return {"strategy": strategy, "exit_policy": policy, "capital": capital, "outcome": "inconclusive",
                "metrics": {"trade_count": 0}}

    monkeypatch.setattr(ev, "run_walk_forward_evaluation", fake)
    monkeypatch.setattr(ev, "REPORT_DIR", tmp_path)
    from config import universe

    monkeypatch.setattr(universe, "get_active_watchlist", lambda: ["AAA", "BBB", "CCC"])
    with pytest.raises(SystemExit) as exit_info:
        command("--evaluate", "CompositeTrend", "--policy", "daily_atr_2x_4x_v1",
                "--from", "2026-08-03", "--to", "2026-09-25", "--eval-symbols", "2")
    assert exit_info.value.code == 0
    assert calls == [10_000.0, 100_000.0]
    printed = json.loads(capsys.readouterr().out)
    assert [row["capital"] for row in printed] == [10_000.0, 100_000.0]
    assert len(list(tmp_path.glob("evaluation_*.json"))) == 1
