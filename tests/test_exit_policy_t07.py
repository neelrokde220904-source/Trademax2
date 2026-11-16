"""T07: shared exit policies, structure stops, time exits, per-position policy persistence."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

from backtester.engine import BacktestEngine
from config.settings import settings
from operations.control_state import ControlStateStore
from risk.exit_policy import (
    ATR_EXIT_POLICIES, CHANDELIER_ATR22_POLICY, DAILY_ATR22_POLICY, INTRADAY_ATR_POLICY,
    atr_before, gap_fill_price, get_atr_exit_policy, next_session_date, ratcheted_stop,
    stop_target_hit, structure_stop, time_exit_due,
)
from risk.stop_loss_manager import LEGACY_ATR_TIMEFRAME, StopLossManager

IST = pytz.timezone("Asia/Kolkata")


def ist(year, month, day, hour=10, minute=0):
    return IST.localize(datetime(year, month, day, hour, minute))


# ── policies ─────────────────────────────────────────────────────────────────

def test_policy_registry_and_timeframes_are_unambiguous():
    assert set(ATR_EXIT_POLICIES) == {
        "daily_atr_baseline_v1", "daily_atr_2x_4x_v1", "intraday_atr_2x_4x_v1",
        "daily_atr22_2x_4x_v1", "chandelier_atr22_k3_v1",
    }
    assert INTRADAY_ATR_POLICY.atr_timeframe == "INTRADAY_15M_14" and INTRADAY_ATR_POLICY.atr_period == 14
    assert DAILY_ATR22_POLICY.atr_period == 22 and CHANDELIER_ATR22_POLICY.trailing_multiplier == 3.0
    assert {p.atr_timeframe for p in ATR_EXIT_POLICIES.values()} == {
        "DAILY_14_COMPLETED_SESSIONS", "DAILY_22_COMPLETED_SESSIONS", "INTRADAY_15M_14",
    }
    assert get_atr_exit_policy("daily_atr_baseline_v1").stop_multiplier == 1.0  # live default unchanged
    assert CHANDELIER_ATR22_POLICY.levels(1000, 20, "BUY") == (940.0, 1120.0)
    with pytest.raises(ValueError):
        get_atr_exit_policy("nope")


def _bars(n=40, minutes=None, start="2026-10-01"):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    index = (
        pd.date_range(start, periods=n, freq=f"{minutes}min") if minutes
        else pd.date_range(start, periods=n, freq="D")
    )
    return pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
    }, index=index)


def test_atr_never_uses_bars_at_or_after_the_entry():
    bars = _bars(40)
    entry = bars.index[30] + pd.Timedelta(hours=9, minutes=30)  # during day 30
    base = atr_before(bars, 14, entry)
    poisoned = bars.copy()
    poisoned.iloc[30:, poisoned.columns.get_indexer(["high"])] += 500  # future / in-progress bars
    poisoned.iloc[30:, poisoned.columns.get_indexer(["low"])] -= 500
    assert atr_before(poisoned, 14, entry) == pytest.approx(base)
    # Day 30 is still in progress at the entry, so day 29 is the last completed session.
    assert atr_before(bars.iloc[:30], 14, bars.index[30]) == pytest.approx(base)


def test_intraday_atr_counts_a_15m_bar_only_after_it_completes():
    bars = _bars(60, minutes=15, start="2026-11-12 09:15")
    entry = bars.index[40] + pd.Timedelta(minutes=7)  # bar 40 is half done
    value = atr_before(bars, 14, entry, bar_minutes=15)
    assert value == pytest.approx(atr_before(bars.iloc[:40], 14, bars.index[40], bar_minutes=15))
    with pytest.raises(ValueError, match="needs"):
        atr_before(bars.iloc[:5], 14, bars.index[4], bar_minutes=15)


def test_atr22_needs_more_history_than_atr14():
    bars = _bars(20)
    assert atr_before(bars, 14, bars.index[-1] + pd.Timedelta(days=1)) > 0
    with pytest.raises(ValueError):
        atr_before(bars, 22, bars.index[-1] + pd.Timedelta(days=1))


# ── structure stop ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("side,entry,level,expected", [
    ("BUY", 100.0, 98.0, 97.5),    # level - 0.25 ATR, distance 2.5 = 1.25 ATR
    ("SELL", 100.0, 102.0, 102.5),
])
def test_structure_stop_is_beyond_the_level_and_tick_aligned(side, entry, level, expected):
    assert structure_stop(side, entry=entry, structural_level=level, atr=2.0, buffer_mult=0.25, tick=0.05) == expected


@pytest.mark.parametrize("side,entry,level", [
    ("BUY", 100.0, 99.9), ("BUY", 100.0, 93.0), ("SELL", 100.0, 100.1), ("SELL", 100.0, 107.0),
])
def test_structure_stop_outside_half_to_three_atr_skips_the_candidate(side, entry, level):
    assert structure_stop(side, entry=entry, structural_level=level, atr=2.0, buffer_mult=0.25, tick=0.05) is None


def test_structure_stop_rounds_away_from_the_entry():
    stop = structure_stop("BUY", entry=100.0, structural_level=98.07, atr=2.0, buffer_mult=0.0, tick=0.05)
    assert stop == 98.05
    stop = structure_stop("SELL", entry=100.0, structural_level=101.93, atr=2.0, buffer_mult=0.0, tick=0.05)
    assert stop == 101.95
    with pytest.raises(ValueError):
        structure_stop("HOLD", entry=100, structural_level=98, atr=2, buffer_mult=0, tick=0.05)


# ── shared exit math ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("action,stop,target,high,low,expected", [
    ("BUY", 95, 110, 105, 96, None), ("BUY", 95, 110, 105, 94, "STOP_LOSS"),
    ("BUY", 95, 110, 111, 96, "TARGET"), ("BUY", 95, 110, 111, 94, "STOP_LOSS"),
    ("SELL", 105, 90, 104, 91, None), ("SELL", 105, 90, 106, 91, "STOP_LOSS"),
    ("SELL", 105, 90, 104, 89, "TARGET"), ("BUY", 0, 0, 200, 1, None),
])
def test_stop_target_hit(action, stop, target, high, low, expected):
    assert stop_target_hit(action, stop, target, high, low) == expected


def test_gap_through_the_stop_fills_at_the_gap_price(monkeypatch):
    assert gap_fill_price("BUY", "STOP_LOSS", 95, 90) == 90
    assert gap_fill_price("BUY", "STOP_LOSS", 95, 97) == 95
    assert gap_fill_price("SELL", "STOP_LOSS", 105, 110) == 110
    assert gap_fill_price("BUY", "TARGET", 110, 115) == 115

    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "min_net_reward_risk", 1.0)
    engine = BacktestEngine(initial_capital=100_000)
    engine._process_signal({
        "symbol": "T", "action": "BUY", "price": 100, "stop_loss": 96, "target": 108, "atr": 2,
        "atr_exit_policy": "daily_atr_baseline_v1",
    }, pd.Timestamp("2026-11-12T10:00:00+05:30"), 100_000, 0)
    index = pd.date_range("2026-11-12T10:15:00+05:30", periods=1, freq="15min")
    data = {"T": pd.DataFrame({"open": [90.0], "high": [92.0], "low": [89.0], "close": [91.0],
                               "volume": [1000]}, index=index)}
    engine._check_exits(data, index[0])
    assert engine._closed_trades[0].exit_price == 90.0  # the gap price, not the 96 stop
    assert engine._closed_trades[0].exit_reason == "STOP_LOSS"


def test_simulation_and_live_manager_agree_on_the_same_levels():
    manager = StopLossManager()
    manager.register_trade({
        "symbol": "T", "action": "BUY", "price": 100, "stop_loss": 96, "target": 108, "atr": 2,
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
    })
    data = manager._tracking["T"]
    for price in (95.9, 96.0, 100.0, 108.0, 108.1):
        sim = stop_target_hit("BUY", 96, 108, high=price, low=price)
        live = manager._price_exit("T", price, data)
        assert (sim is None) == (live is None)
        if live:
            assert live["exit_type"] == sim


# ── per-position policy persistence ──────────────────────────────────────────

def test_new_trades_must_name_policy_and_timeframe():
    manager = StopLossManager()
    base = {"symbol": "T", "action": "BUY", "price": 100, "stop_loss": 96, "target": 108, "atr": 2}
    with pytest.raises(ValueError, match="required for new trades"):
        manager.register_trade(base)
    with pytest.raises(ValueError, match="required for new trades"):
        manager.register_trade({**base, "exit_policy": "daily_atr_baseline_v1"})
    with pytest.raises(ValueError, match="does not match policy"):
        manager.register_trade({
            **base, "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "INTRADAY_15M_14",
        })
    with pytest.raises(ValueError, match="Unknown ATR exit policy"):
        manager.register_trade({**base, "exit_policy": "made_up", "atr_timeframe": "X"})
    manager.register_trade(base, restoring=True)  # legacy positions have no policy
    assert manager.snapshot()["T"]["atr_timeframe"] == LEGACY_ATR_TIMEFRAME


def test_trailing_multiplier_comes_from_the_position_policy(tmp_path):
    from broker.portfolio_tracker import PortfolioTracker
    import risk.stop_loss_manager as module

    manager = StopLossManager()
    for name, policy in (("A", "daily_atr_baseline_v1"), ("B", "chandelier_atr22_k3_v1")):
        manager.register_trade({
            "symbol": name, "action": "BUY", "price": 100, "stop_loss": 90, "target": 150, "atr": 2,
            "exit_policy": policy, "atr_timeframe": get_atr_exit_policy(policy).atr_timeframe,
        })
        manager._tracking[name]["highest_since_entry"] = 120
    manager._check_exit("A", 119, manager._tracking["A"])
    manager._check_exit("B", 119, manager._tracking["B"])
    assert manager._tracking["A"]["stop_loss"] == 116  # 120 - 2 x 2
    assert manager._tracking["B"]["stop_loss"] == 114  # 120 - 3 x 2


def test_a_restored_stop_is_never_loosened_and_the_policy_survives(tmp_path):
    store = ControlStateStore(tmp_path / "stops.sqlite")
    manager = StopLossManager(store)
    manager.register_trade({
        "symbol": "T", "action": "BUY", "price": 100, "stop_loss": 95, "target": 130, "atr": 2,
        "entry_trade_ids": [1], "exit_policy": "chandelier_atr22_k3_v1",
        "atr_timeframe": "DAILY_22_COMPLETED_SESSIONS",
    })
    manager._tracking["T"]["highest_since_entry"] = 120
    manager._tracking["T"]["stop_loss"] = 112
    manager._persist()
    position = {
        "quantity": 5, "avg_price": 100.0, "stop_loss": 95.0, "target": 130.0, "atr": 2.0,
        "entry_trade_ids": [1], "holding_period": "SWING", "entry_time": datetime(2026, 11, 12, 10, 0),
        "exit_policy": "chandelier_atr22_k3_v1", "atr_timeframe": "DAILY_22_COMPLETED_SESSIONS",
    }
    restored = StopLossManager(store)
    restored.restore_from_positions({"T": position})
    snap = restored.snapshot()["T"]
    assert snap["stop_loss"] == 112  # not reset to the ledger's 95
    assert snap["exit_policy"] == "chandelier_atr22_k3_v1"
    assert snap["atr_timeframe"] == "DAILY_22_COMPLETED_SESSIONS"
    assert ratcheted_stop("BUY", 112, 105, 100, 2, 3.0) == 112  # a lower favourable price cannot loosen it


def test_unknown_stored_policy_falls_back_to_legacy_rules_without_dropping_protection():
    manager = StopLossManager()
    invalid = manager.restore_from_positions({"T": {
        "quantity": 5, "avg_price": 100.0, "stop_loss": 95.0, "target": 130.0, "atr": 2.0,
        "entry_trade_ids": [1], "holding_period": "INTRADAY", "exit_policy": "retired_policy_v0",
    }})
    assert invalid == []
    assert manager.snapshot()["T"]["exit_policy"] is None and manager.snapshot()["T"]["stop_loss"] == 95.0


# ── time exits ───────────────────────────────────────────────────────────────

def test_intraday_time_exit_is_1515_or_the_next_open():
    entry = ist(2026, 11, 12, 10, 0)
    assert time_exit_due("INTRADAY", entry, ist(2026, 11, 12, 15, 14)) is None
    assert time_exit_due("INTRADAY", entry, ist(2026, 11, 12, 15, 15)) == "INTRADAY_TIME_EXIT"
    assert time_exit_due("INTRADAY", entry, ist(2026, 11, 13, 9, 14)) is None
    assert time_exit_due("INTRADAY", entry, ist(2026, 11, 13, 9, 15)) == "OVERDUE_INTRADAY_EXIT"


def test_btst_exits_thirty_minutes_after_the_next_session_opens():
    entry = ist(2026, 11, 12, 15, 10)  # Thursday
    assert time_exit_due("BTST", entry, ist(2026, 11, 12, 15, 20)) is None
    assert time_exit_due("BTST", entry, ist(2026, 11, 13, 9, 44)) is None
    assert time_exit_due("BTST", entry, ist(2026, 11, 13, 9, 45)) == "BTST_TIME_EXIT"
    friday = ist(2026, 11, 13, 15, 10)  # the next session is Monday
    assert next_session_date(friday.date()) == datetime(2026, 11, 16).date()
    assert time_exit_due("BTST", friday, ist(2026, 11, 13, 15, 25)) is None
    assert time_exit_due("BTST", friday, ist(2026, 11, 14, 11, 0)) is None  # weekend
    assert time_exit_due("BTST", friday, ist(2026, 11, 16, 9, 45)) == "BTST_TIME_EXIT"


def test_btst_skips_holidays_and_stays_due_when_the_calendar_is_unknown():
    assert next_session_date(datetime(2026, 11, 23).date()) == datetime(2026, 11, 25).date()  # 24 Nov holiday
    entry = IST.localize(datetime(2031, 6, 2, 15, 10))
    with pytest.raises(ValueError, match="not on file"):
        next_session_date(entry.date())
    # Missing reference data never defers an exit.
    assert time_exit_due("BTST", entry, IST.localize(datetime(2031, 6, 3, 9, 45))) == "BTST_TIME_EXIT"


def test_swing_and_positional_use_maximum_holding_days(monkeypatch):
    monkeypatch.setattr(settings, "swing_max_holding_days", 5)
    monkeypatch.setattr(settings, "positional_max_holding_days", 30)
    entry = ist(2026, 11, 2, 10, 0)
    assert time_exit_due("SWING", entry, ist(2026, 11, 6, 15, 0)) is None
    assert time_exit_due("SWING", entry, ist(2026, 11, 7, 9, 30)) == "MAX_HOLDING_PERIOD"
    assert time_exit_due("POSITIONAL", entry, ist(2026, 11, 20, 15, 0)) is None
    assert time_exit_due("POSITIONAL", entry, ist(2026, 12, 2, 9, 30)) == "MAX_HOLDING_PERIOD"
    assert time_exit_due("DELIVERY", entry, ist(2027, 3, 1, 10, 0)) is None  # legacy rows keep their rules


def test_failed_thesis_needs_four_completed_bars_without_half_r():
    entry = ist(2026, 11, 12, 10, 0)
    now = ist(2026, 11, 12, 11, 15)
    assert time_exit_due("INTRADAY", entry, now, 4, bars_elapsed=3, max_favourable_r=0.1) is None
    assert time_exit_due("INTRADAY", entry, now, 4, bars_elapsed=4, max_favourable_r=0.49) == "FAILED_THESIS"
    assert time_exit_due("INTRADAY", entry, now, 4, bars_elapsed=4, max_favourable_r=0.5) is None
    assert time_exit_due("SWING", entry, now, 4, bars_elapsed=9, max_favourable_r=0.0) == "FAILED_THESIS"
    assert time_exit_due("SWING", entry, now) is None  # not measured -> not triggered


def test_manager_time_exit_covers_btst_and_swing_but_not_legacy_delivery(monkeypatch):
    import risk.stop_loss_manager as module

    manager = StopLossManager()
    positions = {
        "BT": ("BTST", ist(2026, 11, 12, 15, 10)),
        "SW": ("SWING", ist(2026, 10, 20, 10, 0)),
        "LG": ("DELIVERY", ist(2026, 10, 20, 10, 0)),
    }
    for symbol, (horizon, entry) in positions.items():
        manager.register_trade({
            "symbol": symbol, "action": "BUY", "price": 100, "stop_loss": 90, "target": 150, "atr": 2,
            "holding_period": horizon, "entry_time": entry,
            "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        })
    now = ist(2026, 11, 13, 9, 50)
    due = {s: manager._time_exit(s, 100.0, manager._tracking[s], now) for s in positions}
    assert due["BT"]["exit_type"] == "TIME_EXIT" and due["BT"]["exit_action"] == "SELL"
    assert due["SW"]["exit_type"] == "TIME_EXIT"
    assert due["LG"] is None
