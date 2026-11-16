import pytest

from broker.portfolio_tracker import PortfolioTracker


def test_daily_loss_halt_uses_day_start_equity_and_latches(monkeypatch):
    from config.settings import settings

    tracker = PortfolioTracker()
    tracker._day_start_equity = 10_000.0
    tracker._high_water_mark = 10_000.0
    tracker._persisted_high_water_mark = 10_000.0
    tracker._daily_baseline_valid = True
    tracker._available_capital = 0.0
    tracker._positions = {
        "TEST": {
            "quantity": 100, "avg_price": 100.0, "ltp": 98.0, "pnl": -200.0,
        }
    }

    monkeypatch.setattr(settings, "max_daily_loss_pct", 0.02)
    tracker._revalue()

    assert tracker.daily_loss_pct == pytest.approx(0.02)
    assert tracker.has_hit_daily_loss_limit()

    tracker.mark_price("TEST", 99.0)
    assert tracker.daily_loss_pct == pytest.approx(0.01)
    assert tracker.has_hit_daily_loss_limit()


def test_unknown_daily_baseline_fails_closed():
    tracker = PortfolioTracker()
    tracker._daily_baseline_valid = False

    assert tracker.has_hit_daily_loss_limit()


def test_portfolio_drawdown_pause_persists_across_tracker_restart(tmp_path):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._high_water_mark = 25_000.0
    tracker._persisted_high_water_mark = 25_000.0
    tracker._day_start_equity = 22_500.0
    tracker._daily_baseline_valid = True
    tracker._available_capital = 0.0
    tracker._positions = {
        "TEST": {
            "quantity": 250, "avg_price": 100.0,
            "ltp": 90.0, "pnl": -2_500.0,
        }
    }

    tracker._revalue()

    assert tracker.portfolio_drawdown_pct == pytest.approx(0.10)
    assert tracker.has_hit_portfolio_drawdown_limit()

    restored = PortfolioTracker(ControlStateStore(store.path))
    restored._total_value = 22_500.0
    assert restored.has_hit_portfolio_drawdown_limit()


def test_portfolio_drawdown_reset_requires_explicit_call(tmp_path):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._high_water_mark = 25_000.0
    tracker._persisted_high_water_mark = 25_000.0
    tracker._portfolio_drawdown_halted = True
    tracker._drawdown_state_loaded = True

    assert tracker.has_hit_portfolio_drawdown_limit()
    tracker.reset_portfolio_drawdown_halt()
    assert not tracker.has_hit_portfolio_drawdown_limit()
    assert ControlStateStore(store.path).read("portfolio_drawdown")["halted"] is False


@pytest.mark.parametrize(("amount", "cash_after"), [(5_000.0, 30_000.0), (-5_000.0, 20_000.0)])
def test_external_cash_flows_pause_entries_until_fresh_equity_sync(
    tmp_path, amount, cash_after,
):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._high_water_mark = 25_000.0
    tracker._persisted_high_water_mark = 25_000.0
    tracker.record_external_cash_flow("statement-row-1", amount, "broker_statement")

    assert store.read("portfolio_drawdown")["high_water_mark"] == 25_000.0
    assert tracker.cash_flow_reconciliation_pending
    assert tracker.has_hit_portfolio_drawdown_limit()

    tracker._available_capital = cash_after
    assert tracker.update_unrealised_pnl()
    assert not tracker.cash_flow_reconciliation_pending
    assert not tracker.has_hit_portfolio_drawdown_limit()
    assert tracker.portfolio_drawdown_pct == pytest.approx(0.0)
    assert tracker.snapshot()["drawdown_equity"] == pytest.approx(25_000.0)
    assert tracker.daily_loss_pct == pytest.approx(0.0)
    assert not tracker.has_hit_daily_loss_limit()


def test_external_cash_flow_without_persisted_baseline_fails_closed_until_review(tmp_path):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.record_external_cash_flow("untracked-baseline", 1_000.0, "broker_statement")
    tracker = PortfolioTracker(store)
    tracker._available_capital += 1_000.0

    assert tracker.update_unrealised_pnl()
    assert tracker.has_hit_portfolio_drawdown_limit()
    assert tracker._drawdown_baseline_unavailable
    with pytest.raises(ValueError, match="reviewed drawdown baseline"):
        tracker.record_external_cash_flow("another-flow", 500.0, "broker_statement")

    tracker.reset_portfolio_drawdown_halt()
    assert not tracker.has_hit_portfolio_drawdown_limit()
    assert not tracker._drawdown_baseline_unavailable


def test_external_cash_flow_references_are_idempotent_and_conflicts_rejected(tmp_path):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    assert store.record_external_cash_flow("flow-1", 1_000, "operator") == 1
    assert store.record_external_cash_flow("flow-1", 1_000, "operator") == 1
    assert store.external_cash_flow_snapshot() == (1_000.0, 1, 0)
    with pytest.raises(ValueError, match="conflicts"):
        store.record_external_cash_flow("flow-1", -1_000, "operator")


def test_drawdown_latch_cannot_be_reset_during_cash_flow_reconciliation(tmp_path):
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._portfolio_drawdown_halted = True
    tracker._drawdown_state_loaded = True
    tracker.record_external_cash_flow("flow-1", 1_000, "operator")
    with pytest.raises(ValueError, match="unreconciled"):
        tracker.reset_portfolio_drawdown_halt()
