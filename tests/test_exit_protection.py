from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from config.settings import settings
from operations.control_state import ControlStateStore
from risk.stop_loss_manager import StopLossManager
from scheduler import market_scheduler


@pytest.fixture
def exit_runtime(tmp_path, monkeypatch):
    from broker.portfolio_tracker import PortfolioTracker
    from risk.dead_mans_switch import DeadMansSwitch
    from risk import stop_loss_manager as stop_module

    store = ControlStateStore(tmp_path / "controls.sqlite")
    manager = StopLossManager(store)
    tracker = PortfolioTracker()
    tracker._positions = {
        "TEST": {
            "quantity": 5, "avg_price": 100, "ltp": 95,
            "stop_loss": 96, "target": 120, "holding_period": "INTRADAY",
            "mark_at": datetime.now(stop_module.IST).isoformat(),
        }
    }
    manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 96, "target": 120,
    })
    monkeypatch.setattr(market_scheduler, "stop_loss_manager", manager)
    monkeypatch.setattr(market_scheduler, "portfolio_tracker", tracker)
    monkeypatch.setattr(stop_module, "portfolio_tracker", tracker)
    monkeypatch.setattr(market_scheduler, "dead_mans_switch", DeadMansSwitch(store=store))
    monkeypatch.setattr(tracker, "update_unrealised_pnl", lambda: None)
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: None)
    monkeypatch.setattr(market_scheduler.telegram_notifier, "notify_error", AsyncMock())
    monkeypatch.setattr(settings, "paper_trading", True)
    return market_scheduler.MarketScheduler(store), manager, tracker


def test_uncertain_exit_is_durable_and_never_automatically_resubmitted(exit_runtime, monkeypatch):
    import asyncio
    from broker.base_broker import OrderSubmissionUncertainError

    scheduler, manager, _tracker = exit_runtime
    monkeypatch.setattr(settings, "paper_trading", False)
    calls = []

    def submit(*_args):
        calls.append(True)
        raise OrderSubmissionUncertainError("timeout after sending")

    monkeypatch.setattr(scheduler, "_place_exit_order", submit)
    monkeypatch.setattr(market_scheduler.angel_client, "get_order_book", lambda: [])
    asyncio.run(scheduler._check_stops())
    restored = StopLossManager(manager._store)
    restored.restore_from_positions(_tracker.positions)
    assert restored.pending_exits()
    asyncio.run(scheduler._check_stops())
    assert len(calls) == 1
    assert market_scheduler.dead_mans_switch.is_halted


def test_unidentified_submitted_exit_cannot_retry(exit_runtime, monkeypatch):
    import asyncio

    scheduler, manager, _tracker = exit_runtime
    monkeypatch.setattr(settings, "paper_trading", False)
    calls = []
    monkeypatch.setattr(scheduler, "_place_exit_order", lambda *_args: calls.append(True) or {"status": "PENDING"})
    monkeypatch.setattr(market_scheduler.angel_client, "get_order_book", lambda: [])
    asyncio.run(scheduler._check_stops())
    assert manager.pending_exits()
    asyncio.run(scheduler._check_stops())
    assert calls == [True]


def test_partial_fill_keeps_remaining_quantity_protected(exit_runtime, monkeypatch):
    import asyncio

    scheduler, manager, tracker = exit_runtime

    def submit(*_args):
        tracker.positions["TEST"]["quantity"] = 3
        return {"status": "PARTIAL", "order_id": "partial-1", "filled_quantity": 2}

    monkeypatch.setattr(scheduler, "_place_exit_order", submit)
    asyncio.run(scheduler._check_stops())
    assert tracker.positions["TEST"]["quantity"] == 3
    assert manager.snapshot()["TEST"]["stop_loss"] == 96
    assert manager.pending_exits() == {"TEST": "partial-1"}


def test_time_exit_remains_due_after_1600(monkeypatch):
    from risk import stop_loss_manager as module

    class AfterClose(datetime):
        @classmethod
        def now(cls, tz=None):
            return module.IST.localize(datetime(2026, 11, 13, 16, 1))

    monkeypatch.setattr(module, "datetime", AfterClose)
    manager = StopLossManager()
    manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "price": 100, "action": "BUY",
        "stop_loss": 90, "target": 120,
    })
    assert manager._check_exit("TEST", 100, manager._tracking["TEST"])["exit_type"] == "TIME_EXIT"


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_protective_exit_uses_a_tick_aligned_limit_not_market(side, monkeypatch):
    from broker import order_manager
    from config import instruments

    calls = []
    monkeypatch.setattr(instruments, "get_token", lambda *_args: "token")
    monkeypatch.setattr(instruments, "get_instrument", lambda *_args: {"tick_size": 10})
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_args: {"price": 100.03})
    monkeypatch.setattr(order_manager, "place_order", lambda **kwargs: calls.append(kwargs) or {"status": "EXECUTED"})
    monkeypatch.setattr(settings, "protective_exit_collar_pct", 0.003)
    market_scheduler.MarketScheduler._place_exit_order("TEST", side, {
        "quantity": -5 if side == "BUY" else 5, "ltp": 1,
        "holding_period": "INTRADAY",
    })
    assert calls[0]["order_type"] == "LIMIT"
    assert calls[0]["price"] == pytest.approx(100.3 if side == "BUY" else 99.8)
    assert not calls[0]["allow_requested_price_reference"]
    assert calls[0]["is_exit"] is True


def test_protective_exit_cannot_use_stale_position_mark(monkeypatch):
    from broker import order_manager
    from config import instruments

    monkeypatch.setattr(instruments, "get_token", lambda *_args: "token")
    monkeypatch.setattr(instruments, "get_instrument", lambda *_args: {"tick_size": 10})
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_args: None)
    monkeypatch.setattr(order_manager, "place_order", lambda **_kwargs: pytest.fail("No fresh reference"))
    assert market_scheduler.MarketScheduler._place_exit_order(
        "TEST", "SELL", {"quantity": 5, "ltp": 100},
    ) is None


def test_live_submission_timeout_is_not_reported_as_safe_retry(monkeypatch):
    from broker import order_manager
    from broker.base_broker import OrderSubmissionUncertainError

    def submit(_params):
        raise TimeoutError("connection lost after send")

    monkeypatch.setattr(order_manager.angel_client, "place_order", submit)
    with pytest.raises(OrderSubmissionUncertainError):
        order_manager._live_trade(
            {"ordertype": "LIMIT"}, "TEST", "SELL", 5, 100, None, None,
            "NSE", "risk_exit", 0, "test",
        )


def test_explicit_live_rejection_allows_a_later_retry(monkeypatch):
    from broker import order_manager
    from broker.base_broker import BrokerRejectedError

    def submit(_params):
        raise BrokerRejectedError("rejected")

    monkeypatch.setattr(order_manager.angel_client, "place_order", submit)
    assert order_manager._live_trade(
        {"ordertype": "LIMIT"}, "TEST", "SELL", 5, 100, None, None,
        "NSE", "risk_exit", 0, "test",
    ) is None


def test_cancelled_partial_exit_syncs_remaining_inventory_before_retry(exit_runtime, monkeypatch):
    import asyncio

    scheduler, manager, tracker = exit_runtime
    monkeypatch.setattr(settings, "paper_trading", False)
    manager.mark_exit_pending("TEST", "part-cancelled")
    monkeypatch.setattr(market_scheduler.angel_client, "get_order_book", lambda: [{
        "orderid": "part-cancelled", "status": "CANCELLED",
    }])

    def sync():
        tracker.positions["TEST"]["quantity"] = 3

    monkeypatch.setattr(tracker, "sync_from_broker", sync)
    asyncio.run(scheduler._reconcile_pending_exits())
    assert tracker.positions["TEST"]["quantity"] == 3
    assert not manager.pending_exits()
    assert manager.snapshot()["TEST"]["stop_loss"] == 96


def test_unexpected_exit_failure_halts_and_preserves_intent(exit_runtime, monkeypatch):
    scheduler, manager, tracker = exit_runtime

    def submit(*_args):
        raise RuntimeError("unexpected router failure")

    monkeypatch.setattr(scheduler, "_place_exit_order", submit)
    assert scheduler._submit_protective_exit("TEST", "SELL", tracker.positions["TEST"]) is None
    assert manager.pending_exits()["TEST"].startswith("UNRESOLVED:")
    assert market_scheduler.dead_mans_switch.is_halted


def test_broker_filled_status_is_reconciled(exit_runtime, monkeypatch):
    import asyncio

    scheduler, manager, tracker = exit_runtime
    monkeypatch.setattr(settings, "paper_trading", False)
    manager.mark_exit_pending("TEST", "filled")
    monkeypatch.setattr(market_scheduler.angel_client, "get_order_book", lambda: [{
        "orderid": "filled", "status": "FILLED",
    }])
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: tracker.positions.clear())
    asyncio.run(scheduler._reconcile_pending_exits())
    assert not manager.snapshot()


def test_restore_does_not_apply_previous_lots_stop_to_new_entry(tmp_path):
    store = ControlStateStore(tmp_path / "controls.sqlite")
    manager = StopLossManager(store)
    manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 95, "target": 120, "entry_trade_ids": [1],
    })
    manager._tracking["TEST"]["stop_loss"] = 110
    manager._tracking["TEST"]["highest_since_entry"] = 115
    manager._persist()
    restored = StopLossManager(store)
    assert restored.restore_from_positions({
        "TEST": {
            "quantity": 5, "avg_price": 100, "stop_loss": 96,
            "target": 130, "entry_trade_ids": [2],
        },
    }) == []
    assert restored.snapshot()["TEST"]["stop_loss"] == 96
    assert restored.snapshot()["TEST"]["target"] == 130


@pytest.mark.parametrize("response", [
    "broker-123", {"status": True, "data": {"orderid": "broker-123"}},
])
def test_angel_sdk_order_identifier_is_submission_not_fill(response, monkeypatch):
    from types import SimpleNamespace
    from broker.angel_client import AngelClient

    monkeypatch.setattr(
        AngelClient, "api", property(lambda _self: SimpleNamespace(placeOrder=lambda _params: response)),
    )
    result = AngelClient().place_order({"tradingsymbol": "TEST-EQ"})
    assert result["data"]["orderid"] == "broker-123"
    assert result.get("status") is True


@pytest.mark.parametrize("response", [None, {}, {"status": True, "data": {}}])
def test_angel_sdk_missing_identifier_requires_reconciliation(response, monkeypatch):
    from types import SimpleNamespace
    from broker.angel_client import AngelClient
    from broker.base_broker import OrderSubmissionUncertainError

    monkeypatch.setattr(
        AngelClient, "api", property(lambda _self: SimpleNamespace(placeOrder=lambda _params: response)),
    )
    with pytest.raises(OrderSubmissionUncertainError):
        AngelClient().place_order({"tradingsymbol": "TEST-EQ"})


def test_angel_sdk_explicit_rejection_is_distinct_from_uncertainty(monkeypatch):
    from types import SimpleNamespace
    from broker.angel_client import AngelClient
    from broker.base_broker import BrokerRejectedError

    monkeypatch.setattr(
        AngelClient, "api", property(lambda _self: SimpleNamespace(
            placeOrder=lambda _params: {"status": False},
        )),
    )
    with pytest.raises(BrokerRejectedError):
        AngelClient().place_order({"tradingsymbol": "TEST-EQ"})
