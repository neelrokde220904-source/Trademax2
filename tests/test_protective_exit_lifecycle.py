"""Regression tests for tick-triggered exits, mark freshness, overdue time exits and exit-intent recovery."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytz

from broker.base_broker import OrderSubmissionUncertainError
from broker.websocket_feed import TickEventBus
from config.settings import settings
from operations.control_state import ControlStateStore
from risk import stop_loss_manager as stop_module
from risk.stop_loss_manager import IST, StopLossManager, mark_is_fresh
from scheduler import market_scheduler

TOKEN = "tok-TEST"


def open_position(**overrides):
    position = {
        "quantity": 5, "avg_price": 100.0, "ltp": 100.0, "stop_loss": 96.0, "target": 120.0,
        "holding_period": "INTRADAY", "mark_at": datetime.now(IST).isoformat(), "mark_source": "ws_tick",
        "entry_trade_ids": [7],
    }
    position.update(overrides)
    return position


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from broker import order_manager
    from broker.portfolio_tracker import PortfolioTracker
    from config import instruments
    from risk.dead_mans_switch import DeadMansSwitch

    store = ControlStateStore(tmp_path / "controls.sqlite")
    manager = StopLossManager(store)
    tracker = PortfolioTracker(store)
    tracker._positions = {"TEST": open_position()}
    manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "action": "BUY", "price": 100, "stop_loss": 96, "target": 120,
        "entry_trade_ids": [7],
    })
    switch = DeadMansSwitch(store=store)
    switch._notifier = SimpleNamespace(send_message=AsyncMock(return_value=True))
    monkeypatch.setattr(market_scheduler, "stop_loss_manager", manager)
    monkeypatch.setattr(market_scheduler, "portfolio_tracker", tracker)
    monkeypatch.setattr(stop_module, "portfolio_tracker", tracker)
    monkeypatch.setattr(market_scheduler, "dead_mans_switch", switch)
    monkeypatch.setattr(tracker, "update_unrealised_pnl", lambda: None)
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: None)
    monkeypatch.setattr(market_scheduler.telegram_notifier, "notify_error", AsyncMock())
    monkeypatch.setattr(market_scheduler.telegram_notifier, "notify_trade_closed", AsyncMock())
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(instruments, "get_token", lambda symbol, *_args: TOKEN if symbol == "TEST" else None)
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 95.0, "source": "ws_tick", "age_seconds": 0.2},
    )
    return SimpleNamespace(
        scheduler=market_scheduler.MarketScheduler(store), manager=manager,
        tracker=tracker, switch=switch,
    )


def closing_exit(runtime, calls):
    def submit(symbol, action, _position):
        calls.append((symbol, action))
        runtime.tracker._positions.pop(symbol, None)
        return {"status": "EXECUTED", "order_id": "paper-1"}

    return submit


async def wait_for(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("Condition was not met in time")
        await asyncio.sleep(0.01)


def freeze_stop_clock(monkeypatch, moment):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(stop_module, "datetime", Frozen)


# ---- tick-triggered exits ----

def test_feed_thread_tick_exits_without_waiting_for_the_scheduled_check(runtime, monkeypatch):
    bus = TickEventBus("angel")
    bus.set_connected("angel", True)
    monkeypatch.setattr(market_scheduler, "tick_event_bus", bus)
    scheduler = runtime.scheduler
    monkeypatch.setattr(scheduler, "scheduler", SimpleNamespace(start=lambda: None, shutdown=lambda **_kw: None))
    calls = []
    monkeypatch.setattr(scheduler, "_place_exit_order", closing_exit(runtime, calls))

    async def scenario():
        scheduler.start()
        assert scheduler.protection_status()["tick_triggers_active"]
        await asyncio.to_thread(bus.publish, "angel", {"token": TOKEN, "ltp": 95.0})
        await wait_for(lambda: "TEST" not in runtime.manager.snapshot())
        scheduler.stop()
        assert not scheduler.protection_status()["tick_triggers_active"]
        runtime.tracker._positions["TEST"] = open_position()
        runtime.manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
            "symbol": "TEST", "action": "BUY", "price": 100, "stop_loss": 96, "target": 120,
        })
        await asyncio.to_thread(bus.publish, "angel", {"token": TOKEN, "ltp": 95.0})
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert calls == [("TEST", "SELL")]
    assert scheduler._exit_outcomes["TEST"]["source"] == "tick"
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "CLOSED"


def test_start_outside_an_event_loop_keeps_the_scheduled_check_only(runtime, monkeypatch):
    scheduler = runtime.scheduler
    monkeypatch.setattr(scheduler, "scheduler", SimpleNamespace(start=lambda: None, shutdown=lambda **_kw: None))
    scheduler.start()
    assert not scheduler.protection_status()["tick_triggers_active"]
    scheduler.stop()


def test_duplicate_ticks_coalesce_into_one_attempt_with_retry_backoff(runtime, monkeypatch):
    scheduler = runtime.scheduler
    calls = []
    monkeypatch.setattr(scheduler, "_place_exit_order", lambda symbol, *_args: calls.append(symbol) or None)

    async def scenario():
        scheduler._loop = asyncio.get_running_loop()
        for _ in range(3):
            scheduler._on_protective_tick({"token": TOKEN, "ltp": 95.0})
        await wait_for(lambda: calls and not scheduler._tick_exits_in_flight)
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 94.0})
        await asyncio.sleep(0.05)
        assert calls == ["TEST"]
        scheduler._tick_retry_after["TEST"] = 0.0
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 94.0})
        await wait_for(lambda: len(calls) == 2)

    asyncio.run(scenario())
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "NOT_FILLED"
    assert not runtime.manager.pending_exits()


def test_tick_without_a_fresh_confirmed_crossing_never_exits(runtime, monkeypatch):
    from broker import order_manager

    scheduler = runtime.scheduler
    calls = []
    monkeypatch.setattr(scheduler, "_place_exit_order", lambda symbol, *_args: calls.append(symbol) or None)

    async def scenario():
        scheduler._loop = asyncio.get_running_loop()
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 100.0})
        scheduler._on_protective_tick({"token": "unknown", "ltp": 1.0})
        assert not scheduler._tick_exits_in_flight
        monkeypatch.setattr(order_manager, "_market_reference", lambda *_args: None)
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 95.0})
        await wait_for(lambda: not scheduler._tick_exits_in_flight)
        monkeypatch.setattr(order_manager, "_market_reference", lambda *_args: {"price": 100.0})
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 95.0})
        await wait_for(lambda: not scheduler._tick_exits_in_flight)
        runtime.manager.mark_exit_pending("TEST", "UNRESOLVED:prior")
        scheduler._on_protective_tick({"token": TOKEN, "ltp": 95.0})
        assert not scheduler._tick_exits_in_flight

    asyncio.run(scenario())
    assert calls == []
    assert "TEST" not in scheduler._tick_retry_after


# ---- mark freshness and time exits ----

@pytest.mark.parametrize(("mark_offset", "ltp", "fresh"), [
    (0, 100.0, True), (59, 100.0, True), (61, 100.0, False),
    (-3, 100.0, True), (-10, 100.0, False), (None, 100.0, False), (0, 0.0, False),
])
def test_mark_freshness_window(mark_offset, ltp, fresh):
    now = IST.localize(datetime(2026, 11, 16, 11, 0))
    mark_at = None if mark_offset is None else (now - timedelta(seconds=mark_offset)).isoformat()
    assert mark_is_fresh({"ltp": ltp, "mark_at": mark_at}, now) is fresh


def test_stale_mark_never_triggers_or_ratchets_a_stop_but_time_exit_stays_due(monkeypatch):
    moment = IST.localize(datetime(2026, 11, 16, 11, 0))
    freeze_stop_clock(monkeypatch, moment)
    manager = StopLossManager()
    manager.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "action": "BUY", "price": 100, "stop_loss": 96, "target": 120, "atr": 2,
    })
    positions = {}
    monkeypatch.setattr(stop_module, "portfolio_tracker", SimpleNamespace(positions=positions))
    stale = (moment - timedelta(minutes=10)).isoformat()

    positions["TEST"] = {"quantity": 5, "ltp": 130.0, "mark_at": stale}
    assert manager.check_all_positions() == []
    assert manager.snapshot()["TEST"]["highest_since_entry"] == 100
    positions["TEST"] = {"quantity": 5, "ltp": 90.0, "mark_at": stale}
    assert manager.check_all_positions() == []
    assert manager.last_check()["unverified_marks"] == ["TEST"]
    assert manager.snapshot()["TEST"]["stop_loss"] == 96

    positions["TEST"] = {"quantity": 5, "ltp": 90.0, "mark_at": moment.isoformat()}
    assert manager.check_all_positions()[0]["exit_type"] == "STOP_LOSS"
    assert manager.last_check()["unverified_marks"] == []

    freeze_stop_clock(monkeypatch, IST.localize(datetime(2026, 11, 16, 15, 20)))
    positions["TEST"] = {"quantity": 5, "ltp": 90.0, "mark_at": stale}
    assert manager.check_all_positions()[0]["exit_type"] == "TIME_EXIT"


@pytest.mark.parametrize("entry_time", [
    datetime(2026, 11, 13, 14, 0),
    pytz.utc.localize(datetime(2026, 11, 13, 8, 30)),
    "2026-11-13T14:00:00+05:30",
    "2026-11-13T14:00:00",
])
def test_carried_intraday_position_is_due_at_the_next_session_open(entry_time):
    data = {"action": "BUY", "holding_period": "INTRADAY", "entry_time": entry_time}

    def monday(hour, minute):
        return IST.localize(datetime(2026, 11, 16, hour, minute))

    assert StopLossManager._time_exit("TEST", 100.0, data, monday(9, 14)) is None
    due = StopLossManager._time_exit("TEST", 100.0, data, monday(9, 15))
    assert due["exit_type"] == "TIME_EXIT" and due["exit_action"] == "SELL"
    assert "carried from 2026-11-13" in due["reason"]


def test_same_day_and_delivery_positions_are_not_overdue():
    now = IST.localize(datetime(2026, 11, 16, 10, 0))
    same_day = {
        "action": "SELL", "holding_period": "INTRADAY",
        "entry_time": IST.localize(datetime(2026, 11, 16, 9, 20)),
    }
    delivery = {"action": "BUY", "holding_period": "DELIVERY", "entry_time": datetime(2026, 11, 13, 14, 0)}
    assert StopLossManager._time_exit("TEST", 100.0, same_day, now) is None
    assert StopLossManager._time_exit("TEST", 100.0, delivery, now) is None
    close = StopLossManager._time_exit("TEST", 100.0, same_day, IST.localize(datetime(2026, 11, 16, 15, 15)))
    assert close["exit_action"] == "BUY"


# ---- exit-intent recovery ----

def test_paper_intent_is_settled_from_the_ledger_without_the_broker_order_book(runtime, monkeypatch):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    manager.begin_exit("TEST")
    monkeypatch.setattr(
        market_scheduler.angel_client, "get_order_book",
        lambda: pytest.fail("Paper exits must not query the broker order book"),
    )
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: tracker._positions.clear())
    asyncio.run(scheduler._reconcile_pending_exits())
    assert "TEST" not in manager.snapshot()
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "CLOSED"
    assert not runtime.switch.is_halted


def test_paper_uncertain_exit_resumes_protection_and_retries_with_the_halt_kept(runtime, monkeypatch):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    tracker._positions["TEST"].update(ltp=95.0, entry_trade_ids=[7])
    calls = []

    def submit(symbol, _action, _position):
        calls.append(symbol)
        if len(calls) == 1:
            raise OrderSubmissionUncertainError("timeout after sending")
        tracker._positions.clear()
        return {"status": "EXECUTED"}

    monkeypatch.setattr(scheduler, "_place_exit_order", submit)
    asyncio.run(scheduler._check_stops())
    assert manager.pending_exits()["TEST"].startswith("UNRESOLVED:")
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "UNRESOLVED"
    assert runtime.switch.is_halted

    asyncio.run(scheduler._check_stops())
    assert calls == ["TEST", "TEST"]
    assert "TEST" not in manager.snapshot()
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "CLOSED"
    assert runtime.switch.is_halted


def test_paper_intent_on_a_position_that_changed_side_halts_and_stays_pending(runtime, monkeypatch):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    manager.begin_exit("TEST")
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: tracker._positions["TEST"].update(quantity=-5))
    asyncio.run(scheduler._reconcile_pending_exits())
    assert manager.pending_exits()["TEST"].startswith("UNRESOLVED:")
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "SIDE_MISMATCH"
    assert runtime.switch.is_halted


@pytest.mark.parametrize("original_ids,remaining_ids,holding_period", [
    ([], [7], "INTRADAY"),
    ([7], [], "INTRADAY"),
    ([7], [7, 8], "INTRADAY"),
    ([7], [7], "DELIVERY"),
])
def test_paper_intent_needs_matching_lots_and_horizon_to_resume(
    runtime, original_ids, remaining_ids, holding_period,
):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    manager.bind_entry_identity("TEST", {"entry_trade_ids": original_ids})
    tracker._positions["TEST"].update(
        entry_trade_ids=remaining_ids, holding_period=holding_period,
    )
    manager.begin_exit("TEST")
    pending = manager.pending_exits()
    asyncio.run(scheduler._reconcile_pending_exits())
    assert manager.pending_exits() == pending
    assert manager.snapshot()["TEST"]["entry_trade_ids"] == original_ids
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "IDENTITY_MISMATCH"
    assert runtime.switch.is_halted


def test_paper_intent_resumes_for_remaining_original_lots(runtime):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    manager.bind_entry_identity("TEST", {"entry_trade_ids": [6, 7]})
    manager.begin_exit("TEST")
    tracker._positions["TEST"].update(quantity=3, entry_trade_ids=[7])
    asyncio.run(scheduler._reconcile_pending_exits())
    assert not manager.pending_exits()
    assert manager.snapshot()["TEST"]["entry_trade_ids"] == [7]
    assert manager.snapshot()["TEST"]["stop_loss"] == 96
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "PROTECTION_RESUMED"


def test_paper_intent_is_kept_when_the_ledger_cannot_be_read(runtime, monkeypatch):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    manager.begin_exit("TEST")

    def unreadable():
        raise OSError("database is locked")

    monkeypatch.setattr(tracker, "sync_from_broker", unreadable)
    asyncio.run(scheduler._reconcile_pending_exits())
    assert manager.pending_exits()["TEST"].startswith("UNRESOLVED:")
    assert runtime.switch.is_halted


def test_strategy_exit_waits_while_a_protective_exit_is_pending(runtime, monkeypatch):
    from agents import graph

    monkeypatch.setattr(stop_module, "stop_loss_manager", runtime.manager)
    runtime.manager.begin_exit("TEST")
    monkeypatch.setattr(graph, "place_order", lambda **_kw: pytest.fail("A second exit must not be sent"))
    result = graph._order_router_node({
        "execution_enabled": True,
        "trading_decisions": [{
            "symbol": "TEST", "action": "SELL", "quantity": 5, "price": 95,
            "holding_period": "INTRADAY",
        }],
        "current_positions": {"TEST": {"quantity": 5}},
    })
    assert result["executed_orders"] == []
    assert any("deferred: a protective exit awaits reconciliation" in error for error in result["errors"])


def test_live_entries_fail_closed_until_the_live_order_lifecycle_is_validated(monkeypatch):
    from agents import graph

    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    monkeypatch.setattr(graph, "place_order", lambda **_kw: pytest.fail("A live entry must not be sent"))
    assert graph.LIVE_ORDER_LIFECYCLE_VALIDATED is False
    result = graph._order_router_node({
        "execution_enabled": True,
        "trading_decisions": [{
            "symbol": "NEW", "action": "BUY", "quantity": 1, "price": 100,
            "stop_loss": 95, "holding_period": "INTRADAY",
        }],
        "current_positions": {},
    })
    assert result["executed_orders"] == []
    assert any(error.startswith("[OrderRouter] Live entry blocked for NEW") for error in result["errors"])


# ---- feed coverage, health and visibility ----

def test_held_positions_are_subscribed_first_and_feed_gaps_are_reported(runtime, monkeypatch):
    from config import instruments, universe

    scheduler, tracker = runtime.scheduler, runtime.tracker
    tracker._positions = {"HELD": open_position(), "NOTOKEN": open_position()}
    watchlist = [f"W{index}" for index in range(60)]
    tokens = {"HELD": "t-held", **{symbol: f"t-{symbol}" for symbol in watchlist}}
    monkeypatch.setattr(instruments, "get_token", lambda symbol, *_args: tokens.get(symbol))
    monkeypatch.setattr(universe, "get_active_watchlist", lambda *_args: watchlist)
    subscriptions = []
    feed = SimpleNamespace(
        connect=lambda: None,
        subscribe=lambda exchange, values: subscriptions.append((exchange, list(values))),
    )
    monkeypatch.setattr(market_scheduler, "ws_feed", feed)
    monkeypatch.setattr(market_scheduler, "angel_client", SimpleNamespace(has_credentials=True))

    asyncio.run(scheduler._connect_ws())
    exchange, subscribed = subscriptions[-1]
    assert exchange == "nse_cm"
    assert subscribed[0] == "t-held" and len(subscribed) == market_scheduler.MAX_WS_TOKENS
    assert scheduler.protection_status()["unsubscribed_positions"] == ["NOTOKEN"]
    market_scheduler.telegram_notifier.notify_error.assert_awaited_once()

    def unavailable(*_args):
        raise universe.UniverseUnavailableError("membership is stale")

    monkeypatch.setattr(universe, "get_active_watchlist", unavailable)
    asyncio.run(scheduler._connect_ws())
    assert subscriptions[-1][1] == ["t-held"]


def test_protective_check_failures_reach_the_dead_mans_switch(runtime, monkeypatch):
    scheduler = runtime.scheduler
    results = iter([RuntimeError, RuntimeError, [], RuntimeError, RuntimeError, RuntimeError])

    def check():
        result = next(results)
        if result is RuntimeError:
            raise RuntimeError("stop store unavailable")
        return result

    monkeypatch.setattr(runtime.manager, "check_all_positions", check)
    for _ in range(4):
        asyncio.run(scheduler._check_stops())
    assert not runtime.switch.is_halted
    assert scheduler.protection_status()["last_check"]["error"] == "RuntimeError"
    asyncio.run(scheduler._check_stops())
    asyncio.run(scheduler._check_stops())
    assert runtime.switch.is_halted
    assert "protective_exits" in runtime.switch.halt_reason


def test_protection_status_reports_coverage_freshness_and_pending_exits(runtime):
    scheduler, manager, tracker = runtime.scheduler, runtime.manager, runtime.tracker
    tracker._positions["BARE"] = open_position(mark_at=None, ltp=50.0, holding_period="DELIVERY")

    rows = {row["symbol"]: row for row in scheduler.protection_status()["positions"]}
    assert rows["TEST"]["tracked"] and rows["TEST"]["stop_loss"] == 96 and rows["TEST"]["mark_fresh"]
    assert rows["TEST"]["pending_exit"] is None
    assert not rows["BARE"]["tracked"] and not rows["BARE"]["mark_fresh"]
    assert rows["BARE"]["mark_age_seconds"] is None

    manager.begin_exit("TEST")
    status = scheduler.protection_status()
    assert {row["symbol"]: row for row in status["positions"]}["TEST"]["pending_exit"] == "unresolved_intent"
    manager.mark_exit_pending("TEST", "broker-1")
    status = scheduler.protection_status()
    assert {row["symbol"]: row for row in status["positions"]}["TEST"]["pending_exit"] == "submitted_order"
    json.dumps(status)
