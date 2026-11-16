"""T15: the eight critical journeys of plan section 9, driven through real code with mocks and replay.

Each journey composes the production modules (ledger, stop manager, scheduler, router, reconciliation) instead
of re-testing a unit. Fixtures from neighbouring suites are reused so these stay in step with the real wiring.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from broker import order_manager
from broker.base_broker import OrderSubmissionUncertainError
from broker.portfolio_tracker import PortfolioTracker
from broker.websocket_feed import TickEventBus
from config.settings import settings
from database import db
from database.models import Trade
from operations.control_state import ControlStateStore
from operations.paper_reconciliation import reconcile_paper
from risk import stop_loss_manager as stop_module
from risk.stop_loss_manager import StopLossManager
from scheduler import market_scheduler
from scheduler.market_scheduler import MarketScheduler
from tests.test_candidate_contract import _entry_state
from tests.test_ledgers_t09 import _clock, paper_db  # noqa: F401  (fixture)
from tests.test_phase1_safety import _buy, _reference, database  # noqa: F401  (fixture)
from tests.test_protective_exit_lifecycle import (  # noqa: F401  (fixture)
    TOKEN, closing_exit, open_position, runtime, wait_for,
)
from tests.sentiment_fixtures import stub_persisted_review_verification

MIDDAY = datetime.now(stop_module.IST).replace(hour=11, minute=0, second=0, microsecond=0)
POLICY = {"exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS"}


def _place(symbol, action, quantity, *, product="INTRADAY", stop=None, target=None, **kwargs):
    from broker.paper_ledger import position_book_identity, replay_orders, summarize_positions

    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED", Trade.is_paper == True).all()
        book = position_book_identity(summarize_positions(replay_orders(trades)))
    kwargs.setdefault("expected_paper_book", book)
    return order_manager.place_order(
        symbol, f"tok-{symbol}", action, quantity, price=100, product_type=product,
        stop_loss=stop, target=target, strategy="journey", **kwargs,
    )


# ── 1. restart with long, short, delivery and partially closed positions ──────

def test_j1_restart_restores_exact_protection_and_unprotected_positions_block_resume(database, monkeypatch, tmp_path):
    _reference(monkeypatch)
    assert _place("LONG", "BUY", 5, stop=95, target=112)
    assert _place("DELIV", "BUY", 4, product="DELIVERY", stop=90, target=130)
    assert _place("PART", "BUY", 10, stop=96, target=110)
    from broker.paper_ledger import replay_orders

    with db.get_db() as session:
        part_ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["PART"]]
    assert _place("PART", "SELL", 4, is_exit=True, expected_entry_trade_ids=part_ids)
    assert _place("SHORT", "SELL", 3, stop=106, target=88)

    store = ControlStateStore(tmp_path / "restart.sqlite3")
    tracker = PortfolioTracker(store)
    tracker.sync_from_broker()
    first = StopLossManager(store)
    assert first.restore_from_positions(tracker.positions) == []
    first._tracking["LONG"]["stop_loss"] = 98.0  # a ratchet earned before the restart
    first._persist()

    restarted = StopLossManager(store)
    tracker2 = PortfolioTracker(store)
    tracker2.sync_from_broker()
    assert restarted.restore_from_positions(tracker2.positions) == []
    snap = restarted.snapshot()
    assert snap["LONG"]["stop_loss"] == 98.0  # never loosened back to the ledger's 95
    assert snap["DELIV"]["stop_loss"] == 90 and snap["DELIV"]["holding_period"] in {"DELIVERY", "SWING"}
    assert snap["SHORT"]["stop_loss"] == 106 and snap["SHORT"]["target"] == 88
    assert snap["PART"]["stop_loss"] == 96 and tracker2.positions["PART"]["quantity"] == 6  # the remainder stays protected
    report = reconcile_paper(tracker_positions=tracker2.positions, stop_snapshot=snap, now=MIDDAY)
    assert report["status"] != "fail", report["issues"]

    # An open position with no tracked protection is a reconciliation failure and blocks resume.
    missing = {k: v for k, v in snap.items() if k != "LONG"}
    report = reconcile_paper(tracker_positions=tracker2.positions, stop_snapshot=missing, now=MIDDAY)
    assert report["status"] == "fail"
    assert any(i["code"] == "missing_stop" and i["symbol"] == "LONG" for i in report["issues"])
    scheduler = MarketScheduler(store)
    scheduler.pause_trading()
    monkeypatch.setattr(scheduler, "reconcile_paper", lambda: report)
    with pytest.raises(ValueError, match="missing_stop"):
        scheduler.resume_trading()
    assert scheduler.is_paused is True


# ── 2. fresh feed -> reviewed candidate -> risk-approved entry -> audited fill -> protected ──

def test_j2_reviewed_entry_is_audited_and_protected_immediately(database, monkeypatch, tmp_path):
    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    result = graph._order_router_node(state)
    assert len(calls) == 1, result["errors"]
    call = calls[0]
    assert call["stop_loss"] == 97.0 and call["target"] == 109.0 and call["is_exit"] is False
    assert call["exit_policy"] and call["atr_timeframe"] and call["horizon"] == "INTRADAY"

    # The fill is then registered with a stop before anything else can happen to the position.
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(portfolio_tracker, "daily_entry_count", lambda *_a, **_k: 0)
    _reference(monkeypatch)
    fill = _place("PROT", "BUY", 5, stop=95, target=112, horizon="INTRADAY", **POLICY)
    assert fill and fill["status"] == "EXECUTED"
    store = ControlStateStore(tmp_path / "j2.sqlite3")
    manager = StopLossManager(store)
    tracker = PortfolioTracker(store)
    monkeypatch.setattr(market_scheduler, "stop_loss_manager", manager)
    monkeypatch.setattr(market_scheduler, "portfolio_tracker", tracker)
    scheduler = MarketScheduler(store)
    scheduler._track_executed_trades({"executed_orders": [{
        "symbol": "PROT", "action": "BUY", "price": 100, "stop_loss": 95, "target": 112, "atr": 3.0,
        "entry_trade_ids": [fill["trade_id"]], "holding_period": "INTRADAY", "is_exit": False, **POLICY,
    }]})
    assert manager.snapshot()["PROT"]["stop_loss"] == 95
    assert reconcile_paper(
        tracker_positions=tracker.positions, stop_snapshot=manager.snapshot(), now=MIDDAY,
    )["status"] != "fail"


# ── 3. an AI review is missing or failed -> no entry, exits keep working ─────

def test_j3_missing_review_rejects_the_entry_and_alerts_but_exits_still_route(database, monkeypatch, tmp_path):
    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    state["debate_results"] = {}  # the required review never arrived
    state["current_positions"] = {"HELD": {"quantity": 5, "entry_trade_ids": [1], "holding_period": "INTRADAY"}}
    state["trading_decisions"].append(
        {"symbol": "HELD", "action": "SELL", "quantity": 5, "price": 100, "holding_period": "INTRADAY"},
    )
    result = graph._order_router_node(state)
    assert [c["symbol"] for c in calls] == ["HELD"] and calls[0]["is_exit"] is True
    assert "[OrderRouter] New entry lacks a complete aligned debate review: OK" in result["errors"]


# ── 4. stop/target/cut-off race and uncertain outcomes -> one exit, safe retry ─

def test_j4_a_tick_exit_runs_once_even_under_a_burst_and_within_one_second(runtime, monkeypatch):
    bus = TickEventBus("angel")
    bus.set_connected("angel", True)
    monkeypatch.setattr(market_scheduler, "tick_event_bus", bus)
    scheduler = runtime.scheduler
    monkeypatch.setattr(scheduler, "scheduler", SimpleNamespace(start=lambda: None, shutdown=lambda **_kw: None))
    calls, timing = [], {}

    def submit(symbol, action, position):
        timing["exit"] = time.monotonic()
        calls.append((symbol, action))
        runtime.tracker._positions.pop(symbol, None)
        return {"status": "EXECUTED", "order_id": "paper-1"}

    monkeypatch.setattr(scheduler, "_place_exit_order", submit)

    async def scenario():
        scheduler.start()
        timing["tick"] = time.monotonic()
        for _ in range(25):  # a burst of duplicate crossing ticks from the feed thread
            await asyncio.to_thread(bus.publish, "angel", {"token": TOKEN, "ltp": 95.0})
        await wait_for(lambda: bool(calls))
        await asyncio.sleep(0.1)
        scheduler.stop()

    asyncio.run(scenario())
    assert calls == [("TEST", "SELL")]  # exactly one exit
    assert timing["exit"] - timing["tick"] <= 1.0


def test_j4_uncertain_exit_keeps_protection_and_retries_without_a_second_order(runtime, monkeypatch):
    scheduler = runtime.scheduler
    attempts = []

    def flaky(symbol, action, position):
        attempts.append(symbol)
        raise OrderSubmissionUncertainError("broker did not answer")

    monkeypatch.setattr(scheduler, "_place_exit_order", flaky)
    exit_info = runtime.manager.price_trigger("TEST", 95.0)
    assert exit_info is not None
    asyncio.run(scheduler._execute_exit_signal(exit_info, source="check"))
    assert attempts == ["TEST"]
    assert "TEST" in runtime.manager.snapshot()  # still tracked: the position is not assumed closed
    assert runtime.tracker.positions.get("TEST") is not None


# ── 5. stale or disconnected feed -> no stale entry and no invented exit price ─

def test_j5_stale_reference_prevents_entries_and_never_triggers_exits(database, monkeypatch, runtime):
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_a: None)  # no fresh tick
    assert _buy(5, symbol="STALE") is None
    with db.get_db() as session:
        assert session.query(Trade).filter(Trade.symbol == "STALE").count() == 0

    class Midday(datetime):
        @classmethod
        def now(cls, tz=None):
            return MIDDAY if tz is None else MIDDAY.astimezone(tz)

    monkeypatch.setattr(stop_module, "datetime", Midday)
    stale = open_position(mark_at=(MIDDAY - timedelta(minutes=10)).isoformat(), ltp=94.0)
    runtime.tracker._positions["TEST"] = stale
    assert runtime.manager.check_all_positions() == []  # a stale mark triggers nothing and invents no price


def test_j5_feed_staleness_reaches_the_dead_mans_switch(monkeypatch):
    from risk.dead_mans_switch import DeadMansSwitch

    switch = DeadMansSwitch(store=None)
    threshold = DeadMansSwitch.DEFAULT_THRESHOLDS["websocket"]
    for _ in range(threshold - 1):
        switch.record_failure("websocket", "no ticks for 120 s")
    assert not switch.is_halted
    switch.record_failure("websocket", "no ticks for 120 s")
    assert switch.is_halted and "websocket" in switch.halt_reason


# ── 6. loss / drawdown / exposure limits -> persistent block, exits remain ───

def test_j6_daily_loss_blocks_entries_across_restart_while_exits_remain(database, monkeypatch, tmp_path):
    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: True)
    state["current_positions"] = {"HELD": {"quantity": 5, "entry_trade_ids": [1], "holding_period": "INTRADAY"}}
    state["trading_decisions"].append(
        {"symbol": "HELD", "action": "SELL", "quantity": 5, "price": 100, "holding_period": "INTRADAY"},
    )
    result = graph._order_router_node(state)
    assert [c["symbol"] for c in calls] == ["HELD"], result["errors"]

    store = ControlStateStore(tmp_path / "halt.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._day_start_equity, tracker._daily_baseline_valid, tracker._daily_pnl = 10_000.0, True, -250.0
    assert tracker.has_hit_daily_loss_limit() is True
    restarted = PortfolioTracker(ControlStateStore(store.path))
    restarted._day_start_equity, restarted._daily_baseline_valid = 10_000.0, True
    assert restarted.has_hit_daily_loss_limit() is True  # the latch survives a restart


def test_j6_sizing_never_exceeds_the_per_position_cap_or_available_cash():
    from risk.position_sizer import PositionSizer

    quantity, reason = PositionSizer().size(
        {"action": "BUY", "price": 100.0, "stop_loss": 99.0, "target": 102.0},
        portfolio_value=10_000.0, available_capital=10_000.0, current_exposure=0.0,
    )
    assert reason is None or quantity == 0
    assert quantity * 100.0 <= 10_000.0 * settings.max_capital_per_trade_pct + 1e-6
    blocked, why = PositionSizer().size(
        {"action": "BUY", "price": 100.0, "stop_loss": 99.0, "target": 102.0},
        portfolio_value=10_000.0, available_capital=10_000.0, current_exposure=10_000.0 * settings.max_total_exposure_pct,
    )
    assert blocked == 0 and why


# ── 7. swing/BTST/US/crypto calendars, settled cash and independent accounting ─

def test_j7_markets_keep_independent_calendars_and_the_india_squareoff_does_not_touch_them(monkeypatch):
    from broker import alpaca_paper_adapter as alp
    from data import nse_calendar, us_calendar

    sunday_utc = datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc)
    assert us_calendar.is_open(sunday_utc) is False
    monday_open = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)  # 10:00 New York
    assert us_calendar.is_open(monday_open) is True
    # NSE trading holiday on 24 Nov 2026 (as listed in the shipped calendar) does not close the US market.
    assert nse_calendar.is_nse_holiday(datetime(2026, 11, 24).date()) is True
    assert us_calendar.is_open(datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)) is True
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", True)
    adapter = alp.AlpacaPaperAdapter(SimpleNamespace(
        paper=True, api_key="k", secret_key="s", trading_client=SimpleNamespace(_base_url="https://paper-api.alpaca.markets"),
    ))
    assert adapter.flatten_all_intraday() == []  # the 15:15 IST square-off never closes US positions

    from broker import crypto_sim

    assert crypto_sim.session_is_open(sunday_utc) is True  # crypto has no holiday calendar


def test_j7_delivery_proceeds_are_unspendable_until_settlement_and_crypto_is_a_separate_book(paper_db, monkeypatch):
    from broker.paper_ledger import replay_orders
    from broker import crypto_sim

    _clock(monkeypatch, datetime(2026, 11, 13, 14, 30))
    assert order_manager.place_order(
        "TEST", "token", "BUY", 50, price=100, product_type="DELIVERY", horizon="SWING",
        stop_loss=95, target=120, expected_paper_book={},
    )
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    assert order_manager.place_order(
        "TEST", "token", "SELL", 50, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    tracker = PortfolioTracker()
    tracker.sync_from_broker()
    assert tracker.snapshot()["unsettled_cash"] == 5000.0
    _clock(monkeypatch, datetime(2026, 11, 16, 10, 0))  # next trading day
    tracker.sync_from_broker()
    assert tracker.unsettled_cash == 0.0

    # A crypto simulation trade changes nothing in the India book.
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    before = tracker.snapshot()
    sim = crypto_sim.CryptoSim(1000, prices=crypto_sim.PriceSource(use_public=False))
    sim.place_order("BTCUSDT", "BUY", 0.0005, now=datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc),
                    portfolio_equity_inr=1_000_000)
    tracker.sync_from_broker()
    assert tracker.snapshot()["settled_cash"] == before["settled_cash"]
    assert "BTCUSDT" not in tracker.positions


# ── 8. identical inputs -> identical research results ───────────────────────

def test_j8_identical_versioned_inputs_reproduce_the_same_evaluation(database, monkeypatch):
    from research import evaluation as ev
    from tests.test_evaluation_t10 import synthetic_bars

    bars = synthetic_bars(drift=0.0015)
    common = dict(
        strategy="CompositeTrend", policy="daily_atr_baseline_v1", universe=list(bars), capital=100_000.0,
        start="2026-08-03", end="2026-09-25", folds=2, purge_bars=4, embargo_bars=4, bars=bars,
    )
    monkeypatch.setattr(settings, "min_net_reward_risk", 1.0)
    first = ev.run_walk_forward_evaluation(**common)
    second = ev.run_walk_forward_evaluation(**common)
    assert first["config_digest"] == second["config_digest"]
    assert first["metrics"] == second["metrics"]
    assert set(first) == set(second)


# ── readiness documents stay consistent with the registry ───────────────────

DOCS = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs"


def test_markets_readiness_lists_every_sleeve_and_enables_none():
    from strategies.contracts import STRATEGY_CONTRACTS

    text = (DOCS / "READINESS_MARKETS.md").read_text()
    for contract in STRATEGY_CONTRACTS.values():
        if contract.enabled_setting:
            assert contract.enabled_setting.upper() in text, contract.name
            assert getattr(settings, contract.enabled_setting) is False  # nothing ships enabled
    for market in ("FOREIGN_US_PAPER_ENABLED", "CRYPTO_SIM_ENABLED"):
        assert market in text
    assert "`blocked`" in text and "ready_for_operator_decision" in text


def test_india_readiness_names_every_required_pass_criterion():
    text = (DOCS / "READINESS_INDIA.md").read_text()
    for phrase in ("≤ 30 s", "SCAN_ENTRY_DEADLINE_SECONDS", "review_contract", "reference_age_seconds ≤ 1.0",
                   "15:15", "reconciliation", "paper only"):
        assert phrase.lower() in text.lower(), phrase


def test_operations_doc_has_runbook_capabilities_and_no_promised_returns():
    from operations.capabilities import CAPABILITIES

    text = (DOCS / "OPERATIONS.md").read_text()
    assert "Beginner runbook" in text and "No promised returns" in text and "Unsupported features" in text
    for capability in CAPABILITIES:
        assert f"| `{capability['name']}` |" in text, capability["name"]
