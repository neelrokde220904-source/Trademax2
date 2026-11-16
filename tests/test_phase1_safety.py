"""Phase 1 safety layer: one identity per symbol, halt latching, budgets, deadlines, calendars."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from broker import order_manager
from broker.paper_ledger import position_book_identity, replay_orders, summarize_positions
from broker.portfolio_tracker import PortfolioTracker
from config.settings import settings
from database import db
from database.models import AlertOutbox, Base, LLMUsage, PaperFillAudit, Trade
from operations.control_state import ControlStateStore
from scheduler import market_scheduler as ms


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'phase1.sqlite'}", connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "max_daily_entries", 100)
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 100)
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 0)
    yield engine
    engine.dispose()


def _reference(monkeypatch, price=100.0, top=None):
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_a: {
        "price": price, "source": "ws_tick", "age_seconds": 1.0, "top_quantity": top,
    })


def _book():
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED", Trade.is_paper == True).all()
        return position_book_identity(summarize_positions(replay_orders(trades)))


def _buy(quantity=5, symbol="TEST", **kwargs):
    kwargs.setdefault("expected_paper_book", _book())
    return order_manager.place_order(
        symbol, "token", "BUY", quantity, price=100, product_type="INTRADAY",
        stop_loss=95, strategy="test", **kwargs,
    )


# ── 1.1 one open identity per symbol ───────────────────────────────────────────

def test_paper_ledger_rejects_a_second_entry_even_with_a_matching_book(database, monkeypatch):
    _reference(monkeypatch)
    assert _buy() is not None
    assert _buy() is None  # the book snapshot matches, so only the new rule can refuse it
    with db.get_db() as session:
        assert session.query(Trade).count() == 1
        assert sorted(a.outcome for a in session.query(PaperFillAudit)) == ["EXISTING_POSITION", "FILLED"]


def test_restore_of_mixed_lots_uses_tightest_stop_and_intraday(database):
    from risk.portfolio_stop_risk import StopRiskUnavailable, open_stop_risk
    from risk.stop_loss_manager import StopLossManager

    now = datetime.now()
    lots = [
        Trade(id=1, timestamp=now - timedelta(hours=2), symbol="MIX", action="BUY", quantity=5, price=100,
              holding_period="DELIVERY", stop_loss=94, target=120),
        Trade(id=2, timestamp=now - timedelta(hours=1), symbol="MIX", action="BUY", quantity=5, price=100,
              holding_period="INTRADAY", stop_loss=97, target=110),
    ]
    position = summarize_positions(replay_orders(lots))["MIX"]
    assert position["mixed_lots"] is True
    assert position["stop_loss"] == 97 and position["target"] == 110
    assert position["holding_period"] == "INTRADAY"

    manager = StopLossManager()
    manager.restore_from_positions({"MIX": {**position, "ltp": 101}})
    protection = manager.snapshot()["MIX"]
    assert protection["mixed_lots"] is True
    assert protection["stop_loss"] == 97 and protection["holding_period"] == "INTRADAY"
    with pytest.raises(StopRiskUnavailable):
        open_stop_risk({"MIX": {**position, "ltp": 101}}, manager.snapshot())


def test_mixed_short_lots_pick_the_highest_priced_protection_that_is_tightest():
    now = datetime.now()
    lots = [
        Trade(id=1, timestamp=now - timedelta(hours=2), symbol="S", action="SELL", quantity=5, price=100,
              holding_period="DELIVERY", stop_loss=106, target=90),
        Trade(id=2, timestamp=now - timedelta(hours=1), symbol="S", action="SELL", quantity=5, price=100,
              holding_period="INTRADAY", stop_loss=103, target=95),
    ]
    position = summarize_positions(replay_orders(lots))["S"]
    assert position["stop_loss"] == 103 and position["target"] == 95


# ── 1.2 daily halt latch ───────────────────────────────────────────────────────

def _tracker(store=None):
    tracker = PortfolioTracker(store)
    tracker._day_start_equity = 10_000.0
    tracker._daily_baseline_valid = True
    return tracker


def test_transient_invalid_baseline_blocks_then_unblocks(tmp_path):
    store = ControlStateStore(tmp_path / "c.sqlite3")
    tracker = _tracker(store)
    tracker._daily_baseline_valid = False
    assert tracker.has_hit_daily_loss_limit() is True
    assert store.read("daily_loss_halt") is None  # nothing persisted
    tracker._daily_baseline_valid = True
    assert tracker.has_hit_daily_loss_limit() is False


def test_real_daily_loss_stays_latched_across_restart(tmp_path):
    store = ControlStateStore(tmp_path / "c.sqlite3")
    tracker = _tracker(store)
    tracker._daily_pnl = -250.0
    assert tracker.has_hit_daily_loss_limit() is True
    restarted = _tracker(ControlStateStore(store.path))
    assert restarted.has_hit_daily_loss_limit() is True
    assert store.read("daily_loss_halt")["reason"] == "daily_loss_limit"


def test_stored_baseline_unavailable_record_and_other_dates_are_ignored(tmp_path):
    store = ControlStateStore(tmp_path / "c.sqlite3")
    today = datetime.now(ms.IST).strftime("%Y-%m-%d")
    store.write("daily_loss_halt", {"date": today, "halted": True, "reason": "baseline_unavailable"})
    assert _tracker(store).has_hit_daily_loss_limit() is False
    store.write("daily_loss_halt", {"date": "2000-01-01", "halted": True, "reason": "daily_loss_limit"})
    assert _tracker(store).has_hit_daily_loss_limit() is False


# ── 1.3 / 1.4 paid scans that cannot lead to an entry are skipped ────────────

@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "sched.sqlite3")
    instance = ms.MarketScheduler(store=store)
    instance._trading_paused = False
    calls = []

    async def pipeline(*_args, **_kwargs):
        calls.append(1)
        return {"errors": [], "executed_orders": [], "trading_decisions": []}

    monkeypatch.setattr(ms, "run_trading_pipeline", pipeline)
    monkeypatch.setattr(ms, "dead_mans_switch", type("D", (), {
        "check_allowed": staticmethod(lambda: (True, "")), "record_success": staticmethod(lambda *_: None),
        "record_failure": staticmethod(lambda *_: None),
    })())
    monkeypatch.setattr(ms, "portfolio_tracker", type("T", (), {
        "has_hit_portfolio_drawdown_limit": staticmethod(lambda: False),
        "has_hit_daily_loss_limit": staticmethod(lambda: False),
    })())
    monkeypatch.setattr(instance, "reconcile_paper", lambda: {"status": "pass", "issues": []})
    monkeypatch.setattr(ms.telegram_notifier, "notify_risk_alert", AsyncMock())
    monkeypatch.setattr(ms.telegram_notifier, "notify_error", AsyncMock())
    monkeypatch.setattr(settings, "paper_trading", True)
    instance.calls = calls
    return instance


def _spend(session, usd, status="success"):
    session.add(LLMUsage(
        id=str(time.time_ns()), run_id="r", mode="economy", model="m", task="t",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None), status=status, estimated_usd=usd,
    ))


def test_scan_runs_when_every_gate_is_open(scheduler, database):
    asyncio.run(scheduler._run_scan(watchlist=["TCS"]))
    assert scheduler.calls == [1]
    assert scheduler.last_scan["status"] == "completed"
    assert scheduler.last_successful_analysis is not None


def test_daily_halt_skips_paid_scan_and_records_why(scheduler, database, monkeypatch):
    monkeypatch.setattr(ms.portfolio_tracker, "has_hit_daily_loss_limit", staticmethod(lambda: True))
    asyncio.run(scheduler._run_scan(watchlist=["TCS"]))
    assert scheduler.calls == []
    assert scheduler.last_skip["reason"] == "daily_halt"
    assert scheduler._store.read("scan_status")["last_skip"]["reason"] == "daily_halt"
    assert not scheduler.is_paused


def test_budget_exhaustion_and_unknown_pricing_skip_the_scan(scheduler, database):
    with db.get_db() as session:
        _spend(session, 1.0)  # about ₹90 at the default rate, above the ₹50 budget
    asyncio.run(scheduler._run_scan(watchlist=["TCS"]))
    assert scheduler.calls == [] and scheduler.last_skip["reason"] == "llm_budget"
    ms.telegram_notifier.notify_risk_alert.assert_awaited_once()
    assert ms.telegram_notifier.notify_risk_alert.await_args.kwargs["dedup_key"].startswith("llm_budget:")


def test_unknown_pricing_counts_as_exceeded(scheduler, database):
    with db.get_db() as session:
        _spend(session, None)
    asyncio.run(scheduler._run_scan(watchlist=["TCS"]))
    assert scheduler.calls == [] and scheduler.last_skip["reason"] == "llm_budget"


def test_budget_does_not_block_an_analysis_only_scan_or_exits(scheduler, database, monkeypatch):
    with db.get_db() as session:
        _spend(session, 5.0)
    scheduler._trading_paused = True
    asyncio.run(scheduler._run_scan(analysis_only=True, watchlist=["TCS"]))
    assert scheduler.calls == [1]


def test_scan_status_survives_a_restart(scheduler, database):
    asyncio.run(scheduler._run_scan(watchlist=["TCS"]))
    restarted = ms.MarketScheduler(store=ControlStateStore(scheduler._store.path))
    assert restarted.last_scan["status"] == "completed"
    status = restarted.scan_status()
    assert "next_trigger" in status and "next_eligible_reason" in status


def test_router_errors_do_not_degrade_a_scan_but_other_errors_do():
    assert not ms.MarketScheduler._scan_is_degraded({"errors": ["[OrderRouter] Entry blocked for X: gate"]})
    assert ms.MarketScheduler._scan_is_degraded({"errors": ["[Technical] boom"]})


def test_llm_profile_cannot_change_mid_run(database, monkeypatch):
    from config import llm_runtime as runtime

    monkeypatch.setattr(settings, "anthropic_api_key", "k")
    monkeypatch.setattr(settings, "deepseek_api_key", "k")
    with runtime.llm_run("scan") as context:
        with pytest.raises(ValueError, match="Cannot change an active LLM run profile"):
            with runtime.llm_run("scan", profile=runtime.profile_for(
                "economy" if context.profile.mode == "claude" else "claude", 9,
            )):
                pass
        runtime.select_mode("economy" if context.profile.mode == "claude" else "claude", 0)
        assert runtime.current_profile() == context.profile


# ── scan entry deadline: entries only ─────────────────────────────────────────

def test_deadline_blocks_entries_but_exits_still_route(monkeypatch):
    from agents import graph

    placed = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: placed.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": 1, "quantity": kw["quantity"],
        "fill_price": 100.0,
    })
    monkeypatch.setattr(graph, "get_token", lambda _s: "token")
    monkeypatch.setattr(settings, "paper_trading", True)
    from risk import stop_loss_manager as slm

    monkeypatch.setattr(slm.stop_loss_manager, "pending_exits", lambda: {})
    state = {
        "execution_enabled": True,
        "scan_started_monotonic": time.monotonic() - settings.scan_entry_deadline_seconds - 5,
        "current_positions": {"HELD": {
            "quantity": 5, "entry_trade_ids": [1], "holding_period": "INTRADAY",
        }},
        "trading_decisions": [
            {"symbol": "NEW", "action": "BUY", "quantity": 1, "price": 100, "stop_loss": 95,
             "target": 110, "holding_period": "INTRADAY"},
            {"symbol": "HELD", "action": "SELL", "quantity": 5, "price": 100, "holding_period": "INTRADAY"},
        ],
    }
    result = graph._order_router_node(state)
    assert "[OrderRouter] Entry blocked for NEW: scan exceeded entry deadline" in result["errors"]
    assert [call["symbol"] for call in placed] == ["HELD"]
    assert placed[0]["is_exit"] is True


def test_scan_deadline_error_is_classified():
    from operations.scan_outcomes import classify_router_error

    code, _reason, kind = classify_router_error(
        "[OrderRouter] Entry blocked for NEW: scan exceeded entry deadline",
    )
    assert (code, kind) == ("scan_deadline", "blocked")
    code, _reason, kind = classify_router_error(
        "[OrderRouter] Entry blocked for NEW: an open position already exists",
    )
    assert (code, kind) == ("existing_position", "blocked")


# ── 1.5 data fixes ───────────────────────────────────────────────────────────

def _contract(expiry, name="NIFTY"):
    return {
        "exch_seg": "NFO", "name": name, "symbol": f"{name}{expiry}24000CE", "instrumenttype": "OPTIDX",
        "strike": "2400000", "lotsize": "75", "token": f"t{expiry}", "expiry": expiry,
    }


def test_expiry_is_read_from_contracts_not_a_fixed_weekday():
    from data.instrument_master import is_fno_expiry_day

    contracts = [_contract("17NOV2026"), _contract("23NOV2026"), _contract("01DEC2026")]
    assert is_fno_expiry_day("NIFTY", date(2026, 11, 17), contracts) is True  # a Tuesday
    assert is_fno_expiry_day("NIFTY", date(2026, 11, 23), contracts) is True  # Monday (holiday shift)
    assert is_fno_expiry_day("NIFTY", date(2026, 11, 19), contracts) is False  # a Thursday
    assert is_fno_expiry_day("BANKNIFTY", date(2026, 11, 17), contracts) is None  # no contracts: unknown


def test_unavailable_contract_master_gives_unknown_expiry(monkeypatch):
    from data import instrument_master
    from agents.india_specific_agent import IndiaSpecificAgent

    def unavailable(*_a, **_k):
        raise instrument_master.InstrumentMasterUnavailableError("stale")

    monkeypatch.setattr(instrument_master, "find_nfo_options", unavailable)
    assert IndiaSpecificAgent.__new__(IndiaSpecificAgent)._is_expiry_day() is None


def test_geopolitical_fallback_reads_nested_macro_data():
    from agents.geopolitical_nlp_agent import GeopoliticalNLPAgent

    agent = GeopoliticalNLPAgent.__new__(GeopoliticalNLPAgent)
    spike = {"macro_signals": {"overall": {}, "data": {
        "crude_oil": {"price": 90, "change_pct": 6.0}, "usdinr": {"rate": 90.0},
    }}}
    result = agent._rule_based_analysis(spike)
    assert [r["category"] for r in result["identified_risks"]] == ["crude_oil_price_increase"]
    assert result["identified_risks"][0]["severity"] == "HIGH"
    assert agent._rule_based_analysis({"macro_signals": {}})["identified_risks"] == []
    flat = {"macro_signals": {"crude_oil": {"change_pct": 9.0}}}  # the old, wrong shape reads nothing
    assert agent._rule_based_analysis(flat)["identified_risks"] == []


def test_delta_hedger_logs_the_delta_key(monkeypatch):
    from agents import graph

    class Hedger:
        def calculate_portfolio_greeks(self, *_a, **_k):
            return {"delta": 1.25}

    import risk.ai_options_delta_hedger as module

    monkeypatch.setattr(module, "delta_hedger", Hedger(), raising=False)
    source = open(graph.__file__).read()
    assert "portfolio_greeks.get('delta', 0)" in source and "portfolio_delta" not in source


def test_unknown_calendar_year_blocks_entries_but_not_exits(monkeypatch):
    from data.nse_calendar import calendar_year_known, is_nse_holiday
    from risk.daily_portfolio_risk import RiskHistoryUnavailable, _session_dates

    assert calendar_year_known(2026) and not calendar_year_known(2031)
    assert is_nse_holiday("2026-01-26")
    with pytest.raises(RiskHistoryUnavailable, match="2031"):
        _session_dates("2031-06-03T10:00:00+05:30")

    from risk.risk_engine import RiskEngine

    class FixedNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2031, 6, 3, 11, 0, tzinfo=tz)

    import risk.risk_engine as engine_module

    monkeypatch.setattr(engine_module, "datetime", FixedNow)
    allowed, reason = RiskEngine()._check_nse_holiday({})
    assert not allowed and "2031" in reason

    # Exits never consult the calendar: the protective exit path has no calendar import.
    import risk.stop_loss_manager as stop_module

    assert "nse_calendar" not in open(stop_module.__file__).read()


def test_override_holidays_merge_with_the_file(monkeypatch):
    from data.nse_calendar import is_nse_holiday

    monkeypatch.setattr(settings, "nse_holidays", ["2026-11-13"])
    assert is_nse_holiday("2026-11-13") and is_nse_holiday("2026-01-26")


# ── 1.6 partial fills / re-pricing ───────────────────────────────────────────

def test_partial_fill_model_is_off_by_default_and_floors_the_quantity():
    fill = order_manager.simulate_paper_fill
    assert fill("BUY", "LIMIT", 100, 100, 0, quantity=10, available_quantity=4) == ("FILLED", 100.0, 10)
    assert fill("BUY", "LIMIT", 100, 100, 0, quantity=10, available_quantity=4, allow_partial=True) == (
        "PARTIALLY_FILLED", 100.0, 4,
    )
    assert fill("BUY", "LIMIT", 100, 100, 0, quantity=10, available_quantity=0, allow_partial=True)[0] == "NO_DEPTH"
    assert fill("BUY", "LIMIT", 100, 100, 0, quantity=10, available_quantity=50, allow_partial=True)[2] == 10
    assert fill("BUY", "LIMIT", 100, 100, 0, quantity=10, available_quantity=None, allow_partial=True)[2] == 10


def test_partial_entry_creates_and_protects_only_the_filled_quantity(database, monkeypatch):
    monkeypatch.setattr(settings, "paper_partial_fill_enabled", True)
    _reference(monkeypatch, top={"BUY": 3, "SELL": 3})
    result = _buy(quantity=10)
    assert result["quantity"] == 3 and result["requested_quantity"] == 10 and result["partial"] is True
    with db.get_db() as session:
        trade = session.query(Trade).one()
        assert trade.quantity == 3
        assert session.query(PaperFillAudit).one().quantity == 3
        position = summarize_positions(replay_orders([trade]))["TEST"]
    assert position["quantity"] == 3 and position["stop_loss"] == 95


def test_exits_are_never_partially_filled(database, monkeypatch):
    monkeypatch.setattr(settings, "paper_partial_fill_enabled", True)
    _reference(monkeypatch, top=None)
    assert _buy(quantity=5) is not None
    _reference(monkeypatch, top={"BUY": 1, "SELL": 1})
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    result = order_manager.place_order(
        "TEST", "token", "SELL", 5, price=100, product_type="INTRADAY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    assert result["quantity"] == 5


def test_protective_exit_reprices_at_most_three_steps_and_alerts_once(monkeypatch, tmp_path):
    from broker.portfolio_tracker import PortfolioTracker
    from risk.dead_mans_switch import DeadMansSwitch
    from risk.stop_loss_manager import StopLossManager

    store = ControlStateStore(tmp_path / "exit.sqlite3")
    scheduler = ms.MarketScheduler()
    monkeypatch.setattr(ms, "stop_loss_manager", StopLossManager(store))
    monkeypatch.setattr(ms, "portfolio_tracker", PortfolioTracker(store))
    monkeypatch.setattr(ms, "dead_mans_switch", DeadMansSwitch(store=store))
    collars = []
    monkeypatch.setattr(
        scheduler, "_place_exit_order",
        lambda _s, _a, position: collars.append(position["_collar_multiple"]) or None,
    )
    alerts = []
    monkeypatch.setattr(scheduler, "_fire_alert", lambda text, **kw: alerts.append(kw["dedup_key"]))
    position = {"quantity": 5, "avg_price": 100.0, "stop_loss": 95.0, "holding_period": "INTRADAY",
                "entry_trade_ids": [1]}
    for _ in range(5):
        assert scheduler._submit_protective_exit("TEST", "SELL", position) is None
    assert collars == [1, 2, 3, 3, 3]
    assert max(collars) * settings.protective_exit_collar_pct <= 0.009 + 1e-9
    assert set(alerts) == {"exit_unfilled:TEST"}
    assert ms.stop_loss_manager.pending_exits() == {}  # no duplicate or stuck exit intent


# ── 1.7 order rate limiter / cancel entries ───────────────────────────────────

def test_rate_limiter_refuses_entries_over_the_window_but_never_drops_exits():
    from operations.order_rate_limiter import OrderRateLimiter

    clock = {"now": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    limiter = OrderRateLimiter(per_second=5, per_minute=100, clock=lambda: clock["now"], sleep=sleep)
    assert all(limiter.acquire(is_exit=False) for _ in range(5))
    assert limiter.acquire(is_exit=False) is False  # sixth entry in one second
    assert limiter.acquire(is_exit=True) is True  # the exit waits, then goes ahead
    assert sum(sleeps) <= 1.0 + 1e-9
    clock["now"] += 2.0
    assert limiter.acquire(is_exit=False) is True


def test_rate_limiter_per_minute_window():
    from operations.order_rate_limiter import OrderRateLimiter

    clock = {"now": 0.0}
    limiter = OrderRateLimiter(per_second=5, per_minute=10, clock=lambda: clock["now"], sleep=lambda s: None)
    granted = 0
    for _ in range(12):
        clock["now"] += 0.5
        granted += limiter.acquire(is_exit=False)
    assert granted == 10
    clock["now"] += 61
    assert limiter.acquire(is_exit=False) is True


def test_halt_cancels_resting_entries_but_not_exits(monkeypatch):
    cancelled = []
    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(order_manager, "get_open_orders", lambda: [
        {"orderid": "E1", "tradingsymbol": "NEW", "transactiontype": "BUY", "orderstatus": "open"},
        {"orderid": "X1", "tradingsymbol": "HELD", "transactiontype": "SELL", "orderstatus": "open"},
        {"orderid": "D1", "tradingsymbol": "NEW", "transactiontype": "BUY", "orderstatus": "complete"},
    ])
    monkeypatch.setattr(order_manager, "cancel_order", lambda oid, *_a: cancelled.append(oid) or True)
    result = order_manager.cancel_pending_entries({"HELD": {"quantity": 5}})
    assert result == ["E1"] and cancelled == ["E1"]


def test_loss_halt_latch_triggers_entry_cancellation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(order_manager, "cancel_pending_entries", lambda positions: calls.append(positions) or [])
    tracker = _tracker(ControlStateStore(tmp_path / "c.sqlite3"))
    tracker._daily_pnl = -300.0
    assert tracker.has_hit_daily_loss_limit()
    assert tracker.has_hit_daily_loss_limit()
    assert len(calls) == 1  # only when the latch first engages


# ── 6.2 outbox ───────────────────────────────────────────────────────────────

class _Bot:
    def __init__(self, fail=0):
        self.fail, self.sent = fail, []

    async def send_message(self, **kwargs):
        if self.fail:
            self.fail -= 1
            from telegram.error import NetworkError

            raise NetworkError("down")
        self.sent.append(kwargs["text"])


@pytest.fixture
def notifier(database, monkeypatch):
    from notifications.telegram_notifier import TelegramNotifier

    monkeypatch.setattr(settings, "telegram_bot_token", "x")
    monkeypatch.setattr(settings, "telegram_chat_id", "1")
    instance = TelegramNotifier()
    instance._bot = _Bot()
    return instance


def test_alert_is_stored_before_delivery_and_retried_after_an_outage(notifier):
    notifier._bot = _Bot(fail=1)
    assert asyncio.run(notifier.send_message("halt", dedup_key="k", severity="critical")) is False
    with db.get_db() as session:
        row = session.query(AlertOutbox).one()
        assert row.delivered_at is None and row.attempts == 1 and row.last_error == "NetworkError"
        row.last_attempt_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
    from notifications.outbox import undelivered_critical

    assert len(undelivered_critical()) == 1
    assert asyncio.run(notifier.retry_outbox()) == 1
    assert notifier._bot.sent == ["halt"]
    assert undelivered_critical() == []


def test_dedup_suppresses_repeats_for_30_minutes_and_pairs_recovery(notifier):
    assert asyncio.run(notifier.send_message("down", dedup_key="ws", severity="critical")) is True
    assert asyncio.run(notifier.send_message("down again", dedup_key="ws", severity="critical")) is False
    assert asyncio.run(notifier.send_message("up", dedup_key="ws:recovered")) is True
    assert asyncio.run(notifier.send_message("down once more", dedup_key="ws", severity="critical")) is True
    assert notifier._bot.sent == ["down", "up", "down once more"]


def test_retry_gives_up_after_ten_attempts_and_backs_off(notifier):
    from notifications import outbox

    notifier._bot = _Bot(fail=99)
    asyncio.run(notifier.send_message("never", severity="critical"))
    assert outbox.backoff_seconds(1) == 60 and outbox.backoff_seconds(3) == 240
    assert outbox.backoff_seconds(20) == outbox.BACKOFF_CAP_SECONDS
    assert outbox.due_for_retry() == []  # still inside the backoff window
    with db.get_db() as session:
        row = session.query(AlertOutbox).one()
        row.attempts = outbox.MAX_ATTEMPTS
        row.last_attempt_at = None
    assert outbox.due_for_retry() == []


# ── 1.8 capability inventory ──────────────────────────────────────────────────

def test_every_graph_node_is_in_the_capability_inventory():
    from agents.graph import build_trading_graph
    from operations.capabilities import CAPABILITIES, capability_names

    nodes = set(build_trading_graph().get_graph().nodes) - {"__start__", "__end__"}
    assert nodes <= capability_names(), nodes - capability_names()
    assert all(item["status"] in {"active", "advisory", "experimental", "disabled"} for item in CAPABILITIES)
    by_name = {item["name"]: item for item in CAPABILITIES}
    assert by_name["hmm_regime"]["status"] == "advisory"
    assert by_name["marl_optimizer"]["status"] == "experimental"
    assert by_name["kafka_events"]["status"] == "experimental"
    assert by_name["signal_gate"]["status"] == "disabled"
    assert by_name["strategy_selector"]["status"] == "disabled"


# ── settings bounds and preflight ─────────────────────────────────────────────

@pytest.mark.parametrize("field,bad", [
    ("scan_entry_deadline_seconds", 60), ("scan_entry_deadline_seconds", 3600),
    ("llm_daily_budget_inr", 0), ("llm_daily_budget_inr", 6000),
    ("order_rate_per_second", 0), ("order_rate_per_minute", 500),
    ("protective_exit_max_reprice_steps", 4), ("protective_exit_max_reprice_steps", 0),
])
def test_new_settings_reject_out_of_range_values(field, bad):
    from pydantic import ValidationError

    from config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: bad})


def test_nse_holidays_env_accepts_csv_and_rejects_bad_dates(monkeypatch):
    from pydantic import ValidationError

    from config.settings import Settings

    assert Settings(_env_file=None, nse_holidays="2026-11-13, 2026-11-14").nse_holidays == [
        "2026-11-13", "2026-11-14",
    ]
    with pytest.raises(ValueError):
        Settings(_env_file=None, nse_holidays="13/11/2026")


def test_preflight_flags_a_missing_calendar_year(monkeypatch):
    from operations import preflight
    import data.nse_calendar as calendar

    monkeypatch.setattr(calendar, "calendar_year_known", lambda year: False)
    result = preflight.run_preflight()
    assert any("No NSE holiday calendar" in error for error in result.errors)
