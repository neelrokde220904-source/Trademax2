"""Restart safety and local paper-operation contracts, without external calls."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from dashboard.backend import app as backend
from operations.control_state import ControlStateStore
from risk.dead_mans_switch import DeadMansSwitch
from scheduler import market_scheduler as scheduler_module
from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state
from tests.test_llm_runtime import llm_database


def test_missing_state_is_paused_without_writes(tmp_path):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    assert store.paused()
    assert not store.path.exists()
    assert scheduler_module.MarketScheduler(store=store).is_paused


def test_pause_and_halt_survive_new_instances(tmp_path):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    scheduler = scheduler_module.MarketScheduler(store=store)
    switch = DeadMansSwitch(store=store)
    store.set_paused(False)
    scheduler.pause_trading()
    switch.record_failure("langgraph", "one")
    switch.record_failure("langgraph", "two")
    switch.record_failure("langgraph", "three")
    recovered = DeadMansSwitch(store=ControlStateStore(store.path))
    assert scheduler_module.MarketScheduler(store=store).is_paused
    assert recovered.is_halted
    assert recovered.get_status()["failure_counts"]["langgraph"] == 3
    recovered.resume_trading("reviewed while entries paused")
    assert not DeadMansSwitch(store=store).is_halted
    assert store.paused()


@pytest.mark.parametrize("payload", ['{"paused":"false"}', '[]', 'not json'])
def test_corrupt_state_fails_closed(tmp_path, payload):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(True)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE controls SET payload=? WHERE name='entries'", (payload,))
    with pytest.raises(ValueError):
        scheduler_module.MarketScheduler(store=store)


def test_failed_persistence_never_clears_halt(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    switch = DeadMansSwitch(store=store)
    switch.force_halt("maintenance")
    def fail(*_args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(store, "write", fail)
    with pytest.raises(OSError):
        switch.resume_trading("reviewed")
    assert switch.is_halted


@pytest.fixture
def operator_client(llm_database, tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    switch = DeadMansSwitch(store=store)
    scheduler = scheduler_module.MarketScheduler(store=store)
    class Notifier:
        async def send_message(self, _message):
            return True
    switch._notifier = Notifier()
    monkeypatch.setattr(backend, "dead_mans_switch", switch)
    monkeypatch.setattr(scheduler_module, "dead_mans_switch", switch)
    monkeypatch.setattr(scheduler_module, "market_scheduler", scheduler)
    monkeypatch.setattr(settings, "paper_trading", True)
    with TestClient(backend.app, base_url="http://127.0.0.1",
                    client=("127.0.0.1", 50000)) as client:
        yield client, scheduler, switch


HEADERS = {"Origin": "http://127.0.0.1:8000", "X-Dashboard-Control": "1"}


def test_dashboard_feed_connect_prioritizes_held_positions(operator_client, monkeypatch):
    from broker import angel_client as angel_module
    from broker.websocket_feed import ws_feed
    from config import instruments, universe

    client, scheduler, _ = operator_client
    broker = SimpleNamespace(has_credentials=True, ensure_session=lambda: None)
    monkeypatch.setattr(angel_module, "angel_client", broker)
    monkeypatch.setattr(scheduler_module, "angel_client", broker)
    monkeypatch.setattr(scheduler_module.portfolio_tracker, "_positions", {
        "HELD": {"quantity": 1}, "NO_TOKEN": {"quantity": 1},
    })
    monkeypatch.setattr(universe, "get_active_watchlist", lambda: [f"W{i}" for i in range(60)])
    monkeypatch.setattr(
        instruments, "get_token", lambda symbol: None if symbol == "NO_TOKEN" else f"t-{symbol}",
    )
    subscriptions, alerts = [], []
    monkeypatch.setattr(ws_feed, "connect", lambda: None)
    monkeypatch.setattr(ws_feed, "subscribe", lambda _exchange, tokens: subscriptions.append(tokens))
    monkeypatch.setattr(ws_feed, "get_status", lambda: {"connected": True})

    async def notify(reason):
        alerts.append(reason)

    monkeypatch.setattr(scheduler_module.telegram_notifier, "notify_error", notify)
    response = client.post("/api/operations/feed-connect", json={}, headers=HEADERS)
    assert response.status_code == 200
    assert subscriptions[-1][0] == "t-HELD"
    assert len(subscriptions[-1]) == 50
    assert scheduler.unsubscribed_positions == ["NO_TOKEN"]
    assert len(alerts) == 1 and "NO_TOKEN" in alerts[0]

    def unavailable():
        raise universe.UniverseUnavailableError("membership unavailable")

    monkeypatch.setattr(universe, "get_active_watchlist", unavailable)
    assert client.post("/api/operations/feed-connect", json={}, headers=HEADERS).status_code == 200
    assert subscriptions[-1] == ["t-HELD"]


@pytest.mark.parametrize("missing_tokens,status", [(True, 409), (False, 502)])
def test_dashboard_feed_connection_failures_remain_explicit(
    operator_client, monkeypatch, missing_tokens, status,
):
    from broker import angel_client as angel_module
    from broker.websocket_feed import ws_feed
    from config import instruments, universe

    client, _, _ = operator_client
    monkeypatch.setattr(angel_module, "angel_client", SimpleNamespace(has_credentials=True))
    monkeypatch.setattr(scheduler_module.portfolio_tracker, "_positions", {})
    monkeypatch.setattr(universe, "get_active_watchlist", lambda: ["TEST"])
    monkeypatch.setattr(instruments, "get_token", lambda _symbol: None if missing_tokens else "token")

    def failed_connect():
        if missing_tokens:
            pytest.fail("Missing tokens must block the connection attempt")
        raise RuntimeError("feed unavailable")

    monkeypatch.setattr(ws_feed, "connect", failed_connect)
    response = client.post("/api/operations/feed-connect", json={}, headers=HEADERS)
    assert response.status_code == status
    assert "detail" in response.json()


def test_control_recovery_requires_review_and_stays_paused(operator_client):
    client, scheduler, switch = operator_client
    switch.force_halt("maintenance")
    assert client.post("/api/control", json={"action": "reset_halt"}, headers=HEADERS).status_code == 409
    assert client.post("/api/control", json={"action": "resume", "reviewed": True}, headers=HEADERS).status_code == 409
    assert client.post("/api/control", json={"action": "reset_halt", "reviewed": True}, headers=HEADERS).status_code == 200
    assert scheduler.is_paused
    assert not switch.is_halted
    assert client.post("/api/control", json={"action": "resume", "reviewed": True}, headers=HEADERS).status_code == 200
    assert not scheduler.is_paused
    assert client.post("/api/control", json={"action": "halt"}, headers=HEADERS).status_code == 200
    assert scheduler.is_paused and switch.is_halted


def test_drawdown_latch_reset_requires_review_and_keeps_entries_paused(operator_client, monkeypatch):
    client, scheduler, _ = operator_client
    scheduler.pause_trading()
    reset_calls = []
    monkeypatch.setattr(
        backend.portfolio_tracker, "has_hit_portfolio_drawdown_limit", lambda: True,
    )
    monkeypatch.setattr(
        backend.portfolio_tracker, "reset_portfolio_drawdown_halt",
        lambda: reset_calls.append("reset"),
    )

    denied = client.post("/api/control", json={"action": "reset_drawdown"}, headers=HEADERS)
    assert denied.status_code == 409
    assert not reset_calls

    response = client.post(
        "/api/control", json={"action": "reset_drawdown", "reviewed": True}, headers=HEADERS,
    )
    assert response.status_code == 200
    assert reset_calls == ["reset"]
    assert scheduler.is_paused


def test_control_rejects_untrusted_and_live_recovery(operator_client, monkeypatch):
    client, scheduler, switch = operator_client
    assert client.post("/api/control", json={"action": "resume", "reviewed": True}).status_code == 403
    assert client.post("/api/control", json={"action": "live"}, headers=HEADERS).status_code == 422
    monkeypatch.setattr(settings, "paper_trading", False)
    assert client.post("/api/control", json={"action": "resume", "reviewed": True}, headers=HEADERS).status_code == 409
    assert scheduler.is_paused


def test_operations_do_not_expose_credentials(operator_client):
    client, _, _ = operator_client
    response = client.get("/api/operations")
    assert response.status_code == 200
    assert response.json()["feed"]["tls_verified"]
    assert response.json()["portfolio_risk"]["max_daily_entries_per_symbol"] == settings.max_daily_entries_per_symbol
    assert response.json()["portfolio_risk"]["symbol_reentry_cooldown_minutes"] == settings.symbol_reentry_cooldown_minutes
    protection = response.json()["protection"]
    assert {"tick_triggers_active", "last_check", "unsubscribed_positions", "positions"} <= protection.keys()
    assert any("live entries are blocked" in item.lower() for item in response.json()["limitations"])
    assert "telegram_chat_id" not in response.text
    assert "telegram_bot_token" not in response.text
    assert "test-economy" not in response.text


def test_pnl_endpoints_subtract_estimated_paper_order_costs(operator_client):
    from datetime import datetime

    from backtester.costs import calculate_indian_transaction_cost
    from database.db import get_db
    from database.models import DailyPnL, Trade

    client, _, _ = operator_client
    day = datetime.now().strftime("%Y-%m-%d")
    with get_db() as db:
        db.add_all([
            Trade(
                timestamp=datetime.now().replace(hour=10, minute=0),
                symbol="TEST", action="BUY", quantity=1, price=100,
                holding_period="INTRADAY", status="EXECUTED", is_paper=True,
            ),
            Trade(
                timestamp=datetime.now().replace(hour=11, minute=0),
                symbol="TEST", action="SELL", quantity=1, price=110,
                holding_period="INTRADAY", status="EXECUTED", is_paper=True,
            ),
            DailyPnL(
                date=day, portfolio_value=25_010, realised_pnl=10,
                unrealised_pnl=0, trades_count=1, win_count=1, loss_count=0,
            ),
        ])

    costs = (
        calculate_indian_transaction_cost(100, 1, is_buy=True, is_intraday=True)
        + calculate_indian_transaction_cost(110, 1, is_buy=False, is_intraday=True)
    )
    pnl = client.get("/api/pnl/daily?days=1").json()
    assert pnl[0]["gross_pnl"] == 10
    assert pnl[0]["transaction_costs"] == pytest.approx(round(costs, 2))
    assert pnl[0]["pnl"] == pytest.approx(10 - costs)
    assert pnl[0]["portfolio_value"] == pytest.approx(
        round(settings.initial_capital + 10 - costs, 2),
    )
    curve = client.get("/api/equity-curve").json()
    assert curve[-1]["value"] == pytest.approx(round(settings.initial_capital + 10 - costs, 2))


def test_paused_scan_never_calls_graph(operator_client, monkeypatch):
    _, scheduler, _ = operator_client
    async def forbidden(*_args, **_kwargs):
        pytest.fail("A paused scan must not call the graph")
    monkeypatch.setattr(scheduler_module, "run_trading_pipeline", forbidden)
    asyncio.run(scheduler._run_scan())
    assert scheduler.last_scan is None


def test_telegram_bot_recipient_is_rejected(monkeypatch):
    from notifications.telegram_notifier import TelegramNotifier
    monkeypatch.setattr(settings, "telegram_bot_token", "test-token")
    monkeypatch.setattr(settings, "telegram_chat_id", "123")
    notifier = TelegramNotifier()
    class Bot:
        async def get_me(self):
            return SimpleNamespace(id=123)
        async def send_message(self, **_kwargs):
            pytest.fail("Must not send to the bot itself")
    notifier._bot = Bot()
    assert not asyncio.run(notifier.test_delivery())
    assert notifier.get_status()["delivery_status"] == "recipient_is_bot"


def test_telegram_test_sends_actual_message(monkeypatch):
    from notifications.telegram_notifier import TelegramNotifier
    monkeypatch.setattr(settings, "telegram_bot_token", "test-token")
    monkeypatch.setattr(settings, "telegram_chat_id", "456")
    notifier = TelegramNotifier()
    messages = []
    class Bot:
        async def get_me(self):
            return SimpleNamespace(id=123)
        async def get_chat(self, _chat_id):
            return SimpleNamespace(type="private")
        async def send_message(self, **kwargs):
            messages.append(kwargs)
    notifier._bot = Bot()
    assert asyncio.run(notifier.test_delivery())
    assert len(messages) == 1
    assert notifier.get_status()["delivery_status"] == "delivered"


def test_inflight_entry_is_blocked_but_exits_are_not(tmp_path, monkeypatch):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    stub_persisted_review_verification(monkeypatch)
    from agents import graph
    from operations import control_state as controls
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Paused entry submitted"))
    evidence = sentiment_state(["TEST"])
    result = graph._order_router_node({
        "execution_enabled": True, "current_positions": {},
        **evidence,
        "watchlist": ["TEST"], "screener_results": {"qualified": ["TEST"], "rejected": []},
        "debate_results": {"TEST": reviewed_debate(evidence, "TEST", "BUY")},
        "technical_signals": evidence["technical_signals"],
        "trading_decisions": [reviewed_decision(evidence, {"symbol": "TEST", "action": "BUY", "quantity": 1,
                               "price": 100, "holding_period": "INTRADAY",
                               })],
    })
    assert not result["executed_orders"]
    assert "operational safety controls" in result["errors"][0]


def test_rejected_broker_authentication_never_looks_ready(monkeypatch):
    from broker.angel_client import AngelClient
    client = AngelClient()
    monkeypatch.setattr(client, "authenticate", lambda: False)
    with pytest.raises(RuntimeError, match="authentication was rejected"):
        client.ensure_session()
    assert not client.is_authenticated


def test_session_refresh_can_reenter_authentication_without_deadlock(monkeypatch):
    from broker.angel_client import AngelClient
    client = AngelClient()
    def authenticate():
        with client._lock:
            return True
    monkeypatch.setattr(client, "authenticate", authenticate)
    assert client.refresh_session()


def test_equity_watchlist_resolves_only_explicit_eq_series(monkeypatch):
    from config import instruments
    eq = {"token": "1", "symbol": "TEST-EQ", "exchange": "NSE"}
    other = {"token": "2", "symbol": "OTHER-BE", "exchange": "NSE"}
    monkeypatch.setattr(instruments, "_CACHE", {"NSE:TEST-EQ": eq, "NSE:OTHER-BE": other})
    assert instruments.get_token("TEST") == "1"
    assert instruments.get_token("TEST-EQ") == "1"
    assert instruments.get_instrument("TEST") == eq
    assert instruments.get_token("OTHER") is None
    assert instruments.get_token("TEST", "NFO") is None


def test_supervised_analysis_disables_execution_and_preserves_halt(operator_client, monkeypatch):
    _, scheduler, switch = operator_client
    switch.force_halt("maintenance")
    seen = []
    async def pipeline(watchlist, *, execute_orders, on_execution_complete):
        seen.append((watchlist, execute_orders))
        return {"executed_orders": [], "errors": []}
    monkeypatch.setattr(scheduler_module, "run_trading_pipeline", pipeline)
    asyncio.run(scheduler._run_scan(analysis_only=True, watchlist=["TEST"]))
    assert seen == [(["TEST"], False)]
    assert switch.is_halted and scheduler.is_paused
    assert scheduler.last_scan["analysis_only"]
    assert scheduler.last_scan["status"] == "completed"
    assert not scheduler.scan_in_progress


def test_supervised_analysis_rejects_unpaused_or_live(operator_client, monkeypatch):
    client, scheduler, _ = operator_client
    scheduler._trading_paused = False
    assert client.post("/api/operations/analysis", json={}, headers=HEADERS).status_code == 409
    scheduler._trading_paused = True
    monkeypatch.setattr(settings, "paper_trading", False)
    assert client.post("/api/operations/analysis", json={}, headers=HEADERS).status_code == 409


def test_regime_classification_is_local_read_only_and_fails_closed(operator_client, monkeypatch):
    from risk import hmm_regime_classifier as hmm

    client, scheduler, _ = operator_client
    unfitted = hmm.HMMRegimeClassifier(model_path="/nonexistent/model.json")
    monkeypatch.setattr(hmm, "hmm_classifier", unfitted)
    assert client.post("/api/operations/regime", json={}).status_code == 403
    assert client.post("/api/operations/regime", json={}, headers=HEADERS).status_code == 409

    class Fitted:
        is_fitted = True
        load_error = None
        result = {
            "regime": "LOW_VOL_TREND", "confidence": 0.9, "panic_risk": 0.01, "as_of": "2026-11-10",
            "degraded": False, "model_stale": False,
            "action_directive": {"allow_new_positions": True, "description": "Trending market"},
        }

        def classify_latest(self):
            return dict(self.result)

    fitted = Fitted()
    monkeypatch.setattr(hmm, "hmm_classifier", fitted)
    response = client.post("/api/operations/regime", json={}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["regime"] == "LOW_VOL_TREND" and response.json()["entry_block_reason"] is None
    fitted.result = {**fitted.result, "degraded": True, "reason": "features are 9 days old"}
    response = client.post("/api/operations/regime", json={}, headers=HEADERS)
    assert response.status_code == 409 and "9 days old" in response.json()["detail"]
    assert scheduler.is_paused


def test_scan_decisions_are_persisted_and_explained(operator_client, monkeypatch):
    client, scheduler, _ = operator_client
    scheduler._trading_paused = False
    monkeypatch.setattr(scheduler, "reconcile_paper", lambda: {"status": "pass", "issues": []})

    async def pipeline(_watchlist, *, execute_orders, on_execution_complete):
        assert execute_orders
        return {
            "trading_decisions": [
                {"symbol": "TCS", "action": "SELL", "quantity": 1, "price": 3000, "confidence": 61},
                {"symbol": "INFY", "action": "BUY", "quantity": 2, "price": 1500, "confidence": 72},
            ],
            "executed_orders": [],
            "errors": [
                "[OrderRouter] Naked short sale blocked for TCS",
                "[OrderRouter] Order not filled for INFY (see paper fill audit / logs)",
            ],
        }
    monkeypatch.setattr(scheduler_module, "run_trading_pipeline", pipeline)
    asyncio.run(scheduler._run_scan(watchlist=["TCS", "INFY"]))
    assert scheduler.last_scan["decisions"] == 2
    assert scheduler.last_scan["blocked"] == 1 and scheduler.last_scan["failed"] == 1
    assert len(scheduler.last_scan["block_reasons"]) == 2

    signals = {row["symbol"]: row for row in client.get("/api/signals").json()}
    assert signals["TCS"]["outcome"] == "blocked" and "short selling" in signals["TCS"]["block_reason"]
    assert signals["INFY"]["outcome"] == "failed" and signals["INFY"]["quantity"] == 2
    assert signals["INFY"]["strategy"] == "portfolio_manager"


def test_scheduled_stops_are_registered_before_research_capture(operator_client, monkeypatch, tmp_path):
    from agents import graph as trading_graph
    from langgraph.graph import END, StateGraph
    from research import tracking

    _, scheduler, _ = operator_client
    scheduler._trading_paused = False
    monkeypatch.setattr(scheduler, "reconcile_paper", lambda: {"status": "pass", "issues": []})
    events = []
    trade = {"symbol": "TEST", "action": "BUY", "price": 100, "stop_loss": 95, "target": 110}
    workflow = StateGraph(dict)
    workflow.add_node("execute", lambda state: {**state, "executed_orders": [trade]})
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", END)
    monkeypatch.setattr(trading_graph, "build_trading_graph", workflow.compile)
    monkeypatch.setattr(trading_graph, "__file__", str(tmp_path / "agents" / "graph.py"))
    monkeypatch.setattr(scheduler_module.stop_loss_manager, "register_trade", lambda _trade: events.append("stop"))
    monkeypatch.setattr(scheduler_module.portfolio_tracker, "sync_from_broker", lambda: events.append("sync"))
    monkeypatch.setattr(scheduler_module, "record_scan_decisions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking, "record_scan", lambda *_args: (
        events.append("capture") or {"snapshot_status": "complete"}
    ))

    async def notify(_trade):
        events.append("notification")

    monkeypatch.setattr(scheduler_module.telegram_notifier, "notify_trade_executed", notify)
    asyncio.run(scheduler._run_scan(watchlist=["TEST"]))
    assert scheduler.last_scan["status"] == "completed"
    assert events == ["stop", "sync", "capture", "notification"]


def test_market_watchlist_is_read_only_feed_data_with_sparklines(operator_client, monkeypatch):
    from datetime import datetime, timedelta

    from broker import websocket_feed
    from config import instruments

    client, _, _ = operator_client
    now = datetime.now(backend.IST)
    ticks = {"1": {"ltp": 102.0, "close": 100.0, "open": 99.0, "high": 103.0, "low": 98.0,
                   "volume": 10, "timestamp": (now - timedelta(seconds=90)).isoformat(),
                   "exchange_timestamp": (now - timedelta(seconds=90)).isoformat()}}

    class Feed:
        def get_all_ticks(self):
            return {token: dict(tick) for token, tick in ticks.items()}

        def get_status(self):
            return {"connected": True, "fresh": True}

    monkeypatch.setattr(websocket_feed, "ws_feed", Feed())
    monkeypatch.setattr(instruments, "get_token", {"AAA": "1"}.get)
    monkeypatch.setattr(settings, "default_watchlist", ["AAA", "BBB"])
    monkeypatch.setattr(settings, "trading_universe_mode", "custom")
    monkeypatch.setattr(backend, "_price_history", {})
    backend._sample_watchlist_prices(now)
    ticks["1"] = {**ticks["1"], "ltp": 104.0, "timestamp": (now - timedelta(seconds=5)).isoformat(),
                  "exchange_timestamp": (now - timedelta(seconds=5)).isoformat()}
    backend._sample_watchlist_prices(now)
    ticks["2"] = {"ltp": 50.0, "timestamp": (now - timedelta(minutes=10)).isoformat()}
    backend._sample_watchlist_prices(now)

    payload = client.get("/api/market/watchlist").json()
    rows = {row["symbol"]: row for row in payload["symbols"]}
    assert rows["AAA"]["change_pct"] == 4.0 and rows["AAA"]["fresh"]
    assert [point["p"] for point in rows["AAA"]["sparkline"]] == [102.0, 104.0]
    assert not rows["BBB"]["resolved"] and rows["BBB"]["sparkline"] == []
    assert "2" not in backend._price_history
    assert client.post("/api/market/watchlist", json={}).status_code == 405


def test_market_session_follows_ist_hours_and_holidays():
    from datetime import datetime

    def at(day, hour, minute):
        return backend._market_session(backend.IST.localize(datetime(2026, 11, day, hour, minute)))

    assert at(10, 10, 0)["state"] == "holiday"
    assert at(14, 10, 0)["state"] == "weekend"
    assert at(11, 9, 5)["state"] == "pre_open"
    assert at(11, 10, 0)["entry_window_open"]
    assert at(11, 14, 50)["is_open"] and not at(11, 14, 50)["entry_window_open"]
    assert at(11, 15, 45)["state"] == "closed"
