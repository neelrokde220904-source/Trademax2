"""T14: redaction, circuit-breaker wiring, scan status and dashboard payloads."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from config.settings import settings
from dashboard.backend import app as backend
from operations import redaction
from operations.control_state import ControlStateStore
from risk.dead_mans_switch import DeadMansSwitch
from scheduler import market_scheduler as ms

FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.c2lnbmF0dXJlMTIzNDU"


# ── redaction ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,secret", [
    (f"token is {FAKE_JWT} ok", FAKE_JWT),
    ("sent Authorization: Bearer abcdef1234567890XYZ end", "abcdef1234567890XYZ"),
    ("headers {'Authorization': 'Bearer abcdef1234567890XYZ'}", "abcdef1234567890XYZ"),
    ("headers {'X-PrivateKey': 'KEY-VALUE-123'}", "KEY-VALUE-123"),
    ("url?api_key=SUPERSECRETKEY&x=1", "SUPERSECRETKEY"),
    ("totp=123456 accepted", "123456"),
    ("mpin: 4321 wrong", "4321"),
    ("login ok for client A123456 today", "A123456"),
    ("Client ID: B987654.", "B987654"),
    ("password='hunter2xyz'", "hunter2xyz"),
])
def test_credentials_are_masked_and_the_value_never_survives(raw, secret):
    cleaned = redaction.redact(raw)
    assert secret not in cleaned and redaction.MASK in cleaned


def test_ordinary_text_is_left_alone():
    for text in ("Order submitted for TCS; confirmation of fills is still required.",
                 "client count 12 at 09:15", "Scan skipped: daily loss halt active",
                 "[OrderRouter] Entry blocked for X: scan exceeded entry deadline"):
        assert redaction.redact(text) == text


def test_configured_secrets_are_masked_wherever_they_appear(monkeypatch):
    monkeypatch.setattr(settings, "angel_api_key", "kJ9fPq2LzXw8")
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:ABCdefGhIJKlmn")
    text = "calling with kJ9fPq2LzXw8 and bot 123456:ABCdefGhIJKlmn done"
    cleaned = redaction.redact(text)
    assert "kJ9fPq2LzXw8" not in cleaned and "ABCdefGhIJKlmn" not in cleaned
    assert redaction.redact(text, include_configured=False) != cleaned


def test_the_loguru_patcher_masks_a_logged_line():
    lines = []
    handler = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        logger.patch(redaction.loguru_patcher).warning("auth failed for token {}", FAKE_JWT)
    finally:
        logger.remove(handler)
    assert lines and FAKE_JWT not in lines[0] and redaction.MASK in lines[0]


def test_setup_logging_installs_the_patcher_and_disables_variable_dumps(capsys):
    import inspect

    import main

    main.setup_logging(file_logging=False)
    logger.warning("auth failed for token {}", FAKE_JWT)
    err = capsys.readouterr().err
    assert FAKE_JWT not in err and "REDACTED" in err
    source = inspect.getsource(main.setup_logging)
    assert "diagnose=False" in source and "backtrace=False" in source and "loguru_patcher" in source


def test_the_sdk_file_logger_is_silenced(tmp_path):
    import logzero

    target = tmp_path / "sdk.log"
    logzero.logfile(str(target), loglevel=logging.ERROR)
    redaction.silence_sdk_logging()
    assert not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(target)
        for h in logzero.logger.handlers
    )
    assert logzero.logger.level == logging.CRITICAL
    assert logging.getLogger("SmartApi").level == logging.WARNING


def test_scrub_file_masks_in_place_and_reports_counts_only(tmp_path):
    path = tmp_path / "app.log"
    path.write_text(f"ok line\nheader X-PrivateKey: ABCDEFGH\nJWT {FAKE_JWT}\nfine\n")
    assert redaction.scrub_file(path) == 2
    cleaned = path.read_text()
    assert "ABCDEFGH" not in cleaned and FAKE_JWT not in cleaned and "ok line" in cleaned
    assert redaction.scrub_file(path) == 0


def test_authentication_does_not_log_the_client_id(monkeypatch):
    from broker import angel_client as module

    monkeypatch.setattr(settings, "angel_api_key", "api-key-value-1")
    monkeypatch.setattr(settings, "angel_client_id", "Z123456")
    monkeypatch.setattr(settings, "angel_mpin", "9999")
    monkeypatch.setattr(settings, "angel_totp_secret", "JBSWY3DPEHPK3PXP")

    class FakeApi:
        def __init__(self, api_key):
            pass

        def generateSession(self, *_a):
            return {"status": True, "data": {"jwtToken": FAKE_JWT, "refreshToken": "refresh-value-1"}}

        def getfeedToken(self):
            return "feed-token-value-1"

    monkeypatch.setattr(module, "SmartConnect", FakeApi)
    lines = []
    handler = logger.add(lambda m: lines.append(str(m)), format="{message}")
    try:
        assert module.AngelClient().authenticate() is True
    finally:
        logger.remove(handler)
    joined = "\n".join(lines)
    assert "Z123456" not in joined and FAKE_JWT not in joined and "Angel One authenticated" in joined


# ── circuit-breaker wiring ───────────────────────────────────────────────────

class FakeSwitch:
    def __init__(self):
        self.events = []

    def record_failure(self, component, error=""):
        self.events.append(("failure", component, error))
        return False

    def record_success(self, component):
        self.events.append(("success", component))


@pytest.fixture
def switch(monkeypatch):
    import risk.dead_mans_switch as module

    fake = FakeSwitch()
    monkeypatch.setattr(module, "dead_mans_switch", fake)
    return fake


def _feed(monkeypatch, market_open=True):
    from broker.websocket_feed import WebSocketFeed

    feed = WebSocketFeed()
    monkeypatch.setattr(WebSocketFeed, "_market_is_open", staticmethod(lambda: market_open))
    return feed


def test_feed_outages_count_once_a_minute_and_recover_on_a_fresh_tick(switch, monkeypatch):
    feed = _feed(monkeypatch)
    feed.report_failure("feed disconnected")
    feed.report_failure("feed disconnected")   # same outage, inside a minute
    assert [e for e in switch.events if e[0] == "failure"] == [("failure", "websocket", "feed disconnected")]
    feed._last_failure_report -= 61
    feed.report_failure("feed disconnected")
    assert len([e for e in switch.events if e[0] == "failure"]) == 2
    feed._report_recovery()
    feed._report_recovery()
    assert switch.events.count(("success", "websocket")) == 1


def test_feed_failures_outside_market_hours_or_during_shutdown_are_ignored(switch, monkeypatch):
    feed = _feed(monkeypatch, market_open=False)
    feed.report_failure("closed at night")
    feed = _feed(monkeypatch, market_open=True)
    feed._stop_event.set()
    feed.report_failure("operator disconnect")
    assert switch.events == []


def test_a_good_tick_reports_recovery(switch, monkeypatch):
    feed = _feed(monkeypatch)
    feed.report_failure("feed stale")
    now = datetime.now(ms.IST)
    feed._on_data(None, {
        "token": "1", "last_traded_price": 10000, "exchange_timestamp": int(now.timestamp() * 1000),
    })
    assert ("success", "websocket") in switch.events and feed._feed_failed is False


def test_stale_feed_is_reported_by_the_watchdog(switch, monkeypatch):
    from broker.websocket_feed import ws_feed

    reported = []
    monkeypatch.setattr(ws_feed, "get_status", lambda: {"tick_age_seconds": 300.0})
    monkeypatch.setattr(ws_feed, "report_failure", lambda reason: reported.append(reason))
    ms.MarketScheduler._check_feed_staleness()
    assert reported and "stale" in reported[0]
    reported.clear()
    monkeypatch.setattr(ws_feed, "get_status", lambda: {"tick_age_seconds": 5.0})
    ms.MarketScheduler._check_feed_staleness()
    monkeypatch.setattr(ws_feed, "get_status", lambda: {"tick_age_seconds": None})
    ms.MarketScheduler._check_feed_staleness()
    assert reported == []


def _hmm_state(monkeypatch, *, raises=False, degraded=False, load_error=None):
    from agents import graph
    from risk import hmm_regime_classifier as hmm

    class Classifier:
        def __init__(self):
            self.load_error = load_error

        def classify_latest(self):
            if raises:
                raise RuntimeError("model broke")
            return {"degraded": degraded, "regime": "LOW_VOL_TREND", "reason": "stale"}

        def unavailable(self, reason):
            return {"degraded": True, "reason": reason}

    monkeypatch.setattr(hmm, "hmm_classifier", Classifier())
    return graph._hmm_regime_node({})


def test_hmm_errors_reach_the_breaker_but_an_unfitted_model_does_not(switch, monkeypatch):
    _hmm_state(monkeypatch, raises=True)
    assert switch.events[-1][:2] == ("failure", "hmm_classifier")
    switch.events.clear()
    _hmm_state(monkeypatch, load_error="corrupt model file")
    assert switch.events[-1][:2] == ("failure", "hmm_classifier")
    switch.events.clear()
    _hmm_state(monkeypatch, degraded=True)  # stale/not fitted: already blocks entries, not a fault
    assert switch.events == []
    _hmm_state(monkeypatch)
    assert switch.events == [("success", "hmm_classifier")]


def test_llm_provider_health_reaches_the_breaker_and_the_old_name_is_an_alias(tmp_path, switch):
    from agents import llm_client

    llm_client._dms("failure", "TimeoutError")
    llm_client._dms("success")
    assert switch.events == [("failure", "llm_provider", "TimeoutError"), ("success", "llm_provider")]

    store = ControlStateStore(tmp_path / "dms.sqlite3")
    real = DeadMansSwitch(store=store)
    real.record_failure("anthropic_api", "old name")
    real.record_failure("llm_provider", "new name")
    assert real.get_status()["failure_counts"] == {"llm_provider": 2}
    assert real.get_status()["thresholds"]["llm_provider"] == 5
    assert "anthropic_api" not in real.get_status()["thresholds"]
    store.write("dead_man", {"halted": False, "halt_reason": "", "halt_time": None,
                             "failure_counts": {"anthropic_api": 3}})
    assert DeadMansSwitch(store=store).get_status()["failure_counts"] == {"llm_provider": 3}
    real.record_success("anthropic_api")
    assert real.get_status()["failure_counts"]["llm_provider"] == 0


def test_a_failed_llm_completion_is_reported(monkeypatch, switch):
    from agents import llm_client
    from config.llm_runtime import llm_run, profile_for

    monkeypatch.setattr(settings, "deepseek_api_key", "k-deepseek")
    monkeypatch.setattr(
        llm_client, "_deepseek_request", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("x")),
    )
    with llm_run("test", profile=profile_for("economy")):
        with pytest.raises(RuntimeError):
            llm_client.complete("fast", "hi")
    assert ("failure", "llm_provider", "TimeoutError") in switch.events


def test_exit_paths_never_consult_the_circuit_breaker():
    import inspect

    import broker.order_manager as orders
    import risk.stop_loss_manager as stops

    for function in (
        ms.MarketScheduler._check_stops, ms.MarketScheduler._intraday_squareoff,
        ms.MarketScheduler._execute_exit_signal, ms.MarketScheduler._submit_protective_exit,
        ms.MarketScheduler._run_tick_exit, ms.MarketScheduler._reconcile_pending_exits,
    ):
        source = inspect.getsource(function)
        assert "check_allowed" not in source and "is_halted" not in source, function.__name__
    assert "check_allowed" not in inspect.getsource(stops) and "is_halted" not in inspect.getsource(orders)


# ── scan status ──────────────────────────────────────────────────────────────

def test_eligibility_names_the_reason_separately_from_the_next_trigger(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "s.sqlite3")
    scheduler = ms.MarketScheduler(store=store)
    scheduler._trading_paused = True
    status = scheduler.scan_status()
    assert status["will_run"] is False and status["next_eligible_reason"] == "entries are paused"
    assert "next_trigger" in status  # reported independently of eligibility
    scheduler._trading_paused = False
    monkeypatch.setattr(ms, "dead_mans_switch", SimpleNamespace(check_allowed=lambda: (False, "websocket x5")))
    assert "circuit breaker" in scheduler.scan_status()["next_eligible_reason"]


def test_every_skip_reason_is_recorded_and_persisted(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "s.sqlite3")
    scheduler = ms.MarketScheduler(store=store)
    monkeypatch.setattr(ms, "dead_mans_switch", SimpleNamespace(check_allowed=lambda: (True, "")))
    scheduler._trading_paused = True
    asyncio.run(scheduler._run_scan())
    assert scheduler.last_skip["reason"] == "paused"
    scheduler._trading_paused = False
    scheduler.scan_in_progress = True
    asyncio.run(scheduler._run_scan(watchlist=["X"]))
    assert scheduler.last_skip["reason"] == "scan_in_progress"
    scheduler.scan_in_progress = False
    monkeypatch.setattr(ms, "dead_mans_switch", SimpleNamespace(check_allowed=lambda: (False, "halted")))
    asyncio.run(scheduler._run_scan(watchlist=["X"]))
    assert scheduler.last_skip["reason"] == "circuit_breaker"

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return ms.IST.localize(datetime(2026, 11, 13, 11, 30))

    monkeypatch.setattr(ms, "datetime", Fixed)
    scheduler.last_regime = "LOW_VOL_TREND"
    asyncio.run(scheduler._adaptive_scan())
    assert scheduler.last_skip["reason"] == "regime_slot"
    scheduler.last_regime = "SYSTEMIC_PANIC"
    asyncio.run(scheduler._adaptive_scan())
    assert scheduler.last_skip["reason"] == "regime_panic"
    assert ControlStateStore(store.path).read("scan_status")["last_skip"]["reason"] == "regime_panic"


# ── dashboard payloads ───────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    scheduler = ms.MarketScheduler(store=store)
    monkeypatch.setattr(backend, "dead_mans_switch", DeadMansSwitch(store=store))
    monkeypatch.setattr(ms, "market_scheduler", scheduler)
    with TestClient(backend.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as http:
        yield http, scheduler


def test_health_and_operations_expose_scan_status_and_budget(client):
    http, scheduler = client
    scheduler._record_skip("daily_halt", "daily loss halt")
    health = http.get("/api/health").json()
    assert health["scan_status"]["last_skip"]["reason"] == "daily_halt"
    assert {"will_run", "next_eligible_reason", "next_trigger"} <= set(health["scan_status"])
    operations = http.get("/api/operations").json()
    assert operations["scan"]["last_skip"]["reason"] == "daily_halt"
    assert operations["llm_budget"]["budget_inr"] == settings.llm_daily_budget_inr
    assert {"spend_inr", "exceeded"} <= set(operations["llm_budget"])
    assert {"us", "crypto"} <= set(operations["foreign_markets"])
    names = {item["name"] for item in operations["capabilities"]}
    assert {"hmm_regime", "signal_gate", "order_router"} <= names
    assert operations["undelivered_critical_alerts"] == []


def test_undelivered_critical_alerts_are_listed(client):
    from notifications import outbox

    outbox.enqueue("halt", parse_mode=None, dedup_key="k", severity="critical")
    outbox.enqueue("fyi", parse_mode=None, dedup_key="j", severity="info")
    http, _ = client
    alerts = http.get("/api/operations").json()["undelivered_critical_alerts"]
    assert [a["text"] for a in alerts] == ["halt"]


def test_signals_expose_the_review_contract_and_operations_list_the_sleeves(client):
    import json

    from database import db
    from database.models import Signal

    report = {"contract_version": "review-v2", "passed": False, "failed_requirements": ["debate_synthesis"],
              "required_agents": {"technical": True, "debate_synthesis": False},
              "required_inputs": {"risk_approval": True}}
    with db.get_db() as session:
        session.add(Signal(agent_name="portfolio_manager", symbol="TCS", signal="BUY", confidence=70,
                           data_json=json.dumps({"outcome": "blocked", "review_contract": report,
                                                 "holding_period": "INTRADAY"})))
        session.add(Signal(agent_name="portfolio_manager", symbol="INFY", signal="BUY",
                           data_json=json.dumps({"outcome": "executed"})))
    http, _ = client
    rows = {row["symbol"]: row for row in http.get("/api/signals").json()}
    assert rows["TCS"]["review_contract"]["failed_requirements"] == ["debate_synthesis"]
    assert rows["TCS"]["holding_period"] == "INTRADAY" and rows["INFY"]["review_contract"] is None
    sleeves = {item["name"]: item for item in http.get("/api/operations").json()["sleeves"]}
    assert sleeves["CompositeTrend"]["enabled"] is True and sleeves["CompositeTrend"]["book"] == "core"
    assert sleeves["ORBIntraday"]["enabled"] is False and sleeves["ETFRotation"]["segment"] == "ETF"
    assert not any(item["enabled"] for name, item in sleeves.items() if name != "CompositeTrend")
