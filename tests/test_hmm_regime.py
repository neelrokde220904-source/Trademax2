"""HMM regime model: fitting, JSON persistence, load validation and entry gating."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from risk import hmm_regime_classifier as hmm
from risk.hmm_regime_classifier import (
    FEATURES, MODEL_FORMAT, HMMRegimeClassifier, entry_regime_block_reason,
)
from tests.sentiment_fixtures import (
    reviewed_debate, reviewed_decision, sentiment_state, stub_persisted_review_verification,
)
from tests.test_llm_runtime import llm_database

REGIME_MEANS = {
    "LOW_VOL_TREND": (0.10, 0.02, 12.0),
    "BULL_EUPHORIA": (0.15, 0.06, 15.0),
    "HIGH_VOL_MEAN_REVERT": (0.24, -0.04, 21.0),
    "SYSTEMIC_PANIC": (0.55, -0.18, 42.0),
}


def _sector_data():
    return {
        "sector": "financial services",
        "sector_source": "yfinance.Ticker.info",
        "sector_observed_at": datetime.now(timezone.utc).isoformat(),
    }


def synthetic_features(days: int = 720, block: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    names = list(REGIME_MEANS)
    rows = [
        np.asarray(REGIME_MEANS[names[(day // block) % len(names)]]) * (1 + rng.normal(0, 0.04, 3))
        for day in range(days)
    ]
    index = pd.bdate_range(end=pd.Timestamp("2026-01-30"), periods=days)
    return pd.DataFrame(rows, columns=list(FEATURES), index=index)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("hmm") / "model.json"
    classifier = HMMRegimeClassifier(model_path=str(path))
    features = synthetic_features()
    classifier.fit(features, n_restarts=3)
    classifier.save()
    return classifier, features, path


def test_fit_labels_states_and_round_trips_through_json(fitted):
    classifier, features, path = fitted
    payload = json.loads(path.read_text())
    assert payload["format"] == MODEL_FORMAT
    assert sorted(classifier.regime_map.values()) == sorted(REGIME_MEANS)
    decoded = classifier.decode(features)
    panic_vix = features["india_vix"][decoded == "SYSTEMIC_PANIC"].mean()
    calm_vix = features["india_vix"][decoded == "LOW_VOL_TREND"].mean()
    assert panic_vix > 35 > 14 > calm_vix

    reloaded = HMMRegimeClassifier(model_path=str(path))
    assert reloaded.is_fitted
    today = features.index[-1].date() + timedelta(days=1)
    assert reloaded.predict_regime(features, today=today) == classifier.predict_regime(features, today=today)
    assert reloaded.decode(features).equals(decoded)


def test_invalid_or_pickled_models_are_refused(fitted, tmp_path):
    _, _, path = fitted
    pickled = tmp_path / "model.pkl"
    pickled.write_bytes(b"\x80\x04K\x01.")
    assert not HMMRegimeClassifier(model_path=str(pickled)).is_fitted

    payload = json.loads(path.read_text())
    payload["features"] = ["momentum_20d", "realized_vol_20d", "india_vix"]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    classifier = HMMRegimeClassifier(model_path=str(tampered))
    assert not classifier.is_fitted
    assert classifier.load_error
    result = classifier.classify_latest(date(2026, 2, 2))
    assert result["regime"] == "UNAVAILABLE"
    assert entry_regime_block_reason(result)


def test_entry_gate_requires_fresh_non_panic_classification(fitted):
    classifier, features, _ = fitted
    calm = features[classifier.decode(features) == "LOW_VOL_TREND"]
    fresh = classifier.predict_regime(calm, today=calm.index[-1].date() + timedelta(days=1))
    assert fresh["regime"] == "LOW_VOL_TREND" and not fresh["degraded"]
    assert entry_regime_block_reason(fresh) is None

    stale = classifier.predict_regime(calm, today=calm.index[-1].date() + timedelta(days=30))
    assert stale["degraded"] and not stale["action_directive"]["allow_new_positions"]
    assert entry_regime_block_reason(stale)

    panic = features[classifier.decode(features) == "SYSTEMIC_PANIC"]
    crisis = classifier.predict_regime(panic, today=panic.index[-1].date() + timedelta(days=1))
    assert crisis["regime"] == "SYSTEMIC_PANIC"
    assert entry_regime_block_reason(crisis)
    assert entry_regime_block_reason(None)


@pytest.mark.parametrize("cap", [None, 0, -0.01, True, 1.01, float("nan"), float("inf"), "0.1", 10 ** 400])
def test_entry_regime_rejects_missing_or_invalid_position_cap(cap):
    regime = {"degraded": False, "action_directive": {
        "allow_new_positions": True, "max_position_size_pct": cap,
    }}
    with pytest.raises(ValueError, match="position-size cap"):
        hmm.entry_regime_position_cap(regime)


@pytest.mark.parametrize(("allowed", "hedge", "strategy", "blocked"), [
    (["trend_following"], False, "CompositeTrend", False),
    (["mean_reversion"], False, "CompositeTrend", True),
    (["momentum_breakout"], False, "CompositeTrend", True),
    (["trend_following"], True, "CompositeTrend", True),
    (["trend_following"], None, "CompositeTrend", True),
    (None, False, "CompositeTrend", True),
    (["trend_following"], False, "UnknownStrategy", True),
])
def test_hmm_strategy_and_hedge_directives_are_explicit(allowed, hedge, strategy, blocked):
    regime = {"degraded": False, "action_directive": {
        "allow_new_positions": True, "allowed_strategies": allowed, "hedge_required": hedge,
        "max_position_size_pct": 0.1,
    }}
    assert bool(hmm.entry_regime_strategy_block_reason(regime, strategy)) is blocked


def test_unavailable_regime_blocks_entries_but_not_exits(tmp_path, monkeypatch):
    stub_persisted_review_verification(monkeypatch)
    from agents import graph
    from broker.portfolio_tracker import portfolio_tracker
    from config.settings import settings
    from operations import control_state as controls
    from risk import dead_mans_switch as dms
    from operations.control_state import ControlStateStore

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(False)
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(dms, "dead_mans_switch", dms.DeadMansSwitch(store=store))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "update_unrealised_pnl", lambda: True)
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    calls = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: calls.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": len(calls),
    })
    regime = HMMRegimeClassifier(model_path=str(tmp_path / "missing.json")).unavailable("not fitted")
    decision = {
        "action": "BUY", "quantity": 1, "price": 100, "stop_loss": 95,
        "target": 110, "holding_period": "DELIVERY",
    }

    evidence = sentiment_state(["NEW"])
    entry = graph._order_router_node({
        "execution_enabled": True, "hmm_regime": regime, "current_positions": {},
        **evidence,
        "portfolio_snapshot_valid": True,
        "watchlist": ["NEW"], "screener_results": {"qualified": ["NEW"], "rejected": []},
        "debate_results": {"NEW": reviewed_debate(evidence, "NEW", "BUY")},
        "trading_decisions": [reviewed_decision(evidence, {**decision, "symbol": "NEW"})],
    })
    assert not calls and "New entry blocked for NEW" in entry["errors"][0]

    exit_result = graph._order_router_node({
        "execution_enabled": True, "hmm_regime": regime,
        "current_positions": {"OLD": {"quantity": 2}},
        "trading_decisions": [{**decision, "symbol": "OLD", "action": "SELL", "quantity": 2}],
    })
    assert [call["symbol"] for call in calls] == ["OLD"]
    assert calls[0]["allow_requested_price_reference"] is False
    assert calls[0]["is_exit"] is True
    assert exit_result["executed_orders"][0]["is_exit"]


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ({"price": 100, "stop_loss": 105, "target": 110, "holding_period": "INTRADAY"},
         "BUY requires stop_loss < price < target"),
        ({"price": 100, "stop_loss": 95, "target": 0, "holding_period": "INTRADAY"},
         "target must be a positive finite number"),
        ({"price": 100, "stop_loss": 95, "target": 110, "holding_period": "UNKNOWN"},
         "holding_period must be one of INTRADAY, SWING, BTST, POSITIONAL"),
    ],
)
def test_entry_decisions_require_valid_execution_fields(decision, expected):
    from agents.graph import _validate_entry_decision

    assert _validate_entry_decision(decision, "BUY") == expected


def test_new_entry_rejects_market_order_type():
    from agents.graph import _validate_entry_decision

    assert _validate_entry_decision({
        "price": 100, "stop_loss": 95, "target": 110,
        "holding_period": "INTRADAY", "order_type": "MARKET",
    }, "BUY") == "new entries require LIMIT orders"


def test_short_entry_requires_target_below_price_and_stop_above():
    from agents.graph import _validate_entry_decision

    assert _validate_entry_decision({
        "price": 100, "stop_loss": 105, "target": 90, "holding_period": "INTRADAY",
    }, "SELL") is None
    assert _validate_entry_decision({
        "price": 100, "stop_loss": 95, "target": 110, "holding_period": "INTRADAY",
    }, "SELL") == "SELL requires target < price < stop_loss"


def test_order_router_never_routes_entry_with_invalid_protective_levels(tmp_path, monkeypatch):
    stub_persisted_review_verification(monkeypatch)
    from agents import graph
    from config.settings import settings
    from operations import control_state as controls
    from operations.control_state import ControlStateStore
    from broker.portfolio_tracker import portfolio_tracker
    from risk import dead_mans_switch as dms
    from risk import hmm_regime_classifier as hmm

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(False)
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(dms, "dead_mans_switch", dms.DeadMansSwitch(store=store))
    monkeypatch.setattr(hmm, "entry_regime_block_reason", lambda _regime: None)
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(portfolio_tracker, "daily_entry_count", lambda: 0)
    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "update_unrealised_pnl", lambda: True)
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Invalid entry was routed"))

    evidence = sentiment_state(["TEST"])
    evidence["technical_signals"]["TEST"] = {
        **evidence["technical_signals"]["TEST"],
        "signal": "BUY", "strategy": "CompositeTrend", "stop_loss": 105, "target": 110,
        "indicators": {"close": 100, "daily_atr_14": 3.0},
    }
    result = graph._order_router_node({
        "execution_enabled": True,
        "portfolio_snapshot_valid": True,
        **evidence,
        "watchlist": ["TEST"], "screener_results": {"qualified": ["TEST"], "rejected": []},
        "hmm_regime": {"degraded": False, "action_directive": {
            "allow_new_positions": True, "max_position_size_pct": 0.1,
            "allowed_strategies": ["trend_following"], "hedge_required": False,
        }},
        "current_positions": {},
        "risk_assessment": {"TEST": {
            "approved": True, "deterministic_action": "BUY",
            "stop_loss": 105, "target": 110, "vix_factor": 1.0,
            **_sector_data(),
        }},
        "debate_results": {"TEST": reviewed_debate(evidence, "TEST", "BUY")},
        "trading_decisions": [reviewed_decision(evidence, {
            "symbol": "TEST", "action": "BUY", "quantity": 1,
            "price": 100, "stop_loss": 105, "target": 110,
            "holding_period": "INTRADAY",
        })],
    })
    assert not result["executed_orders"]
    assert "BUY requires stop_loss < price < target" in result["errors"][0]


def test_adaptive_scan_follows_last_completed_regime(tmp_path, monkeypatch):
    import asyncio

    from operations.control_state import ControlStateStore
    from scheduler import market_scheduler as scheduler_module

    scheduler = scheduler_module.MarketScheduler(store=ControlStateStore(tmp_path / "controls.sqlite3"))
    scans = []
    async def run_scan():
        scans.append(scheduler.last_regime)
    monkeypatch.setattr(scheduler, "_run_scan", run_scan)

    scheduler.last_regime, scheduler.last_vix = "SYSTEMIC_PANIC", 18.0
    asyncio.run(scheduler._adaptive_scan())
    scheduler.last_regime, scheduler.last_vix = "HIGH_VOL_MEAN_REVERT", 31.0
    asyncio.run(scheduler._adaptive_scan())
    assert scans == []
    scheduler.last_vix = 20.0
    asyncio.run(scheduler._adaptive_scan())
    assert scans == ["HIGH_VOL_MEAN_REVERT"]


class FakeCandleApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.params = []

    def getCandleData(self, params):
        self.params.append(params)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _patch_candles(monkeypatch, responses):
    from types import SimpleNamespace

    from data import market_data

    api = FakeCandleApi(responses)
    monkeypatch.setattr(market_data, "angel_client", SimpleNamespace(api=api))
    monkeypatch.setattr(market_data.candle_rate_limiter, "call", lambda action, *args: action(*args))
    monkeypatch.setattr(market_data.time, "sleep", lambda _seconds: None)
    return market_data, api


def test_daily_history_spans_whole_days_and_retries_rate_limits(monkeypatch):
    rows = [["2020-03-16T00:00:00+05:30", 1, 1, 1, 9200.0, 0], ["2020-03-17T00:00:00+05:30", 1, 1, 1, 8967.0, 0]]
    market_data, api = _patch_candles(monkeypatch, [
        RuntimeError("Couldn't parse the JSON response: b'Access denied because of exceeding access rate'"),
        {"status": True, "data": rows},
    ])
    frame = market_data.fetch_index_daily_history("99926000", date(2020, 3, 16), date(2020, 3, 17))
    assert list(frame["close"]) == [9200.0, 8967.0]
    assert api.params[-1]["fromdate"] == "2020-03-16 00:00"
    assert api.params[-1]["todate"] == "2020-03-17 23:59"


def test_daily_history_refuses_failed_chunks_and_gaps(monkeypatch):
    market_data, _ = _patch_candles(monkeypatch, [{"status": False, "errorcode": "AB1004", "message": "busy"}])
    with pytest.raises(RuntimeError, match="AB1004"):
        market_data.fetch_index_daily_history("99926000", date(2020, 1, 1), date(2020, 2, 1))

    gap = [["2020-01-01T00:00:00+05:30", 1, 1, 1, 1.0, 0], ["2020-01-20T00:00:00+05:30", 1, 1, 1, 1.0, 0]]
    market_data, _ = _patch_candles(monkeypatch, [{"status": True, "data": gap}])
    with pytest.raises(RuntimeError, match="19-day gap"):
        market_data.fetch_index_daily_history("99926000", date(2020, 1, 1), date(2020, 2, 1))


def test_elevated_volatility_state_is_never_labelled_euphoric():
    # State profiles observed when fitting Angel Nifty/VIX history from 2017-10.
    raw = np.array([
        [0.085, 0.022, 12.6],   # calm
        [0.426, -0.028, 40.0],  # crisis
        [0.173, 0.016, 20.0],   # elevated volatility, weakly positive momentum
        [0.125, -0.010, 14.9],  # moderate, drifting lower
    ])
    scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    labels = HMMRegimeClassifier(model_path="/nonexistent/model.json")._label_states(raw, scaled)
    assert labels == {0: "LOW_VOL_TREND", 1: "SYSTEMIC_PANIC", 2: "HIGH_VOL_MEAN_REVERT", 3: "HIGH_VOL_MEAN_REVERT"}
    raw[3, 1] = 0.03
    labels = HMMRegimeClassifier(model_path="/nonexistent/model.json")._label_states(raw, scaled)
    assert labels[3] == "BULL_EUPHORIA" and labels[2] == "HIGH_VOL_MEAN_REVERT"


def test_status_exposes_directive_and_decoded_history(fitted, monkeypatch):
    from data import market_data
    from config.settings import settings

    _, _, path = fitted
    classifier = HMMRegimeClassifier(model_path=str(path))
    assert classifier.status()["history"] == [] and classifier.status()["latest"] is None
    days = pd.bdate_range(end=pd.Timestamp("2026-03-02"), periods=320)
    rng = np.random.default_rng(3)
    frames = {
        settings.nifty50_token: pd.DataFrame({
            "datetime": days, "close": 20000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))),
        }),
        settings.india_vix_token: pd.DataFrame({"datetime": days, "close": 15 + rng.normal(0, 1, len(days))}),
    }
    monkeypatch.setattr(market_data, "fetch_index_daily_history", lambda token, *_: frames[token])
    result = classifier.classify_latest(date(2026, 3, 3))
    assert not result["degraded"]

    status = classifier.status()
    latest = status["latest"]
    assert latest["directive"]["allow_new_positions"] is not None
    assert abs(sum(latest["regime_probs"].values()) - 1) < 1e-6
    history = status["history"]
    assert 0 < len(history) <= hmm.REGIME_HISTORY_SESSIONS
    assert history[-1]["date"] == "2026-03-02"
    assert {row["regime"] for row in history} <= set(classifier.regime_map.values())
    assert all(row["nifty_close"] > 0 and row["india_vix"] > 0 for row in history)


def test_paper_intraday_short_entries_require_protective_stop(tmp_path, monkeypatch, llm_database):
    from datetime import datetime as real_datetime

    from agents import graph
    from data import indicators
    from config.settings import settings
    from operations import control_state as controls
    from operations.control_state import ControlStateStore
    from broker.portfolio_tracker import portfolio_tracker
    from risk import dead_mans_switch as dms
    from risk import hmm_regime_classifier as hmm
    from risk import daily_portfolio_risk as daily_risk
    from agents.portfolio_manager_agent import PortfolioManagerAgent
    from operations.candidate_reviews import persist_candidate_reviews

    class MidSession(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return graph.IST.localize(real_datetime(2026, 11, 12, 11, 0))

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(False)
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(dms, "dead_mans_switch", dms.DeadMansSwitch(store=store))
    monkeypatch.setattr(hmm, "entry_regime_block_reason", lambda _regime: None)
    monkeypatch.setattr(graph, "datetime", MidSession)
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(portfolio_tracker, "daily_entry_count", lambda: 0)
    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "update_unrealised_pnl", lambda: True)
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    monkeypatch.setattr(indicators, "average_daily_turnover_inr", lambda *_args, **_kwargs: 1_000_000_000.0)
    monkeypatch.setattr(daily_risk, "daily_returns", lambda *_args: pd.Series(np.linspace(-0.01, 0.01, 20)))
    calls = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: calls.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": len(calls),
    })
    short = {"action": "SELL", "quantity": 10, "price": 100, "stop_loss": 103, "target": 90, "holding_period": "INTRADAY"}

    def route(*decisions, positions=None, cap=0.1, target=90,
              portfolio_value=25_000, available_capital=25_000,
              allowed=None, hedge=False):
        evidence = sentiment_state([decision["symbol"] for decision in decisions])
        evidence["technical_signals"] = {
            decision["symbol"]: {
                **evidence["technical_signals"][decision["symbol"]],
                "signal": "SELL", "strategy": "CompositeTrend", "stop_loss": 103, "target": target,
                "indicators": {"close": 100, "daily_atr_14": 3.0},
            }
            for decision in decisions
        }
        state = {
            "execution_enabled": True,
            "portfolio_snapshot_valid": True,
            **evidence,
            "watchlist": [decision["symbol"] for decision in decisions],
            "screener_results": {
                "qualified": [decision["symbol"] for decision in decisions], "rejected": [],
            },
            "current_positions": positions or {},
            "risk_assessment": {
                decision["symbol"]: {
                    "approved": True, "deterministic_action": "SELL",
                    "stop_loss": 103, "target": target, "holding_period": "INTRADAY",
                    "vix_factor": 1.0,
                    **_sector_data(),
                }
                for decision in decisions
            },
            "debate_results": {
                decision["symbol"]: reviewed_debate(evidence, decision["symbol"], "SELL")
                for decision in decisions
            },
            "hmm_regime": {
                "degraded": False, "action_directive": {
                    "allow_new_positions": True, "max_position_size_pct": cap,
                    "allowed_strategies": ["trend_following"] if allowed is None else allowed,
                    "hedge_required": hedge,
                },
            },
            "available_capital": available_capital,
            "portfolio_value": portfolio_value,
            "trading_decisions": [reviewed_decision(evidence, decision) for decision in decisions],
        }
        unique = {decision["candidate_id"]: decision for decision in state["trading_decisions"]}
        context = PortfolioManagerAgent._portfolio_context(state, state["risk_assessment"])
        persist_candidate_reviews(state, list(unique.values()), context)
        return graph._order_router_node(state)

    result = route(
        {**short, "symbol": "OK"},
        {**short, "symbol": "BADSTOP", "stop_loss": 97},
        {**short, "symbol": "DELIV", "holding_period": "DELIVERY"},
    )
    assert [call["symbol"] for call in calls] == ["OK"]
    assert calls[0]["quantity"] == 24
    assert calls[0]["target"] == 90
    assert calls[0]["product_type"] == "INTRADAY" and calls[0]["allow_requested_price_reference"] is False
    assert result["executed_orders"][0]["is_exit"] is False
    assert any("BADSTOP" in error and "stop-loss" in error for error in result["errors"])
    assert "[OrderRouter] Naked short sale blocked for DELIV" in result["errors"]

    monkeypatch.setattr(settings, "trademax_broker_mode", "zerodha_only")
    alternate_broker_paper = route({**short, "symbol": "ALT"})
    assert alternate_broker_paper["executed_orders"]
    assert calls[-1]["expected_paper_book"] == {}
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")

    duplicate = route({**short, "symbol": "DUP"}, {**short, "symbol": "DUP"})
    assert [call["symbol"] for call in calls].count("DUP") == 1
    assert any(
        "safe maximum" in error or "Cash, per-position" in error or "Adversely correlated" in error
        or "an open position already exists" in error
        for error in duplicate["errors"]
    ), duplicate["errors"]

    original_max_positions = settings.max_positions
    monkeypatch.setattr(settings, "max_positions", 1)
    before = len(calls)
    capped_positions = route({**short, "symbol": "FIRST"}, {**short, "symbol": "SECOND"})
    assert [call["symbol"] for call in calls[before:]] == ["FIRST"]
    assert any("max positions" in error.lower() for error in capped_positions["errors"])
    before = len(calls)
    monkeypatch.setattr(portfolio_tracker, "_positions", {
        "EXISTING": {"quantity": 1, "ltp": 100},
    })
    already_full = route(
        {**short, "symbol": "THIRD"},
        positions={"EXISTING": {"quantity": 1, "ltp": 100}},
    )
    assert len(calls) == before
    assert any("max positions" in error.lower() for error in already_full["errors"])
    monkeypatch.setattr(portfolio_tracker, "_positions", {})
    def rejected_then_filled(**kwargs):
        calls.append(kwargs)
        if kwargs["symbol"] == "UNFILLED":
            return None
        return {"symbol": kwargs["symbol"], "status": "EXECUTED", "trade_id": len(calls)}
    monkeypatch.setattr(graph, "place_order", rejected_then_filled)
    before = len(calls)
    uncapped = route({**short, "symbol": "UNFILLED"}, {**short, "symbol": "FILLED"})
    assert [call["symbol"] for call in calls[before:]] == ["UNFILLED", "FILLED"]
    assert [order["symbol"] for order in uncapped["executed_orders"]] == ["FILLED"]
    monkeypatch.setattr(settings, "max_positions", original_max_positions)
    before = len(calls)
    correlated = route({**short, "symbol": "BATCH1"}, {**short, "symbol": "BATCH2"})
    assert [call["symbol"] for call in calls[before:]] == ["BATCH1"]
    assert any("Adversely correlated" in error for error in correlated["errors"])

    missing_cap = route({**short, "symbol": "NOCAP"}, cap=None)
    assert not missing_cap["executed_orders"]
    assert any("position-size cap" in error for error in missing_cap["errors"])

    wrong_strategy = route({**short, "symbol": "WRONGREGIME"}, allowed=["mean_reversion"])
    assert not wrong_strategy["executed_orders"]
    assert any("does not permit" in error for error in wrong_strategy["errors"])
    hedge_required = route({**short, "symbol": "HEDGE"}, hedge=True)
    assert not hedge_required["executed_orders"]
    assert any("requires a hedge" in error for error in hedge_required["errors"])

    regime_limited = route({**short, "symbol": "REGIME", "target": 80}, cap=0.04, target=80)
    assert regime_limited["executed_orders"], regime_limited["errors"]
    assert calls[-1]["quantity"] == 9

    from risk.portfolio_stop_risk import entry_stop_risk
    from risk.stop_loss_manager import stop_loss_manager

    monkeypatch.setattr(daily_risk, "correlation_block_reason", lambda *_args: None)
    monkeypatch.setattr(settings, "max_open_stop_risk_pct", 0.006)
    before = len(calls)
    budgeted = route(
        {**short, "symbol": "RISK1", "target": 80},
        {**short, "symbol": "RISK2", "target": 80},
        target=80,
    )
    budget_calls = calls[before:]
    assert len(budget_calls) == 2, budgeted["errors"]
    assert budget_calls[0]["expected_paper_book"] == {}
    assert budget_calls[1]["expected_paper_book"]["RISK1"]["quantity"] == -budget_calls[0]["quantity"]
    assert budget_calls[1]["expected_paper_book"]["RISK1"]["entry_trade_ids"]
    assert budget_calls[1]["quantity"] < budget_calls[0]["quantity"]
    assert sum(entry_stop_risk({
        "action": "SELL", "price": 100, "stop_loss": 103, "holding_period": "INTRADAY",
    }, call["quantity"]) for call in budget_calls) <= 25_000 * 0.006

    held = {
        "quantity": -10, "ltp": 90, "avg_price": 100, "stop_loss": 105,
        "entry_trade_ids": [1], "holding_period": "INTRADAY",
        **_sector_data(),
    }
    protection = {
        "action": "SELL", "stop_loss": 105, "entry_trade_ids": [1],
        "holding_period": "INTRADAY", "exit_pending_order_id": None,
    }
    monkeypatch.setattr(portfolio_tracker, "_positions", {"HELD": held})
    monkeypatch.setattr(portfolio_tracker, "_total_value", 25_000)
    monkeypatch.setattr(stop_loss_manager, "snapshot", lambda: {})
    missing_protection = route({**short, "symbol": "MISSING"}, positions={"HELD": held})
    assert not missing_protection["executed_orders"]
    assert any("active stop is missing" in error for error in missing_protection["errors"])
    protected_exit = route(
        {**short, "symbol": "HELD", "action": "BUY", "quantity": 10},
        positions={"HELD": held},
    )
    assert protected_exit["executed_orders"][0]["is_exit"] is True

    monkeypatch.setattr(stop_loss_manager, "snapshot", lambda: {"HELD": protection})
    monkeypatch.setattr(settings, "max_open_stop_risk_pct", 0.004)
    exhausted = route({**short, "symbol": "EXHAUSTED"}, positions={"HELD": held})
    assert not exhausted["executed_orders"]
    assert any("risk budget" in error.lower() for error in exhausted["errors"])
    held["ltp"] = 103
    remaining = route({**short, "symbol": "REMAINING", "target": 80}, positions={"HELD": held}, target=80)
    assert remaining["executed_orders"], remaining["errors"]
    assert calls[-1]["quantity"] < 24
    changed = route(
        {**short, "symbol": "CHANGED"},
        positions={"HELD": {**held, "entry_trade_ids": [2]}},
    )
    assert not changed["executed_orders"]
    assert any("identities changed" in error for error in changed["errors"])

    fresh_held = {
        "quantity": -125, "ltp": 100, "avg_price": 100, "stop_loss": 100.01,
        "entry_trade_ids": [1], "holding_period": "INTRADAY",
    }
    fresh_protection = {
        "action": "SELL", "stop_loss": 100.01, "entry_trade_ids": [1],
        "holding_period": "INTRADAY", "exit_pending_order_id": None,
    }
    monkeypatch.setattr(portfolio_tracker, "_positions", {"HELD": fresh_held})
    monkeypatch.setattr(portfolio_tracker, "_total_value", 25_000)
    monkeypatch.setattr(portfolio_tracker, "_available_capital", 25_000)
    monkeypatch.setattr(stop_loss_manager, "snapshot", lambda: {"HELD": fresh_protection})
    fresh_mark_gate = route(
        {**short, "symbol": "FRESHMARK"},
        positions={"HELD": {**fresh_held, "ltp": 1}},
        portfolio_value=100_000, available_capital=100_000,
    )
    assert not fresh_mark_gate["executed_orders"]
    assert any("risk sizing" in error.lower() for error in fresh_mark_gate["errors"])

    monkeypatch.setattr(portfolio_tracker, "_positions", {})
    monkeypatch.setattr(settings, "max_open_stop_risk_pct", 0.02)

    monkeypatch.setattr(indicators, "average_daily_turnover_inr", lambda *_args, **_kwargs: 100_000.0)
    participation_limited = route({**short, "symbol": "PART"})
    assert [call["symbol"] for call in calls].count("PART") == 0
    assert any("exceeds 1.00% of average daily turnover" in error for error in participation_limited["errors"])

    monkeypatch.setattr(portfolio_tracker, "daily_entry_count", lambda: 0)
    monkeypatch.setattr(settings, "max_sector_exposure_pct", 0.05)
    sector_limited = route({**short, "symbol": "SECTOR"})
    assert not sector_limited["executed_orders"]
    assert any("sector exposure" in error.lower() for error in sector_limited["errors"])

    monkeypatch.setattr(settings, "max_sector_exposure_pct", 0.25)
    monkeypatch.setattr(settings, "max_portfolio_var_pct", 0.0001)
    var_limited = route({**short, "symbol": "VAR"})
    assert not var_limited["executed_orders"]
    assert any("portfolio var" in error.lower() for error in var_limited["errors"])

    monkeypatch.setattr(
        portfolio_tracker, "daily_entry_count", lambda: settings.max_daily_entries
    )
    existing_call_count = len(calls)
    capped = route({**short, "symbol": "CAPPED"})
    assert len(calls) == existing_call_count
    assert any("Daily entry cap reached" in error for error in capped["errors"])

    monkeypatch.setattr(settings, "paper_trading", False)
    live = route({**short, "symbol": "LIVE"})
    assert "[OrderRouter] Naked short sale blocked for LIVE" in live["errors"]


@pytest.mark.parametrize("screener,watchlist", [
    ({}, ["TEST"]),
    ({"qualified": ["OTHER"], "rejected": ["TEST"]}, ["TEST"]),
    ({"qualified": ["TEST"], "rejected": ["TEST"]}, ["TEST"]),
    ({"qualified": "TEST", "rejected": []}, ["TEST"]),
    ({"qualified": ["TEST"], "rejected": []}, ["OTHER"]),
])
def test_router_blocks_entries_without_valid_screening_evidence(screener, watchlist, monkeypatch):
    from agents import graph

    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unscreened entry submitted"))
    result = graph._order_router_node({
        "execution_enabled": True, "current_positions": {}, "watchlist": watchlist,
        **sentiment_state(["TEST"]),
        "screener_results": screener,
        "debate_results": {"TEST": {"review_complete": True, "action": "BUY"}},
        "trading_decisions": [{
            "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
            "holding_period": "INTRADAY", "ai_review_passed": True,
        }],
    })
    assert result["executed_orders"] == []
    assert any("screener" in error.lower() for error in result["errors"])
