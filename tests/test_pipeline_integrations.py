"""Verify scheduled graph adapters against their real optional-component APIs."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents import graph
from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state


def test_queued_macro_events_use_existing_stream_api(monkeypatch):
    from data import event_driven_kafka_stream as stream
    instance = stream.EventDrivenKafkaStream()
    asyncio.run(instance._handle_event({"type": "RBI_RATE_DECISION"}, None))
    monkeypatch.setattr(stream, "event_stream", instance)
    result = graph._kafka_events_node({})
    assert not result.get("errors")
    assert result["active_macro_event"]["type"] == "RBI_RATE_DECISION"


def test_depth_adapter_uses_normalized_live_quote_fields(monkeypatch):
    from data import l2_order_flow_analyzer as depth
    monkeypatch.setattr(depth, "l2_analyzer", depth.L2OrderFlowAnalyzer())
    result = graph._l2_order_flow_node({
        "watchlist": ["TEST"],
        "live_quotes": {"TEST": {
            "ltp": 100, "best_bid": [{"quantity": 10, "orders": 1}],
            "best_ask": [{"quantity": 5, "orders": 1}],
        }},
    })
    assert not result.get("errors")
    assert result["ofi_signals"]["TEST"]["ofi"] > 0
    assert result["ofi_signals"]["TEST"]["signal"] == "INSUFFICIENT_DATA"


def test_untrained_hmm_is_explicitly_unavailable(monkeypatch):
    from risk import hmm_regime_classifier as hmm
    monkeypatch.setattr(hmm, "hmm_classifier", hmm.HMMRegimeClassifier(model_path="/nonexistent/model.json"))
    result = graph._hmm_regime_node({})
    assert result["hmm_regime"]["regime"] == "UNAVAILABLE"
    assert result["hmm_regime"]["degraded"]
    assert result["hmm_regime"]["action_directive"]["allow_new_positions"] is False
    assert result["errors"]


def test_hrp_adapter_works_without_trained_rl_and_retains_cash():
    result = graph._marl_optimizer_node({
        "market_data": {"TEST": pd.DataFrame({"close": np.arange(100, 140, dtype=float)})},
        "technical_signals": {"TEST": {"signal": "BUY"}},
        "available_capital": 10000,
    })
    assert not result.get("errors")
    assert result["optimal_allocations"]["TEST"] <= 0.12
    assert sum(result["optimal_allocations"].values()) <= 1


@pytest.mark.parametrize("level", [15.2, 23819.75, float("nan")])
def test_market_data_validates_fetched_vix(monkeypatch, level):
    from agents import market_data_agent as market
    monkeypatch.setattr(market, "fetch_multiple", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(market, "fetch_india_vix", lambda: (level, "angel_one"))
    monkeypatch.setattr(market, "portfolio_tracker", SimpleNamespace(
        sync_from_broker=lambda: None, update_unrealised_pnl=lambda: True,
        total_value=10000, available_capital=10000, positions={},
    ))
    result = market.MarketDataAgent().run({"watchlist": []})
    if level == 15.2:
        assert result["india_vix"] == 15.2
        assert result["india_vix_source"] == "angel_one"
        assert not result["errors"]
    else:
        assert result["india_vix"] == 0
        assert result["errors"]


def test_market_data_fetches_held_symbols_for_portfolio_risk(monkeypatch):
    from agents import market_data_agent as market

    calls = []
    monkeypatch.setattr(market, "fetch_multiple", lambda symbols, **kwargs: calls.append((symbols, kwargs)) or {})
    monkeypatch.setattr(market, "fetch_india_vix", lambda: (15.2, "angel_one"))
    monkeypatch.setattr(market, "get_token", lambda _symbol: None)
    monkeypatch.setattr(market, "portfolio_tracker", SimpleNamespace(
        sync_from_broker=lambda: None, update_unrealised_pnl=lambda: True,
        total_value=10000, available_capital=10000,
        positions={"HELD": {"quantity": 1, "ltp": 100}},
    ))
    market.MarketDataAgent().run({"watchlist": ["TEST", "HELD"]})
    assert calls == [(["TEST", "HELD"], {"interval": "15m", "days": 45})]


def test_market_data_carries_provider_provenance_separately_from_frame(monkeypatch):
    from agents import market_data_agent as market

    frame = pd.DataFrame({"close": [100.0]})
    frame.attrs["history_provenance"] = {"provider": "angel_one", "symbol": "TEST"}
    monkeypatch.setattr(market, "fetch_multiple", lambda *_args, **_kwargs: {"TEST": frame})
    monkeypatch.setattr(market, "fetch_india_vix", lambda: (15.2, "angel_one"))
    monkeypatch.setattr(market, "get_token", lambda _symbol: None)
    monkeypatch.setattr(market, "portfolio_tracker", SimpleNamespace(
        sync_from_broker=lambda: None, update_unrealised_pnl=lambda: True,
        total_value=10000, available_capital=10000, positions={},
    ))

    result = market.MarketDataAgent().run({"watchlist": ["TEST"]})
    provenance = result["market_data_provenance"]["TEST"]
    assert provenance["provider"] == "angel_one"
    assert provenance["symbol"] == "TEST"
    assert len(provenance["bars_sha256"]) == 64
    frame.attrs["history_provenance"]["provider"] = "changed"
    assert result["market_data_provenance"]["TEST"]["provider"] == "angel_one"


def test_market_data_reports_one_bad_history_without_losing_other_provenance(monkeypatch):
    from agents import market_data_agent as market

    bad = pd.DataFrame({"unsupported": [object()]})
    bad.attrs["history_provenance"] = {"provider": "angel_one", "symbol": "BAD"}
    good = pd.DataFrame({"close": [100.0]})
    good.attrs["history_provenance"] = {"provider": "angel_one", "symbol": "GOOD"}
    monkeypatch.setattr(market, "fetch_multiple", lambda *_args, **_kwargs: {"BAD": bad, "GOOD": good})
    monkeypatch.setattr(market, "fetch_india_vix", lambda: (15.2, "angel_one"))
    monkeypatch.setattr(market, "get_token", lambda _symbol: None)
    monkeypatch.setattr(market, "portfolio_tracker", SimpleNamespace(
        sync_from_broker=lambda: None, update_unrealised_pnl=lambda: True,
        total_value=10000, available_capital=10000, positions={},
    ))

    result = market.MarketDataAgent().run({"watchlist": ["BAD", "GOOD"]})
    assert any("BAD: Historical data fingerprint failed" in error for error in result["errors"])
    assert "bars_sha256" not in result["market_data_provenance"]["BAD"]
    assert len(result["market_data_provenance"]["GOOD"]["bars_sha256"]) == 64


@pytest.mark.parametrize("failure", ["sync", "stale"])
def test_market_data_reports_unusable_portfolio_snapshot(monkeypatch, failure):
    from agents import market_data_agent as market

    def sync():
        if failure == "sync":
            raise RuntimeError("sync failed")
    monkeypatch.setattr(market, "fetch_multiple", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(market, "fetch_india_vix", lambda: (15.2, "angel_one"))
    monkeypatch.setattr(market, "get_token", lambda _symbol: None)
    monkeypatch.setattr(market, "portfolio_tracker", SimpleNamespace(
        sync_from_broker=sync, update_unrealised_pnl=lambda: False,
        total_value=10000, available_capital=10000, positions={},
    ))
    result = market.MarketDataAgent().run({"watchlist": ["TEST"]})
    assert result["portfolio_snapshot_valid"] is False
    assert result["errors"]


def test_portfolio_marks_require_a_fresh_reference_for_every_position(monkeypatch):
    from broker import order_manager
    from broker.portfolio_tracker import PortfolioTracker
    from config import instruments

    tracker = PortfolioTracker()
    tracker._positions = {
        "FRESH": {"quantity": 2, "avg_price": 100.0, "ltp": 100.0},
        "STALE": {"quantity": 1, "avg_price": 100.0, "ltp": 90.0},
        "UNKNOWN": {"quantity": 1, "avg_price": 100.0, "ltp": 100.0},
    }
    monkeypatch.setattr(tracker, "_revalue", lambda: None)
    monkeypatch.setattr(instruments, "get_token", lambda symbol: None if symbol == "UNKNOWN" else symbol)
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda symbol, *_args: {"price": 110.0} if symbol == "FRESH" else None,
    )
    assert tracker.update_unrealised_pnl() is False
    assert tracker._positions["FRESH"]["pnl"] == 20.0
    assert tracker._positions["STALE"]["ltp"] == 90.0
    tracker._positions = {"FRESH": tracker._positions["FRESH"]}
    assert tracker.update_unrealised_pnl() is True
    tracker._positions = {}
    assert tracker.update_unrealised_pnl() is True


def test_risk_manager_blocks_an_unusable_portfolio_snapshot(monkeypatch):
    from agents.base_agent import create_initial_state
    from agents.risk_manager_agent import RiskManagerAgent
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "has_hit_portfolio_drawdown_limit", lambda: False)
    state = create_initial_state(["TEST"])
    state["india_vix"] = 15
    state["portfolio_snapshot_valid"] = False
    result = RiskManagerAgent().run(state)
    assert result["risk_assessment"]["blocked"]
    assert "portfolio" in result["risk_assessment"]["reason"].lower()


@pytest.mark.parametrize("snapshot,refresh", [(False, True), (True, False), (True, "error")])
def test_router_blocks_entry_when_portfolio_evidence_fails(
    tmp_path, monkeypatch, snapshot, refresh,
):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    stub_persisted_review_verification(monkeypatch)
    from broker.portfolio_tracker import portfolio_tracker
    from operations import control_state as controls
    from operations.control_state import ControlStateStore
    from risk import dead_mans_switch as dms

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(False)
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(dms, "dead_mans_switch", dms.DeadMansSwitch(store=store))
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    def refresh_marks():
        if refresh == "error":
            raise RuntimeError("mark refresh failed")
        return refresh

    monkeypatch.setattr(portfolio_tracker, "update_unrealised_pnl", refresh_marks)
    evidence = sentiment_state(["TEST"])
    result = graph._order_router_node({
        "execution_enabled": True, "current_positions": {},
        **evidence,
        "portfolio_snapshot_valid": snapshot,
        "watchlist": ["TEST"], "screener_results": {"qualified": ["TEST"], "rejected": []},
        "debate_results": {"TEST": reviewed_debate(evidence, "TEST", "BUY")},
        "trading_decisions": [reviewed_decision(evidence, {
            "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        })],
    })
    assert not result["executed_orders"]
    assert "portfolio" in result["errors"][0].lower()


def test_india_vix_uses_only_fresh_yahoo_fallback(monkeypatch):
    from data import market_data

    now = market_data.datetime.now(market_data.IST)
    monkeypatch.setattr(
        market_data, "fetch_index_data",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["datetime", "close"]),
    )

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "^INDIAVIX"

        def history(self, **kwargs):
            assert kwargs == {"period": "1d", "interval": "1m"}
            return pd.DataFrame(
                {"Close": [15.5]},
                index=pd.DatetimeIndex([now - pd.Timedelta(minutes=2)]),
            )

    monkeypatch.setattr(market_data.yf, "Ticker", Ticker)
    value, source = market_data.fetch_india_vix()
    assert value == 15.5
    assert source == "yfinance"


def test_india_vix_rejects_stale_yahoo_fallback(monkeypatch):
    from data import market_data

    now = market_data.datetime.now(market_data.IST)
    monkeypatch.setattr(
        market_data, "fetch_index_data",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["datetime", "close"]),
    )

    class Ticker:
        def history(self, **_kwargs):
            return pd.DataFrame(
                {"Close": [15.5]},
                index=pd.DatetimeIndex([now - pd.Timedelta(minutes=30)]),
            )

    monkeypatch.setattr(market_data.yf, "Ticker", Ticker)
    with pytest.raises(RuntimeError, match="No fresh India VIX"):
        market_data.fetch_india_vix()


def test_missing_vix_blocks_risk_manager_without_approvals(monkeypatch):
    from agents.risk_manager_agent import RiskManagerAgent
    from broker.portfolio_tracker import portfolio_tracker
    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    result = RiskManagerAgent().run({"india_vix": 0, "current_positions": {}})
    assert result["risk_assessment"]["blocked"]
    assert "unavailable" in result["risk_assessment"]["reason"]


def test_hmm_default_path_is_a_model_file_not_working_directory(monkeypatch):
    from config.settings import settings
    from risk.hmm_regime_classifier import HMMRegimeClassifier
    monkeypatch.setattr(settings, "hmm_model_path", "")
    classifier = HMMRegimeClassifier()
    assert classifier.model_path.name == "hmm_regime_classifier.json"
