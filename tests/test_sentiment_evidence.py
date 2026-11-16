"""Sentiment failures must not masquerade as successful HOLD reviews."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.test_llm_runtime import llm_database
from tests.test_agents import portfolio_manager_state
from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state


@pytest.mark.parametrize("reply", [
    "not json",
    json.dumps({"signal": "HOLD", "confidence": 10}),
    json.dumps({"signal": "BUY", "confidence": 101, "sentiment_score": 1,
                "key_events": [], "reasoning": "x"}),
    json.dumps({"signal": "HOLD", "confidence": 50, "sentiment_score": float("nan"),
                "key_events": [], "reasoning": "x"}),
])
def test_invalid_model_output_is_unavailable_not_hold(llm_database, reply):
    from agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()
    agent._llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=reply))
    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}

    result = agent.run(state)
    assert "TEST" not in result["sentiment_signals"]
    assert result["errors"]


@pytest.mark.parametrize("failure", ["no_news", "no_model", "provider_error"])
def test_unavailable_sentiment_is_explicit(llm_database, failure):
    from agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()
    def invoke(_):
        raise RuntimeError("model unavailable")
    agent._llm = None if failure == "no_model" else SimpleNamespace(invoke=invoke)
    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}
    if failure == "no_news":
        state["news_snapshot"]["by_symbol"]["TEST"] = []

    result = agent.run(state)
    assert "TEST" not in result["sentiment_signals"]
    assert result["errors"]


def test_valid_hold_is_preserved(llm_database):
    from agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()
    response = {
        "signal": "HOLD", "confidence": 0, "sentiment_score": 0,
        "key_events": [], "reasoning": "No directional signal.",
    }
    agent._llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(response)))
    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}

    result = agent.run(state)
    signal = result["sentiment_signals"]["TEST"]
    assert {key: signal[key] for key in response} == response
    state["sentiment_signals"] = result["sentiment_signals"]
    from agents.sentiment_agent import validate_candidate_sentiment
    assert validate_candidate_sentiment(state, "TEST") == signal
    assert not result["errors"]


def test_missing_sentiment_blocks_debate_before_calling_models(llm_database, monkeypatch):
    from agents import bull_bear_synthesis_agent as module

    monkeypatch.setattr(module, "_call_llm", lambda *_args: pytest.fail("Debate must not run"))
    state = {
        "watchlist": ["TEST"], "technical_signals": {"TEST": {"signal": "BUY"}},
        "sentiment_signals": {},
    }

    result = module.BullBearSynthesisAgent().run(state)
    assert not result["debate_results"]
    assert any("sentiment" in error for error in result["errors"])


@pytest.mark.parametrize("failure", [
    "cross_run", "cross_candidate", "altered_news", "stale_snapshot", "future_snapshot",
    "old_article", "unknown_publication", "unavailable_source", "wrong_query", "missing_evidence",
])
def test_candidate_sentiment_rejects_stale_unbound_or_invalid_evidence(failure):
    from agents.sentiment_agent import validate_candidate_sentiment

    state = sentiment_state(["TEST"])
    snapshot = state["news_snapshot"]
    article = snapshot["by_symbol"]["TEST"][0]
    evidence = state["sentiment_signals"]["TEST"]["evidence"]
    if failure == "cross_run":
        state["llm_profile"]["run_id"] = "other-run"
    elif failure == "cross_candidate":
        evidence["symbol"] = "OTHER"
    elif failure == "altered_news":
        article["title"] = "Modified headline"
    elif failure in {"stale_snapshot", "future_snapshot"}:
        delta = timedelta(minutes=-31 if failure == "stale_snapshot" else 1)
        stamp = (datetime.now(timezone.utc) + delta).isoformat()
        snapshot["as_of"] = evidence["snapshot_as_of"] = article["observed_at"] = stamp
    elif failure == "old_article":
        article["published_at"] = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    elif failure == "unknown_publication":
        article["publication_time_status"] = "unknown"
    elif failure == "unavailable_source":
        snapshot["source_status"]["rss:economic_times"] = "error"
    elif failure == "wrong_query":
        article["queried_symbol"] = "OTHER"
    else:
        del state["sentiment_signals"]["TEST"]["evidence"]

    with pytest.raises(ValueError):
        validate_candidate_sentiment(state, "TEST")


@pytest.mark.parametrize("signals", [{}, {"TEST": {"signal": "HOLD"}}, None])
def test_missing_or_invalid_sentiment_blocks_router_entry(signals, monkeypatch):
    from agents import graph

    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))
    result = graph._order_router_node({
        "execution_enabled": True, "current_positions": {},
        "sentiment_signals": signals,
        "trading_decisions": [{
            "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
            "holding_period": "INTRADAY", "ai_review_passed": True,
        }],
    })

    assert not result["executed_orders"]
    assert any("sentiment" in error.lower() for error in result["errors"])


def test_portfolio_manager_does_not_call_model_without_sentiment(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["sentiment_signals"] = {}
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Model must not run"))

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("sentiment" in error.lower() for error in result["errors"])


def test_debate_records_the_sentiment_reviewed(llm_database, monkeypatch):
    from agents.bull_bear_synthesis_agent import BullBearSynthesisAgent
    from agents.review_context import candidate_review_identity, technical_review_digest
    from agents.sentiment_evidence import sentiment_digest

    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}
    agent = BullBearSynthesisAgent()
    monkeypatch.setattr(agent, "run_debate", lambda *_args: {
        "review_complete": True, "action": "BUY", "conviction_score": 80,
    })

    result = agent.run(state)
    assert result["debate_results"]["TEST"]["sentiment_review_digest"] == sentiment_digest(
        state["sentiment_signals"]["TEST"]
    )
    assert result["debate_results"]["TEST"]["technical_review_digest"] == technical_review_digest(
        state["technical_signals"]["TEST"]
    )
    assert result["debate_results"]["TEST"]["candidate_id"] == candidate_review_identity(state, "TEST")


def test_debate_rejects_technical_signal_changed_during_review(llm_database, monkeypatch):
    from agents.bull_bear_synthesis_agent import BullBearSynthesisAgent

    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}
    agent = BullBearSynthesisAgent()

    def debate(*_args):
        state["technical_signals"]["TEST"]["reasoning"] = "Changed while debating"
        return {"review_complete": True, "action": "BUY", "conviction_score": 80}

    monkeypatch.setattr(agent, "run_debate", debate)
    result = agent.run(state)

    assert not result["debate_results"]
    assert any("technical" in error.lower() and "changed" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("strategy", [None, "UnknownStrategy"])
def test_debate_abstains_without_declared_strategy_contract(llm_database, monkeypatch, strategy):
    from agents.bull_bear_synthesis_agent import BullBearSynthesisAgent

    state = {"watchlist": ["TEST"], **sentiment_state(["TEST"])}
    state["technical_signals"]["TEST"]["strategy"] = strategy
    agent = BullBearSynthesisAgent()
    monkeypatch.setattr(agent, "run_debate", lambda *_args: pytest.fail("Unknown strategy must not invoke debate"))

    result = agent.run(state)

    assert not result["debate_results"]
    assert any("strategy" in error.lower() for error in result["errors"])


def test_candidate_identity_binds_scan_symbol_strategy_and_technical():
    from agents.review_context import candidate_review_identity
    from strategies.contracts import COMPOSITE_TREND_REVIEW

    state = sentiment_state(["TEST", "OTHER"])
    first = candidate_review_identity(state, "TEST")
    assert first != candidate_review_identity(state, "OTHER")
    state["llm_profile"]["run_id"] = "different-run"
    assert first != candidate_review_identity(state, "TEST")
    state["llm_profile"]["run_id"] = "test-run"
    state["technical_signals"]["TEST"]["signal"] = "SELL"
    assert first != candidate_review_identity(state, "TEST")
    assert {"technical", "sentiment", "bull", "bear", "debate_synthesis", "portfolio_manager"} == set(
        COMPOSITE_TREND_REVIEW.required
    )
    assert "fundamental" in COMPOSITE_TREND_REVIEW.optional
    assert {
        "verified_15m_history", "current_attributable_news", "passed_quantitative_screen",
        "risk_approval", "valid_portfolio_snapshot",
    } == set(COMPOSITE_TREND_REVIEW.required_inputs)


def test_portfolio_manager_rejects_sentiment_changed_after_debate(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["sentiment_signals"]["GOOD"]["reasoning"] = "Changed after debate"
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Model must not run"))

    result = agent.run(state)
    assert not result["trading_decisions"]
    assert any("does not match" in error for error in result["errors"])


def test_portfolio_manager_rejects_technical_signal_changed_after_debate(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["technical_signals"]["GOOD"]["reasoning"] = "Changed after debate"
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Changed input must not reach model"))

    result = agent.run(state)

    assert not result["trading_decisions"]
    assert any("technical" in error.lower() for error in result["errors"])


def test_portfolio_manager_rejects_candidate_identity_changed_after_debate(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["debate_results"]["GOOD"]["candidate_id"] = "different-candidate"
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Changed identity must not reach model"))

    result = agent.run(state)

    assert not result["trading_decisions"]
    assert any("candidate identity" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("changed", ["signal", "decision"])
def test_router_rejects_sentiment_changed_after_review(changed, monkeypatch):
    from agents import graph

    state = sentiment_state(["TEST"])
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "debate_results": {"TEST": reviewed_debate(state, "TEST", "BUY")},
        "trading_decisions": [decision],
    })
    if changed == "signal":
        state["sentiment_signals"]["TEST"]["reasoning"] = "Changed after review"
    else:
        decision["sentiment_review_digest"] = "wrong"
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)
    assert not result["executed_orders"]
    assert any("debate" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("conviction", [None, False, float("nan"), 69.99, 101])
def test_router_rechecks_debate_conviction_before_entry(conviction, monkeypatch):
    from agents import graph

    state = sentiment_state(["TEST"])
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    debate = reviewed_debate(state, "TEST", "BUY")
    debate["conviction_score"] = conviction
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "debate_results": {"TEST": debate}, "trading_decisions": [decision],
    })
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)

    assert not result["executed_orders"]
    assert any("debate" in error.lower() for error in result["errors"])


def test_router_rejects_debate_changed_after_portfolio_review(monkeypatch):
    from agents import graph

    state = sentiment_state(["TEST"])
    debate = reviewed_debate(state, "TEST", "BUY")
    state["debate_results"] = {"TEST": debate}
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    debate["reasoning"] = "Changed after portfolio review"
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "trading_decisions": [decision],
    })
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)

    assert not result["executed_orders"]
    assert any("debate" in error.lower() for error in result["errors"])


def test_router_rejects_technical_signal_changed_after_portfolio_review(monkeypatch):
    from agents import graph

    state = sentiment_state(["TEST"])
    state["debate_results"] = {"TEST": reviewed_debate(state, "TEST", "BUY")}
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    state["technical_signals"]["TEST"]["reasoning"] = "Changed after review"
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "trading_decisions": [decision],
    })
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)

    assert not result["executed_orders"]
    assert any("technical" in error.lower() for error in result["errors"])


def test_router_rejects_changed_candidate_identity(monkeypatch):
    from agents import graph

    state = sentiment_state(["TEST"])
    state["debate_results"] = {"TEST": reviewed_debate(state, "TEST", "BUY")}
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    decision["candidate_id"] = "different-candidate"
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "trading_decisions": [decision],
    })
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)

    assert not result["executed_orders"]
    assert any("candidate identity" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("technical", [None, [], "invalid"])
def test_router_does_not_block_exit_on_invalid_technical_input(technical, monkeypatch):
    from agents import graph

    calls = []
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    monkeypatch.setattr(graph, "place_order", lambda **kwargs: calls.append(kwargs) or {
        "symbol": kwargs["symbol"], "status": "EXECUTED",
    })
    result = graph._order_router_node({
        "execution_enabled": True, "current_positions": {"TEST": {"quantity": 2}},
        "technical_signals": technical,
        "trading_decisions": [{
            "symbol": "TEST", "action": "SELL", "quantity": 2, "price": 100,
            "holding_period": "INTRADAY",
        }],
    })
    assert len(calls) == 1 and calls[0]["is_exit"] is True
    assert result["executed_orders"][0]["is_exit"] is True
