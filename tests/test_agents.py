"""Tests for agents base state and agent creation."""

from datetime import datetime

import pytest

from agents.base_agent import TradingState, create_initial_state
from tests.test_llm_runtime import llm_database
from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state


def test_create_initial_state():
    """create_initial_state should return a valid TradingState."""
    state = create_initial_state(["RELIANCE", "TCS"])

    assert state["watchlist"] == ["RELIANCE", "TCS"]
    assert state["agent_logs"] == []
    assert state["errors"] == []
    assert state["market_data"] == {}
    assert state["technical_signals"] == {}
    assert state["trading_decisions"] == []


def test_initial_state_has_all_keys():
    """State dict should contain all required keys."""
    state = create_initial_state([])
    required_keys = [
        "watchlist", "market_data", "technical_signals", "fundamental_signals",
        "sentiment_signals", "options_signals", "macro_signals",
        "india_signals", "risk_assessment", "trading_decisions",
        "executed_orders", "current_positions", "review_contract_results",
        "agent_logs", "errors",
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"


def test_screener_evidence_survives_langgraph_state():
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(TradingState)
    workflow.add_node("screen", lambda _state: {
        "watchlist": ["GOOD"], "screener_results": {"qualified": ["GOOD"], "rejected": []},
    })
    workflow.set_entry_point("screen")
    workflow.add_edge("screen", END)
    result = workflow.compile().invoke(create_initial_state(["GOOD"]))
    assert result["screener_results"] == {"qualified": ["GOOD"], "rejected": []}


@pytest.fixture
def portfolio_manager_state(llm_database, monkeypatch):
    from agents import portfolio_manager_agent as module

    class MarketTime(datetime):
        @classmethod
        def now(cls, tz=None):
            midday = datetime(2026, 11, 12, 11, 0)
            return tz.localize(midday) if tz else midday

    monkeypatch.setattr(module, "datetime", MarketTime)
    state = create_initial_state(["GOOD"])
    state["risk_assessment"] = {"GOOD": {"approved": True, "deterministic_action": "BUY"}}
    state["technical_signals"] = {"GOOD": {"signal": "BUY"}}
    state.update(sentiment_state(["GOOD", "BANNED", "LATE"]))
    state["debate_results"] = {"GOOD": reviewed_debate(state, "GOOD", "BUY")}
    return module.PortfolioManagerAgent(), state


def test_portfolio_manager_rejects_malformed_decisions_without_losing_valid_ones(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    valid = {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}
    decisions = [valid, {"symbol": "MISSING_ACTION"}, {"action": "SELL"}, None, "BUY"]
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: decisions)

    result = agent.run(state)

    assert result["trading_decisions"] == [reviewed_decision(state, valid)]
    assert len(result["errors"]) == 4
    assert all("Invalid" in error for error in result["errors"])


def test_portfolio_manager_keeps_fo_ban_filter(portfolio_manager_state, monkeypatch):
    agent, state = portfolio_manager_state
    valid = {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}
    banned = {"symbol": "BANNED", "action": "BUY", "quantity": 1, "price": 100}
    state["india_signals"] = {"fo_ban_list": ["BANNED"]}
    state["risk_assessment"]["BANNED"] = {"approved": True, "deterministic_action": "BUY"}
    state["debate_results"]["BANNED"] = reviewed_debate(state, "BANNED", "BUY")
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [valid, banned])

    result = agent.run(state)

    assert result["trading_decisions"] == [reviewed_decision(state, valid)]
    assert result["errors"] == []


@pytest.mark.parametrize("decisions", [None, "not a list", {"symbol": "GOOD", "action": "BUY"}])
def test_portfolio_manager_rejects_non_list_decision_response(
    portfolio_manager_state, monkeypatch, decisions,
):
    agent, state = portfolio_manager_state
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: decisions)

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("invalid" in error.lower() for error in result["errors"])


def test_portfolio_manager_blocks_when_required_model_fails(portfolio_manager_state, monkeypatch):
    agent, state = portfolio_manager_state
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: None)

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("blocked" in error.lower() for error in result["errors"])


def test_portfolio_manager_requires_complete_aligned_debate(portfolio_manager_state, monkeypatch):
    agent, state = portfolio_manager_state
    state["debate_results"]["GOOD"]["review_complete"] = False
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Model must not be called"))

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("debate" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("conviction", [None, False, float("nan"), -1, 69.99, 101])
def test_portfolio_manager_blocks_invalid_or_low_debate_conviction(
    portfolio_manager_state, monkeypatch, conviction,
):
    agent, state = portfolio_manager_state
    state["debate_results"]["GOOD"]["conviction_score"] = conviction
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: pytest.fail("Low conviction must not reach model"))

    result = agent.run(state)

    assert result["trading_decisions"] == []


@pytest.mark.parametrize("conviction, valid", [(69.99, False), (70, True), (100, True), (100.01, False)])
def test_debate_conviction_boundary_is_explicit(conviction, valid):
    import hashlib
    from agents.review_context import valid_entry_debate

    context_json = "{}"
    debate = {
        "review_complete": True, "action": "BUY", "conviction_score": conviction,
        "reasoning": "Both sides considered.",
        "arguments": {
            "bull": {"_review_succeeded": True, "thesis": "Bull argument."},
            "bear": {"_review_succeeded": True, "counter_thesis": "Bear argument."},
        },
        "debate_context_sha256": hashlib.sha256(context_json.encode("utf-8")).hexdigest(),
        "debate_context_json": context_json,
    }
    assert valid_entry_debate(debate, "BUY") is valid


@pytest.mark.parametrize("reasoning", [None, "", "   ", 1])
def test_debate_requires_nonempty_reasoning(reasoning):
    from agents.review_context import valid_entry_debate

    assert not valid_entry_debate({
        "review_complete": True, "action": "BUY", "conviction_score": 80,
        "reasoning": reasoning,
    }, "BUY")


def test_portfolio_manager_rejects_debate_changed_during_review(portfolio_manager_state, monkeypatch):
    agent, state = portfolio_manager_state

    def decide(*_args):
        state["debate_results"]["GOOD"]["reasoning"] = "Changed while model was running"
        return [{"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}]

    monkeypatch.setattr(agent, "_llm_decide", decide)
    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("changed" in error.lower() for error in result["errors"])


def test_portfolio_manager_rejects_technical_signal_changed_during_review(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state

    def decide(*_args):
        state["technical_signals"]["GOOD"]["reasoning"] = "Changed while model was running"
        return [{"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}]

    monkeypatch.setattr(agent, "_llm_decide", decide)
    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("technical" in error.lower() and "changed" in error.lower() for error in result["errors"])


def test_debate_marks_incomplete_review_as_non_actionable(llm_database, monkeypatch):
    from agents.bull_bear_synthesis_agent import BullBearSynthesisAgent

    agent = BullBearSynthesisAgent()
    monkeypatch.setattr(agent, "_run_bull_researcher", lambda *_args: {"_review_succeeded": False})
    monkeypatch.setattr(agent, "_run_bear_researcher", lambda *_args: {"_review_succeeded": True})
    monkeypatch.setattr(agent, "_synthesize_debate", lambda *_args: {
        "_review_succeeded": True, "action": "BUY", "conviction_score": 90,
    })

    result = agent.run_debate("GOOD", {}, {})

    assert result["review_complete"] is False
    assert result["action"] == "HOLD"
    assert result["conviction_score"] == 0


def test_debate_marks_complete_review_only_after_all_three_models(llm_database, monkeypatch):
    from agents.bull_bear_synthesis_agent import BullBearSynthesisAgent

    agent = BullBearSynthesisAgent()
    monkeypatch.setattr(agent, "_run_bull_researcher", lambda *_args: {"_review_succeeded": True})
    monkeypatch.setattr(agent, "_run_bear_researcher", lambda *_args: {"_review_succeeded": True})
    monkeypatch.setattr(agent, "_synthesize_debate", lambda *_args: {
        "_review_succeeded": True, "action": "BUY", "conviction_score": 90,
    })

    result = agent.run_debate("GOOD", {}, {})

    assert result["review_complete"] is True
    assert result["action"] == "BUY"


def test_debate_preserves_complete_evidence_and_arguments(llm_database, monkeypatch):
    import hashlib
    import json
    from agents import bull_bear_synthesis_agent as module

    calls = []
    bull = {
        "thesis": "x" * 2500 + "BULL_END", "bull_confidence": 80, "probability": 0.6,
        "supporting_factors": ["support"], "key_catalysts": ["catalyst"],
    }
    bear = {
        "counter_thesis": "y" * 2500 + "BEAR_END", "bear_confidence": 80, "probability": 0.6,
        "risk_factors": ["risk"], "bull_case_weaknesses": ["weakness"],
    }
    verdict = {"action": "BUY", "conviction_score": 80, "reasoning": "reviewed"}
    replies = iter([bull, bear, verdict])
    monkeypatch.setattr(module, "_call_llm", lambda _role, prompt: calls.append(prompt) or json.dumps(next(replies)))
    state = create_initial_state(["GOOD"])
    reports = {"technical": {"signal": "BUY", "reasoning": "z" * 3500 + "REPORT_END"}}

    agent = module.BullBearSynthesisAgent()
    context = agent._debate_context("GOOD", reports, state)
    result = agent.run_debate("GOOD", reports, state)
    assert result["review_complete"]
    assert result["arguments"]["bull"]["thesis"] == bull["thesis"]
    assert result["arguments"]["bear"]["counter_thesis"] == bear["counter_thesis"]
    assert result["debate_context_json"] == context
    assert result["debate_context_sha256"] == hashlib.sha256(context.encode("utf-8")).hexdigest()
    assert context in calls[0] and context in calls[1]
    assert "REPORT_END" in calls[0] and "REPORT_END" in calls[1]
    assert "BULL_END" in calls[1] and "BULL_END" in calls[2]
    assert "BEAR_END" in calls[2]


def test_debate_abstains_when_optional_context_changes_before_bear(llm_database, monkeypatch):
    import json
    from agents import bull_bear_synthesis_agent as module

    state = create_initial_state(["GOOD"])
    state.update(sentiment_state(["GOOD"]))
    state["fundamental_signals"] = {"GOOD": {"reasoning": "original"}}
    calls = []

    def bull(_role, _prompt):
        calls.append("bull")
        state["fundamental_signals"]["GOOD"]["reasoning"] = "changed"
        return json.dumps({
            "thesis": "Bull", "bull_confidence": 80, "probability": 0.6,
            "supporting_factors": [], "key_catalysts": [],
        })

    monkeypatch.setattr(module, "_call_llm", bull)
    result = module.BullBearSynthesisAgent().run(state)

    assert calls == ["bull"]
    assert not result["debate_results"]
    assert any("context changed" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("part", ["bull", "bear", "context_hash"])
def test_debate_requires_both_arguments_and_context_hash(part):
    from agents.review_context import valid_entry_debate
    from tests.sentiment_fixtures import reviewed_debate

    state = sentiment_state(["GOOD"])
    debate = reviewed_debate(state, "GOOD", "BUY")
    if part == "context_hash":
        debate["debate_context_sha256"] = "not-a-hash"
    else:
        debate["arguments"][part] = {}

    assert not valid_entry_debate(debate, "BUY")


def test_debate_rejects_oversized_evidence_before_a_model_call(llm_database, monkeypatch):
    from agents import bull_bear_synthesis_agent as module

    monkeypatch.setattr(module, "_call_llm", lambda *_args: pytest.fail("Oversized debate must not be sent"))
    state = create_initial_state(["GOOD"])
    state.update(sentiment_state(["GOOD"]))
    state["technical_signals"] = {"GOOD": {"signal": "BUY", "reasoning": "x" * 100_000}}

    result = module.BullBearSynthesisAgent().run(state)
    assert "GOOD" not in result["debate_results"]
    assert any("context" in error.lower() for error in result["errors"])


def test_debate_rejects_oversized_bull_argument_before_bear_call(llm_database, monkeypatch):
    import json
    from agents import bull_bear_synthesis_agent as module

    calls = []
    bull = {
        "thesis": "x" * 20_000, "bull_confidence": 80, "probability": 0.6,
        "supporting_factors": ["support"], "key_catalysts": ["catalyst"],
    }
    monkeypatch.setattr(module, "_call_llm", lambda _role, prompt: calls.append(prompt) or json.dumps(bull))
    state = create_initial_state(["GOOD"])
    state["technical_signals"] = {"GOOD": {"signal": "BUY"}}
    state.update(sentiment_state(["GOOD"]))

    result = module.BullBearSynthesisAgent().run(state)
    assert len(calls) == 1
    assert "GOOD" not in result["debate_results"]
    assert any("context" in error.lower() for error in result["errors"])


def test_portfolio_manager_reports_missing_decisions_field(portfolio_manager_state, monkeypatch):
    from types import SimpleNamespace

    agent, state = portfolio_manager_state

    class FakeLLM:
        def invoke(self, _prompt):
            return SimpleNamespace(content='{"market_view": "NEUTRAL"}')

    monkeypatch.setattr(agent, "_llm", FakeLLM())
    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("Invalid" in error for error in result["errors"])


def test_portfolio_prompt_contains_complete_evidence_for_every_approved_candidate(
    portfolio_manager_state, monkeypatch,
):
    import json
    from types import SimpleNamespace

    agent, state = portfolio_manager_state
    state["risk_assessment"]["LATE"] = {"approved": True, "deterministic_action": "BUY"}
    state["technical_signals"]["UNRELATED"] = {"reasoning": "x" * 5000}
    state["technical_signals"]["LATE"]["reasoning"] = "late technical evidence"
    state["fundamental_signals"]["LATE"] = {"signal": "HOLD", "reasoning": "fundamental evidence"}
    state["sentiment_signals"]["LATE"].update(signal="SELL", reasoning="sentiment evidence")
    state["debate_results"]["LATE"] = {
        **reviewed_debate(state, "LATE", "BUY"), "winning_argument": "bull",
    }
    prompts = []

    class FakeLLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content='{"decisions": []}')

    monkeypatch.setattr(agent, "_llm", FakeLLM())
    assert agent.run(state)["trading_decisions"] == []
    assert len(prompts) == 1
    context_text = prompts[0].split("REVIEW_CONTEXT_BEGIN\n", 1)[1].split("\nREVIEW_CONTEXT_END", 1)[0]
    context = json.loads(context_text)
    assert set(context["candidates"]) == {"GOOD", "LATE"}
    assert context["candidates"]["GOOD"]["fundamental"] is None
    late = context["candidates"]["LATE"]
    assert late["technical"]["reasoning"] == "late technical evidence"
    assert late["fundamental"]["reasoning"] == "fundamental evidence"
    assert late["sentiment"]["reasoning"] == "sentiment evidence"
    assert late["debate"]["winning_argument"] == "bull"
    assert late["risk"]["approved"] is True
    assert "UNRELATED" not in context["candidates"]


def test_portfolio_manager_blocks_oversized_context_instead_of_truncating(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["technical_signals"]["GOOD"]["reasoning"] = "x" * 100_000

    class FakeLLM:
        def invoke(self, _prompt):
            pytest.fail("Oversized review must not be sent")

    monkeypatch.setattr(agent, "_llm", FakeLLM())
    result = agent.run(state)
    assert result["trading_decisions"] == []
    assert any("blocked" in error.lower() for error in result["errors"])


def test_portfolio_manager_blocks_nonfinite_review_evidence(portfolio_manager_state, monkeypatch):
    agent, state = portfolio_manager_state
    state["technical_signals"]["GOOD"]["indicators"] = {"rsi": float("nan")}

    class FakeLLM:
        def invoke(self, _prompt):
            pytest.fail("Invalid review JSON must not be sent")

    monkeypatch.setattr(agent, "_llm", FakeLLM())
    result = agent.run(state)
    assert result["trading_decisions"] == []
    assert any("valid JSON" in error for error in result["errors"])


def test_review_context_preserves_supported_scalars_and_rejects_dataframe():
    import json
    import numpy as np
    import pandas as pd
    from agents.review_context import ReviewContextError, serialize_review_context

    text = serialize_review_context(
        {"count": np.int64(2), "value": np.float32(1.5), "as_of": datetime(2026, 11, 12, 11)},
        max_bytes=1000,
    )
    assert json.loads(text) == {"count": 2, "value": 1.5, "as_of": "2026-11-12T11:00:00"}
    with pytest.raises(ReviewContextError, match="valid JSON"):
        serialize_review_context({"data": pd.DataFrame({"value": [1, 2]})}, max_bytes=1000)
    with pytest.raises(ReviewContextError, match="valid JSON"):
        serialize_review_context({"as_of": pd.NaT}, max_bytes=1000)
