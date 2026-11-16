"""Durable candidate review records for the paper entry path."""

import json

import pytest

from tests.test_agents import portfolio_manager_state
from tests.test_llm_runtime import llm_database


def test_portfolio_review_persists_candidate_and_complete_context(portfolio_manager_state, monkeypatch):
    from database.db import get_db
    from database.models import CandidateReview

    agent, state = portfolio_manager_state
    decision = {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [decision])

    result = agent.run(state)

    assert len(result["trading_decisions"]) == 1
    accepted = result["trading_decisions"][0]
    with get_db() as db:
        row = db.get(CandidateReview, accepted["candidate_id"])
        assert row is not None
        assert row.run_id == "test-run" and row.symbol == "GOOD"
        assert row.contract_version == "review-v2"
        context = json.loads(row.portfolio_context_json)
        assert context["candidates"]["GOOD"]["review_contract"] == {
            "strategy": "CompositeTrend",
            "version": "review-v2",
            "required_agents": [
                "technical", "sentiment", "bull", "bear", "debate_synthesis", "portfolio_manager",
            ],
            "optional_agents": ["fundamental", "options", "macro", "india_specific", "geopolitical"],
            "required_inputs": [
                "verified_15m_history", "current_attributable_news", "passed_quantitative_screen",
                "risk_approval", "valid_portfolio_snapshot",
            ],
        }
        assert context["candidates"]["GOOD"]["technical"] == state["technical_signals"]["GOOD"]
        debate = state["debate_results"]["GOOD"]
        assert context["candidates"]["GOOD"]["debate"] == {
            key: value for key, value in debate.items() if key != "debate_context_json"
        }
        assert context["candidates"]["GOOD"]["debate"]["arguments"]["bull"]["thesis"] == "Bull argument."
        assert context["candidates"]["GOOD"]["debate"]["arguments"]["bear"]["counter_thesis"] == "Bear argument."
        assert json.loads(row.decision_json) == accepted
        assert json.loads(row.decision_json)["debate_context_json"] == debate["debate_context_json"]


def test_portfolio_review_abstains_when_evidence_cannot_be_persisted(portfolio_manager_state, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    from operations import candidate_reviews

    agent, state = portfolio_manager_state
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [
        {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100},
    ])
    def fail(*_args):
        raise SQLAlchemyError("database unavailable")
    monkeypatch.setattr(candidate_reviews, "persist_candidate_reviews", fail)

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("persist" in error.lower() for error in result["errors"])


def test_portfolio_review_abstains_when_optional_context_changes_during_model_call(
    portfolio_manager_state, monkeypatch,
):
    agent, state = portfolio_manager_state
    state["fundamental_signals"] = {"GOOD": {"signal": "HOLD", "reasoning": "original"}}
    def decide(*_args):
        state["fundamental_signals"]["GOOD"]["reasoning"] = "changed"
        return [{"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}]
    monkeypatch.setattr(agent, "_llm_decide", decide)

    result = agent.run(state)

    assert result["trading_decisions"] == []
    assert any("context changed" in error.lower() for error in result["errors"])


def test_stored_context_is_exactly_what_model_received(portfolio_manager_state, monkeypatch):
    from types import SimpleNamespace
    from database.db import get_db
    from database.models import CandidateReview

    agent, state = portfolio_manager_state
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content='{"decisions":[{"symbol":"GOOD","action":"BUY","quantity":1,"price":100}]}')

    monkeypatch.setattr(agent, "_llm", Model())
    result = agent.run(state)
    assert len(result["trading_decisions"]) == 1
    context = prompts[0].split("REVIEW_CONTEXT_BEGIN\n", 1)[1].split("\nREVIEW_CONTEXT_END", 1)[0]
    with get_db() as db:
        row = db.get(CandidateReview, result["trading_decisions"][0]["candidate_id"])
        assert row.portfolio_context_json == context


def test_actual_debate_context_and_arguments_are_durable(portfolio_manager_state, monkeypatch):
    from agents import bull_bear_synthesis_agent as debate_module
    from database.db import get_db
    from database.models import CandidateReview

    agent, state = portfolio_manager_state
    responses = iter([
        {"thesis": "Bull case", "bull_confidence": 80, "probability": 0.6,
         "supporting_factors": [], "key_catalysts": []},
        {"counter_thesis": "Bear case", "bear_confidence": 70, "probability": 0.5,
         "risk_factors": [], "bull_case_weaknesses": []},
        {"action": "BUY", "conviction_score": 80, "reasoning": "Both cases considered."},
    ])
    monkeypatch.setattr(debate_module, "_call_llm", lambda *_args: json.dumps(next(responses)))
    debate = debate_module.BullBearSynthesisAgent().run(state)
    state["debate_results"] = debate["debate_results"]
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [
        {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100},
    ])

    decision = agent.run(state)["trading_decisions"][0]
    with get_db() as db:
        row = db.get(CandidateReview, decision["candidate_id"])
        stored_decision = json.loads(row.decision_json)
        stored_portfolio_context = json.loads(row.portfolio_context_json)
        debate_context = json.loads(stored_decision["debate_context_json"])
        stored_debate = stored_portfolio_context["candidates"]["GOOD"]["debate"]
        assert debate_context["reports"]["technical"] == state["technical_signals"]["GOOD"]
        assert debate_context["reports"]["sentiment"] == state["sentiment_signals"]["GOOD"]
        assert stored_debate["arguments"]["bull"]["thesis"] == "Bull case"
        assert stored_debate["arguments"]["bear"]["counter_thesis"] == "Bear case"


def test_conflicting_repeat_candidate_abstains_without_overwriting(portfolio_manager_state, monkeypatch):
    from database.db import get_db
    from database.models import CandidateReview

    agent, state = portfolio_manager_state
    decision = {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100}
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [decision])
    first = agent.run(state)
    assert len(first["trading_decisions"]) == 1
    decision["quantity"] = 2
    second = agent.run(state)

    assert second["trading_decisions"] == []
    assert any("persistence" in error.lower() for error in second["errors"])
    with get_db() as db:
        row = db.get(CandidateReview, first["trading_decisions"][0]["candidate_id"])
        assert json.loads(row.decision_json)["quantity"] == 1


def test_candidate_review_batch_rolls_back_on_conflict(portfolio_manager_state, monkeypatch):
    from database.db import get_db
    from database.models import CandidateReview
    from operations.candidate_reviews import persist_candidate_reviews

    agent, state = portfolio_manager_state
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [
        {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100},
    ])
    original = agent.run(state)["trading_decisions"][0]
    with get_db() as db:
        original_context = db.get(CandidateReview, original["candidate_id"]).portfolio_context_json
    new = {**original, "candidate_id": "a" * 64, "symbol": "GOOD"}
    conflict = {**original, "quantity": 2}

    with pytest.raises(ValueError, match="conflicts"):
        persist_candidate_reviews(state, [new, conflict], original_context)
    with get_db() as db:
        assert db.get(CandidateReview, new["candidate_id"]) is None


@pytest.mark.parametrize("change", ["optional", "risk", "decision", "checksum", "missing", "database"])
def test_router_blocks_changed_or_unavailable_persisted_review(portfolio_manager_state, monkeypatch, change):
    from contextlib import contextmanager
    from sqlalchemy.exc import SQLAlchemyError
    from agents import graph
    from database.db import get_db
    from database.models import CandidateReview
    from operations import candidate_reviews

    agent, state = portfolio_manager_state
    state["fundamental_signals"] = {"GOOD": {"reasoning": "original"}}
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [
        {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100},
    ])
    decision = agent.run(state)["trading_decisions"][0]
    state.update({
        "execution_enabled": True, "watchlist": ["GOOD"],
        "screener_results": {"qualified": ["GOOD"], "rejected": []},
        "trading_decisions": [decision],
    })
    if change == "optional":
        state["fundamental_signals"]["GOOD"]["reasoning"] = "changed"
    elif change == "risk":
        state["risk_assessment"]["GOOD"]["max_quantity"] = 2
    elif change == "decision":
        decision["quantity"] = 2
    elif change in {"checksum", "missing"}:
        with get_db() as db:
            row = db.get(CandidateReview, decision["candidate_id"])
            if change == "checksum":
                row.portfolio_context_sha256 = "0" * 64
            else:
                db.delete(row)
    else:
        @contextmanager
        def unavailable():
            raise SQLAlchemyError("database unavailable")
            yield
        monkeypatch.setattr(candidate_reviews, "get_db", unavailable)
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))

    result = graph._order_router_node(state)

    assert result["executed_orders"] == []
    assert any("stored candidate review" in error.lower() for error in result["errors"])


def test_router_accepts_matching_review_before_operational_gate(portfolio_manager_state, monkeypatch):
    from types import SimpleNamespace
    from agents import graph
    from operations import control_state
    from risk import dead_mans_switch

    agent, state = portfolio_manager_state
    monkeypatch.setattr(agent, "_llm_decide", lambda *_args: [
        {"symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100},
    ])
    state.update({
        "execution_enabled": True, "watchlist": ["GOOD"],
        "screener_results": {"qualified": ["GOOD"], "rejected": []},
    })
    state["trading_decisions"] = agent.run(state)["trading_decisions"]
    monkeypatch.setattr(control_state, "control_state", SimpleNamespace(paused=lambda: True))
    monkeypatch.setattr(dead_mans_switch, "dead_mans_switch", SimpleNamespace(is_halted=False))
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Paused entry submitted"))

    result = graph._order_router_node(state)

    assert not result["executed_orders"]
    assert any("operational safety controls" in error.lower() for error in result["errors"])


def _review_contract_state(state):
    from tests.sentiment_fixtures import reviewed_decision

    state.update({
        "execution_enabled": True,
        "watchlist": ["GOOD"],
        "screener_results": {"qualified": ["GOOD"], "rejected": []},
        "portfolio_snapshot_valid": True,
    })
    state["trading_decisions"] = [reviewed_decision(state, {
        "symbol": "GOOD", "action": "BUY", "quantity": 1, "price": 100,
    })]
    return state


def test_review_contract_report_passes_with_required_evidence_and_optional_agents_absent(
    portfolio_manager_state,
):
    from agents.review_context import strategy_review_contract_report

    _, state = portfolio_manager_state
    state = _review_contract_state(state)

    report = strategy_review_contract_report(state, "GOOD", state["trading_decisions"][0])

    assert report["passed"] is True
    assert all(report["required_agents"].values())
    assert all(report["required_inputs"].values())
    assert "fundamental" in report["optional_agents"]


@pytest.mark.parametrize(
    "requirement",
    [
        "technical", "sentiment", "bull", "bear", "debate_synthesis", "portfolio_manager",
        "verified_15m_history", "current_attributable_news", "passed_quantitative_screen",
        "risk_approval", "valid_portfolio_snapshot",
    ],
)
def test_review_contract_report_identifies_each_missing_required_item(
    portfolio_manager_state, requirement,
):
    from agents.review_context import strategy_review_contract_report

    _, state = portfolio_manager_state
    state = _review_contract_state(state)
    decision = state["trading_decisions"][0]
    if requirement == "technical":
        state["technical_signals"]["GOOD"]["signal"] = "INVALID"
    elif requirement == "sentiment":
        state["sentiment_signals"].pop("GOOD")
    elif requirement == "bull":
        state["debate_results"]["GOOD"]["arguments"]["bull"]["_review_succeeded"] = False
    elif requirement == "bear":
        state["debate_results"]["GOOD"]["arguments"]["bear"]["_review_succeeded"] = False
    elif requirement == "debate_synthesis":
        state["debate_results"]["GOOD"]["review_complete"] = False
    elif requirement == "portfolio_manager":
        decision["ai_review_passed"] = False
    elif requirement == "verified_15m_history":
        state["technical_signals"]["GOOD"]["history_provenance"]["bars_sha256"] = "invalid"
    elif requirement == "current_attributable_news":
        state["news_snapshot"]["by_symbol"]["GOOD"] = []
    elif requirement == "passed_quantitative_screen":
        state["screener_results"]["qualified"] = []
    elif requirement == "risk_approval":
        state["risk_assessment"]["GOOD"]["approved"] = False
    else:
        state["portfolio_snapshot_valid"] = False

    report = strategy_review_contract_report(state, "GOOD", decision)

    assert report["passed"] is False
    assert requirement in report["failed_requirements"]


def test_router_records_contract_failure_and_exit_skips_entry_contract(
    portfolio_manager_state, monkeypatch,
):
    from agents import graph

    _, state = portfolio_manager_state
    state = _review_contract_state(state)
    state["screener_results"]["qualified"] = []
    entry_result = graph._order_router_node(state)

    assert not entry_result["executed_orders"]
    assert "passed_quantitative_screen" in entry_result["review_contract_results"]["GOOD"]["failed_requirements"]

    state["current_positions"] = {
        "GOOD": {"quantity": 1, "entry_trade_ids": [1], "holding_period": "DELIVERY"},
    }
    state["trading_decisions"] = [{
        "symbol": "GOOD", "action": "SELL", "quantity": 1, "price": 100,
        "holding_period": "DELIVERY",
    }]
    monkeypatch.setattr(
        graph, "place_order",
        lambda **_kwargs: {"symbol": "GOOD", "action": "SELL", "quantity": 1, "fill_price": 100},
    )
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    from risk.stop_loss_manager import stop_loss_manager
    monkeypatch.setattr(stop_loss_manager, "pending_exits", lambda: {})
    monkeypatch.setattr(
        "agents.review_context.strategy_review_contract_report",
        lambda *_args: pytest.fail("Exit must not require an entry review contract"),
    )

    exit_result = graph._order_router_node(state)

    assert len(exit_result["executed_orders"]) == 1
    assert exit_result["review_contract_results"] == {}
