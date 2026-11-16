"""Replay must consume captured inputs rather than today's providers."""

import pandas as pd
import pytest

from tests.test_llm_runtime import llm_database


def _forbid_provider(*_args, **_kwargs):
    raise AssertionError("A replay contacted a live provider")


def test_fundamental_inputs_are_frozen(llm_database, monkeypatch):
    from agents.fundamental_agent import FundamentalAgent
    from config.llm_runtime import llm_run, profile_for

    agent = FundamentalAgent()
    monkeypatch.setattr(agent, "_llm", None)
    monkeypatch.setattr(agent, "_fetch_fundamentals", lambda _symbol: {
        "pe_ratio": 12, "roe": 0.2, "debt_to_equity": 10,
    })
    recorded = agent.run({"watchlist": ["TEST"]})
    monkeypatch.setattr(agent, "_fetch_fundamentals", _forbid_provider)
    with llm_run("research_test", profile=profile_for("rules")):
        replayed = agent.run({
            "watchlist": ["TEST"], "research_mode": True,
            "research_fundamental_inputs": recorded["research_fundamental_inputs"],
        })
    assert recorded["fundamental_signals"] == {} and recorded["errors"]
    assert replayed["fundamental_signals"]["TEST"]["signal"] == "BUY"
    with pytest.raises(ValueError, match="fundamental"):
        agent.run({"watchlist": ["TEST"], "research_mode": True})


def test_options_inputs_are_frozen(monkeypatch):
    from agents import options_agent

    chain = pd.DataFrame([
        {"strike": 100, "option_type": "CE", "oi": 100, "ltp": 2, "symbol": "CE"},
        {"strike": 100, "option_type": "PE", "oi": 130, "ltp": 3, "symbol": "PE"},
    ])
    monkeypatch.setattr(options_agent, "fetch_options_chain", lambda _symbol: chain)
    state = {"live_quotes": {"NIFTY": {"ltp": 100}, "BANKNIFTY": {"ltp": 100}}}
    recorded = options_agent.OptionsAgent().run(state)
    monkeypatch.setattr(options_agent, "fetch_options_chain", _forbid_provider)
    replayed = options_agent.OptionsAgent().run({
        **state, "research_mode": True,
        "research_options_inputs": recorded["research_options_inputs"],
    })
    assert replayed["options_signals"] == recorded["options_signals"]
    with pytest.raises(ValueError, match="options"):
        options_agent.OptionsAgent().run({"research_mode": True})


def test_macro_inputs_are_frozen(monkeypatch):
    from agents.macro_agent import MacroAgent

    agent = MacroAgent()
    methods = {
        "_fetch_global_cues": {"sp500": {"change_pct": 1}},
        "_fetch_usdinr": {"rate": 80},
        "_fetch_crude_oil": {"change_pct": 0},
        "_fetch_fii_dii": [],
    }
    for name, result in methods.items():
        monkeypatch.setattr(agent, name, lambda result=result: result)
    recorded = agent.run({"india_vix": 15})
    for name in methods:
        monkeypatch.setattr(agent, name, _forbid_provider)
    replayed = agent.run({
        "india_vix": 15, "research_mode": True,
        "research_macro_inputs": recorded["research_macro_inputs"],
    })
    assert replayed["macro_signals"] == recorded["macro_signals"]
    with pytest.raises(ValueError, match="macro"):
        agent.run({"research_mode": True})


def test_india_inputs_and_expiry_are_frozen(monkeypatch):
    from agents.india_specific_agent import IndiaSpecificAgent

    agent = IndiaSpecificAgent()
    methods = {
        "_fetch_fo_ban_list": ["BANNED"],
        "_fetch_bulk_deals": [],
        "_fetch_block_deals": [],
        "_check_promoter_pledging": {},
        "_check_circuit_limits": {},
        "_is_expiry_day": True,
    }
    for name, result in methods.items():
        monkeypatch.setattr(agent, name, lambda *_args, result=result: result)
    recorded = agent.run({"watchlist": ["TEST"]})
    for name in methods:
        monkeypatch.setattr(agent, name, _forbid_provider)
    replayed = agent.run({
        "watchlist": ["TEST"], "research_mode": True,
        "research_india_inputs": recorded["research_india_inputs"],
    })
    assert replayed["india_signals"] == recorded["india_signals"]
    with pytest.raises(ValueError, match="India"):
        agent.run({"research_mode": True})


def test_comparison_profiles_do_not_override_dashboard_selection(llm_database, monkeypatch):
    from agents.llm_client import _use_universal_factory
    from config import llm_runtime

    before = llm_runtime.selected_profile()
    monkeypatch.setattr(llm_runtime.settings, "trademax_llm_provider", "openai")
    with llm_runtime.llm_run("research", profile=llm_runtime.profile_for("economy")):
        assert llm_runtime.current_profile().mode == "economy"
        assert not _use_universal_factory("decision")
    with llm_runtime.llm_run("research", profile=llm_runtime.profile_for("rules")):
        assert llm_runtime.current_profile().mode == "rules"
        assert not _use_universal_factory("decision")
    assert llm_runtime.selected_profile() == before
