"""Fundamental model failures must not become fabricated analyst signals."""

import pytest

from tests.test_llm_runtime import llm_database


class _Model:
    def __init__(self, response):
        self.response = response

    def invoke(self, _prompt):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize("response", [
    RuntimeError("provider unavailable"),
    "not json",
    '{"signal":"BUY","confidence":101,"reasoning":"x","key_risks":[]}',
    '{"signal":"BUY","confidence":60,"reasoning":"","key_risks":[]}',
    '{"signal":"BUY","confidence":60,"reasoning":"x","key_risks":"unknown"}',
])
def test_fundamental_model_failure_is_not_rules_fallback(llm_database, monkeypatch, response):
    from agents.fundamental_agent import FundamentalAgent

    agent = FundamentalAgent()
    monkeypatch.setattr(agent, "_llm", _Model(response))
    monkeypatch.setattr(agent, "_fetch_fundamentals", lambda _symbol: {"pe_ratio": 12, "roe": 0.2})
    result = agent.run({"watchlist": ["TEST"]})
    assert "TEST" not in result["fundamental_signals"]
    assert result["errors"]


def test_missing_fundamental_model_or_metrics_is_reported(llm_database, monkeypatch):
    from agents.fundamental_agent import FundamentalAgent

    agent = FundamentalAgent()
    monkeypatch.setattr(agent, "_llm", None)
    monkeypatch.setattr(agent, "_fetch_fundamentals", lambda _symbol: {"pe_ratio": 12})
    result = agent.run({"watchlist": ["TEST"]})
    assert result["fundamental_signals"] == {} and result["errors"]
    monkeypatch.setattr(agent, "_llm", _Model('{"signal":"HOLD","confidence":30,"reasoning":"x","key_risks":[]}'))
    monkeypatch.setattr(agent, "_fetch_fundamentals", lambda _symbol: {"pe_ratio": None})
    result = agent.run({"watchlist": ["TEST"]})
    assert result["fundamental_signals"] == {} and result["errors"]


def test_valid_fundamental_model_response_is_preserved(llm_database, monkeypatch):
    from agents.fundamental_agent import FundamentalAgent

    agent = FundamentalAgent()
    monkeypatch.setattr(agent, "_llm", _Model(
        '{"signal":"HOLD","confidence":40,"reasoning":"Mixed data","key_risks":["Debt"]}'
    ))
    monkeypatch.setattr(agent, "_fetch_fundamentals", lambda _symbol: {"pe_ratio": 12})
    result = agent.run({"watchlist": ["TEST"]})
    assert result["fundamental_signals"]["TEST"]["signal"] == "HOLD"
    assert not result["errors"]
