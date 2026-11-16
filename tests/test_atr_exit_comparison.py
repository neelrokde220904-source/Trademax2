from copy import deepcopy

import pandas as pd
import pytest

from tests.test_llm_runtime import llm_database
from tests.test_research_comparison import _snapshot, _forbid


def test_atr_study_uses_captured_inputs_without_models_or_broker_orders(llm_database, monkeypatch):
    from agents import llm_client
    from broker import order_manager
    from research import atr_exit_comparison as study, tracking
    from risk.exit_policy import BASELINE_ATR_POLICY, RESEARCH_ATR_POLICY

    monkeypatch.setattr(study.settings, "paper_trading", True)
    monkeypatch.setattr(study.settings, "min_net_reward_risk", 1.0)
    monkeypatch.setattr(llm_client, "complete", _forbid)
    monkeypatch.setattr(llm_client, "_claude_request", _forbid)
    monkeypatch.setattr(llm_client, "_deepseek_request", _forbid)
    monkeypatch.setattr(order_manager, "place_order", _forbid)
    signal = {"TEST": {
        "signal": "BUY", "strategy": "CompositeTrend",
        "indicators": {"close": 100, "daily_atr_14": 2},
        "stop_loss": 98, "target": 105,
    }}
    seen = []

    def analyze(state):
        seen.append(state["research_as_of"])
        return {"technical_signals": deepcopy(signal), "errors": []}

    monkeypatch.setattr(study, "technical_agent", analyze)
    bars = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000,
    }]
    for identifier, when, rows in [
        ("first", "2026-06-05T10:31:00+05:30", bars),
        ("second", "2026-06-05T11:16:00+05:30", bars + [{
            "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
            "open": 100.0, "high": 106.0, "low": 99.0, "close": 105.0, "volume": 1000,
        }, {
            "datetime": pd.Timestamp("2026-06-05T11:00:00+05:30"),
            "open": 105.0, "high": 106.0, "low": 103.0, "close": 104.0, "volume": 1000,
        }]),
    ]:
        state, observed = _snapshot(when, rows)
        tracking.record_scan(state, identifier, observed_at=observed)
    result = study.run_atr_exit_comparison(["second", "first"])
    variants = {item["mode"]: item for item in result["variants"]}
    assert len(seen) == 2
    assert result["verdict"] == "inconclusive_not_for_promotion"
    baseline = variants[BASELINE_ATR_POLICY.name]
    candidate = variants[RESEARCH_ATR_POLICY.name]
    assert len(baseline["closed_trades"]) == 1
    assert baseline["closed_trades"][0]["exit_reason"] == "TARGET"
    assert len(candidate["open_positions"]) == 1
    position = candidate["open_positions"][0]
    assert position["atr"] == 2
    assert position["target"] - position["entry_price"] == pytest.approx(8)
    assert position["stop_loss"] == pytest.approx(102)
    assert baseline["closed_trades"][0]["costs"] > 0
    assert signal["TEST"]["stop_loss"] == 98


def test_atr_study_rejects_live_mode_and_duplicate_snapshots(monkeypatch):
    from research import atr_exit_comparison as study

    monkeypatch.setattr(study, "load_snapshot", _forbid)
    monkeypatch.setattr(study.settings, "paper_trading", False)
    with pytest.raises(ValueError, match="paper-trading"):
        study.run_atr_exit_comparison(["first", "second"])
    monkeypatch.setattr(study.settings, "paper_trading", True)
    with pytest.raises(ValueError, match="unique"):
        study.run_atr_exit_comparison(["first", "first"])
