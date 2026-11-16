"""Scan decision outcomes: matching, plain-English block reasons and persistence."""

from __future__ import annotations

from operations.scan_outcomes import (
    classify_router_error, scan_decision_outcomes, summarize_outcomes,
    summarize_review_rejections, summarize_risk_rejections,
)


def _decision(symbol, action="BUY", **extra):
    return {"symbol": symbol, "action": action, "quantity": 2, "price": 100.0,
            "confidence": 70, "reasoning": "test", **extra}


def test_each_decision_is_matched_to_its_fill_or_blocking_rule():
    result = {
        "trading_decisions": [
            _decision("M&M"), _decision("M&MFIN"), _decision("TCS", "SELL"),
            _decision("INFY"), _decision("WIPRO"),
        ],
        "executed_orders": [{"symbol": "M&M", "action": "BUY", "fill_price": 101.5}],
        "errors": [
            "[OrderRouter] Naked short sale blocked for TCS",
            "[OrderRouter] Order not filled for INFY (see paper fill audit / logs)",
            "[OrderRouter] New entry blocked for M&MFIN: HMM unavailable",
            "[Technical] unrelated failure for WIPRO",
        ],
    }
    outcomes = {item["symbol"]: item for item in scan_decision_outcomes(result, analysis_only=False)}
    assert outcomes["M&M"]["outcome"] == "executed" and outcomes["M&M"]["fill_price"] == 101.5
    assert outcomes["M&MFIN"]["block_code"] == "regime_block"
    assert outcomes["TCS"]["outcome"] == "blocked"
    assert "short selling is disabled" in outcomes["TCS"]["block_reason"]
    assert outcomes["INFY"]["outcome"] == "failed" and outcomes["INFY"]["block_code"] == "not_filled"
    assert outcomes["WIPRO"]["outcome"] == "not_executed"

    summary = summarize_outcomes(list(outcomes.values()))
    assert summary["decisions"] == 5 and summary["executed"] == 1
    assert summary["blocked"] == 2 and summary["failed"] == 1


def test_analysis_only_decisions_are_never_reported_as_orders():
    outcomes = scan_decision_outcomes(
        {"trading_decisions": [_decision("RELIANCE")], "executed_orders": [], "errors": []},
        analysis_only=True,
    )
    assert outcomes[0]["outcome"] == "analysis_only"
    assert summarize_outcomes(outcomes)["executed"] == 0


def test_decision_outcome_includes_review_contract_report():
    report = {
        "strategy": "CompositeTrend", "contract_version": "review-v2",
        "required_agents": {"sentiment": False},
        "optional_agents": ["fundamental"], "required_inputs": {},
        "failed_requirements": ["sentiment"], "passed": False,
    }
    outcomes = scan_decision_outcomes({
        "trading_decisions": [_decision("TCS")], "executed_orders": [], "errors": [],
        "review_contract_results": {"TCS": report},
    }, analysis_only=False)

    assert outcomes[0]["review_contract"] == report


def test_operational_block_is_not_mistaken_for_regime_block():
    code, _, kind = classify_router_error(
        "[OrderRouter] New entry blocked by operational safety controls: TCS"
    )
    assert (code, kind) == ("safety_controls", "blocked")
    assert classify_router_error("[OrderRouter] something new")[0] == "other"


def test_ai_review_failures_have_explicit_safety_reasons():
    cases = [
        ("Stored candidate review unavailable for TCS: stored review is missing", "review_record"),
        ("New entry lacks a successful portfolio-manager review: TCS", "portfolio_review"),
        ("New entry has invalid debate evidence: TCS", "debate_review"),
        ("New entry lacks a complete aligned debate review: TCS", "debate_review"),
        ("Historical data evidence missing, expired, or changed: TCS", "historical_data_evidence"),
        ("New entry has changed technical evidence: TCS", "technical_review"),
        ("New entry has mismatched candidate identity: TCS", "candidate_identity"),
    ]
    for message, expected_code in cases:
        code, reason, kind = classify_router_error(f"[OrderRouter] {message}")
        assert code == expected_code
        assert reason
        assert kind == "blocked"
    code, _, kind = classify_router_error(
        "[OrderRouter] Required strategy review contract failed for TCS: missing input"
    )
    assert (code, kind) == ("strategy_review_contract", "blocked")


def test_predecision_ai_review_failures_are_summarized_without_raw_errors():
    result = {"errors": [
        "[PortfolioManagerAgent] 09:31:00 — GOOD: Required sentiment is unavailable or invalid.",
        "[PortfolioManagerAgent] 09:31:01 — OTHER: Required sentiment is unavailable or invalid.",
        "[PortfolioManagerAgent] 09:31:02 — Candidate review persistence failed; new entries blocked: OperationalError",
        "[BullBearDebate] Error debating GOOD: debate context changed during review",
        "[BullBearDebate] GOOD: required technical evidence is invalid: historical data cannot be revalidated",
        "[UnrelatedAgent] Required sentiment is unavailable or invalid",
        "unrecognized error with sensitive details",
    ]}
    assert summarize_review_rejections(result) == {
        "Required sentiment review unavailable or invalid": 2,
        "Could not save required AI review evidence": 1,
        "Debate evidence changed during review": 1,
        "Historical data evidence is missing, expired, invalid, or changed": 1,
    }
    assert summarize_review_rejections({"errors": None}) == {}


def test_portfolio_snapshot_and_mark_errors_are_explained():
    cases = [
        ("Portfolio snapshot unavailable; new entry blocked: TCS", "portfolio_snapshot", "blocked"),
        ("Fresh portfolio marks unavailable; new entry blocked: TCS", "portfolio_marks", "blocked"),
        ("Portfolio mark refresh failed for TCS: error", "portfolio_refresh_error", "failed"),
        ("Portfolio stop risk unavailable for TCS: active stop is missing", "portfolio_stop_risk", "blocked"),
        ("Paper position identity unavailable for TCS", "paper_book", "blocked"),
        ("Paper fill identity unavailable after TCS", "paper_fill_identity", "failed"),
        ("Entry blocked by risk sizing for TCS: Portfolio stop-risk budget is below one share",
         "portfolio_stop_budget", "blocked"),
        ("Entry blocked by risk sizing for TCS: No safe position size", "risk_sizing", "blocked"),
    ]
    for message, code, kind in cases:
        actual_code, reason, actual_kind = classify_router_error(f"[OrderRouter] {message}")
        assert (actual_code, actual_kind) == (code, kind)
        assert reason


def test_predecision_risk_rejections_are_explained():
    result = {"risk_assessment": {
        "blocked": False,
        "ONE": {"approved": False, "rejection_reason": "Daily risk history unavailable"},
        "TWO": {"approved": False, "rejection_reason": "Daily risk history unavailable"},
        "THREE": {"approved": True},
    }}
    assert summarize_risk_rejections(result) == {"Daily risk history unavailable": 2}
    assert summarize_risk_rejections({"risk_assessment": {
        "blocked": True, "reason": "Daily loss limit exceeded",
    }}) == {"Daily loss limit exceeded": 1}
    assert summarize_risk_rejections({}) == {"Risk assessment unavailable": 1}
    assert classify_router_error("[OrderRouter] Daily risk history unavailable for X")[0] == "risk_history"
    assert classify_router_error("[OrderRouter] Entry blocked for X: Adversely correlated")[0] == "portfolio_correlation"
