"""Explain and persist what happened to each portfolio-manager decision in a scan."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from loguru import logger

DECISION_AGENT = "portfolio_manager"

# (message fragment, code, plain-English reason, kind); first match wins.
# "blocked" means a safety rule stopped the order by design; "failed" means it could not be completed.
_ROUTER_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("Naked short sale blocked", "no_naked_shorts",
     "Sell signal without a holding — short selling is disabled", "blocked"),
    ("operational safety controls", "safety_controls",
     "Entries paused or circuit breaker tripped", "blocked"),
    ("Portfolio snapshot unavailable", "portfolio_snapshot",
     "Portfolio synchronization or position marks unavailable", "blocked"),
    ("Fresh portfolio marks unavailable", "portfolio_marks",
     "Fresh prices for open positions are unavailable", "blocked"),
    ("Portfolio mark refresh failed", "portfolio_refresh_error",
     "Could not refresh open-position prices", "failed"),
    ("Portfolio stop risk unavailable", "portfolio_stop_risk",
     "Open-position stop protection cannot be verified", "blocked"),
    ("Paper position identity unavailable", "paper_book",
     "Paper position identities are unavailable for the risk check", "blocked"),
    ("Paper fill identity unavailable", "paper_fill_identity",
     "Paper fill identity is unavailable; further entries are blocked", "failed"),
    ("Portfolio stop-risk budget", "portfolio_stop_budget",
     "Aggregate stop-risk budget has no room for another share", "blocked"),
    ("Entry blocked by risk sizing", "risk_sizing",
     "Entry exceeds a position-sizing or trade-economics limit", "blocked"),
    ("Daily risk history unavailable", "risk_history",
     "Daily risk history is incomplete or unavailable", "blocked"),
    ("Adversely correlated", "portfolio_correlation",
     "Correlated portfolio exposure exceeds the safety limit", "blocked"),
    ("Signal gate", "signal_gate", "The calibrated signal gate did not approve the entry", "blocked"),
    ("Max positions", "position_limit", "Maximum open-position count reached", "blocked"),
    ("an open position already exists", "existing_position",
     "The symbol already has an open position; one identity per symbol", "blocked"),
    ("scan exceeded entry deadline", "scan_deadline",
     "The scan took too long; its prices are no longer fresh enough to enter", "blocked"),
    ("corporate action", "corporate_action",
     "An unprocessed corporate action blocks new entries in this symbol", "blocked"),
    ("Unsupported horizon", "horizon_contract",
     "Horizon is not allowed by the strategy contract", "blocked"),
    ("Candidate contract invalid", "candidate_contract",
     "The trade candidate failed contract validation", "blocked"),
    ("screener eligibility", "screener",
     "Candidate did not pass the quantitative screen", "blocked"),
    ("Required sentiment unavailable", "sentiment",
     "Required sentiment review is unavailable or invalid", "blocked"),
    ("Required strategy review contract failed", "strategy_review_contract",
     "A required strategy review or input is missing, invalid, or changed", "blocked"),
    ("Stored candidate review unavailable", "review_record",
     "Stored AI review is missing, unavailable, or no longer matches", "blocked"),
    ("successful portfolio-manager review", "portfolio_review",
     "Required portfolio-manager review is missing", "blocked"),
    ("debate evidence", "debate_review",
     "Required debate evidence is missing, invalid, or changed", "blocked"),
    ("aligned debate review", "debate_review",
     "Required debate review is incomplete or does not match", "blocked"),
    ("Historical data evidence", "historical_data_evidence",
     "Historical data evidence is missing, expired, invalid, or changed", "blocked"),
    ("technical evidence", "technical_review",
     "Reviewed technical evidence is missing, invalid, or changed", "blocked"),
    ("candidate identity", "candidate_identity",
     "Candidate identity does not match the reviewed trade", "blocked"),
    ("Live entry blocked", "time_machine",
     "Time Machine evidence missing or rejected", "blocked"),
    ("before 09:30", "before_entry_window", "Before the 09:30 IST entry window", "blocked"),
    ("entry cutoff", "after_entry_cutoff", "After the 14:45 IST intraday entry cutoff", "blocked"),
    ("Exit quantity exceeds", "exit_exceeds_position", "Exit larger than the held position", "blocked"),
    ("New entry blocked for", "regime_block", "Market-regime gate blocked new entries", "blocked"),
    ("Invalid", "invalid_decision", "Decision had invalid order details", "blocked"),
    ("Order not filled", "not_filled", "Paper fill simulation did not fill the order", "failed"),
    ("not yet confirmed filled", "unconfirmed", "Order submitted but fill not confirmed", "failed"),
    ("Token not found", "routing_unavailable", "No broker route or instrument token", "failed"),
    ("No safe broker route", "routing_unavailable", "No broker route or instrument token", "failed"),
    ("No configured router", "routing_unavailable", "No broker route or instrument token", "failed"),
    ("execution error", "execution_error", "Execution raised an error", "failed"),
)


def classify_router_error(message: str) -> tuple[str, str, str]:
    """Return (code, plain-English reason, kind) for an order-router error message."""
    for fragment, code, reason, kind in _ROUTER_RULES:
        if fragment in message:
            return code, reason, kind
    return "other", "Order was not completed", "failed"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def scan_decision_outcomes(result: dict[str, Any], *, analysis_only: bool) -> list[dict[str, Any]]:
    """Match every decision to its executed order or the router error that stopped it."""
    executed = list(result.get("executed_orders") or [])
    router_errors = [
        error for error in result.get("errors") or []
        if isinstance(error, str) and error.startswith("[OrderRouter]")
    ]
    used_orders: set[int] = set()
    used_errors: set[int] = set()
    outcomes: list[dict[str, Any]] = []
    for decision in result.get("trading_decisions") or []:
        if not isinstance(decision, dict):
            continue
        symbol = str(decision.get("symbol") or "")
        action = str(decision.get("action") or "").upper()
        outcome = {
            "symbol": symbol,
            "action": action,
            "quantity": decision.get("quantity"),
            "price": _number(decision.get("price")),
            "stop_loss": _number(decision.get("stop_loss")),
            "target": _number(decision.get("target")),
            "holding_period": decision.get("holding_period"),
            "confidence": _number(decision.get("confidence")),
            "reasoning": str(decision.get("reasoning") or "")[:2000],
            "outcome": "not_executed",
            "block_code": None,
            "block_reason": None,
        }
        contract_results = result.get("review_contract_results")
        if isinstance(contract_results, dict):
            contract_report = contract_results.get(symbol)
            if isinstance(contract_report, dict):
                outcome["review_contract"] = contract_report
        if analysis_only:
            outcome["outcome"] = "analysis_only"
            outcome["block_reason"] = "Analysis-only run — no orders were sent"
            outcomes.append(outcome)
            continue
        order_index = next((
            index for index, order in enumerate(executed)
            if index not in used_orders and order.get("symbol") == symbol
            and str(order.get("action") or "").upper() == action
        ), None)
        if order_index is not None:
            used_orders.add(order_index)
            outcome["outcome"] = "executed"
            outcome["fill_price"] = _number(executed[order_index].get("fill_price"))
            outcomes.append(outcome)
            continue
        if symbol:
            pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])")
            error_index = next((
                index for index, error in enumerate(router_errors)
                if index not in used_errors and pattern.search(error)
            ), None)
            if error_index is not None:
                used_errors.add(error_index)
                code, reason, kind = classify_router_error(router_errors[error_index])
                outcome.update(outcome=kind, block_code=code, block_reason=reason)
        outcomes.append(outcome)
    return outcomes


def summarize_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the dashboard's last-scan card."""
    reasons = Counter(item["block_reason"] for item in outcomes if item["outcome"] in {"blocked", "failed"})
    return {
        "decisions": len(outcomes),
        "executed": sum(item["outcome"] == "executed" for item in outcomes),
        "blocked": sum(item["outcome"] == "blocked" for item in outcomes),
        "failed": sum(item["outcome"] == "failed" for item in outcomes),
        "block_reasons": dict(reasons.most_common()),
    }


def summarize_risk_rejections(result: dict[str, Any]) -> dict[str, int]:
    """Explain risk-gate rejections, including candidates that never became decisions."""
    assessment = result.get("risk_assessment")
    if not isinstance(assessment, dict):
        return {"Risk assessment unavailable": 1}
    reasons: Counter[str] = Counter()
    if assessment.get("blocked") is True:
        reason = assessment.get("reason")
        reasons[str(reason)[:200] if reason else "Risk manager blocked entries"] += 1
    for symbol, item in assessment.items():
        if symbol in {"blocked", "reason"}:
            continue
        if not isinstance(item, dict):
            reasons["Risk assessment invalid"] += 1
        elif item.get("approved") is False:
            reason = item.get("rejection_reason")
            reasons[str(reason)[:200] if reason else "Risk manager rejected candidate"] += 1
    return dict(reasons.most_common())


def summarize_review_rejections(result: dict[str, Any]) -> dict[str, int]:
    """Summarize known AI-review failures even when no decision was created."""
    rules = (
        ("historical data", "Historical data evidence is missing, expired, invalid, or changed"),
        ("candidate review persistence failed", "Could not save required AI review evidence"),
        ("portfolio review context changed", "Portfolio review evidence changed"),
        ("portfolio-manager review blocked", "Portfolio review context unavailable or too large"),
        ("required portfolio-manager model is unavailable", "Portfolio-manager model unavailable"),
        ("invalid or missing portfolio-manager response", "Portfolio-manager response invalid or missing"),
        ("required sentiment is unavailable or invalid", "Required sentiment review unavailable or invalid"),
        ("required debate is incomplete", "Required debate review incomplete or below threshold"),
        ("debate sentiment evidence does not match", "Sentiment changed after debate"),
        ("debate technical evidence does not match", "Technical evidence changed after debate"),
        ("debate candidate identity does not match", "Debate candidate identity did not match"),
        ("context changed during review", "Debate evidence changed during review"),
    )
    errors = result.get("errors")
    if not isinstance(errors, list):
        return {}
    reasons: Counter[str] = Counter()
    for error in errors:
        if not isinstance(error, str) or not error.startswith(("[PortfolioManagerAgent]", "[BullBearDebate]")):
            continue
        message = error.lower()
        for fragment, reason in rules:
            if fragment in message:
                reasons[reason] += 1
                break
    return dict(reasons.most_common())


def record_scan_decisions(
    outcomes: list[dict[str, Any]], *, scan_started_at: str, analysis_only: bool,
) -> int:
    """Persist decisions as Signal rows; never raises so a scan cannot fail on bookkeeping."""
    if not outcomes:
        return 0
    try:
        from database.db import get_db
        from database.models import Signal

        with get_db() as db:
            for item in outcomes:
                db.add(Signal(
                    agent_name=DECISION_AGENT,
                    symbol=item["symbol"][:50] or "UNKNOWN",
                    signal=(item["action"] or "UNKNOWN")[:20],
                    confidence=item["confidence"],
                    reasoning=item["reasoning"] or None,
                    data_json=json.dumps(
                        {**item, "scan_started_at": scan_started_at, "analysis_only": analysis_only},
                        default=str,
                    ),
                ))
        return len(outcomes)
    except Exception as exc:
        logger.error("[Scheduler] Could not record scan decisions: {}", type(exc).__name__)
        return 0
