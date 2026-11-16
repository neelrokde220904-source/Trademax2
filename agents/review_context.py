"""Serialize complete, bounded evidence for model review."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

CONVICTION_THRESHOLD = 70
MAX_DEBATE_CONTEXT_BYTES = 16_000
MAX_REVIEW_DECISION_BYTES = 64_000
MAX_HISTORY_CAPTURE_AGE = timedelta(minutes=30)


class ReviewContextError(ValueError):
    """The complete review context cannot be safely supplied."""


class HistoryEvidenceError(ReviewContextError):
    """Historical bars or their recorded provenance cannot be verified."""


def valid_entry_debate(value: Any, action: str) -> bool:
    """Validate the uncalibrated debate verdict before an entry is considered."""
    if not isinstance(value, dict) or value.get("review_complete") is not True:
        return False
    conviction = value.get("conviction_score")
    reasoning = value.get("reasoning")
    arguments = value.get("arguments")
    bull = arguments.get("bull") if isinstance(arguments, dict) else None
    bear = arguments.get("bear") if isinstance(arguments, dict) else None
    context_hash = value.get("debate_context_sha256")
    context_json = value.get("debate_context_json")
    valid_context = False
    if isinstance(context_json, str) and isinstance(context_hash, str):
        try:
            encoded = context_json.encode("utf-8")
            valid_context = (
                len(encoded) <= MAX_DEBATE_CONTEXT_BYTES
                and hashlib.sha256(encoded).hexdigest() == context_hash
                and isinstance(json.loads(context_json), dict)
            )
        except (UnicodeError, ValueError):
            valid_context = False
    return (
        action in {"BUY", "SELL"}
        and value.get("action") == action
        and type(conviction) in (int, float)
        and math.isfinite(conviction)
        and CONVICTION_THRESHOLD <= conviction <= 100
        and isinstance(reasoning, str)
        and bool(reasoning.strip())
        and isinstance(bull, dict) and bull.get("_review_succeeded") is True
        and isinstance(bull.get("thesis"), str) and bool(bull["thesis"].strip())
        and isinstance(bear, dict) and bear.get("_review_succeeded") is True
        and isinstance(bear.get("counter_thesis"), str) and bool(bear["counter_thesis"].strip())
        and valid_context
    )


def _json_value(value: Any) -> Any:
    if value is pd.NaT or isinstance(value, np.datetime64) and np.isnat(value):
        raise ValueError("Invalid timestamp in review context")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite decimal in review context")
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported review context type: {type(value).__name__}")


def serialize_review_context(value: Any, *, max_bytes: int, sort_keys: bool = False) -> str:
    try:
        text = json.dumps(
            value, default=_json_value, allow_nan=False, sort_keys=sort_keys, separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewContextError("review context cannot be serialized as valid JSON") from exc
    if len(text.encode("utf-8")) > max_bytes:
        raise ReviewContextError(f"complete review context exceeds {max_bytes} bytes")
    return text


def _review_digest(value: Any) -> str:
    text = serialize_review_context(value, max_bytes=16_000, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def debate_review_digest(value: Any) -> str:
    """Fingerprint the complete debate verdict, not just its selected action."""
    text = serialize_review_context(value, max_bytes=MAX_REVIEW_DECISION_BYTES, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def technical_review_digest(value: Any) -> str:
    """Fingerprint a present, recognizable candidate technical signal."""
    if not isinstance(value, dict) or value.get("signal") not in {
        "BUY", "STRONG_BUY", "SELL", "STRONG_SELL", "HOLD",
    }:
        raise ReviewContextError("candidate technical signal is unavailable or invalid")
    return _review_digest(value)


def candidate_review_identity(state: dict[str, Any], symbol: str) -> str:
    """Bind the declared review contract to one scan, symbol and technical output."""
    from strategies.contracts import require_strategy_review_contract

    profile = state.get("llm_profile")
    run_id = profile.get("run_id") if isinstance(profile, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ReviewContextError("candidate scan identifier is unavailable")
    if not isinstance(symbol, str) or not symbol or symbol != symbol.strip().upper():
        raise ReviewContextError("candidate symbol is invalid")
    signals = state.get("technical_signals")
    technical = signals.get(symbol) if isinstance(signals, dict) else None
    provenance = technical.get("history_provenance") if isinstance(technical, dict) else None
    sources = state.get("market_data_provenance")
    from operations.history_evidence import history_digest

    histories = state.get("market_data")
    if (
        not isinstance(provenance, dict)
        or not isinstance(sources, dict)
        or sources.get(symbol) != provenance
        or not isinstance(histories, dict)
        or provenance.get("symbol") != symbol
        or provenance.get("provider") not in ("angel_one", "yahoo_finance")
        or provenance.get("status") != "available"
        or provenance.get("exchange") != "NSE"
        or provenance.get("interval") != "15m"
        or not isinstance(provenance.get("bars_sha256"), str)
    ):
        raise HistoryEvidenceError("historical data provenance does not match the candidate")
    frame = histories.get(symbol)
    if (
        not isinstance(frame, pd.DataFrame)
        or type(provenance.get("returned_rows")) is not int
        or len(frame) != provenance["returned_rows"]
    ):
        raise HistoryEvidenceError("historical data row count does not match provenance")
    try:
        retrieved_at = datetime.fromisoformat(provenance["retrieved_at"])
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("timezone missing")
        age = datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)
        if age < timedelta(0) or age > MAX_HISTORY_CAPTURE_AGE:
            raise ValueError("history capture time is stale or in the future")
        current_digest = history_digest(frame)
    except (TypeError, ValueError, KeyError) as exc:
        raise HistoryEvidenceError("historical data cannot be revalidated") from exc
    if current_digest != provenance["bars_sha256"]:
        raise HistoryEvidenceError("historical data changed after capture")
    try:
        contract = require_strategy_review_contract(technical)
    except ValueError as exc:
        raise ReviewContextError(str(exc)) from exc
    return _review_digest({
        "run_id": run_id, "symbol": symbol, "strategy": contract.strategy,
        "contract_version": contract.version, "technical_digest": technical_review_digest(technical),
    })


def strategy_review_contract_report(
    state: dict[str, Any], symbol: str, decision: Any,
) -> dict[str, Any]:
    """Report the declared required reviews and inputs for one entry candidate."""
    technical_signals = state.get("technical_signals")
    technical = technical_signals.get(symbol) if isinstance(technical_signals, dict) else None
    strategy = technical.get("strategy") if isinstance(technical, dict) else None
    from strategies.contracts import require_strategy_review_contract

    try:
        contract = require_strategy_review_contract(technical)
    except ValueError:
        return {
            "strategy": strategy,
            "contract_version": None,
            "required_agents": {},
            "optional_agents": [],
            "required_inputs": {},
            "failed_requirements": ["declared_strategy_contract"],
            "passed": False,
        }

    sentiment_ok = False
    sentiment_hash = None
    try:
        from agents.sentiment_agent import validate_candidate_sentiment
        from agents.sentiment_evidence import sentiment_digest

        sentiment_hash = sentiment_digest(validate_candidate_sentiment(state, symbol))
        sentiment_ok = True
    except (TypeError, ValueError):
        pass

    technical_ok = False
    candidate_id = None
    technical_hash = None
    try:
        technical_hash = technical_review_digest(technical)
        candidate_id = candidate_review_identity(state, symbol)
        technical_ok = True
    except ReviewContextError:
        pass

    debates = state.get("debate_results")
    debate = debates.get(symbol) if isinstance(debates, dict) else None
    arguments = debate.get("arguments") if isinstance(debate, dict) else None
    bull = arguments.get("bull") if isinstance(arguments, dict) else None
    bear = arguments.get("bear") if isinstance(arguments, dict) else None
    decision_action = decision.get("action") if isinstance(decision, dict) else None
    debate_ok = valid_entry_debate(debate, decision_action)
    debate_hash = None
    if isinstance(debate, dict):
        try:
            debate_hash = debate_review_digest(debate)
        except ReviewContextError:
            pass

    required_agents = {
        "technical": technical_ok,
        "sentiment": sentiment_ok,
        "bull": (
            isinstance(bull, dict) and bull.get("_review_succeeded") is True
            and isinstance(bull.get("thesis"), str) and bool(bull["thesis"].strip())
        ),
        "bear": (
            isinstance(bear, dict) and bear.get("_review_succeeded") is True
            and isinstance(bear.get("counter_thesis"), str) and bool(bear["counter_thesis"].strip())
        ),
        "debate_synthesis": debate_ok,
        "portfolio_manager": (
            isinstance(decision, dict) and decision.get("ai_review_passed") is True
            and candidate_id is not None and technical_hash is not None
            and sentiment_hash is not None and debate_hash is not None
            and decision.get("candidate_id") == candidate_id
            and decision.get("sentiment_review_digest") == sentiment_hash
            and decision.get("technical_review_digest") == technical_hash
            and decision.get("debate_review_digest") == debate_hash
            and isinstance(debate, dict)
            and decision.get("debate_context_json") == debate.get("debate_context_json")
        ),
    }

    screening = state.get("screener_results")
    qualified = screening.get("qualified") if isinstance(screening, dict) else None
    rejected = screening.get("rejected") if isinstance(screening, dict) else None
    watchlist = state.get("watchlist")
    risk_assessment = state.get("risk_assessment")
    risk = risk_assessment.get(symbol) if isinstance(risk_assessment, dict) else None
    required_inputs = {
        "verified_15m_history": candidate_id is not None,
        "current_attributable_news": sentiment_ok,
        "passed_quantitative_screen": (
            isinstance(qualified, list) and isinstance(rejected, list)
            and isinstance(watchlist, list) and symbol in qualified
            and symbol not in rejected and symbol in watchlist
        ),
        "risk_approval": isinstance(risk, dict) and risk.get("approved") is True,
        "valid_portfolio_snapshot": state.get("portfolio_snapshot_valid") is True,
    }
    failed = [
        requirement for requirement in contract.required if required_agents.get(requirement) is not True
    ]
    failed.extend(
        requirement for requirement in contract.required_inputs
        if required_inputs.get(requirement) is not True
    )
    return {
        "strategy": contract.strategy,
        "contract_version": contract.version,
        "required_agents": required_agents,
        "optional_agents": list(contract.optional),
        "required_inputs": required_inputs,
        "failed_requirements": failed,
        "passed": not failed,
    }
