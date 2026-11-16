"""Persist reviewed candidates before they can be returned for entry routing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agents.review_context import MAX_REVIEW_DECISION_BYTES, serialize_review_context
from database.db import get_db
from database.models import CandidateReview
from strategies.contracts import require_strategy_review_contract


class CandidateReviewUnavailable(ValueError):
    """A stored review is missing, invalid, or does not match the entry."""


def persist_candidate_reviews(
    state: dict[str, Any], decisions: list[dict[str, Any]], portfolio_context_json: str,
) -> None:
    """Atomically save bounded review evidence; conflicting identities fail closed."""
    profile = state.get("llm_profile")
    run_id = profile.get("run_id") if isinstance(profile, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("review scan identifier is unavailable")
    context_hash = hashlib.sha256(portfolio_context_json.encode("utf-8")).hexdigest()
    prepared = []
    seen: set[str] = set()
    for decision in decisions:
        candidate_id = decision.get("candidate_id")
        symbol = decision.get("symbol")
        if not isinstance(candidate_id, str) or len(candidate_id) != 64 or not isinstance(symbol, str):
            raise ValueError("review candidate identity is invalid")
        if candidate_id in seen:
            raise ValueError("multiple decisions for the same candidate")
        seen.add(candidate_id)
        technical_signals = state.get("technical_signals")
        technical = technical_signals.get(symbol) if isinstance(technical_signals, dict) else None
        contract = require_strategy_review_contract(technical)
        decision_json = serialize_review_context(decision, max_bytes=MAX_REVIEW_DECISION_BYTES, sort_keys=True)
        prepared.append((candidate_id, symbol, contract.version, decision_json))

    with get_db() as db:
        for candidate_id, symbol, contract_version, decision_json in prepared:
            decision_hash = hashlib.sha256(decision_json.encode("utf-8")).hexdigest()
            existing = db.get(CandidateReview, candidate_id)
            if existing is not None:
                if (
                    existing.run_id != run_id or existing.symbol != symbol
                    or existing.contract_version != contract_version
                    or existing.portfolio_context_json != portfolio_context_json
                    or existing.decision_json != decision_json
                    or existing.portfolio_context_sha256 != context_hash
                    or existing.decision_sha256 != decision_hash
                ):
                    raise ValueError("existing review candidate conflicts with this decision")
                continue
            db.add(CandidateReview(
                candidate_id=candidate_id, run_id=run_id, symbol=symbol,
                contract_version=contract_version,
                portfolio_context_json=portfolio_context_json, portfolio_context_sha256=context_hash,
                decision_json=decision_json, decision_sha256=decision_hash,
            ))


def verify_candidate_review(state: dict[str, Any], decision: dict[str, Any]) -> None:
    """Require the committed review and current complete portfolio context to match."""
    from agents.portfolio_manager_agent import PortfolioManagerAgent

    candidate_id = decision.get("candidate_id")
    profile = state.get("llm_profile")
    run_id = profile.get("run_id") if isinstance(profile, dict) else None
    technical_signals = state.get("technical_signals")
    technical = (
        technical_signals.get(decision.get("symbol"))
        if isinstance(technical_signals, dict) else None
    )
    try:
        contract_version = require_strategy_review_contract(technical).version
    except ValueError as exc:
        raise CandidateReviewUnavailable("strategy review contract is unavailable") from exc
    with get_db() as db:
        record = db.get(CandidateReview, candidate_id)
        if record is None:
            raise CandidateReviewUnavailable("stored review is missing")
        context_json = record.portfolio_context_json
        decision_json = record.decision_json
        if not isinstance(context_json, str) or not isinstance(decision_json, str):
            raise CandidateReviewUnavailable("stored review is invalid")
        if (
            record.run_id != run_id
            or record.symbol != decision.get("symbol")
            or record.contract_version != contract_version
            or hashlib.sha256(context_json.encode("utf-8")).hexdigest() != record.portfolio_context_sha256
            or hashlib.sha256(decision_json.encode("utf-8")).hexdigest() != record.decision_sha256
            or serialize_review_context(decision, max_bytes=MAX_REVIEW_DECISION_BYTES, sort_keys=True) != decision_json
        ):
            raise CandidateReviewUnavailable("stored review does not match the decision")
        try:
            context = json.loads(context_json)
            candidates = context["candidates"]
            if not isinstance(candidates, dict) or decision.get("symbol") not in candidates:
                raise ValueError("invalid candidates")
            risk = state.get("risk_assessment")
            if not isinstance(risk, dict):
                raise ValueError("risk assessment unavailable")
            approved = {symbol: risk[symbol] for symbol in candidates}
            current_context = PortfolioManagerAgent._portfolio_context(state, approved)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CandidateReviewUnavailable("portfolio review context is unavailable or invalid") from exc
        if current_context != context_json:
            raise CandidateReviewUnavailable("portfolio review context changed after review")
