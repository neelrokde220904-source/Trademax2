"""Portfolio Manager Agent — final decision aggregator using Claude Opus."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytz
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings
from agents.llm_client import LLMClient
from agents.review_context import (
    ReviewContextError, candidate_review_identity, debate_review_digest, serialize_review_context,
    technical_review_digest, valid_entry_debate,
)
from config.llm_runtime import current_context, llm_operation

IST = pytz.timezone("Asia/Kolkata")
MAX_REVIEW_CONTEXT_BYTES = 32_000


def _decision_time(state: TradingState) -> datetime:
    if not state.get("research_mode"):
        return datetime.now(IST)
    timestamp = datetime.fromisoformat(state["research_as_of"])
    if timestamp.tzinfo is None:
        raise ValueError("Research decision time must include a timezone")
    return timestamp.astimezone(IST)


class PortfolioManagerAgent(BaseAgent):
    """Final decision-making node — aggregates all signals and generates trade orders.

    Uses Claude Opus (most capable model) for the final decision.
    """

    def __init__(self) -> None:
        super().__init__("PortfolioManagerAgent")
        self._llm = LLMClient("decision", "portfolio_manager", max_tokens=4096, temperature=0.2)

    @llm_operation("portfolio_manager")
    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        risk = state.get("risk_assessment", {})

        # If globally blocked, no decisions
        if risk.get("blocked"):
            logs.append(self.log("Trading blocked: {}", risk.get("reason", "Unknown")))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        # Check time window
        now = _decision_time(state)
        no_trade_start = now.replace(hour=9, minute=15 + settings.no_trade_window_open_minutes, second=0)
        no_trade_end = now.replace(hour=15, minute=30 - settings.no_trade_window_close_minutes, second=0)

        if now < no_trade_start:
            logs.append(self.log("In opening no-trade window (first {} mins).", settings.no_trade_window_open_minutes))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        if now > no_trade_end:
            logs.append(self.log("In closing no-trade window (last {} mins).", settings.no_trade_window_close_minutes))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        # Gather approved symbols
        debate_results = state.get("debate_results", {})
        technical_signals = state.get("technical_signals", {})
        approved = {}
        debate_digests = {}
        for symbol, data in risk.items():
            if not isinstance(data, dict) or data.get("approved") is not True:
                continue
            if not state.get("research_mode"):
                from agents.sentiment_agent import validate_candidate_sentiment
                from agents.sentiment_evidence import sentiment_digest
                try:
                    candidate_digest = sentiment_digest(validate_candidate_sentiment(state, symbol))
                except (TypeError, ValueError):
                    errors.append(self.log("{}: Required sentiment is unavailable or invalid.", symbol))
                    continue
            debate = debate_results.get(symbol, {})
            if (
                not state.get("research_mode") and isinstance(debate, dict)
                and debate.get("review_complete") is True
                and debate.get("sentiment_review_digest") != candidate_digest
            ):
                errors.append(self.log("{}: Debate sentiment evidence does not match.", symbol))
                continue
            technical = technical_signals.get(symbol, {})
            technical_signal = str(technical.get("signal", "")).upper()
            expected_action = (
                "BUY" if technical_signal in {"BUY", "STRONG_BUY"}
                else "SELL" if technical_signal in {"SELL", "STRONG_SELL"}
                else ""
            )
            if not state.get("research_mode") and not valid_entry_debate(debate, expected_action):
                errors.append(self.log("{}: Required debate is incomplete, invalid, or below threshold.", symbol))
                continue
            if not state.get("research_mode"):
                try:
                    if technical_review_digest(technical) != debate.get("technical_review_digest"):
                        errors.append(self.log("{}: Debate technical evidence does not match.", symbol))
                        continue
                    if candidate_review_identity(state, symbol) != debate.get("candidate_id"):
                        errors.append(self.log("{}: Debate candidate identity does not match.", symbol))
                        continue
                    debate_digests[symbol] = debate_review_digest(debate)
                except ReviewContextError as exc:
                    errors.append(self.log("{}: Technical or debate review blocked: {}", symbol, exc))
                    continue
            if (
                state.get("research_mode")
                and expected_action
                and data.get("deterministic_action") == expected_action
            ) or (
                not state.get("research_mode")
                and
                debate.get("review_complete") is True
                and expected_action
                and debate.get("action") == expected_action
                and data.get("deterministic_action") == expected_action
            ):
                approved[symbol] = data

        if not approved:
            logs.append(self.log("No risk-approved candidates have a complete, aligned AI review."))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        context = current_context()
        research_rules = state.get("research_mode") and context is not None and context.profile.mode == "rules"
        portfolio_context_json = None
        if not state.get("research_mode"):
            try:
                portfolio_context_json = self._portfolio_context(state, approved)
            except ReviewContextError as exc:
                errors.append(self.log("Portfolio-manager review blocked: {}", exc))
                return {"trading_decisions": [], "agent_logs": logs, "errors": errors}
        if research_rules:
            decisions = self._research_rules_decide(state, approved)
        elif not self._llm and not state.get("research_mode"):
            message = "Required portfolio-manager model is unavailable; new entries are blocked."
            errors.append(self.log(message))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}
        else:
            try:
                decisions = self._llm_decide(state, approved, portfolio_context_json)
            except ReviewContextError as exc:
                errors.append(self.log("Portfolio-manager review blocked: {}", exc))
                return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        if not state.get("research_mode"):
            try:
                if self._portfolio_context(state, approved) != portfolio_context_json:
                    errors.append(self.log("Portfolio review context changed during the model call; technical, debate or other evidence may have changed."))
                    return {"trading_decisions": [], "agent_logs": logs, "errors": errors}
            except ReviewContextError as exc:
                errors.append(self.log("Portfolio-manager review blocked: {}", exc))
                return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        if not isinstance(decisions, list):
            message = "Invalid or missing portfolio-manager response; new entries are blocked."
            errors.append(self.log(message))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        fo_ban = state.get("india_signals", {}).get("fo_ban_list", [])
        accepted = []
        for index, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                errors.append(self.log("Invalid portfolio decision {}: expected an object.", index))
                continue
            if not isinstance(decision.get("symbol"), str) or not isinstance(decision.get("action"), str):
                errors.append(self.log("Invalid portfolio decision {}: missing symbol or action.", index))
                continue
            symbol = decision["symbol"].strip().upper()
            action = decision["action"].strip().upper()
            if symbol not in approved:
                errors.append(self.log("Portfolio decision {} is not risk-approved.", index))
                continue
            if isinstance(symbol, str) and symbol in fo_ban:
                continue
            debate = debate_results.get(symbol, {})
            technical_signal = str(technical_signals[symbol].get("signal", "")).upper()
            expected_action = (
                "BUY" if technical_signal in {"BUY", "STRONG_BUY"}
                else "SELL" if technical_signal in {"SELL", "STRONG_SELL"}
                else ""
            )
            if (
                action != expected_action
                or (
                    not state.get("research_mode")
                    and action != debate.get("action")
                )
            ):
                errors.append(self.log("Portfolio decision {} conflicts with the reviewed signal.", index))
                continue
            accepted_decision = {
                **decision,
                "symbol": symbol,
                "action": action,
            }
            if not state.get("research_mode"):
                try:
                    if debate_review_digest(debate) != debate_digests[symbol]:
                        errors.append(self.log("{}: Debate changed during portfolio review.", symbol))
                        continue
                    if technical_review_digest(technical_signals.get(symbol)) != debate.get("technical_review_digest"):
                        errors.append(self.log("{}: Technical evidence changed during portfolio review.", symbol))
                        continue
                    if candidate_review_identity(state, symbol) != debate.get("candidate_id"):
                        errors.append(self.log("{}: Candidate identity changed during portfolio review.", symbol))
                        continue
                except ReviewContextError as exc:
                    errors.append(self.log("{}: Technical or debate review blocked: {}", symbol, exc))
                    continue
                accepted_decision["ai_review_passed"] = True
                accepted_decision["sentiment_review_digest"] = debate["sentiment_review_digest"]
                accepted_decision["debate_review_digest"] = debate_digests[symbol]
                accepted_decision["technical_review_digest"] = debate["technical_review_digest"]
                accepted_decision["candidate_id"] = debate["candidate_id"]
                accepted_decision["debate_context_json"] = debate["debate_context_json"]
            accepted.append(accepted_decision)
        decisions = accepted
        if decisions and not state.get("research_mode"):
            from operations.candidate_reviews import persist_candidate_reviews

            try:
                if portfolio_context_json is None:
                    raise ValueError("review context is unavailable")
                persist_candidate_reviews(state, decisions, portfolio_context_json)
            except (SQLAlchemyError, ReviewContextError, ValueError, OSError) as exc:
                errors.append(self.log("Candidate review persistence failed; new entries blocked: {}", type(exc).__name__))
                return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        logs.append(self.log("Generated {} trading decisions.", len(decisions)))
        for d in decisions:
            logs.append(
                self.log(
                    "Decision: {} {} × {} @ ₹{} (conf: {}%)",
                    d["action"], d.get("quantity", 0), d["symbol"],
                    d.get("price", 0), d.get("confidence", 0),
                )
            )

        return {
            "trading_decisions": decisions,
            "agent_logs": logs,
            "errors": errors,
        }

    @staticmethod
    def _portfolio_context(state: TradingState, approved: dict) -> str:
        """Serialize the exact candidate context supplied to the portfolio model."""
        india_vix = state.get("india_vix", 0)
        india_signals = state.get("india_signals", {})
        from strategies.contracts import require_strategy_review_contract

        def candidate_signal(field: str, symbol: str) -> Any:
            values = state.get(field, {})
            if not isinstance(values, dict):
                return None
            value = values.get(symbol, values.get("overall"))
            if field == "debate_results" and isinstance(value, dict):
                return {key: item for key, item in value.items() if key != "debate_context_json"}
            return value

        def candidate_contract(symbol: str) -> dict[str, Any]:
            contract = require_strategy_review_contract(
                candidate_signal("technical_signals", symbol)
            )
            return {
                "strategy": contract.strategy,
                "version": contract.version,
                "required_agents": list(contract.required),
                "optional_agents": list(contract.optional),
                "required_inputs": list(contract.required_inputs),
            }

        context = {
            "portfolio_value": state.get("portfolio_value"),
            "available_capital": state.get("available_capital"),
            "current_positions": state.get("current_positions"),
            "india_vix": india_vix,
            "hmm_regime": state.get("hmm_regime"),
            "macro": state.get("macro_signals"),
            "geopolitical_risks": state.get("geopolitical_risks"),
            "is_expiry_day": india_signals.get("is_expiry_day"),
            "fo_ban_list": india_signals.get("fo_ban_list", []),
            "candidates": {
                symbol: {
                    "review_contract": candidate_contract(symbol),
                    "risk": risk,
                    "technical": candidate_signal("technical_signals", symbol),
                    "fundamental": candidate_signal("fundamental_signals", symbol),
                    "sentiment": candidate_signal("sentiment_signals", symbol),
                    "options": candidate_signal("options_signals", symbol),
                    "india": candidate_signal("india_signals", symbol),
                    "debate": candidate_signal("debate_results", symbol),
                    "allocation": candidate_signal("optimal_allocations", symbol),
                }
                for symbol, risk in approved.items()
            },
        }
        return serialize_review_context(context, max_bytes=MAX_REVIEW_CONTEXT_BYTES)

    def _llm_decide(
        self, state: TradingState, approved: dict, context_json: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Use Claude Opus for final decision making."""
        india_vix = state.get("india_vix", 0)
        if context_json is None:
            context_json = self._portfolio_context(state, approved)

        prompt = f"""You are the Portfolio Manager of an AI hedge fund trading NSE/BSE Indian markets.
Today is {state.get('date')}. Current time: {_decision_time(state).strftime('%H:%M IST')}.
India VIX: {india_vix} ({'HIGH RISK - reduce size' if india_vix > 20 else 'NORMAL'})

REVIEW_CONTEXT_BEGIN
{context_json}
REVIEW_CONTEXT_END

INDIA-SPECIFIC RULES:
1. NEVER trade stocks in the supplied F&O ban list.
2. On F&O expiry, expect higher volatility.
3. Prefer liquid large-caps for intraday
4. Be conservative — missing a trade is better than a bad trade
5. Only propose the same direction as that symbol's deterministic technical signal.
6. Execution pins price, stop, target, holding period and quantity to deterministic risk controls.

Output ONLY valid JSON (no markdown, no explanation):
{{"decisions": [{{"symbol": "XXX", "action": "BUY/SELL", "quantity": N, "order_type": "LIMIT/MARKET", "price": X.XX, "stop_loss": X.XX, "target": X.XX, "holding_period": "INTRADAY/SWING/BTST/POSITIONAL (the candidate's declared horizon)", "confidence": 0-100, "reasoning": "..."}}], "market_view": "BULLISH/BEARISH/NEUTRAL", "session_notes": "..."}}"""

        try:
            resp = self._llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                decisions = result.get("decisions")
                return decisions if isinstance(decisions, list) else None
        except Exception as exc:
            logger.exception("LLM decision failed: {}", exc)

        return None

    def _research_rules_decide(self, state: TradingState, approved: dict) -> list[dict[str, Any]]:
        """Create a deterministic baseline for isolated research books only."""
        decisions = []
        technical_signals = state.get("technical_signals", {})
        for symbol, risk_data in approved.items():
            technical = technical_signals.get(symbol, {})
            action = (
                "BUY" if str(technical.get("signal", "")).upper() in {"BUY", "STRONG_BUY"}
                else "SELL" if str(technical.get("signal", "")).upper() in {"SELL", "STRONG_SELL"}
                else ""
            )
            if not action:
                continue
            decisions.append({
                "symbol": symbol,
                "action": action,
                "quantity": risk_data.get("max_quantity", 0),
                "order_type": "LIMIT",
                "price": technical.get("indicators", {}).get("close", 0),
                "stop_loss": risk_data.get("stop_loss", 0),
                "target": risk_data.get("target", 0),
                "holding_period": risk_data.get("holding_period", "INTRADAY"),
                "confidence": technical.get("confidence", 0),
                "reasoning": "Isolated rules baseline; never used for scheduled entries.",
            })
        return decisions


def portfolio_manager_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return PortfolioManagerAgent().run(state)
