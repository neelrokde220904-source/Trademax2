"""Bull vs. Bear Adversarial Debate Protocol — the most critical v2 upgrade.

Instead of unstructured LLM conversation, forces each side to produce
machine-readable JSON arguments. Only trades where conviction_score > 70
from the debate proceed to the Risk Manager.

Conviction is an uncalibrated model score, not a probability of success.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from loguru import logger

from agents.base_agent import TradingState
from config.settings import settings

from agents.llm_client import LLMClient, complete
from agents.review_context import (
    CONVICTION_THRESHOLD, MAX_DEBATE_CONTEXT_BYTES, ReviewContextError,
    candidate_review_identity, serialize_review_context, technical_review_digest,
)
from config.llm_runtime import llm_operation


@llm_operation("bull_bear")
def _call_llm(role: str, prompt: str, max_tokens: int = 1500) -> str:
    if not LLMClient(role, "bull_bear"):
        return "{}"
    return complete(role, prompt, max_tokens, task="bull_bear")


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from Claude response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse JSON from LLM response")
    return {}


class BullBearSynthesisAgent:
    """Adversarial debate protocol between Bull and Bear researchers.

    Flow:
    1. Collect all analyst JSON theses
    2. Bull Researcher (claude-sonnet) argues FOR the trade
    3. Bear Researcher (claude-sonnet) argues AGAINST, attacking bull's case
    4. Portfolio Manager (claude-opus) synthesizes the debate
    5. Only conviction_score > 70 proceeds
    """

    def __init__(self) -> None:
        self.fast_model = "fast"
        self.decision_model = "decision"

    @llm_operation("bull_bear")
    def run(self, state: TradingState) -> dict[str, Any]:
        """Run debate for all symbols in watchlist with analyst signals.

        Returns state update with debate_results dict.
        """
        logs: list[str] = []
        errors: list[str] = []
        debate_results: dict[str, Any] = {}

        watchlist = state.get("watchlist", [])
        regime = state.get("hmm_regime", {})

        # If SYSTEMIC_PANIC, skip debate entirely
        if regime.get("regime") == "SYSTEMIC_PANIC":
            logs.append("[BullBearDebate] SYSTEMIC_PANIC regime — all debates skipped, no new positions")
            return {
                "debate_results": {},
                "agent_logs": logs,
                "errors": errors,
            }

        # Collect analyst signals for each symbol
        for symbol in watchlist:
            try:
                if not state.get("research_mode"):
                    from agents.sentiment_agent import validate_candidate_sentiment
                    from agents.sentiment_evidence import sentiment_digest
                    try:
                        sentiment_review_digest = sentiment_digest(validate_candidate_sentiment(state, symbol))
                    except (TypeError, ValueError):
                        errors.append(f"[BullBearDebate] {symbol}: required sentiment is unavailable or invalid")
                        continue
                    try:
                        technical = state.get("technical_signals", {}).get(symbol)
                        technical_digest = technical_review_digest(technical)
                        candidate_id = candidate_review_identity(state, symbol)
                    except (AttributeError, ReviewContextError) as exc:
                        errors.append(f"[BullBearDebate] {symbol}: required technical evidence is invalid: {exc}")
                        continue
                analyst_signals = self._collect_signals(symbol, state)
                if not analyst_signals:
                    continue

                result = self.run_debate(symbol, analyst_signals, state)
                if not state.get("research_mode"):
                    try:
                        if (
                            technical_review_digest(state.get("technical_signals", {}).get(symbol)) != technical_digest
                            or candidate_review_identity(state, symbol) != candidate_id
                        ):
                            errors.append(f"[BullBearDebate] {symbol}: technical evidence changed during review")
                            continue
                    except (AttributeError, ReviewContextError) as exc:
                        errors.append(f"[BullBearDebate] {symbol}: required technical evidence is invalid: {exc}")
                        continue
                    result["sentiment_review_digest"] = sentiment_review_digest
                    result["technical_review_digest"] = technical_digest
                    result["candidate_id"] = candidate_id
                debate_results[symbol] = result

                conviction = result.get("conviction_score", 0)
                action = result.get("action", "HOLD")
                winner = result.get("winning_argument", "neutral")

                logs.append(
                    f"[BullBearDebate] {symbol}: {action} (conviction={conviction}, "
                    f"winner={winner})"
                )

                if conviction >= CONVICTION_THRESHOLD:
                    logs.append(
                        f"[BullBearDebate] {symbol} PASSED threshold ({conviction} >= {CONVICTION_THRESHOLD})"
                    )
                else:
                    logs.append(
                        f"[BullBearDebate] {symbol} REJECTED (conviction {conviction} < {CONVICTION_THRESHOLD})"
                    )

            except Exception as e:
                errors.append(f"[BullBearDebate] Error debating {symbol}: {e}")

        return {
            "debate_results": debate_results,
            "agent_logs": logs,
            "errors": errors,
        }

    @llm_operation("bull_bear")
    def run_debate(self, symbol: str, analyst_signals: dict[str, Any],
                   state: TradingState) -> dict[str, Any]:
        """Run the full Bull vs. Bear debate for a single symbol."""
        context_json = self._debate_context(symbol, analyst_signals, state)

        def verify_context() -> None:
            if self._debate_context(symbol, analyst_signals, state) != context_json:
                raise ReviewContextError("debate context changed during review")

        # Step 1: Bull Researcher argues FOR
        bull_argument = self._run_bull_researcher(symbol, analyst_signals, state, context_json)
        verify_context()

        # Step 2: Bear Researcher argues AGAINST (sees the bull case)
        bear_argument = self._run_bear_researcher(symbol, analyst_signals, bull_argument, state, context_json)
        verify_context()

        # Step 3: Synthesis — Portfolio Manager judges the debate
        verdict = self._synthesize_debate(symbol, bull_argument, bear_argument, state, context_json)
        verify_context()
        review_complete = all((
            bull_argument.get("_review_succeeded") is True,
            bear_argument.get("_review_succeeded") is True,
            verdict.get("_review_succeeded") is True,
        ))
        verdict["review_complete"] = review_complete
        verdict["arguments"] = {"bull": bull_argument, "bear": bear_argument}
        verdict["debate_context_sha256"] = hashlib.sha256(context_json.encode("utf-8")).hexdigest()
        verdict["debate_context_json"] = context_json
        if not review_complete:
            verdict["action"] = "HOLD"
            verdict["conviction_score"] = 0

        return verdict

    def _collect_signals(self, symbol: str, state: TradingState) -> dict[str, Any]:
        """Collect all analyst signals for a symbol."""
        signals: dict[str, Any] = {}

        for key in ("technical_signals", "fundamental_signals", "sentiment_signals",
                     "options_signals", "macro_signals", "india_signals"):
            agent_signals = state.get(key, {})
            if isinstance(agent_signals, dict):
                sym_signal = agent_signals.get(symbol, agent_signals.get("overall", {}))
                if sym_signal:
                    signals[key.replace("_signals", "")] = sym_signal

        return signals

    def _debate_context(self, symbol: str, reports: dict[str, Any], state: TradingState) -> str:
        context = {
            "reports": {
                name: reports.get(name)
                for name in ("technical", "fundamental", "sentiment", "options", "macro", "india")
            },
            "live_quote": state.get("live_quotes", {}).get(symbol),
            "hmm_regime": state.get("hmm_regime"),
            "india_vix": state.get("india_vix"),
            "geopolitical_risks": state.get("geopolitical_risks"),
            "order_flow": state.get("ofi_signals", {}).get(symbol),
            "portfolio_value": state.get("portfolio_value"),
            "available_capital": state.get("available_capital"),
            "current_positions": state.get("current_positions"),
        }
        return serialize_review_context(context, max_bytes=MAX_DEBATE_CONTEXT_BYTES)

    def _run_bull_researcher(self, symbol: str, reports: dict[str, Any],
                             state: TradingState, context_json: str | None = None) -> dict[str, Any]:
        """Bull side — build the strongest possible case FOR buying."""
        reports_str = context_json if context_json is not None else self._debate_context(symbol, reports, state)
        context = json.loads(reports_str)
        ltp = (context.get("live_quote") or {}).get("ltp", "N/A")
        regime = (context.get("hmm_regime") or {}).get("regime", "UNKNOWN")

        prompt = f"""You are the BULL RESEARCHER. Your sole job is to build the strongest 
possible case FOR buying {symbol}.

Analyst reports:
{reports_str}

Current price: ₹{ltp}
Market regime: {regime}

ATTACK any weaknesses in potential bear arguments. Find every reason this stock 
will go up. Be specific with numbers and catalysts.

Respond ONLY in this JSON format:
{{
    "thesis": "One paragraph bull thesis",
    "upside_target": 0.0,
    "probability": 0.75,
    "supporting_factors": ["factor1", "factor2", "factor3"],
    "key_catalysts": ["catalyst1", "catalyst2"],
    "bull_confidence": 75
}}"""

        response = _call_llm(self.fast_model, prompt)
        result = _parse_json_response(response)

        confidence = result.get("bull_confidence")
        probability = result.get("probability")
        valid = (
            isinstance(result.get("thesis"), str) and bool(result["thesis"].strip())
            and type(confidence) in (int, float) and 0 < confidence <= 100
            and type(probability) in (int, float) and 0 <= probability <= 1
            and isinstance(result.get("supporting_factors"), list)
            and isinstance(result.get("key_catalysts"), list)
        )
        if not valid:
            result = {
                "thesis": "Unable to generate bull thesis",
                "upside_target": 0.0,
                "probability": 0.0,
                "supporting_factors": [],
                "key_catalysts": [],
                "bull_confidence": 0,
                "_review_succeeded": False,
            }
        else:
            result["_review_succeeded"] = True

        return result

    def _run_bear_researcher(self, symbol: str, reports: dict[str, Any],
                             bull_argument: dict[str, Any],
                             state: TradingState, context_json: str | None = None) -> dict[str, Any]:
        """Bear side — destroy the bull case and identify all downside risks."""
        reports_str = context_json if context_json is not None else self._debate_context(symbol, reports, state)
        context = json.loads(reports_str)
        ltp = (context.get("live_quote") or {}).get("ltp", "N/A")
        regime = (context.get("hmm_regime") or {}).get("regime", "UNKNOWN")
        bull_str = serialize_review_context(bull_argument, max_bytes=MAX_DEBATE_CONTEXT_BYTES)

        prompt = f"""You are the BEAR RESEARCHER. Your sole job is to destroy the bull case 
for {symbol} and identify all downside risks.

Bull argument to attack:
{bull_str}

Analyst reports:
{reports_str}

Current price: ₹{ltp}
Market regime: {regime}

Be ruthless. Find margin compression, hidden debt, management red flags,
sector headwinds, regulatory risks, global macro threats. Be specific.

Respond ONLY in this JSON format:
{{
    "counter_thesis": "One paragraph bear thesis",
    "downside_risk": 0.0,
    "probability": 0.60,
    "risk_factors": ["risk1", "risk2", "risk3"],
    "bull_case_weaknesses": ["weakness1", "weakness2"],
    "bear_confidence": 60
}}"""

        response = _call_llm(self.fast_model, prompt)
        result = _parse_json_response(response)

        confidence = result.get("bear_confidence")
        probability = result.get("probability")
        valid = (
            isinstance(result.get("counter_thesis"), str) and bool(result["counter_thesis"].strip())
            and type(confidence) in (int, float) and 0 < confidence <= 100
            and type(probability) in (int, float) and 0 <= probability <= 1
            and isinstance(result.get("risk_factors"), list)
            and isinstance(result.get("bull_case_weaknesses"), list)
        )
        if not valid:
            result = {
                "counter_thesis": "Unable to generate bear thesis",
                "downside_risk": 0.0,
                "probability": 0.0,
                "risk_factors": [],
                "bull_case_weaknesses": [],
                "bear_confidence": 0,
                "_review_succeeded": False,
            }
        else:
            result["_review_succeeded"] = True

        return result

    def _synthesize_debate(self, symbol: str,
                           bull: dict[str, Any],
                           bear: dict[str, Any],
                           state: TradingState, context_json: str | None = None) -> dict[str, Any]:
        """Portfolio Manager synthesizes the debate — final judgment.

        Uses claude-opus for the most capable reasoning.
        """
        context = json.loads(context_json) if context_json is not None else {
            "hmm_regime": state.get("hmm_regime"), "india_vix": state.get("india_vix", 0),
        }
        regime = (context.get("hmm_regime") or {}).get("regime", "UNKNOWN")
        vix = context.get("india_vix", 0)

        bull_str = serialize_review_context(bull, max_bytes=MAX_DEBATE_CONTEXT_BYTES)
        bear_str = serialize_review_context(bear, max_bytes=MAX_DEBATE_CONTEXT_BYTES)

        prompt = f"""You are the PORTFOLIO MANAGER making the final trading decision for {symbol}.

You have heard both sides of an adversarial debate:

BULL CASE:
{bull_str}

BEAR CASE:
{bear_str}

CONTEXT:
- Market regime: {regime}
- India VIX: {vix}
- Conviction threshold: {CONVICTION_THRESHOLD} (only invest if your conviction exceeds this)

Evaluate BOTH arguments critically. Consider:
1. Quality of evidence (specific numbers vs vague claims)
2. Relevance to current market regime
3. Risk/reward asymmetry
4. Time horizon alignment

Respond ONLY in this JSON format:
{{
    "action": "BUY",
    "conviction_score": 75,
    "reasoning": "Brief synthesis of why this is the right call",
    "winning_argument": "bull",
    "risk_reward_ratio": 2.5,
    "recommended_position_pct": 0.05,
    "stop_loss_pct": 0.03,
    "time_horizon": "SWING"
}}

Where:
- action: BUY, SELL, or HOLD
- conviction_score: 0-100 (only BUY if > {CONVICTION_THRESHOLD})
- winning_argument: "bull", "bear", or "neutral"
- time_horizon: "INTRADAY", "SWING", or "POSITIONAL"
"""

        response = _call_llm(self.decision_model, prompt)
        result = _parse_json_response(response)

        conviction = result.get("conviction_score")
        valid = (
            result.get("action") in {"BUY", "SELL", "HOLD"}
            and type(conviction) in (int, float) and 0 <= conviction <= 100
            and isinstance(result.get("reasoning"), str) and bool(result["reasoning"].strip())
        )
        if not valid:
            return {
                "action": "HOLD",
                "conviction_score": 0,
                "reasoning": "Failed to synthesize debate",
                "winning_argument": "neutral",
                "_review_succeeded": False,
            }

        result["_review_succeeded"] = True
        # Enforce conviction threshold
        if conviction < CONVICTION_THRESHOLD:
            result["action"] = "HOLD"
            result["reasoning"] = (
                f"Conviction {conviction} below threshold {CONVICTION_THRESHOLD}. "
                + result.get("reasoning", "")
            )

        return result


def bull_bear_synthesis_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node wrapper for the Bull/Bear Debate."""
    agent = BullBearSynthesisAgent()
    return agent.run(state)
