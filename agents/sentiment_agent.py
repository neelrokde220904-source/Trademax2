"""Sentiment Agent — news NLP + social sentiment scoring via LLM."""

from __future__ import annotations

import json
import math
from typing import Any

from agents.base_agent import BaseAgent, TradingState
from agents.llm_client import LLMClient
from agents.sentiment_evidence import candidate_news
from config.llm_runtime import llm_operation
from data.news_fetcher import format_news_for_prompt


def validate_sentiment(signal: Any) -> dict[str, Any]:
    if not isinstance(signal, dict) or signal.get("signal") not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("Invalid sentiment action")
    for key, lower, upper in (("confidence", 0, 100), ("sentiment_score", -100, 100)):
        value = signal.get(key)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid sentiment {key}")
    events = signal.get("key_events")
    reasoning = signal.get("reasoning")
    if not isinstance(events, list) or not all(isinstance(event, str) for event in events):
        raise ValueError("Invalid sentiment key events")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Invalid sentiment reasoning")
    return signal


def validate_candidate_sentiment(state: TradingState, symbol: str) -> dict[str, Any]:
    signals = state.get("sentiment_signals")
    signal = validate_sentiment(signals.get(symbol) if isinstance(signals, dict) else None)
    profile = state.get("llm_profile")
    run_id = profile.get("run_id") if isinstance(profile, dict) else None
    evidence = signal.get("evidence")
    snapshot = state.get("news_snapshot")
    if not isinstance(run_id, str) or not run_id or not isinstance(evidence, dict):
        raise ValueError("Sentiment run evidence is unavailable")
    if (
        evidence.get("run_id") != run_id or evidence.get("symbol") != symbol
        or not isinstance(snapshot, dict) or evidence.get("snapshot_as_of") != snapshot.get("as_of")
    ):
        raise ValueError("Sentiment evidence does not match candidate and scan")
    _, digest = candidate_news(snapshot, symbol)
    if evidence.get("news_digest") != digest:
        raise ValueError("Sentiment news differs from the reviewed snapshot")
    return signal


class SentimentAgent(BaseAgent):
    """Fetches news and uses Claude to score sentiment for each symbol."""

    def __init__(self) -> None:
        super().__init__("SentimentAgent")
        self._llm = LLMClient("fast", "sentiment", max_tokens=1024, temperature=0.1)

    @llm_operation("sentiment")
    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        watchlist = state.get("watchlist", [])

        snapshot = state.get("news_snapshot", {})
        general_news = snapshot.get("general", [])
        by_symbol = snapshot.get("by_symbol", {})
        if not snapshot.get("as_of"):
            errors.append(self.log("Shared news snapshot is unavailable."))
            return {"sentiment_signals": {}, "agent_logs": logs, "errors": errors}

        for symbol in watchlist:
            try:
                if state.get("research_mode"):
                    all_news = by_symbol.get(symbol) or general_news[:5]
                    if not all_news:
                        logs.append(self.log("{}: Sentiment unavailable — no news data.", symbol))
                        continue
                    news_text = format_news_for_prompt(all_news)
                else:
                    profile = state.get("llm_profile")
                    run_id = profile.get("run_id") if isinstance(profile, dict) else None
                    if not isinstance(run_id, str) or not run_id:
                        raise ValueError("Sentiment scan ID is unavailable")
                    news_text, digest = candidate_news(snapshot, symbol)

                if self._llm:
                    signal = validate_sentiment(self._llm_sentiment(symbol, news_text))
                else:
                    errors.append(self.log("{}: Sentiment model is unavailable.", symbol))
                    continue

                if not state.get("research_mode"):
                    signal = {**signal, "evidence": {
                        "run_id": run_id, "symbol": symbol, "snapshot_as_of": snapshot["as_of"],
                        "news_digest": digest,
                    }}
                signals[symbol] = signal

                if signal.get("signal") != "HOLD":
                    logs.append(
                        self.log("{}: {} (sentiment score: {})", symbol, signal["signal"], signal.get("sentiment_score", "N/A"))
                    )

            except Exception as exc:
                errors.append(self.log("{}: Sentiment analysis error — {}", symbol, exc))

        logs.append(self.log("Sentiment analysis complete for {} symbols.", len(signals)))

        return {
            "sentiment_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }

    def _llm_sentiment(self, symbol: str, news_text: str) -> dict[str, Any]:
        """Use Claude to score sentiment."""
        prompt = f"""Analyze sentiment for {symbol} from these Indian financial news articles.
Focus on: earnings announcements, management changes, regulatory issues,
          sector tailwinds/headwinds, FII interest, analyst upgrades/downgrades.
The list may include general market news and search results. Do not assume that
every article specifically concerns {symbol}.

Articles:
{news_text}

Output ONLY valid JSON (no markdown):
{{"sentiment_score": -100, "signal": "BUY/SELL/HOLD", "confidence": 0-100, "key_events": ["..."], "reasoning": "..."}}

Where sentiment_score ranges from -100 (extremely negative) to +100 (extremely positive)."""

        resp = self._llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if not isinstance(content, str):
            raise ValueError("Sentiment response is not text")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Sentiment response contains no JSON object")
        return json.loads(content[start:end])


def sentiment_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return SentimentAgent().run(state)
