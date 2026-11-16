"""Fundamental Agent — P/E, P/B, ROE, debt ratios, promoter holdings via LLM synthesis."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from agents.llm_client import LLMClient
from config.llm_runtime import current_profile, llm_operation

NUMERIC_FIELDS = {
    "pe_ratio", "forward_pe", "pb_ratio", "roe", "roce", "debt_to_equity",
    "revenue_growth", "earnings_growth", "profit_margins", "market_cap",
    "dividend_yield", "book_value", "52w_high", "52w_low", "current_price",
    "target_mean_price",
}


def validate_fundamentals(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Fundamental data is unavailable")
    available = 0
    for key in NUMERIC_FIELDS:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid fundamental metric {key}")
        available += 1
    if not available:
        raise ValueError("No numeric fundamental metrics are available")
    return data


def validate_fundamental_signal(signal: Any) -> dict[str, Any]:
    if not isinstance(signal, dict) or signal.get("signal") not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("Invalid fundamental action")
    confidence = signal.get("confidence")
    if (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence) or not 0 <= confidence <= 100
    ):
        raise ValueError("Invalid fundamental confidence")
    if not isinstance(signal.get("reasoning"), str) or not signal["reasoning"].strip():
        raise ValueError("Invalid fundamental reasoning")
    risks = signal.get("key_risks")
    if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
        raise ValueError("Invalid fundamental risks")
    return signal


class FundamentalAgent(BaseAgent):
    """Fetches fundamentals from yfinance/NSE and uses Claude Haiku to synthesise signals."""

    def __init__(self) -> None:
        super().__init__("FundamentalAgent")
        self._llm = LLMClient("fast", "fundamental", max_tokens=1024, temperature=0.1)

    @llm_operation("fundamental")
    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        watchlist = state.get("watchlist", [])
        captured = state.get("research_fundamental_inputs", {})
        replay = state.get("research_mode", False)
        if replay and any(symbol not in captured for symbol in watchlist):
            raise ValueError("Missing captured fundamental inputs for research replay")
        inputs: dict[str, Any] = {}

        for symbol in watchlist:
            try:
                fundamentals = captured[symbol] if replay else self._fetch_fundamentals(symbol)
                fundamentals = validate_fundamentals(fundamentals)
                inputs[symbol] = fundamentals

                if replay and current_profile().mode == "rules":
                    signal = self._rule_based_signal(symbol, fundamentals)
                elif self._llm:
                    signal = validate_fundamental_signal(self._llm_synthesise(symbol, fundamentals))
                else:
                    raise ValueError("Fundamental model is unavailable")

                signals[symbol] = signal

                if signal.get("signal") != "HOLD":
                    logs.append(
                        self.log("{}: {} (confidence: {}%)", symbol, signal["signal"], signal.get("confidence", 0))
                    )

            except Exception as exc:
                errors.append(self.log("{}: Fundamental analysis error — {}", symbol, exc))

        logs.append(self.log("Fundamental analysis complete for {} symbols.", len(signals)))

        return {
            "fundamental_signals": signals,
            "research_fundamental_inputs": inputs,
            "agent_logs": logs,
            "errors": errors,
        }

    def _fetch_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Fetch fundamental data from yfinance."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info

            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "roce": info.get("returnOnAssets"),  # Approximation
                "debt_to_equity": info.get("debtToEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margins": info.get("profitMargins"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "Unknown"),
                "sector_source": "yfinance.Ticker.info",
                "sector_observed_at": datetime.now(timezone.utc).isoformat(),
                "industry": info.get("industry", "Unknown"),
                "dividend_yield": info.get("dividendYield"),
                "book_value": info.get("bookValue"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice"),
                "target_mean_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey"),
            }
        except Exception as exc:
            logger.warning("yfinance fundamentals unavailable for {} ({})", symbol, type(exc).__name__)
            return None

    def _llm_synthesise(self, symbol: str, data: dict) -> dict[str, Any]:
        """Use Claude Haiku to synthesise fundamental signals."""
        from agents.state_condenser import state_to_prompt_json

        context_json = state_to_prompt_json(data, max_chars=2000)
        prompt = f"""You are a fundamental analyst for Indian stocks (NSE/BSE).
Analyze {symbol} with these metrics:
{context_json}

Indian market context:
- SEBI regulations, GST impact, RBI rate decisions matter
- Promoter holding trends are more significant in India than Western markets
- Sector: {data.get('sector', 'Unknown')} — consider sector-specific India risks

Output ONLY valid JSON (no markdown):
{{"signal": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "...", "key_risks": ["..."]}}"""

        resp = self._llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if not isinstance(content, str):
            raise ValueError("Fundamental response is not text")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Fundamental response contains no JSON object")
        return json.loads(content[start:end])

    def _rule_based_signal(self, symbol: str, data: dict) -> dict[str, Any]:
        """Simple rule-based fundamental scoring when LLM unavailable."""
        score = 0
        reasons = []

        pe = data.get("pe_ratio")
        if pe is not None:
            if pe < 15:
                score += 2
                reasons.append(f"Low P/E ({pe:.1f})")
            elif pe > 40:
                score -= 2
                reasons.append(f"High P/E ({pe:.1f})")

        roe = data.get("roe")
        if roe is not None:
            if roe > 0.15:
                score += 1
                reasons.append(f"Good ROE ({roe:.1%})")
            elif roe < 0.05:
                score -= 1
                reasons.append(f"Low ROE ({roe:.1%})")

        de = data.get("debt_to_equity")
        if de is not None:
            if de > 150:
                score -= 2
                reasons.append(f"High debt ({de:.0f})")
            elif de < 50:
                score += 1

        growth = data.get("revenue_growth")
        if growth is not None and growth > 0.10:
            score += 1
            reasons.append(f"Revenue growing ({growth:.1%})")

        if score >= 3:
            signal = "BUY"
            confidence = min(75, 45 + score * 5)
        elif score <= -3:
            signal = "SELL"
            confidence = min(75, 45 + abs(score) * 5)
        else:
            signal = "HOLD"
            confidence = 30

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": " | ".join(reasons) if reasons else "Mixed fundamentals",
            "key_risks": [],
        }


def fundamental_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return FundamentalAgent().run(state)
