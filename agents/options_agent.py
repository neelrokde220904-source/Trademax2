"""Options Agent — options chain analysis: PCR, IV, max pain, OI analysis."""

from __future__ import annotations

from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings
from data.options_data import (
    calculate_max_pain,
    calculate_pcr,
    detect_unusual_oi,
    fetch_options_chain,
    find_max_oi_strikes,
)


class OptionsAgent(BaseAgent):
    """Analyses NIFTY and BANKNIFTY options chains for market signals."""

    def __init__(self) -> None:
        super().__init__("OptionsAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        live_quotes = state.get("live_quotes", {})
        captured = state.get("research_options_inputs", {})
        replay = state.get("research_mode", False)
        underlyings = ["NIFTY", "BANKNIFTY"]
        if replay and any(
            name not in captured or "chain" not in captured[name] or "spot" not in captured[name]
            for name in underlyings
        ):
            raise ValueError("Missing captured options inputs for research replay")
        inputs: dict[str, Any] = {}

        for underlying in underlyings:
            try:
                chain = captured[underlying]["chain"] if replay else fetch_options_chain(underlying)

                if chain.empty:
                    inputs[underlying] = {"chain": chain, "spot": 0}
                    logs.append(self.log("{}: No options data available.", underlying))
                    continue

                # PCR
                pcr = calculate_pcr(chain)
                if pcr is None:
                    spot = captured[underlying]["spot"] if replay else live_quotes.get(underlying, {}).get("ltp", 0)
                    inputs[underlying] = {"chain": chain, "spot": spot}
                    signals[underlying] = {
                        "signal": "HOLD", "confidence": 0, "data_status": "unavailable",
                        "reasoning": "Valid single-expiry call and put open interest is unavailable.",
                        "pcr": None, "max_pain": None, "support": None, "resistance": None,
                        "unusual_oi_count": 0, "spot": spot,
                    }
                    logs.append(self.log("{}: OI data unavailable; options context abstains.", underlying))
                    continue

                # Get spot price
                spot = captured[underlying]["spot"] if replay else live_quotes.get(underlying, {}).get("ltp", 0)
                if not spot and not replay:
                    # Try to get from Nifty index token
                    spot_token = settings.nifty50_token if underlying == "NIFTY" else settings.banknifty_token
                    from broker.angel_client import angel_client
                    spot = angel_client.get_ltp("NSE", underlying, spot_token) or 0
                inputs[underlying] = {"chain": chain, "spot": spot}

                # Max pain
                max_pain = calculate_max_pain(chain, spot) if spot else None

                # Max OI strikes (support/resistance)
                oi_strikes = find_max_oi_strikes(chain)

                # Unusual OI
                unusual = detect_unusual_oi(chain)

                # Derive signal from PCR
                signal = "HOLD"
                confidence = 30
                reasoning_parts = []

                if pcr is not None:
                    reasoning_parts.append(f"PCR: {pcr:.3f}")
                    if pcr > 1.2:
                        signal = "BUY"  # Contrarian — high put writing = bullish
                        confidence = 65
                        reasoning_parts.append("High PCR (contrarian bullish)")
                    elif pcr < 0.7:
                        signal = "SELL"  # Contrarian — high call writing = bearish
                        confidence = 65
                        reasoning_parts.append("Low PCR (contrarian bearish)")

                if max_pain:
                    reasoning_parts.append(f"Max Pain: {max_pain}")
                    if spot and abs(spot - max_pain) / spot > 0.02:
                        reasoning_parts.append("Spot far from max pain — reversion expected")

                if oi_strikes.get("max_ce_oi_strike"):
                    reasoning_parts.append(f"Resistance (max CE OI): {oi_strikes['max_ce_oi_strike']}")
                if oi_strikes.get("max_pe_oi_strike"):
                    reasoning_parts.append(f"Support (max PE OI): {oi_strikes['max_pe_oi_strike']}")

                if unusual:
                    reasoning_parts.append(f"Unusual OI detected in {len(unusual)} contracts")

                signals[underlying] = {
                    "signal": signal,
                    "confidence": confidence,
                    "data_status": "available",
                    "reasoning": " | ".join(reasoning_parts),
                    "pcr": pcr,
                    "max_pain": max_pain,
                    "support": oi_strikes.get("max_pe_oi_strike"),
                    "resistance": oi_strikes.get("max_ce_oi_strike"),
                    "unusual_oi_count": len(unusual),
                    "spot": spot,
                }

                logs.append(self.log("{}: PCR={}, MaxPain={}, Signal={}", underlying, pcr, max_pain, signal))

            except Exception as exc:
                errors.append(self.log("{}: Options analysis error — {}", underlying, exc))

        return {
            "options_signals": signals,
            "research_options_inputs": inputs,
            "agent_logs": logs,
            "errors": errors,
        }


def options_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return OptionsAgent().run(state)
