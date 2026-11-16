"""Technical Agent — RSI, MACD, EMA, Supertrend, ADX, Bollinger, VWAP analysis."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings
from data.indicators import (
    completed_intraday_bars,
    compute_all_indicators,
    daily_atr,
    get_latest_signals,
)
from strategies.contracts import COMPOSITE_TREND_REVIEW


class TechnicalAgent(BaseAgent):
    """Computes technical indicators and generates signals for all watchlist symbols."""

    def __init__(self) -> None:
        super().__init__("TechnicalAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        market_data = state.get("market_data", {})
        live_quotes = state.get("live_quotes", {})

        if not market_data:
            logs.append(self.log("No market data available. Skipping technical analysis."))
            return {"technical_signals": {}, "agent_logs": logs, "errors": errors}

        watchlist = state.get("watchlist")
        symbols = watchlist if watchlist is not None else list(market_data)
        for symbol in symbols:
            try:
                df = market_data.get(symbol)
                if df is None:
                    logs.append(self.log("{}: No market data.", symbol))
                    continue
                completed = completed_intraday_bars(df, state.get("research_as_of") or None)
                if completed.empty or len(completed) < 30:
                    logs.append(self.log("{}: Insufficient completed data ({} rows).", symbol, len(completed)))
                    continue

                # Compute all indicators
                df_with_indicators = compute_all_indicators(completed)

                # Extract signals
                signal_data = get_latest_signals(df_with_indicators)
                signal_data["strategy"] = COMPOSITE_TREND_REVIEW.strategy
                source_data = state.get("market_data_provenance")
                if source_data is not None and not isinstance(source_data, dict):
                    raise ValueError("Invalid historical data provenance")
                provenance = source_data.get(symbol) if source_data is not None else None
                if provenance is not None and not isinstance(provenance, dict):
                    raise ValueError("Invalid historical data provenance for symbol")
                if isinstance(provenance, dict):
                    signal_data["history_provenance"] = dict(provenance)

                # Entry levels use daily ATR: 15-minute ATR targets are too small to clear costs.
                close = signal_data["indicators"].get("close", 0)
                rsi = signal_data["indicators"].get("rsi_14")
                signal = signal_data.get("signal")
                if signal in ("BUY", "STRONG_BUY") and not (
                    isinstance(rsi, (int, float)) and 40 <= rsi <= 70
                ):
                    signal_data["signal"] = signal_data["composite_signal"] = "HOLD"
                    signal_data["confidence"] = 0
                    signal_data["reasoning"] += " Entry blocked: trend longs require RSI 40–70."
                elif signal in ("SELL", "STRONG_SELL") and not (
                    isinstance(rsi, (int, float)) and 30 <= rsi <= 60
                ):
                    signal_data["signal"] = signal_data["composite_signal"] = "HOLD"
                    signal_data["confidence"] = 0
                    signal_data["reasoning"] += " Entry blocked: trend shorts require RSI 30–60."
                day_atr = daily_atr(df, state.get("research_as_of") or None)
                signal_data["indicators"]["daily_atr_14"] = day_atr
                stop_distance = settings.technical_stop_daily_atr_multiplier * (day_atr or 0)
                target_distance = settings.technical_target_daily_atr_multiplier * (day_atr or 0)

                if day_atr and close:
                    if signal_data["signal"] in ("BUY", "STRONG_BUY"):
                        signal_data["stop_loss"] = round(close - stop_distance, 2)
                        signal_data["target"] = round(close + target_distance, 2)
                    elif signal_data["signal"] in ("SELL", "STRONG_SELL"):
                        signal_data["stop_loss"] = round(close + stop_distance, 2)
                        signal_data["target"] = round(close - target_distance, 2)

                signals[symbol] = signal_data

                if signal_data["signal"] != "HOLD":
                    logs.append(
                        self.log(
                            "{}: {} (confidence: {}%) — {}",
                            symbol,
                            signal_data["signal"],
                            signal_data["confidence"],
                            signal_data["reasoning"],
                        )
                    )

            except Exception as exc:
                errors.append(self.log("{}: Technical analysis error — {}", symbol, exc))

        # Enabled strategy sleeves may supply a candidate for symbols the core strategy left on
        # HOLD. The overlay keeps the symbol's verified history provenance, so the same AI review,
        # risk approval and router gates apply. With no sleeve enabled this loop never runs.
        for symbol, candidate in (state.get("strategy_candidates") or {}).items():
            base = signals.get(symbol)
            if not isinstance(base, dict) or "history_provenance" not in base:
                continue
            if base.get("signal") in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
                continue  # the core strategy owns a symbol it already signals
            side = candidate["side"]
            signals[symbol] = {
                **base,
                "strategy": candidate["strategy"], "signal": side, "composite_signal": side,
                "stop_loss": candidate["stop_loss"], "target": candidate["target"],
                "holding_period": candidate["horizon"], "exit_policy": candidate["exit_policy"],
                "atr_timeframe": candidate["atr_timeframe"],
                "sleeve_candidate_id": candidate["candidate_id"],
                "confidence": 60, "reasoning": candidate["thesis"],
                "indicators": {**base.get("indicators", {}), "policy_atr": candidate["atr_value"]},
            }
            logs.append(self.log("{}: sleeve {} {} ({})", symbol, candidate["strategy"], side, candidate["horizon"]))

        logs.append(self.log("Technical analysis complete for {} symbols.", len(signals)))

        return {
            "technical_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }


def technical_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return TechnicalAgent().run(state)
