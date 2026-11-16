"""Quantitative Screener — pre-filters watchlist with fast numeric checks before LLM agents.

Runs AFTER hmm_regime, BEFORE the analyst fan-out. Reduces the number of symbols
sent to expensive LLM agents by eliminating:
- Stocks whose session-to-date volume is below prior sessions at the same time
- Stocks with ADX < 15 (no trend)
- Stocks trading inside a narrow range (15-minute ATR/Close < SCREENER_MIN_ATR_PCT)
- Stocks on F&O ban list

This is a pure-Python, zero-LLM-cost gate that typically removes 30-60% of the
watchlist, proportionally reducing Anthropic API spend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import pandas as pd
import pandas_ta as ta
from loguru import logger

from agents.base_agent import TradingState
from config.settings import settings
from data.indicators import (
    average_daily_turnover_inr,
    bar_relative_volume,
    best_quote_spread_pct,
    completed_intraday_bars,
    is_current_session_breakout,
    is_vwap_chop,
    session_relative_volume,
)


class QuantitativeScreener:
    """Fast numeric pre-filter before LLM analyst agents."""

    def __init__(
        self,
        min_volume_ratio: float | None = None,
        min_adx: float = 15.0,
        min_atr_pct: float | None = None,
    ) -> None:
        self.min_volume_ratio = (
            settings.screener_min_relative_volume if min_volume_ratio is None else min_volume_ratio
        )
        self.min_adx = min_adx
        self.min_atr_pct = settings.screener_min_atr_pct if min_atr_pct is None else min_atr_pct

    def screen(self, state: TradingState) -> dict[str, Any]:
        """Screen the watchlist and return only qualified symbols.

        Returns state update with filtered watchlist and screening logs.
        """
        watchlist = state.get("watchlist", [])
        market_data = state.get("market_data", {})
        live_quotes = state.get("live_quotes", {})
        regime = state.get("hmm_regime", {})

        if not watchlist:
            return {
                "watchlist": watchlist,
                "screener_results": {
                    "qualified": [], "rejected": [], "reasons": {},
                    "original_count": 0, "qualified_count": 0,
                },
                "agent_logs": ["[QuantScreener] Empty watchlist — no symbols to analyze."],
            }

        qualified: list[str] = []
        rejected: list[str] = []
        reasons: dict[str, str] = {}

        for symbol in watchlist:
            df = market_data.get(symbol)
            if not isinstance(df, pd.DataFrame) or df.empty:
                rejected.append(symbol)
                reasons[symbol] = "no_data"
                continue

            passed, reason = self._check_symbol(
                symbol, df, regime, state.get("research_as_of") or None,
                live_quotes.get(symbol),
            )
            if passed:
                qualified.append(symbol)
            else:
                rejected.append(symbol)
                reasons[symbol] = reason

        pct_removed = len(rejected) / len(watchlist) * 100 if watchlist else 0
        logs = [
            f"[QuantScreener] {len(qualified)}/{len(watchlist)} passed "
            f"({pct_removed:.0f}% filtered out). "
            f"Regime: {regime.get('regime', 'UNKNOWN')}"
        ]

        if rejected:
            logs.append(
                f"[QuantScreener] Rejected: {', '.join(rejected[:10])}"
                + (f" (+{len(rejected) - 10} more)" if len(rejected) > 10 else "")
            )

        return {
            "watchlist": qualified,
            "screener_results": {
                "qualified": qualified,
                "rejected": rejected,
                "reasons": reasons,
                "original_count": len(watchlist),
                "qualified_count": len(qualified),
            },
            "agent_logs": logs,
        }

    def _check_symbol(
        self, symbol: str, df: pd.DataFrame, regime: dict, as_of: str | None = None,
        quote: dict | None = None,
    ) -> tuple[bool, str]:
        """Run all numeric checks on a single symbol."""
        try:
            df = completed_intraday_bars(df, as_of)
            if len(df) < 20:
                return False, "insufficient_bars"

            close = df["close"] if "close" in df.columns else None
            volume = df["volume"] if "volume" in df.columns else None
            high = df["high"] if "high" in df.columns else None
            low = df["low"] if "low" in df.columns else None

            if close is None or volume is None:
                return False, "missing_ohlcv"

            turnover = average_daily_turnover_inr(
                df, as_of,
                minimum_sessions=settings.screener_min_liquidity_sessions,
            )
            if turnover is None:
                return False, "liquidity_history_unavailable"
            if turnover < settings.screener_min_average_daily_turnover_inr:
                return False, "low_daily_turnover"

            spread_pct = best_quote_spread_pct(
                quote, as_of, max_age_seconds=settings.screener_max_quote_age_seconds,
            )
            if spread_pct is None:
                return False, "spread_unavailable"
            if spread_pct > settings.screener_max_spread_pct:
                return False, "wide_spread"

            # Volume check: session-to-date volume vs prior sessions at the same time of day,
            # using closed bars only (the in-progress bar's volume is always tiny).
            relative_volume = session_relative_volume(df, as_of)
            if relative_volume is None:
                return False, "volume_unavailable"
            if relative_volume < self.min_volume_ratio:
                return False, "low_volume"
            if is_vwap_chop(
                df, as_of,
                lookback_bars=settings.screener_vwap_chop_bars,
                band_pct=settings.screener_vwap_chop_band_pct,
            ):
                return False, "vwap_chop"
            if is_current_session_breakout(
                df, as_of, lookback_bars=settings.screener_breakout_lookback_bars,
            ):
                breakout_volume = bar_relative_volume(df, as_of)
                if (
                    breakout_volume is None
                    or breakout_volume < settings.screener_breakout_min_relative_volume
                ):
                    return False, "breakout_relative_volume"

            # ADX check (skip in SYSTEMIC_PANIC — we don't trade anyway)
            if regime.get("regime") != "SYSTEMIC_PANIC" and high is not None and low is not None:
                adx = ta.adx(high, low, close, length=14)
                if adx is not None and len(adx) > 0:
                    adx_col = [c for c in adx.columns if "ADX" in c and "DM" not in c]
                    if adx_col:
                        adx_val = adx[adx_col[0]].iloc[-1]
                        if pd.notna(adx_val) and adx_val < self.min_adx:
                            return False, "low_adx"

            # ATR/Close ratio check — skip too-quiet stocks
            if high is not None and low is not None:
                atr = ta.atr(high, low, close, length=14)
                if atr is not None and len(atr) > 0:
                    atr_val = atr.iloc[-1]
                    close_val = close.iloc[-1]
                    if pd.notna(atr_val) and close_val > 0:
                        atr_pct = atr_val / close_val
                        if atr_pct < self.min_atr_pct:
                            return False, "narrow_range"

            return True, ""

        except Exception as exc:
            logger.exception("[QuantScreener] {} check failed; rejecting symbol.", symbol)
            return False, "screening_error"


def quantitative_screener_node(state: TradingState) -> dict[str, Any]:
    """LangGraph node function for the quantitative screener."""
    screener = QuantitativeScreener()
    return screener.screen(state)
