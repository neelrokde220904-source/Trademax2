"""Market Data Agent — fetches and preprocesses all market data for the pipeline."""

from __future__ import annotations

from typing import Any
import math

import pandas as pd

from agents.base_agent import BaseAgent, TradingState
from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from broker.websocket_feed import ws_feed
from config.instruments import get_token
from data.market_data import fetch_india_vix, fetch_multiple
from operations.history_evidence import history_digest


class MarketDataAgent(BaseAgent):
    """Fetches historical OHLCV, live quotes, VIX, and portfolio state."""

    def __init__(self) -> None:
        super().__init__("MarketDataAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        watchlist = state["watchlist"]
        logs.append(self.log("Fetching market data for {} symbols.", len(watchlist)))

        portfolio_snapshot_valid = False
        try:
            portfolio_tracker.sync_from_broker()
            portfolio_snapshot_valid = portfolio_tracker.update_unrealised_pnl()
            if not portfolio_snapshot_valid:
                errors.append(self.log("Fresh marks unavailable for one or more open positions."))
        except Exception as exc:
            errors.append(self.log("Portfolio sync error: {}", exc))

        # ----- Historical OHLCV -----
        market_data = {}
        market_data_provenance = {}
        try:
            history_symbols = list(dict.fromkeys([*watchlist, *portfolio_tracker.positions]))
            market_data = fetch_multiple(history_symbols, interval="15m", days=45)
            for symbol, frame in market_data.items():
                if not isinstance(frame, pd.DataFrame):
                    errors.append(self.log("{}: Historical data is not a DataFrame.", symbol))
                    continue
                provenance = frame.attrs.get("history_provenance")
                if not isinstance(provenance, dict):
                    errors.append(self.log("{}: Historical data provenance is unavailable.", symbol))
                    continue
                market_data_provenance[symbol] = dict(provenance)
                if not frame.empty:
                    try:
                        market_data_provenance[symbol]["bars_sha256"] = history_digest(frame)
                    except (TypeError, ValueError, OverflowError) as exc:
                        errors.append(self.log("{}: Historical data fingerprint failed: {}", symbol, type(exc).__name__))
            logs.append(self.log("Fetched historical data for {} symbols.", len(market_data)))
        except Exception as exc:
            err = self.log("Historical data fetch failed: {}", exc)
            errors.append(err)

        # ----- Live Quotes from WebSocket -----
        live_quotes = {}
        for sym in watchlist:
            token = get_token(sym)
            if token:
                tick = ws_feed.get_tick(token)
                if tick:
                    live_quotes[sym] = tick
                elif angel_client.has_credentials:
                    # Fallback to REST API LTP
                    ltp = angel_client.get_ltp("NSE", sym, token)
                    if ltp:
                        live_quotes[sym] = {"ltp": ltp, "token": token}

        logs.append(self.log("Got live quotes for {} symbols.", len(live_quotes)))

        # ----- India VIX -----
        india_vix = 0.0
        india_vix_source = None
        try:
            india_vix, india_vix_source = fetch_india_vix()
            if not math.isfinite(india_vix) or not 0 < india_vix <= 200:
                raise ValueError("Invalid India VIX level; refusing an unrelated index value")
            logs.append(self.log("India VIX: {:.2f} ({})", india_vix, india_vix_source))
        except Exception as exc:
            india_vix = 0.0
            errors.append(self.log("India VIX unavailable: {}", type(exc).__name__))

        return {
            "market_data": market_data,
            "market_data_provenance": market_data_provenance,
            "live_quotes": live_quotes,
            "india_vix": india_vix,
            "india_vix_source": india_vix_source,
            "portfolio_value": portfolio_tracker.total_value,
            "available_capital": portfolio_tracker.available_capital,
            "current_positions": portfolio_tracker.positions,
            "portfolio_snapshot_valid": portfolio_snapshot_valid,
            "agent_logs": logs,
            "errors": errors,
        }


def market_data_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return MarketDataAgent().run(state)
