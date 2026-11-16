"""Graph node: collect trade candidates from the strategy sleeves that are explicitly enabled.

With every ``strategy_<name>_enabled`` flag off (the default) the node returns an empty update,
so the pipeline is unchanged for CompositeTrend. Enabled sleeves still go through the same
technical overlay, AI review, risk approval, router gates and exits as any other entry.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from agents.base_agent import TradingState

IST = ZoneInfo("Asia/Kolkata")


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """Plain, JSON-serialisable view of a candidate (state is checkpointed)."""
    return {
        "symbol": candidate.asset.symbol, "segment": candidate.asset.segment,
        "candidate_id": candidate.candidate_id, "strategy": candidate.strategy,
        "strategy_version": candidate.strategy_version, "side": candidate.side,
        "horizon": candidate.horizon, "product": candidate.product,
        "signal_time": candidate.signal_time.isoformat(),
        "expires_at": candidate.expires_at.isoformat(),
        "entry_price": float(candidate.entry_price), "stop_loss": float(candidate.stop_loss),
        "target": float(candidate.target), "exit_policy": candidate.exit_policy,
        "atr_value": float(candidate.atr_value), "atr_timeframe": candidate.atr_timeframe,
        "thesis": candidate.thesis, "tick_size": float(candidate.tick_size),
        "lot_size": candidate.lot_size,
        "review_contract_version": candidate.review_contract_version,
    }


def _market_indicators(state: TradingState, symbols: list[str]) -> tuple[float, float]:
    """Median ADX(14) and Choppiness(14) across the qualified symbols' 15-minute bars."""
    import pandas_ta as ta

    adx_values: list[float] = []
    chop_values: list[float] = []
    for symbol in symbols:
        frame = state.get("market_data", {}).get(symbol)
        if not isinstance(frame, pd.DataFrame) or len(frame) < 60:
            continue
        adx = ta.adx(frame["high"], frame["low"], frame["close"], length=14)
        chop = ta.chop(frame["high"], frame["low"], frame["close"], length=14)
        if adx is not None and "ADX_14" in adx and pd.notna(adx["ADX_14"].iloc[-1]):
            adx_values.append(float(adx["ADX_14"].iloc[-1]))
        if chop is not None and len(chop) and pd.notna(chop.iloc[-1]):
            chop_values.append(float(chop.iloc[-1]))
    if not adx_values or not chop_values:
        raise ValueError("market regime indicators are unavailable")
    return float(pd.Series(adx_values).median()), float(pd.Series(chop_values).median())


def build_bars_by_timeframe(state: TradingState, symbol: str, as_of: datetime) -> dict[str, Any]:
    """Bar sets for the adapters, derived only from the scan's own captured history."""
    from strategies.adapters import daily_from_intraday

    fifteen = state.get("market_data", {}).get(symbol)
    five = state.get("five_minute_bars", {}).get(symbol) if isinstance(state.get("five_minute_bars"), dict) else None
    bars: dict[str, Any] = {"15m": fifteen, "5m": five}
    if isinstance(fifteen, pd.DataFrame):
        bars["1d"] = daily_from_intraday(fifteen, as_of)
    return bars


def strategy_candidates_node(state: TradingState) -> dict[str, Any]:
    from strategies.adapters import enabled_adapters

    adapters = enabled_adapters()
    if not adapters:
        return {}
    logs: list[str] = []
    errors: list[str] = []
    qualified = [
        s for s in (state.get("screener_results") or {}).get("qualified", []) if isinstance(s, str)
    ]
    from config.settings import settings
    from models.signal_gate.strategy_selector import select_strategies

    try:
        adx, chop = _market_indicators(state, qualified)
        vix = float(state.get("india_vix") or 0)
        if vix <= 0:
            raise ValueError("India VIX is unavailable")
        from broker.portfolio_tracker import portfolio_tracker

        regime = (state.get("hmm_regime") or {}).get("regime") or ""
        selection = select_strategies(
            adx=adx, choppiness=chop, india_vix=vix,
            daily_drawdown=portfolio_tracker.daily_loss_pct, hmm_regime=str(regime),
        )
    except (ValueError, TypeError) as exc:
        # Unknown regime: no sleeve candidates (the core strategy is unaffected).
        return {"agent_logs": [f"[StrategyCandidates] Skipped: {exc}"], "strategy_candidates": {}}
    if selection.halt_new_entries:
        return {
            "agent_logs": [f"[StrategyCandidates] No sleeve candidates: {selection.reason}"],
            "strategy_candidates": {},
        }
    as_of = datetime.now(IST)
    candidates: dict[str, dict[str, Any]] = {}
    from config.instruments import get_instrument

    for adapter in adapters:
        if adapter.key not in selection.active_strategies:
            continue
        etf_peers = {
            peer: state["market_data"].get(peer) for peer in settings.etf_rotation_universe
        }
        for symbol in qualified:
            if symbol in candidates:
                continue  # one candidate per symbol per scan
            try:
                bars = build_bars_by_timeframe(state, symbol, as_of)
                if adapter.key == "etf_rotation":
                    bars["1d_peers"] = {
                        peer: build_bars_by_timeframe(state, peer, as_of).get("1d")
                        for peer, frame in etf_peers.items() if frame is not None and peer != symbol
                    }
                tick, lot = Decimal("0.05"), 1
                instrument = get_instrument(symbol, "NSE")
                if instrument:
                    tick = Decimal(str(float(instrument["tick_size"]) / 100)) or tick
                    lot = max(1, int(instrument.get("lotsize") or 1))
                candidate = adapter.generate(symbol, bars, as_of, tick=tick, lot_size=lot)
            except Exception as exc:
                errors.append(f"[StrategyCandidates] {adapter.key} {symbol}: {type(exc).__name__}")
                continue
            if candidate is not None:
                candidates[symbol] = candidate_to_dict(candidate)
                logs.append(
                    f"[StrategyCandidates] {candidate.strategy}: {candidate.side} {symbol} "
                    f"({candidate.horizon})"
                )
    logger.info("[StrategyCandidates] {} sleeve candidate(s).", len(candidates))
    return {"strategy_candidates": candidates, "agent_logs": logs, "errors": errors}
