"""Rule-based regime-to-strategy allocation; no model calls in this gate."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings


@dataclass(frozen=True)
class StrategySelection:
    """Active strategy sleeves and mandatory entry/flatten decisions."""

    active_strategies: tuple[str, ...]
    halt_new_entries: bool
    flatten_intraday: bool
    tighten_swing_stops: bool
    reason: str


def select_strategies(
    *, adx: float, choppiness: float, india_vix: float, daily_drawdown: float,
    hmm_regime: str = "",
) -> StrategySelection:
    """Select strategy sleeves using explicit deterministic regime thresholds."""
    if any(value < 0 for value in (adx, choppiness, india_vix, daily_drawdown)):
        raise ValueError("market indicators and drawdown cannot be negative")
    if india_vix >= settings.vix_halt_threshold or hmm_regime.upper() in {
        "CRISIS", "SYSTEMIC_PANIC", "HIGH_VOLATILITY",
    }:
        return StrategySelection((), True, True, True, "High volatility")
    if daily_drawdown >= settings.max_daily_loss_pct:
        return StrategySelection((), True, False, False, "Daily loss limit")
    if adx > 25 and choppiness < 38.2 and india_vix < 20:
        return StrategySelection(
            ("orb_intraday", "supertrend_adx", "momentum_breakout", "vcp_nr7_swing",
             "btst_strength", "etf_rotation"),
            False, False, False, "Trending regime with normal volatility",
        )
    if adx < 20 and choppiness > 61.8 and india_vix < 22:
        return StrategySelection(
            ("mean_reversion", "nifty_options_seller", "pairs_trading", "etf_rotation"),
            False, False, False, "Range-bound regime with contained volatility",
        )
    return StrategySelection((), True, False, True, "Regime is not sufficiently classified")
