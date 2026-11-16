"""Strategy contract registry: review requirements, horizons, venues and exit policy per strategy."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyReviewContract:
    strategy: str
    version: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    required_inputs: tuple[str, ...]


COMPOSITE_TREND_REVIEW = StrategyReviewContract(
    strategy="CompositeTrend",
    version="review-v2",
    required=("technical", "sentiment", "bull", "bear", "debate_synthesis", "portfolio_manager"),
    optional=("fundamental", "options", "macro", "india_specific", "geopolitical"),
    required_inputs=(
        "verified_15m_history", "current_attributable_news", "passed_quantitative_screen",
        "risk_approval", "valid_portfolio_snapshot",
    ),
)


@dataclass(frozen=True)
class StrategyContract:
    name: str
    version: str
    horizons: tuple[str, ...]
    asset_venues: tuple[str, ...]
    review: StrategyReviewContract | None
    enabled_setting: str | None  # None = always on; otherwise a boolean setting defaulting to False
    default_exit_policy: str
    hmm_tag: str = "trend_following"  # the HMM allowed_strategies entry that permits this strategy
    segment: str = "CASH"
    book: str | None = None  # paper book name; None = the core book

    def is_enabled(self) -> bool:
        if self.enabled_setting is None:
            return True
        from config.settings import settings

        return getattr(settings, self.enabled_setting, False) is True


COMPOSITE_TREND = StrategyContract(
    name="CompositeTrend",
    version="composite-trend-v1",
    horizons=("INTRADAY",),
    asset_venues=("NSE",),
    review=COMPOSITE_TREND_REVIEW,
    enabled_setting=None,
    default_exit_policy="daily_atr_baseline_v1",
)

def _sleeve_review(strategy: str) -> StrategyReviewContract:
    """Sleeves are AI-required exactly like CompositeTrend: same agents and inputs, own strategy name."""
    return StrategyReviewContract(
        strategy=strategy, version=COMPOSITE_TREND_REVIEW.version,
        required=COMPOSITE_TREND_REVIEW.required, optional=COMPOSITE_TREND_REVIEW.optional,
        required_inputs=COMPOSITE_TREND_REVIEW.required_inputs,
    )


def _sleeve(
    name: str, flag: str, horizon: str, policy: str, *, hmm_tag: str = "trend_following",
    segment: str = "CASH",
) -> StrategyContract:
    return StrategyContract(
        name=name, version=f"{name.lower()}-v1", horizons=(horizon,), asset_venues=("NSE",),
        review=_sleeve_review(name), enabled_setting=flag, default_exit_policy=policy,
        hmm_tag=hmm_tag, segment=segment, book=name,
    )


SLEEVE_CONTRACTS: tuple[StrategyContract, ...] = (
    _sleeve("ORBIntraday", "strategy_orb_intraday_enabled", "INTRADAY", "intraday_atr_2x_4x_v1"),
    _sleeve("SupertrendADX", "strategy_supertrend_adx_enabled", "INTRADAY", "intraday_atr_2x_4x_v1"),
    _sleeve("MeanReversion", "strategy_mean_reversion_enabled", "SWING", "daily_atr22_2x_4x_v1",
            hmm_tag="mean_reversion"),
    _sleeve("MomentumBreakout", "strategy_momentum_breakout_enabled", "SWING", "daily_atr22_2x_4x_v1"),
    _sleeve("VCPNR7Swing", "strategy_vcp_nr7_swing_enabled", "SWING", "daily_atr22_2x_4x_v1"),
    _sleeve("BTSTStrength", "strategy_btst_strength_enabled", "BTST", "daily_atr_baseline_v1"),
    _sleeve("ETFRotation", "strategy_etf_rotation_enabled", "POSITIONAL", "chandelier_atr22_k3_v1",
            segment="ETF"),
)

STRATEGY_CONTRACTS: dict[str, StrategyContract] = {
    COMPOSITE_TREND.name: COMPOSITE_TREND,
    **{contract.name: contract for contract in SLEEVE_CONTRACTS},
}


def strategy_contract(name: Any) -> StrategyContract | None:
    return STRATEGY_CONTRACTS.get(name) if isinstance(name, str) else None


def require_strategy_contract(name: Any) -> StrategyContract:
    """Unknown or disabled strategies fail closed (entries only; exits never consult this)."""
    contract = strategy_contract(name)
    if contract is None:
        raise ValueError("No declared contract for candidate strategy")
    if not contract.is_enabled():
        raise ValueError(f"Strategy {contract.name} is disabled")
    return contract


def require_strategy_review_contract(technical: Any) -> StrategyReviewContract:
    """Reject strategies without an explicitly supported review contract."""
    strategy = technical.get("strategy") if isinstance(technical, dict) else None
    contract = strategy_contract(strategy)
    if contract is None or contract.review is None or not contract.is_enabled():
        raise ValueError("No declared review contract for candidate strategy")
    return contract.review
