"""Point-in-time positional screening and drift-band portfolio helpers."""

from __future__ import annotations

from typing import Any


def dcf_value_per_share(
    free_cash_flows: list[float],
    *,
    discount_rate: float,
    terminal_growth_rate: float,
    net_debt: float,
    diluted_shares: float,
) -> float:
    """Calculate a deterministic discounted-cash-flow value from supplied data."""
    if not free_cash_flows or diluted_shares <= 0:
        raise ValueError("positive diluted_shares and at least one cash flow are required")
    if discount_rate <= terminal_growth_rate or terminal_growth_rate < 0:
        raise ValueError("discount_rate must exceed non-negative terminal growth")
    if any(value < 0 for value in free_cash_flows):
        raise ValueError("negative forecast cash flows are not supported")
    projected_value = sum(
        cash_flow / ((1 + discount_rate) ** year)
        for year, cash_flow in enumerate(free_cash_flows, start=1)
    )
    terminal = free_cash_flows[-1] * (1 + terminal_growth_rate) / (
        (discount_rate - terminal_growth_rate) * (1 + discount_rate) ** len(free_cash_flows)
    )
    return (projected_value + terminal - net_debt) / diluted_shares


def rebalance_targets(
    current_weights: dict[str, float], target_weights: dict[str, float], drift_band: float = 0.05
) -> dict[str, float]:
    """Return target weights only for positions outside their allowed drift band."""
    if not 0 <= drift_band <= 1:
        raise ValueError("drift_band must be between 0 and 1")
    changes = {}
    for symbol in set(current_weights) | set(target_weights):
        current = current_weights.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        if current < 0 or target < 0:
            raise ValueError("portfolio weights cannot be negative")
        if abs(current - target) > drift_band:
            changes[symbol] = target
    return changes


def screen_smart_money(
    symbol: str, *, price: float, fundamentals: dict[str, Any],
    flow_data: dict[str, Any], max_valuation_multiple: float = 25.0,
) -> dict[str, Any] | None:
    """Screen a symbol using caller-supplied, point-in-time fundamentals and flows."""
    required = ("published_at", "pe_ratio", "promoter_holding_pct")
    if any(key not in fundamentals for key in required):
        return None
    if price <= 0 or fundamentals["pe_ratio"] <= 0:
        return None
    flow_confirmed = (
        float(flow_data.get("fii_net_buy_inr", 0)) > 0
        or float(flow_data.get("dii_net_buy_inr", 0)) > 0
        or int(flow_data.get("bulk_block_buy_count", 0)) > int(flow_data.get("bulk_block_sell_count", 0))
    )
    if not flow_confirmed or float(fundamentals["pe_ratio"]) > max_valuation_multiple:
        return None
    return {
        "symbol": symbol,
        "price": price,
        "valuation_multiple": float(fundamentals["pe_ratio"]),
        "promoter_holding_pct": float(fundamentals["promoter_holding_pct"]),
        "holding_period": "POSITIONAL",
        "strategy": "SmartMoneyPositional",
        "fundamentals_published_at": fundamentals["published_at"],
    }
