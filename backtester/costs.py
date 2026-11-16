"""Shared deterministic Indian equity transaction-cost calculations."""

from __future__ import annotations

import numpy as np

from config.settings import settings


def calculate_indian_transaction_cost(
    price: float, quantity: int, *, is_buy: bool, is_intraday: bool = False
) -> float:
    """Return estimated one-side NSE equity costs using the configured broker schedule."""
    if price < 0 or quantity < 0:
        raise ValueError("price and quantity must be non-negative")
    turnover = price * quantity
    brokerage = (
        min(
            settings.brokerage_per_order,
            max(settings.brokerage_min_per_order, turnover * settings.brokerage_rate_pct),
        )
        if turnover > 0 else 0.0
    )
    if is_intraday:
        stt = 0.0 if is_buy else turnover * settings.stt_intraday_pct
    else:
        stt = turnover * settings.stt_delivery_pct
    exchange = turnover * settings.exchange_charges_pct
    sebi = turnover * settings.sebi_turnover_fee_pct
    gst = (brokerage + exchange + sebi) * settings.gst_on_brokerage_pct
    stamp_rate = settings.stamp_duty_intraday_pct if is_intraday else settings.stamp_duty_pct
    stamp = turnover * stamp_rate if is_buy else 0.0
    return brokerage + stt + exchange + sebi + gst + stamp


def round_trip_cost(
    entry_price: float, exit_price: float, quantity: int, *,
    is_intraday: bool = True, is_entry_buy: bool = True,
) -> float:
    """Compute entry and exit costs for a complete position."""
    return calculate_indian_transaction_cost(
        entry_price, quantity, is_buy=is_entry_buy, is_intraday=is_intraday
    ) + calculate_indian_transaction_cost(
        exit_price, quantity, is_buy=not is_entry_buy, is_intraday=is_intraday
    )


def round_trip_cost_vectorized(
    entry_price: float, exit_prices: np.ndarray, quantity: int, *,
    is_intraday: bool = True, is_entry_buy: bool = True,
) -> np.ndarray:
    """Vectorized round-trip costs for Monte Carlo paths."""
    exits = np.asarray(exit_prices, dtype=float)
    if entry_price < 0 or quantity < 0 or np.any(exits < 0):
        raise ValueError("prices and quantity must be non-negative")

    def side_cost(price: float | np.ndarray, *, is_buy: bool) -> np.ndarray:
        turnover = np.asarray(price, dtype=float) * quantity
        brokerage = np.where(
            turnover > 0,
            np.minimum(
                settings.brokerage_per_order,
                np.maximum(settings.brokerage_min_per_order, turnover * settings.brokerage_rate_pct),
            ),
            0.0,
        )
        if is_intraday:
            stt = np.zeros_like(turnover) if is_buy else turnover * settings.stt_intraday_pct
        else:
            stt = turnover * settings.stt_delivery_pct
        exchange = turnover * settings.exchange_charges_pct
        sebi = turnover * settings.sebi_turnover_fee_pct
        gst = (brokerage + exchange + sebi) * settings.gst_on_brokerage_pct
        stamp_rate = settings.stamp_duty_intraday_pct if is_intraday else settings.stamp_duty_pct
        stamp = turnover * stamp_rate if is_buy else 0
        return brokerage + stt + exchange + sebi + gst + stamp

    return side_cost(entry_price, is_buy=is_entry_buy) + side_cost(exits, is_buy=not is_entry_buy)
