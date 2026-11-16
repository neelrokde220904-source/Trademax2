import numpy as np
import pytest

from backtester.costs import (
    calculate_indian_transaction_cost,
    round_trip_cost,
    round_trip_cost_vectorized,
)


def test_equity_brokerage_applies_minimum_and_cap():
    assert calculate_indian_transaction_cost(
        1_000, 1, is_buy=True, is_intraday=True,
    ) == pytest.approx(5 + 0.0297 + 0.001 + (5 + 0.0297 + 0.001) * 0.18 + 0.03)
    assert calculate_indian_transaction_cost(
        100_000, 10, is_buy=True, is_intraday=True,
    ) == pytest.approx(20 + 29.7 + 1.0 + (20 + 29.7 + 1.0) * 0.18 + 30.0)
    assert calculate_indian_transaction_cost(
        1_000, 0, is_buy=True, is_intraday=True,
    ) == 0


def test_stamp_duty_and_stt_follow_product_and_side():
    intraday_buy = calculate_indian_transaction_cost(2_000, 1, is_buy=True, is_intraday=True)
    intraday_sell = calculate_indian_transaction_cost(2_000, 1, is_buy=False, is_intraday=True)
    delivery_buy = calculate_indian_transaction_cost(2_000, 1, is_buy=True, is_intraday=False)
    delivery_sell = calculate_indian_transaction_cost(2_000, 1, is_buy=False, is_intraday=False)

    charges = 0.0594 + 0.002 + (5 + 0.0594 + 0.002) * 0.18
    assert intraday_buy == pytest.approx(5 + charges + 0.06)
    assert intraday_sell == pytest.approx(5 + charges + 0.5)
    assert delivery_buy == pytest.approx(5 + charges + 0.3 + 2)
    assert delivery_sell == pytest.approx(5 + charges + 2)


def test_short_round_trip_applies_buy_charges_on_exit():
    expected = (
        calculate_indian_transaction_cost(100, 2, is_buy=False, is_intraday=True)
        + calculate_indian_transaction_cost(90, 2, is_buy=True, is_intraday=True)
    )
    assert round_trip_cost(100, 90, 2, is_intraday=True, is_entry_buy=False) == pytest.approx(expected)


def test_vectorized_round_trip_matches_scalar_costs():
    exits = np.array([99.0, 100.0, 101.0])
    vectorized = round_trip_cost_vectorized(100.0, exits, 3, is_intraday=True)
    expected = np.array([
        round_trip_cost(100.0, float(exit_price), 3, is_intraday=True)
        for exit_price in exits
    ])
    np.testing.assert_allclose(vectorized, expected)
