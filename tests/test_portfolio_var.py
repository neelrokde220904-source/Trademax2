import numpy as np
import pandas as pd
import pytest

from risk.daily_portfolio_risk import RiskHistoryUnavailable
from risk.portfolio_var import historical_portfolio_var_pct


def _returns(values, *, start="2026-09-01"):
    return pd.Series(values, index=pd.date_range(start, periods=20, freq="D"))


def test_portfolio_var_uses_signed_exposures_and_aligned_history():
    adverse = _returns([-0.10, -0.10, *([0.01] * 18)])
    result = historical_portfolio_var_pct(
        {
            "LONG": {"quantity": 10, "ltp": 100},
            "SHORT": {"quantity": -10, "ltp": 100},
        },
        lambda _symbol: adverse,
        1_000,
    )
    assert result == pytest.approx(0.0)

    long_only = historical_portfolio_var_pct(
        {"LONG": {"quantity": 10, "ltp": 100}},
        lambda _symbol: adverse,
        1_000,
    )
    assert long_only == pytest.approx(0.10)


def test_portfolio_var_requires_exactly_aligned_complete_returns():
    first = _returns(np.linspace(-0.02, 0.02, 20))
    second = _returns(np.linspace(-0.02, 0.02, 20), start="2026-09-02")
    with pytest.raises(RiskHistoryUnavailable, match="aligned"):
        historical_portfolio_var_pct(
            {
                "FIRST": {"quantity": 1, "ltp": 100},
                "SECOND": {"quantity": 1, "ltp": 100},
            },
            lambda symbol: {"FIRST": first, "SECOND": second}[symbol],
            1_000,
        )


def test_portfolio_var_rejects_missing_marks_and_nan_returns():
    with pytest.raises(RiskHistoryUnavailable, match="quantity or mark"):
        historical_portfolio_var_pct(
            {"MISSING": {"quantity": 1}},
            lambda _symbol: _returns(np.zeros(20)),
            1_000,
        )

    invalid = _returns(np.zeros(20))
    invalid.iloc[4] = np.nan
    with pytest.raises(RiskHistoryUnavailable, match="returns"):
        historical_portfolio_var_pct(
            {"INVALID": {"quantity": 1, "ltp": 100}},
            lambda _symbol: invalid,
            1_000,
        )
