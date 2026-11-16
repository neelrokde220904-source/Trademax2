"""Historical one-day VaR for the complete signed portfolio exposure."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from risk.daily_portfolio_risk import MIN_DAILY_RETURNS, RiskHistoryUnavailable


def historical_portfolio_var_pct(
    positions: Mapping[str, Any],
    returns_for: Callable[[str], pd.Series],
    equity: float,
) -> float:
    """Return 95% historical one-day loss divided by equity for a projected book."""
    if isinstance(equity, bool) or not isinstance(equity, (int, float)):
        raise RiskHistoryUnavailable("invalid equity for portfolio VaR")
    if not math.isfinite(equity) or equity <= 0:
        raise RiskHistoryUnavailable("invalid equity for portfolio VaR")

    exposures: dict[str, float] = {}
    histories: dict[str, pd.Series] = {}
    for symbol in sorted(positions):
        if not isinstance(symbol, str) or not symbol.strip():
            raise RiskHistoryUnavailable("invalid portfolio symbol")
        position = positions[symbol]
        if not isinstance(position, Mapping):
            raise RiskHistoryUnavailable(f"invalid position for {symbol}")
        quantity = position.get("quantity")
        mark = position.get("ltp")
        if (
            isinstance(quantity, bool) or not isinstance(quantity, (int, float))
            or isinstance(mark, bool) or not isinstance(mark, (int, float))
            or not math.isfinite(quantity) or not math.isfinite(mark)
        ):
            raise RiskHistoryUnavailable(f"invalid quantity or mark for {symbol}")
        if quantity == 0:
            continue
        if mark <= 0:
            raise RiskHistoryUnavailable(f"invalid mark for {symbol}")
        returns = returns_for(symbol)
        try:
            values = returns.to_numpy(dtype=float) if isinstance(returns, pd.Series) else np.array([])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskHistoryUnavailable(f"invalid daily returns for {symbol}") from exc
        if (
            not isinstance(returns, pd.Series)
            or len(returns) != MIN_DAILY_RETURNS
            or not returns.index.is_unique
            or not np.isfinite(values).all()
        ):
            raise RiskHistoryUnavailable(f"invalid daily returns for {symbol}")
        exposures[symbol] = float(quantity) * float(mark)
        histories[symbol] = returns.rename(symbol)

    if not exposures:
        return 0.0
    aligned = pd.concat(list(histories.values()), axis=1, join="inner").sort_index()
    if len(aligned) != MIN_DAILY_RETURNS or aligned.isna().any().any():
        raise RiskHistoryUnavailable("portfolio returns are not aligned across 20 sessions")
    pnl = aligned.to_numpy(dtype=float) @ np.asarray(
        [exposures[symbol] for symbol in aligned.columns], dtype=float,
    )
    if not np.isfinite(pnl).all():
        raise RiskHistoryUnavailable("portfolio PnL history is non-finite")
    return max(0.0, -float(np.percentile(pnl, 5)) / float(equity))
