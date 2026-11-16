"""Risk-budget sizing and execution-friction checks for Indian equities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger

from backtester.costs import round_trip_cost
from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings
from risk.portfolio_stop_risk import entry_stop_risk


class PositionSizer:
    """Size entries from stop distance, cash, exposure, and net trade economics."""

    def size(
        self,
        signal: dict[str, Any],
        *,
        portfolio_value: float | None = None,
        available_capital: float | None = None,
        current_exposure: float | None = None,
        current_symbol_exposure: float = 0.0,
        risk_multiplier: float = 1.0,
        max_position_pct: float | None = None,
        remaining_stop_risk: float | None = None,
        calibrated_p: float | None = None,
        payoff_b: float | None = None,
    ) -> tuple[int, str | None]:
        """Return the largest safe share quantity, or zero with a blocking reason.

        With a calibrated win probability and payoff ratio the result is further reduced to a
        half-Kelly risk budget. Kelly can only shrink the quantity, never enlarge it.
        """
        try:
            price = self._positive(signal.get("price"))
            stop_loss = self._positive(signal.get("stop_loss"))
            target = self._positive(signal.get("target"))
            equity = self._positive(
                portfolio_tracker.total_value if portfolio_value is None else portfolio_value
            )
            available = self._positive(
                portfolio_tracker.available_capital
                if available_capital is None else available_capital
            )
            multiplier = float(risk_multiplier)
        except (TypeError, ValueError, OverflowError):
            return 0, "Price, stop, target, or portfolio values are invalid"

        action = str(signal.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            return 0, "Action must be BUY or SELL"
        if action == "BUY" and not stop_loss < price < target:
            return 0, "BUY requires stop_loss < price < target"
        if action == "SELL" and not target < price < stop_loss:
            return 0, "SELL requires target < price < stop_loss"
        if equity <= 0 or available <= 0:
            return 0, "No positive equity or available capital"
        if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
            return 0, "Risk multiplier must be in (0, 1]"

        exposure = (
            self.exposure_from_positions(portfolio_tracker.positions)
            if current_exposure is None else float(current_exposure)
        )
        if not math.isfinite(exposure) or exposure < 0:
            return 0, "Current exposure is invalid"
        if not math.isfinite(current_symbol_exposure) or current_symbol_exposure < 0:
            return 0, "Current symbol exposure is invalid"

        if max_position_pct is None:
            position_pct = settings.max_capital_per_trade_pct
        else:
            if isinstance(max_position_pct, bool) or not isinstance(max_position_pct, (int, float)):
                return 0, "Position-size cap is invalid"
            try:
                position_pct = float(max_position_pct)
            except (OverflowError, ValueError):
                return 0, "Position-size cap is invalid"
            if not math.isfinite(position_pct) or not 0 < position_pct <= 1:
                return 0, "Position-size cap is invalid"
            position_pct = min(settings.max_capital_per_trade_pct, position_pct)

        slippage = settings.slippage_pct
        if not math.isfinite(slippage) or not 0 <= slippage < 1:
            return 0, "Slippage assumption is invalid"
        max_notional = min(
            available,
            equity * position_pct - current_symbol_exposure,
            equity * settings.max_total_exposure_pct - exposure,
        )
        if max_notional < price * (1 + settings.slippage_pct):
            return 0, "Cash, per-position, or total-exposure cap is below one share"

        max_quantity = math.floor(
            max_notional / (price * (1 + settings.slippage_pct))
        )
        risk_budget = equity * settings.risk_per_trade_pct * multiplier
        if remaining_stop_risk is not None:
            if (
                isinstance(remaining_stop_risk, bool)
                or not isinstance(remaining_stop_risk, (int, float))
                or not math.isfinite(remaining_stop_risk)
                or remaining_stop_risk < 0
            ):
                return 0, "Remaining portfolio stop-risk budget is invalid"
            risk_budget = min(risk_budget, remaining_stop_risk)
        is_intraday = str(signal.get("holding_period", "INTRADAY")).upper() == "INTRADAY"

        for quantity in range(max_quantity, 0, -1):
            entry_fill = price * (1 + slippage if action == "BUY" else 1 - slippage)
            target_fill = target * (1 - slippage if action == "BUY" else 1 + slippage)
            target_fees = round_trip_cost(
                entry_fill, target_fill, quantity,
                is_intraday=is_intraday, is_entry_buy=action == "BUY",
            )
            gross_reward = abs(target - price) * quantity
            net_reward = (
                (target_fill - entry_fill) * quantity
                if action == "BUY" else (entry_fill - target_fill) * quantity
            ) - target_fees
            net_loss = entry_stop_risk(signal, quantity)

            if net_loss > risk_budget:
                continue
            if gross_reward <= 0 or net_reward <= 0:
                return 0, "Estimated fees and slippage consume the target profit"
            friction = gross_reward - (
                (target_fill - entry_fill) * quantity
                if action == "BUY" else (entry_fill - target_fill) * quantity
            ) + target_fees
            if friction / gross_reward > settings.max_cost_to_gross_reward_pct:
                return 0, "Estimated fees and slippage exceed the target-profit cost limit"
            if net_reward / net_loss < settings.min_net_reward_risk:
                return 0, "Net reward-to-risk is below the configured minimum"
            if calibrated_p is not None or payoff_b is not None:
                return self._apply_half_kelly(signal, quantity, equity, calibrated_p, payoff_b)
            return quantity, None

        if remaining_stop_risk is not None and remaining_stop_risk <= equity * settings.risk_per_trade_pct * multiplier:
            return 0, "Portfolio stop-risk budget is below one share"
        return 0, "Stop-distance risk budget is below one share"

    def validate_quantity(
        self,
        signal: dict[str, Any],
        quantity: int,
        *,
        portfolio_value: float | None = None,
        available_capital: float | None = None,
        current_exposure: float | None = None,
        current_symbol_exposure: float = 0.0,
        risk_multiplier: float = 1.0,
        max_position_pct: float | None = None,
        remaining_stop_risk: float | None = None,
    ) -> str | None:
        """Validate a requested quantity against limits and net target economics."""
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return "Quantity must be a positive integer"
        max_quantity, reason = self.size(
            signal,
            portfolio_value=portfolio_value,
            available_capital=available_capital,
            current_exposure=current_exposure,
            current_symbol_exposure=current_symbol_exposure,
            risk_multiplier=risk_multiplier,
            max_position_pct=max_position_pct,
            remaining_stop_risk=remaining_stop_risk,
        )
        if reason:
            return reason
        if quantity > max_quantity:
            return f"Quantity {quantity} exceeds safe maximum {max_quantity}"

        try:
            price = self._positive(signal.get("price"))
            self._positive(signal.get("stop_loss"))
            target = self._positive(signal.get("target"))
        except (TypeError, ValueError, OverflowError):
            return "Price, stop, or target is invalid"
        action = str(signal.get("action", "")).upper()
        slippage = settings.slippage_pct
        entry_fill = price * (1 + slippage if action == "BUY" else 1 - slippage)
        target_fill = target * (1 - slippage if action == "BUY" else 1 + slippage)
        is_intraday = str(signal.get("holding_period", "INTRADAY")).upper() == "INTRADAY"
        target_fees = round_trip_cost(
            entry_fill, target_fill, quantity,
            is_intraday=is_intraday, is_entry_buy=action == "BUY",
        )
        gross_reward = abs(target - price) * quantity
        favorable_move = (
            (target_fill - entry_fill) * quantity
            if action == "BUY" else (entry_fill - target_fill) * quantity
        )
        net_reward = favorable_move - target_fees
        net_loss = entry_stop_risk(signal, quantity)
        friction = gross_reward - favorable_move + target_fees
        if gross_reward <= 0 or net_reward <= 0:
            return "Estimated fees and slippage consume the target profit"
        if friction / gross_reward > settings.max_cost_to_gross_reward_pct:
            return "Estimated fees and slippage exceed the target-profit cost limit"
        if net_reward / net_loss < settings.min_net_reward_risk:
            return "Net reward-to-risk is below the configured minimum"
        return None

    def calculate(
        self,
        signal: dict[str, Any],
        *,
        portfolio_value: float | None = None,
        available_capital: float | None = None,
        current_exposure: float | None = None,
        risk_multiplier: float = 1.0,
    ) -> int:
        """Compatibility wrapper returning only the safe quantity."""
        quantity, reason = self.size(
            signal,
            portfolio_value=portfolio_value,
            available_capital=available_capital,
            current_exposure=current_exposure,
            risk_multiplier=risk_multiplier,
        )
        if reason:
            logger.info("[PositionSizer] {} — {}", signal.get("symbol", "?"), reason)
            return 0
        return quantity

    @staticmethod
    def exposure_from_positions(positions: dict[str, Any]) -> float:
        """Estimate gross open notional using fresh marks or average entry prices."""
        return sum(PositionSizer.exposure_by_symbol(positions).values())

    @staticmethod
    def exposure_by_symbol(positions: dict[str, Any]) -> dict[str, float]:
        """Return validated gross open notional for each symbol."""
        by_symbol: dict[str, float] = {}
        for symbol, position in positions.items():
            if not isinstance(position, dict):
                raise ValueError(f"Position {symbol} is not an object")
            try:
                quantity = abs(float(position.get("quantity") or 0))
                price = float(position.get("ltp") or position.get("avg_price") or 0)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"Position {symbol} has invalid quantity or price") from None
            if not math.isfinite(quantity) or not math.isfinite(price):
                raise ValueError(f"Position {symbol} has non-finite quantity or price")
            if quantity > 0 and price <= 0:
                raise ValueError(f"Position {symbol} has quantity but no positive mark")
            if quantity > 0:
                notional = quantity * price
                by_symbol[symbol] = notional
        return by_symbol

    @staticmethod
    def _positive(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("boolean is not a price")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("value must be positive and finite")
        return number

    @staticmethod
    def kelly_fraction(p: float, b: float) -> float:
        """Full-Kelly fraction f = p - (1 - p) / b (can be negative)."""
        return p - (1.0 - p) / b

    def _apply_half_kelly(
        self, signal: dict[str, Any], quantity: int, equity: float,
        calibrated_p: float | None, payoff_b: float | None,
    ) -> tuple[int, str | None]:
        try:
            p, b = float(calibrated_p), float(payoff_b)
        except (TypeError, ValueError):
            return 0, "kelly_inputs_invalid"
        if not (math.isfinite(p) and math.isfinite(b)) or not 0 < p < 1 or b <= 0:
            return 0, "kelly_inputs_invalid"
        fraction = self.kelly_fraction(p, b)
        if fraction <= 0:
            return 0, "kelly_non_positive"
        budget = 0.5 * fraction * equity  # capital risked at the stop
        reduced = quantity
        while reduced > 0 and entry_stop_risk(signal, reduced) > budget:
            reduced -= 1
        if reduced <= 0:
            return 0, "kelly_budget_below_one_share"
        return min(quantity, reduced), None

    def _kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """Half-Kelly helper retained for reporting/backward-compatible tests."""
        p = max(0.01, min(0.99, win_rate))
        q = 1 - p
        b = max(0.01, win_loss_ratio)
        kelly = (p * b - q) / b
        return max(0, min(0.25, kelly * 0.5))

    def _calculate_var(self, daily_returns: list[float], position_value: float) -> float:
        """Calculate parametric VaR at 95% confidence."""
        returns = np.asarray(daily_returns, dtype=float)
        if not returns.size or not np.isfinite(returns).all():
            return 0.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        if sigma == 0:
            return 0.0
        return position_value * abs(mu - 1.645 * sigma)


position_sizer = PositionSizer()
