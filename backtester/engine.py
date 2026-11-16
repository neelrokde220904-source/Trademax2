"""Backtester Engine — event-driven backtesting with realistic Indian market costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as session_date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from risk.position_sizer import position_sizer
from risk.exit_policy import (
    gap_fill_price, get_atr_exit_policy, ratcheted_stop, stop_target_hit,
)


@dataclass
class BacktestTrade:
    symbol: str
    action: str  # BUY / SELL
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    strategy: str
    holding_period: str = "INTRADAY"
    entry_cost: float = 0.0
    exit_date: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    atr: float = 0.0
    atr_exit_policy: str = ""
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    initial_stop: float = 0.0
    bars_open: int = 0


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[datetime] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0


class BacktestEngine:
    """Event-driven backtester with Indian market cost model.

    Entries use the same risk-budget and net reward/risk calculator as the
    live paper router. The paper portfolio is unlevered; short collateral is
    fully reserved until the position closes.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        max_positions: int | None = None,
        max_capital_pct: float | None = None,
        intraday_rules: bool = False,
    ) -> None:
        self.initial_capital = initial_capital or settings.initial_capital
        self.max_positions = max_positions or settings.max_positions
        self.max_capital_pct = max_capital_pct or settings.max_capital_per_trade_pct

        self._capital = self.initial_capital
        self._positions: dict[str, BacktestTrade] = {}
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._dates: list[datetime] = []
        self._entries_today = 0
        self._entry_date: session_date | None = None
        # Live INTRADAY rules: no entry before 09:30 or from 14:45, flat from 15:15.
        self.intraday_rules = intraday_rules
        self.exposure_bars = 0
        self.total_bars = 0

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame, int], list[dict[str, Any]]],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """Run backtest over historical data.

        Args:
            data: {symbol: DataFrame} with OHLCV, indexed by date.
            strategy_fn: (symbol, df_up_to_bar, bar_index) -> list of signals.
            start_date: Optional start filter (YYYY-MM-DD).
            end_date: Optional end filter (YYYY-MM-DD).

        Returns:
            BacktestResult with trades, equity curve.
        """
        data = self._normalize_data(data)
        self._capital = self.initial_capital
        self._positions = {}
        self._closed_trades = []
        self._equity_curve = [self.initial_capital]
        self._dates = []
        self._entries_today = 0
        self._entry_date = None
        self.exposure_bars = 0
        self.total_bars = 0

        # Build unified date index
        all_dates = sorted({timestamp for df in data.values() for timestamp in df.index})
        previous_dates = {
            timestamp: all_dates[index - 1]
            for index, timestamp in enumerate(all_dates)
            if index > 0
        }
        sorted_dates = all_dates

        if start_date:
            start = self._coerce_timestamp(start_date, sorted_dates)
            sorted_dates = [d for d in sorted_dates if d >= start]
        if end_date:
            end = self._coerce_timestamp(end_date, sorted_dates)
            sorted_dates = [d for d in sorted_dates if d <= end]

        logger.info(
            "[Backtest] Running {} symbols over {} days (₹{:,.0f} capital)",
            len(data), len(sorted_dates), self.initial_capital,
        )

        for index, date in enumerate(sorted_dates):
            if self._entry_date != date.date():
                self._entry_date = date.date()
                self._entries_today = 0

            previous_date = previous_dates.get(date)
            if previous_date is not None:
                for symbol, df in data.items():
                    bar = self._get_bar(df, date)
                    history = df[df.index < date]
                    if bar is None or history.empty or history.index[-1] != previous_date:
                        continue

                    signals = strategy_fn(symbol, history, len(history) - 1)
                    try:
                        open_price = float(bar["open"])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        logger.warning("[Backtest] {} missing a valid open; skipping entries.", symbol)
                        continue
                    if not np.isfinite(open_price) or open_price <= 0:
                        logger.warning("[Backtest] {} has an invalid open; skipping entries.", symbol)
                        continue

                    for signal in signals:
                        executable_signal = {**signal, "price": open_price}
                        equity = self._calculate_equity(data, date, price_field="open")
                        exposure = self._current_exposure(data, date, price_field="open")
                        self._process_signal(executable_signal, date, equity, exposure)

            self._check_exits(data, date)
            if self.intraday_rules:
                self._flatten_intraday(data, date)
            self.total_bars += 1
            if self._positions:
                self.exposure_bars += 1
                for held in self._positions.values():
                    held.bars_open += 1

            # Record equity
            equity = self._calculate_equity(data, date)
            self._equity_curve.append(equity)
            self._dates.append(date)

        # Close remaining positions at last available price
        self._close_all(data, sorted_dates[-1] if sorted_dates else datetime.now())
        if self._equity_curve:
            self._equity_curve[-1] = self._capital

        result = BacktestResult(
            trades=self._closed_trades,
            equity_curve=self._equity_curve,
            dates=self._dates,
            initial_capital=self.initial_capital,
            final_capital=self._capital,
        )

        logger.info(
            "[Backtest] Complete: {} trades, ₹{:,.0f} → ₹{:,.0f}",
            len(result.trades), result.initial_capital, result.final_capital,
        )
        return result

    @staticmethod
    def _normalize_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        normalized: dict[str, pd.DataFrame] = {}
        for symbol, source in data.items():
            frame = source.copy()
            timestamp_column = next(
                (column for column in ("datetime", "date") if column in frame.columns),
                None,
            )
            if timestamp_column is not None:
                index = pd.DatetimeIndex(pd.to_datetime(frame.pop(timestamp_column)))
            elif isinstance(frame.index, pd.DatetimeIndex):
                index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            else:
                raise ValueError(f"{symbol} data must have a DatetimeIndex or datetime/date column")

            if index.tz is None:
                index = index.tz_localize("Asia/Kolkata")
            else:
                index = index.tz_convert("Asia/Kolkata")
            frame.index = index
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            normalized[symbol] = frame
        return normalized

    @staticmethod
    def _coerce_timestamp(value: str, dates: list[datetime]) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        timezone = dates[0].tz if dates else None
        if timezone is not None and timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone)
        elif timezone is None and timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp

    def _process_signal(
        self,
        signal: dict[str, Any],
        date: datetime,
        portfolio_value: float,
        current_exposure: float,
    ) -> None:
        symbol = signal.get("symbol", "")
        action = str(signal.get("action", "")).upper()
        try:
            price = float(signal.get("price", 0))
        except (TypeError, ValueError, OverflowError):
            return

        if not symbol or action not in {"BUY", "SELL"} or not np.isfinite(price) or price <= 0:
            return

        # Skip if already in position
        if symbol in self._positions:
            return

        # Position limit
        if len(self._positions) >= self.max_positions:
            return
        if self._entries_today >= settings.max_daily_entries:
            return

        policy_name = str(signal.get("atr_exit_policy") or "")
        atr = 0.0
        if policy_name:
            policy = get_atr_exit_policy(policy_name)
            atr = float(signal.get("atr") or 0)
            fill_reference = price * (
                1 + settings.slippage_pct if action == "BUY" else 1 - settings.slippage_pct
            )
            limit_price = signal.get("limit_price")
            if limit_price is not None:
                fill_reference = (
                    min(fill_reference, limit_price) if action == "BUY"
                    else max(fill_reference, limit_price)
                )
            stop, target = policy.levels(fill_reference, atr, action)
            signal = {**signal, "stop_loss": stop, "target": target}

        holding_period = str(signal.get("holding_period", "INTRADAY")).upper()
        if self.intraday_rules and holding_period == "INTRADAY" and not self._intraday_entry_allowed(date):
            return
        sizing_signal = {
            **signal,
            "symbol": symbol,
            "action": action,
            "price": price,
            "holding_period": holding_period,
        }
        available_for_sizing = min(self._capital, portfolio_value * self.max_capital_pct)
        quantity, reason = position_sizer.size(
            sizing_signal,
            portfolio_value=portfolio_value,
            available_capital=available_for_sizing,
            current_exposure=current_exposure,
            risk_multiplier=float(signal.get("risk_multiplier", 1.0)),
            max_position_pct=signal.get("hmm_position_cap"),
        )
        if quantity <= 0 or position_sizer.validate_quantity(
            sizing_signal,
            quantity,
            portfolio_value=portfolio_value,
            available_capital=available_for_sizing,
            current_exposure=current_exposure,
            risk_multiplier=float(signal.get("risk_multiplier", 1.0)),
            max_position_pct=signal.get("hmm_position_cap"),
        ):
            logger.debug("[Backtest] {} entry rejected: {}", symbol, reason or "net trade gate")
            return

        # Apply slippage
        slippage = price * settings.slippage_pct
        entry_price = price + slippage if action == "BUY" else price - slippage
        limit_price = signal.get("limit_price")
        if limit_price is not None:
            entry_price = min(entry_price, limit_price) if action == "BUY" else max(entry_price, limit_price)

        cost_entry = self._calculate_costs(
            entry_price, quantity, is_buy=action == "BUY",
            is_intraday=holding_period == "INTRADAY",
        )
        cash_required = entry_price * quantity + cost_entry
        if cash_required > self._capital:
            return
        self._capital -= cash_required

        trade = BacktestTrade(
            symbol=symbol,
            action=action,
            entry_date=date,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.get("stop_loss", 0),
            target=signal.get("target", 0),
            initial_stop=signal.get("stop_loss", 0),
            strategy=signal.get("strategy", "unknown"),
            holding_period=holding_period,
            entry_cost=cost_entry,
            atr=atr,
            atr_exit_policy=policy_name,
            highest_since_entry=entry_price,
            lowest_since_entry=entry_price,
        )
        self._positions[symbol] = trade
        self._entries_today += 1

    def _intraday_entry_allowed(self, date: datetime) -> bool:
        clock = (date.hour, date.minute)
        return (9, 30) <= clock < (14, 45)

    def _flatten_intraday(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        """Close INTRADAY positions at the open of the first bar at or after 15:15."""
        if (date.hour, date.minute) < (15, 15):
            return
        for symbol, trade in list(self._positions.items()):
            if trade.holding_period != "INTRADAY":
                continue
            row = self._get_bar(data[symbol], date) if symbol in data else None
            if row is not None:
                self._close_trade(symbol, float(row.get("open", row["close"])), date, "TIME_EXIT")

    def _check_exits(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        for symbol in list(self._positions.keys()):
            trade = self._positions[symbol]
            df = data.get(symbol)
            if df is None:
                continue

            row = self._get_bar(df, date)
            if row is None:
                continue

            high = float(row.get("high", 0))
            low = float(row.get("low", 0))
            open_price = float(row.get("open", row.get("close", 0)))

            reason = stop_target_hit(trade.action, trade.stop_loss, trade.target, high, low)
            if reason is not None:
                level = trade.stop_loss if reason == "STOP_LOSS" else trade.target
                self._close_trade(
                    symbol, gap_fill_price(trade.action, reason, level, open_price), date, reason,
                )
            elif trade.atr_exit_policy:
                trade.highest_since_entry = max(trade.highest_since_entry, high)
                trade.lowest_since_entry = min(trade.lowest_since_entry, low)
                policy = get_atr_exit_policy(trade.atr_exit_policy)
                # This bar's excursion can only tighten the stop for the next bar.
                trade.stop_loss = ratcheted_stop(
                    trade.action, trade.stop_loss, trade.highest_since_entry,
                    trade.lowest_since_entry, trade.atr, policy.trailing_multiplier,
                )

    def _close_trade(
        self, symbol: str, exit_price: float, date: datetime, reason: str,
        *, limit_price: float | None = None,
    ) -> None:
        trade = self._positions.pop(symbol, None)
        if not trade:
            return

        # Apply slippage on exit
        slippage = exit_price * settings.slippage_pct
        exit_price = exit_price - slippage if trade.action == "BUY" else exit_price + slippage
        if limit_price is not None:
            exit_price = max(exit_price, limit_price) if trade.action == "BUY" else min(exit_price, limit_price)

        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        if trade.action == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        cost_exit = self._calculate_costs(
            exit_price, trade.quantity, is_buy=trade.action == "SELL",
            is_intraday=trade.holding_period == "INTRADAY",
        )
        trade.costs = trade.entry_cost + cost_exit
        trade.net_pnl = trade.pnl - trade.costs

        if trade.action == "BUY":
            self._capital += exit_price * trade.quantity - cost_exit
        else:
            self._capital += trade.entry_price * trade.quantity + trade.pnl - cost_exit
        self._closed_trades.append(trade)

    def _close_all(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        for symbol in list(self._positions.keys()):
            df = data.get(symbol)
            if df is not None:
                row = self._get_bar(df, date)
                if row is not None:
                    self._close_trade(symbol, float(row["close"]), date, "BACKTEST_END")

    def _get_bar(self, df: pd.DataFrame, date: datetime) -> dict | None:
        if date in df.index:
            return df.loc[date].to_dict()
        return None

    def _calculate_equity(
        self, data: dict[str, pd.DataFrame], date: datetime, *, price_field: str = "close"
    ) -> float:
        equity = self._capital
        for symbol, trade in self._positions.items():
            df = data.get(symbol)
            if df is not None:
                row = self._get_bar(df, date)
                if row:
                    mark = float(row.get(price_field, row["close"]))
                else:
                    mark = trade.entry_price
            else:
                mark = trade.entry_price
            if trade.action == "BUY":
                equity += mark * trade.quantity
            else:
                equity += trade.entry_price * trade.quantity
                equity += (trade.entry_price - mark) * trade.quantity
        return equity

    def _current_exposure(
        self, data: dict[str, pd.DataFrame], date: datetime, *, price_field: str = "close"
    ) -> float:
        exposure = 0.0
        for symbol, trade in self._positions.items():
            df = data.get(symbol)
            row = self._get_bar(df, date) if df is not None else None
            mark = float(row.get(price_field, row["close"])) if row else trade.entry_price
            exposure += abs(mark * trade.quantity)
        return exposure

    def _calculate_costs(self, price: float, qty: int, is_buy: bool, is_intraday: bool = False) -> float:
        """Calculate all Indian market transaction costs."""
        from backtester.costs import calculate_indian_transaction_cost

        return calculate_indian_transaction_cost(
            price, qty, is_buy=is_buy, is_intraday=is_intraday
        )

    def run_walk_forward(
        self,
        data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame, int], list[dict[str, Any]]],
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
        optimize_fn: Callable[[dict[str, pd.DataFrame]], Callable] | None = None,
    ) -> WalkForwardResult:
        """Run anchored walk-forward validation.

        Splits data into rolling train/test windows. If optimize_fn is provided,
        it re-optimizes strategy_fn on each train window.

        Args:
            data: {symbol: DataFrame} with OHLCV, indexed by date.
            strategy_fn: Default strategy if optimize_fn is None.
            train_days: Training window size in trading days.
            test_days: Out-of-sample test window size.
            step_days: Step size between windows.
            optimize_fn: (train_data) -> optimized_strategy_fn. Optional.

        Returns:
            WalkForwardResult with per-fold results and aggregated metrics.
        """
        # Build unified sorted date index
        data = self._normalize_data(data)
        sorted_dates = sorted({timestamp for df in data.values() for timestamp in df.index})

        total_days = len(sorted_dates)
        min_required = train_days + test_days
        if total_days < min_required:
            logger.error(
                "[WalkForward] Need {} days, have {}. Cannot proceed.",
                min_required, total_days,
            )
            return WalkForwardResult(folds=[])

        folds: list[BacktestResult] = []
        fold_ranges: list[dict[str, str]] = []
        fold_idx = 0
        start = 0

        while start + train_days < total_days:
            train_end = start + train_days
            test_end = min(train_end + test_days, total_days)

            train_start_date = sorted_dates[start]
            train_end_date = sorted_dates[train_end - 1]
            test_start_date = sorted_dates[train_end]
            test_end_date = sorted_dates[test_end - 1]

            logger.info(
                "[WalkForward] Fold {} — Train: {} to {}, Test: {} to {}",
                fold_idx,
                str(train_start_date)[:10], str(train_end_date)[:10],
                str(test_start_date)[:10], str(test_end_date)[:10],
            )

            # Keep pre-test bars available for indicators/lookback while only
            # simulating fills inside the out-of-sample window.
            def _slice(df: pd.DataFrame, d_start: datetime, d_end: datetime) -> pd.DataFrame:
                return df[(df.index >= d_start) & (df.index <= d_end)].copy()

            train_data = {sym: _slice(df, train_start_date, train_end_date) for sym, df in data.items()}
            train_data = {sym: df for sym, df in train_data.items() if not df.empty}

            # Optimize on train window if optimizer provided
            if optimize_fn is not None:
                try:
                    current_strategy = optimize_fn(train_data)
                except Exception as exc:
                    logger.warning("[WalkForward] Fold {} optimize failed: {}, using default.", fold_idx, exc)
                    current_strategy = strategy_fn
            else:
                current_strategy = strategy_fn

            # Run backtest on OOS test window
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                max_positions=self.max_positions,
                max_capital_pct=self.max_capital_pct,
            )
            result = engine.run(
                data,
                current_strategy,
                start_date=str(test_start_date)[:10],
                end_date=str(test_end_date)[:10],
            )
            folds.append(result)
            fold_ranges.append({
                "fold": fold_idx,
                "train_start": str(train_start_date)[:10],
                "train_end": str(train_end_date)[:10],
                "test_start": str(test_start_date)[:10],
                "test_end": str(test_end_date)[:10],
            })

            fold_idx += 1
            start += step_days

        # Aggregate OOS equity curves
        combined_equity = [self.initial_capital]
        combined_trades: list[Any] = []
        for f in folds:
            if f.equity_curve and len(f.equity_curve) > 1:
                scale = combined_equity[-1] / f.equity_curve[0] if f.equity_curve[0] > 0 else 1.0
                combined_equity.extend([v * scale for v in f.equity_curve[1:]])
            combined_trades.extend(f.trades)

        wf_result = WalkForwardResult(
            folds=folds,
            fold_ranges=fold_ranges,
            combined_equity=combined_equity,
            combined_trades=combined_trades,
            initial_capital=self.initial_capital,
            final_capital=combined_equity[-1] if combined_equity else self.initial_capital,
        )

        logger.info(
            "[WalkForward] {} folds, {} total OOS trades, ₹{:,.0f} → ₹{:,.0f}",
            len(folds), len(combined_trades), self.initial_capital, wf_result.final_capital,
        )
        return wf_result


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""

    folds: list[BacktestResult] = field(default_factory=list)
    fold_ranges: list[dict[str, str]] = field(default_factory=list)
    combined_equity: list[float] = field(default_factory=list)
    combined_trades: list[Any] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_return_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    def per_fold_returns(self) -> list[float]:
        """Return % returns for each fold."""
        returns = []
        for f in self.folds:
            if f.initial_capital > 0:
                returns.append((f.final_capital - f.initial_capital) / f.initial_capital * 100)
            else:
                returns.append(0.0)
        return returns
