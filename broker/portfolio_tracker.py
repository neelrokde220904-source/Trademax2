"""Portfolio Tracker — holdings, positions, PnL tracking."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Any

import pytz
from loguru import logger
from sqlalchemy.orm import Session

from backtester.costs import calculate_indian_transaction_cost
from broker.angel_client import angel_client
from config.settings import settings
from database.db import get_db
from database.models import ClosedTrade, DailyPnL, Portfolio, Trade
from operations.control_state import ControlStateStore, control_state

IST = pytz.timezone("Asia/Kolkata")


class PortfolioTracker:
    """Tracks current portfolio state, positions, and PnL."""

    def __init__(self, store: ControlStateStore | None = None) -> None:
        self._store = store
        self._positions: dict[str, dict[str, Any]] = {}
        self._available_capital: float = settings.initial_capital
        self._unsettled_cash: float = 0.0
        self._previous_snapshot_date: str | None = None
        self._reserved_cash: float = 0.0  # open-order reservations (paper orders fill or fail at once)
        self._total_value: float = settings.initial_capital
        self._realised_pnl: float = 0.0
        self._gross_daily_pnl: float = 0.0
        self._daily_transaction_costs: float = 0.0
        self._total_transaction_costs: float = 0.0
        self._daily_pnl: float = 0.0
        self._day_start_equity: float = settings.initial_capital
        self._daily_baseline_valid = True
        self._daily_halt_date: str | None = None
        self._daily_loss_halted = False
        self._high_water_mark = settings.initial_capital
        self._persisted_high_water_mark = settings.initial_capital
        self._portfolio_drawdown_pct = 0.0
        self._portfolio_drawdown_halted = False
        self._drawdown_state_loaded = False
        self._drawdown_baseline_unavailable = False
        self._cash_flow_baseline_at = datetime.now(IST).astimezone(timezone.utc).isoformat()
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0

    def sync_from_broker(self) -> None:
        """Sync positions and holdings from Angel One."""
        if settings.paper_trading:
            self._sync_paper_positions()
        else:
            self._sync_live_positions()

    def _sync_live_positions(self) -> None:
        """Sync from live Angel One API."""
        self._daily_baseline_valid = False
        positions = angel_client.get_positions()
        holdings = angel_client.get_holdings()
        rms = angel_client.get_rms_limits()

        self._positions = {}

        for pos in positions:
            symbol = pos.get("tradingsymbol", "")
            net_qty = int(pos.get("netqty", 0))
            if net_qty != 0:
                self._positions[symbol] = {
                    "symbol": symbol,
                    "quantity": net_qty,
                    "avg_price": float(pos.get("averageprice", 0)),
                    "ltp": float(pos.get("ltp", 0)),
                    "pnl": float(pos.get("pnl", 0)),
                    "product": pos.get("producttype", ""),
                }

        if rms:
            self._available_capital = float(rms.get("availablecash", self._available_capital))

        logger.info("Synced {} live positions. Available capital: ₹{:.2f}", len(self._positions), self._available_capital)

    def _sync_paper_positions(self) -> None:
        """Rebuild paper positions and realised PnL from the persisted ledger."""
        from broker.paper_ledger import replay_orders, summarize_positions

        day_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        today = datetime.now(IST).strftime("%Y-%m-%d")
        with get_db() as db:
            trades = (
                db.query(Trade)
                .filter(Trade.is_paper == True, Trade.status == "EXECUTED")
                .all()
            )
            replay = replay_orders(trades)
            from broker import cash_ledger

            cash_ledger.backfill_missing(db, self._trade_cost)
            cash_ledger.settle_due(db, datetime.now(IST).date())
            db.flush()
            self._unsettled_cash = cash_ledger.unsettled_cash(db)
            closed = db.query(ClosedTrade).filter(ClosedTrade.is_paper == True).all()
            realised_total = sum(row.gross_pnl for row in closed)
            pnl_today = [
                row.gross_pnl for row in closed if row.exit_timestamp.replace(tzinfo=None) >= day_start
            ]
            self._total_transaction_costs = sum(self._trade_cost(trade) for trade in trades)
            self._daily_transaction_costs = sum(
                self._trade_cost(trade)
                for trade in trades
                if trade.timestamp.replace(tzinfo=None) >= day_start
            )
            previous_snapshot = (
                db.query(DailyPnL)
                .filter(DailyPnL.date < today)
                .order_by(DailyPnL.date.desc())
                .first()
            )
            highest_snapshot = db.query(DailyPnL.portfolio_value).order_by(
                DailyPnL.portfolio_value.desc()
            ).first()
            prior_trade = (
                db.query(Trade.id)
                .filter(
                    Trade.is_paper == True,
                    Trade.status == "EXECUTED",
                    Trade.timestamp < day_start,
                )
                .first()
            )
            self._day_start_equity = (
                float(previous_snapshot.portfolio_value)
                if previous_snapshot is not None
                else settings.initial_capital
            )
            self._daily_baseline_valid = previous_snapshot is not None or prior_trade is None
            self._previous_snapshot_date = previous_snapshot.date if previous_snapshot is not None else None
            flow_count = (
                self._store.external_cash_flow_snapshot()[1]
                if self._store is not None else 0
            )
            if highest_snapshot is not None and flow_count == 0:
                self._high_water_mark = max(
                    self._high_water_mark, float(highest_snapshot[0] or 0)
                )

        previous = self._positions
        self._positions = {}
        closing_marks = self._closing_marks()
        carried_without_mark: list[str] = []
        for symbol, derived in summarize_positions(replay).items():
            prior = previous.get(symbol, {})
            ltp = float(prior.get("ltp") or 0)
            mark_at = prior.get("mark_at") if ltp else None
            mark_source = prior.get("mark_source") if ltp else None
            if not ltp:
                entered = derived.get("entry_time")
                carried = hasattr(entered, "replace") and entered.replace(tzinfo=None) < day_start
                recorded = closing_marks.get(symbol)
                if (
                    recorded and recorded.get("entry_trade_ids") == derived.get("entry_trade_ids")
                    and recorded.get("quantity") == derived["quantity"]
                ):
                    # Continuity with the previous close: the carried gain is not today's gain.
                    # The mark is not fresh (mark_at stays empty), so fresh-mark gates still apply.
                    ltp, mark_source = float(recorded["ltp"]), "previous_close"
                elif carried:
                    carried_without_mark.append(symbol)
            ltp = ltp or float(derived["avg_price"])
            self._positions[symbol] = {
                **derived,
                "ltp": ltp,
                # An unmarked position is valued at cost, which is not a market observation.
                "mark_at": mark_at,
                "mark_source": mark_source,
                "pnl": (ltp - derived["avg_price"]) * derived["quantity"],
                "product": derived["holding_period"],
            }
        if carried_without_mark and self._previous_snapshot_date is not None:
            # The previous close's marked value of these positions is unknown, so the day-start
            # equity cannot be trusted: entries wait (non-latching) rather than guess.
            logger.warning("Carried positions without a recorded closing mark: {}", carried_without_mark)
            self._daily_baseline_valid = False

        self._realised_pnl = sum(pnl_today)
        self._trades_today = len(pnl_today)
        self._wins_today = sum(1 for pnl in pnl_today if pnl > 0)
        self._losses_today = self._trades_today - self._wins_today
        external_flows = (
            self._store.external_cash_flow_snapshot()[0]
            if self._store is not None else 0.0
        )
        self._available_capital = (
            settings.initial_capital + external_flows + realised_total - self._cost_basis()
            - self._total_transaction_costs
        )
        self._revalue()

    def _closing_marks(self) -> dict[str, dict[str, Any]]:
        """Marks recorded with the latest daily snapshot before today (empty when unavailable)."""
        if self._store is None:
            return {}
        try:
            record = self._store.read("closing_marks")
        except Exception:
            return {}
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if (
            not isinstance(record, dict) or not isinstance(record.get("marks"), dict)
            or not isinstance(record.get("date"), str) or record["date"] >= today
        ):
            return {}
        return {
            symbol: mark for symbol, mark in record["marks"].items()
            if isinstance(mark, dict) and isinstance(mark.get("ltp"), (int, float)) and mark["ltp"] > 0
        }

    @staticmethod
    def _trade_cost(trade: Trade) -> float:
        return calculate_indian_transaction_cost(
            price=float(trade.price),
            quantity=int(trade.quantity),
            is_buy=str(trade.action).upper() == "BUY",
            is_intraday=str(trade.holding_period or "INTRADAY").upper() == "INTRADAY",
        )

    def _cost_basis(self) -> float:
        return sum(abs(p["quantity"]) * p["avg_price"] for p in self._positions.values())

    def _revalue(self) -> None:
        # Cash excludes the cost basis of open lots, so value = cash + cost + mark-to-market (long or short).
        unrealised = sum(p["pnl"] for p in self._positions.values())
        self._total_value = self._available_capital + self._cost_basis() + unrealised
        daily_external_flows = 0.0
        if self._store is not None:
            day_start = datetime.now(IST).replace(
                hour=0, minute=0, second=0, microsecond=0,
            ).isoformat()
            daily_external_flows = self._store.external_cash_flows_since(day_start)
        self._daily_pnl = self._total_value - self._day_start_equity - daily_external_flows
        self._gross_daily_pnl = self._daily_pnl + self._daily_transaction_costs
        self._latch_daily_loss_halt()
        self._update_portfolio_drawdown_halt()

    def _ensure_drawdown_state(self) -> None:
        if self._drawdown_state_loaded:
            return
        self._drawdown_state_loaded = True
        if self._store is None:
            return
        state = self._store.read("portfolio_drawdown")
        if state is None:
            if self._store.external_cash_flow_snapshot()[1] > 0:
                self._portfolio_drawdown_halted = True
                self._drawdown_baseline_unavailable = True
                logger.critical(
                    "External cash flows exist but the adjusted drawdown baseline is missing; "
                    "new entries remain paused pending operator review."
                )
            return
        high_water_mark = state.get("high_water_mark")
        halted = state.get("halted")
        baseline_at = state.get("cash_flow_baseline_at")
        if baseline_at is None:
            baseline_at = state.get("updated_at")
        if (
            isinstance(high_water_mark, bool)
            or not isinstance(high_water_mark, (int, float))
            or not math.isfinite(high_water_mark)
            or high_water_mark <= 0
            or type(halted) is not bool
            or not isinstance(baseline_at, str)
        ):
            raise ValueError("Invalid persisted portfolio drawdown control state")
        try:
            baseline_timestamp = datetime.fromisoformat(baseline_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid persisted cash-flow baseline timestamp") from exc
        if baseline_timestamp.tzinfo is None:
            raise ValueError("Persisted cash-flow baseline timestamp must include a timezone")
        self._cash_flow_baseline_at = baseline_timestamp.astimezone(timezone.utc).isoformat()
        self._high_water_mark = max(self._high_water_mark, float(high_water_mark))
        self._persisted_high_water_mark = float(high_water_mark)
        self._portfolio_drawdown_halted = halted

    def _persist_drawdown_state(self) -> None:
        if self._store is None:
            return
        external_flows = self._store.external_cash_flow_snapshot()[0]
        self._store.write("portfolio_drawdown", {
            "high_water_mark": round(self._high_water_mark, 2),
            "drawdown_pct": self._portfolio_drawdown_pct,
            "halted": self._portfolio_drawdown_halted,
            "equity_basis": "net_of_external_cash_flows",
            "net_external_cash_flows": round(external_flows, 2),
            "cash_flow_baseline_at": self._cash_flow_baseline_at,
            "updated_at": datetime.now(IST).isoformat(),
        })
        self._persisted_high_water_mark = self._high_water_mark

    def _update_portfolio_drawdown_halt(self) -> None:
        self._ensure_drawdown_state()
        external_flows = 0.0
        if self._store is not None:
            external_flows, count, reconciled = self._store.external_cash_flow_snapshot()
            if count != reconciled:
                return
        adjusted_equity = self._total_value - external_flows
        if adjusted_equity > self._high_water_mark:
            self._high_water_mark = adjusted_equity
        if self._high_water_mark > 0:
            self._portfolio_drawdown_pct = max(
                0.0, (self._high_water_mark - adjusted_equity) / self._high_water_mark
            )
        newly_halted = (
            self._portfolio_drawdown_pct >= settings.max_portfolio_drawdown_pct
            and not self._portfolio_drawdown_halted
        )
        if newly_halted:
            self._portfolio_drawdown_halted = True
            self._cancel_pending_entries("portfolio drawdown")
            logger.critical(
                "Portfolio drawdown {:.2%} reached {:.2%}; new entries are persistently paused.",
                self._portfolio_drawdown_pct, settings.max_portfolio_drawdown_pct,
            )
        persist_threshold = max(1.0, self._persisted_high_water_mark * 0.001)
        if newly_halted or self._high_water_mark - self._persisted_high_water_mark >= persist_threshold:
            self._persist_drawdown_state()

    def _cancel_pending_entries(self, reason: str) -> None:
        """Resting entries must not fill after a halt latches; exits are never touched."""
        try:
            from broker.order_manager import cancel_pending_entries

            cancel_pending_entries(self._positions)
        except Exception as exc:
            logger.error("Could not cancel pending entries after {}: {}", reason, type(exc).__name__)

    def _ensure_daily_halt_state(self) -> None:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if self._daily_halt_date == today:
            return
        self._daily_halt_date = today
        self._daily_loss_halted = False
        if self._store is not None:
            state = self._store.read("daily_loss_halt")
            if isinstance(state, dict) and state.get("date") == today:
                # A transient baseline problem must not latch for the whole day.
                self._daily_loss_halted = (
                    state.get("halted") is True and state.get("reason") != "baseline_unavailable"
                )

    def _latch_daily_loss_halt(self) -> None:
        self._ensure_daily_halt_state()
        if self._daily_loss_halted:
            return
        loss_limit = settings.max_daily_loss_pct * self._day_start_equity
        if self._daily_baseline_valid and self._daily_pnl <= -loss_limit:
            self._daily_loss_halted = True
            self._cancel_pending_entries("daily loss limit")
            if self._store is not None:
                payload = {
                    "date": self._daily_halt_date,
                    "halted": True,
                    "reason": "daily_loss_limit",
                    "equity": round(self._total_value, 2),
                    "day_start_equity": round(self._day_start_equity, 2),
                    "limit_pct": settings.max_daily_loss_pct,
                }
                try:
                    self._store.write("daily_loss_halt", payload)
                except Exception as exc:
                    logger.error("Could not persist daily loss halt: {}", type(exc).__name__)

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        return self._positions

    @property
    def available_capital(self) -> float:
        """Settled cash less open-order reservations; the only cash that may size an entry."""
        return max(0.0, self._available_capital - self._unsettled_cash - self._reserved_cash)

    @property
    def settled_cash(self) -> float:
        return self.available_capital

    @property
    def unsettled_cash(self) -> float:
        return self._unsettled_cash

    @property
    def total_value(self) -> float:
        return self._total_value

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def daily_entry_count(
        self, symbol: str | None = None, *, as_of: datetime | None = None,
        session: Session | None = None,
    ) -> int:
        """Count distinct entries opened on the IST date, including open positions."""
        from broker.paper_ledger import replay_orders

        at = as_of or datetime.now(IST)
        if at.tzinfo is not None:
            at = at.astimezone(IST)
        day_start = at.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        day_end = day_start + timedelta(days=1)
        with nullcontext(session) if session is not None else get_db() as db:
            trades = (
                db.query(Trade)
                .filter(
                    Trade.is_paper == settings.paper_trading,
                    Trade.status == "EXECUTED",
                )
                .all()
            )
            closed_query = (
                db.query(ClosedTrade.entry_order_id)
                .filter(
                    ClosedTrade.is_paper == settings.paper_trading,
                    ClosedTrade.entry_timestamp >= day_start,
                    ClosedTrade.entry_timestamp < day_end,
                )
            )
            if symbol is not None:
                closed_query = closed_query.filter(ClosedTrade.symbol == symbol)
            closed_entry_ids = {
                row[0]
                for row in closed_query.all()
            }
            replay = replay_orders(trades)
            entries = {trade.id: trade for trade in trades}
            replayed_entry_ids = {
                close.entry_id
                for close in replay.closed
                if (symbol is None or close.symbol == symbol)
                and day_start <= entries[close.entry_id].timestamp.replace(tzinfo=None) < day_end
            }
            open_entry_ids = {
                lot.trade_id
                for lot_symbol, lots in replay.open_lots.items()
                for lot in lots
                if (symbol is None or lot_symbol == symbol)
                and day_start <= lot.timestamp.replace(tzinfo=None) < day_end
            }
        return len(closed_entry_ids | replayed_entry_ids | open_entry_ids)

    def last_symbol_exit_time(self, symbol: str, *, session: Session | None = None) -> datetime | None:
        """Return the latest recorded or FIFO-replayed exit for the symbol."""
        from broker.paper_ledger import replay_orders

        with nullcontext(session) if session is not None else get_db() as db:
            row = (
                db.query(ClosedTrade.exit_timestamp)
                .filter(ClosedTrade.is_paper == settings.paper_trading, ClosedTrade.symbol == symbol)
                .order_by(ClosedTrade.exit_timestamp.desc())
                .first()
            )
            trades = (
                db.query(Trade)
                .filter(
                    Trade.is_paper == settings.paper_trading,
                    Trade.status == "EXECUTED",
                    Trade.symbol == symbol,
                )
                .all()
            )
            timestamps = {trade.id: trade.timestamp for trade in trades}
            exits = [timestamps[close.exit_id] for close in replay_orders(trades).closed]
            if row is not None:
                exits.append(row[0])
            return max(
                (value.astimezone(IST).replace(tzinfo=None) if value.tzinfo else value)
                for value in exits
            ) if exits else None

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def gross_daily_pnl(self) -> float:
        return self._gross_daily_pnl

    @property
    def daily_transaction_costs(self) -> float:
        return self._daily_transaction_costs

    @property
    def daily_loss_pct(self) -> float:
        """Current net daily equity loss as a share of the day's starting equity."""
        if not self._daily_baseline_valid or self._day_start_equity <= 0:
            return 0.0
        return max(0.0, -self._daily_pnl / self._day_start_equity)

    def has_hit_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been breached."""
        self._latch_daily_loss_halt()
        # An invalid baseline blocks entries while it lasts, but is not latched.
        return self._daily_loss_halted or not self._daily_baseline_valid

    @property
    def portfolio_drawdown_pct(self) -> float:
        self._update_portfolio_drawdown_halt()
        return self._portfolio_drawdown_pct

    def has_hit_portfolio_drawdown_limit(self) -> bool:
        self._update_portfolio_drawdown_halt()
        return self.cash_flow_reconciliation_pending or self._portfolio_drawdown_halted

    def reset_portfolio_drawdown_halt(self) -> None:
        """Clear the persistent drawdown latch after explicit operator review."""
        if self.cash_flow_reconciliation_pending:
            raise ValueError("Cannot reset drawdown while external cash flows are unreconciled")
        external_flows = (
            self._store.external_cash_flow_snapshot()[0]
            if self._store is not None else 0.0
        )
        self._high_water_mark = max(1.0, self._total_value - external_flows)
        self._portfolio_drawdown_pct = 0.0
        self._portfolio_drawdown_halted = False
        self._drawdown_baseline_unavailable = False
        self._persist_drawdown_state()

    @property
    def cash_flow_reconciliation_pending(self) -> bool:
        return (
            self._store is not None
            and self._store.external_cash_flow_reconciliation_pending()
        )

    def record_external_cash_flow(self, reference: str, amount: float, source: str) -> int:
        """Record a signed deposit/withdrawal; entries stay blocked until fresh marks sync it."""
        if self._store is None:
            raise RuntimeError("External cash flows require a durable control-state store")
        self._ensure_drawdown_state()
        if self._drawdown_baseline_unavailable:
            raise ValueError("Cannot record cash flow without a reviewed drawdown baseline")
        if self._store.read("portfolio_drawdown") is None:
            self._persist_drawdown_state()
        return self._store.record_external_cash_flow(reference, amount, source)

    def import_angelone_cash_flow_statement(
        self, digest: str, entries: list[tuple[str, float, str]],
    ) -> dict[str, int | str]:
        """Persist reviewed broker transfers after ensuring a durable adjusted-equity baseline."""
        if self._store is None:
            raise RuntimeError("Cash-flow statement imports require a durable control-state store")
        self._ensure_drawdown_state()
        if self._drawdown_baseline_unavailable:
            raise ValueError("Cannot import cash flows without a reviewed drawdown baseline")
        if self._store.read("portfolio_drawdown") is None:
            self._persist_drawdown_state()
        return self._store.import_angelone_cash_flow_statement(
            digest, entries, self._cash_flow_baseline_at,
        )

    def mark_price(
        self, symbol: str, price: float, *, marked_at: datetime | None = None,
        source: str = "ws_tick",
    ) -> None:
        """Apply one valid market tick to an open position and update safety controls."""
        position = self._positions.get(symbol)
        if not position or not math.isfinite(price) or price <= 0:
            return
        position["ltp"] = price
        position["mark_at"] = (marked_at or datetime.now(IST)).isoformat()
        position["mark_source"] = source
        position["pnl"] = (price - float(position["avg_price"])) * int(position["quantity"])
        self._revalue()

    def mark_token_price(self, token: str, price: float) -> None:
        """Mark any open position associated with a broker instrument token."""
        from config.instruments import get_token

        for symbol in tuple(self._positions):
            if get_token(symbol) == str(token):
                self.mark_price(symbol, price)
                return

    def update_unrealised_pnl(self) -> bool:
        """Update marks and report whether every open position has a fresh reference."""
        from broker.order_manager import _market_reference
        from config.instruments import get_token

        flow_snapshot = (
            self._store.external_cash_flow_snapshot()
            if self._store is not None else (0.0, 0, 0)
        )
        all_fresh = True
        for symbol, pos in self._positions.items():
            token = get_token(symbol)
            reference = _market_reference(symbol, token, "NSE") if token else None
            if reference:
                pos["ltp"] = reference["price"]
                age = float(reference.get("age_seconds") or 0)
                pos["mark_at"] = (datetime.now(IST) - timedelta(seconds=max(0.0, age))).isoformat()
                pos["mark_source"] = reference.get("source") or "reference"
            else:
                all_fresh = False
                logger.warning("Fresh portfolio mark unavailable for {}; blocking new entries.", symbol)
            ltp = float(pos.get("ltp") or pos["avg_price"])
            pos["pnl"] = (ltp - pos["avg_price"]) * pos["quantity"]
        self._revalue()
        if (
            all_fresh and self._store is not None
            and flow_snapshot[1] != flow_snapshot[2]
            and self._store.acknowledge_external_cash_flows(flow_snapshot[1])
        ):
            self._update_portfolio_drawdown_halt()
        return all_fresh

    def snapshot(self) -> dict[str, Any]:
        """Return current portfolio state as a dict."""
        external_flows = (
            self._store.external_cash_flow_snapshot()[0]
            if self._store is not None else 0.0
        )
        return {
            "total_value": round(self._total_value, 2),
            "available_capital": round(self.available_capital, 2),
            "settled_cash": round(self.settled_cash, 2),
            "unsettled_cash": round(self._unsettled_cash, 2),
            "invested_value": round(self._total_value - self._available_capital, 2),
            "realised_pnl": round(self._realised_pnl, 2),
            "unrealised_pnl": round(self._gross_daily_pnl - self._realised_pnl, 2),
            "gross_daily_pnl": round(self._gross_daily_pnl, 2),
            "transaction_costs": round(self._daily_transaction_costs, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "day_start_equity": round(self._day_start_equity, 2),
            "daily_loss_baseline_valid": self._daily_baseline_valid,
            "daily_loss_halted": self._daily_loss_halted,
            "portfolio_high_water_mark": round(self._high_water_mark, 2),
            "portfolio_drawdown_pct": self._portfolio_drawdown_pct,
            "portfolio_drawdown_halted": self._portfolio_drawdown_halted,
            "drawdown_equity": round(self._total_value - external_flows, 2),
            "net_external_cash_flows": round(external_flows, 2),
            "cash_flow_reconciliation_pending": self.cash_flow_reconciliation_pending,
            "position_count": self.position_count,
            "positions": self._positions,
            "trades_today": self._trades_today,
            "wins_today": self._wins_today,
            "losses_today": self._losses_today,
        }

    def _persist_closing_marks(self, today: str) -> None:
        """Record the marked value of carried positions so tomorrow's day-start equity is exact."""
        if self._store is None:
            return
        marks = {
            symbol: {
                "ltp": float(position["ltp"]), "quantity": position["quantity"],
                "entry_trade_ids": position.get("entry_trade_ids"),
            }
            for symbol, position in self._positions.items()
            if position.get("mark_at") and float(position.get("ltp") or 0) > 0
        }
        try:
            self._store.write("closing_marks", {"date": today, "marks": marks})
        except Exception as exc:
            logger.error("Could not persist closing marks: {}", type(exc).__name__)

    def save_daily_snapshot(self) -> None:
        """Persist daily PnL to the database."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        self._persist_closing_marks(today)
        with get_db() as db:
            existing = db.query(DailyPnL).filter(DailyPnL.date == today).first()
            if existing:
                existing.portfolio_value = self._total_value
                existing.realised_pnl = self._realised_pnl
                existing.unrealised_pnl = self._gross_daily_pnl - self._realised_pnl
                existing.trades_count = self._trades_today
                existing.win_count = self._wins_today
                existing.loss_count = self._losses_today
            else:
                db.add(DailyPnL(
                    date=today,
                    portfolio_value=self._total_value,
                    realised_pnl=self._realised_pnl,
                    unrealised_pnl=self._gross_daily_pnl - self._realised_pnl,
                    trades_count=self._trades_today,
                    win_count=self._wins_today,
                    loss_count=self._losses_today,
                ))


# Module-level singleton
portfolio_tracker = PortfolioTracker(control_state)
