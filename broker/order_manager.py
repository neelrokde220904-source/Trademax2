"""Order Manager — places, modifies, and cancels orders with paper-trading support."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta
from typing import Any

import pytz
from loguru import logger
from sqlalchemy.orm import Session

from broker.angel_client import angel_client
from broker.base_broker import BrokerRejectedError, OrderSubmissionUncertainError
from config.settings import settings
from database.db import get_db
from database.models import ClosedTrade, PaperFillAudit, Trade
from strategies.candidate import horizon_to_product

IST = pytz.timezone("Asia/Kolkata")
_paper_order_lock = threading.RLock()


def _is_market_hours() -> bool:
    """Check if current time is within NSE market hours (9:15 – 15:30 IST, Mon-Fri)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    today_str = now.strftime("%Y-%m-%d")
    from data.nse_calendar import is_nse_holiday

    if is_nse_holiday(today_str):
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def place_order(
    symbol: str,
    token: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    price: float = 0.0,
    trigger_price: float = 0.0,
    stop_loss: float | None = None,
    target: float | None = None,
    exchange: str = "NSE",
    product_type: str = "INTRADAY",
    strategy: str = "",
    confidence: float = 0.0,
    reasoning: str = "",
    allow_requested_price_reference: bool = False,
    atr: float | None = None,
    is_exit: bool = False,
    expected_entry_trade_ids: list[int] | None = None,
    expected_paper_book: dict[str, dict[str, Any]] | None = None,
    sector: str | None = None,
    sector_source: str | None = None,
    sector_observed_at: str | None = None,
    horizon: str | None = None,
    strategy_version: str | None = None,
    exit_policy: str | None = None,
    atr_timeframe: str | None = None,
    book: str | None = None,
) -> dict[str, Any] | None:
    """Place an order — paper or live depending on config.

    ``allow_requested_price_reference`` lets paper exits fill at the requested
    price when no live reference exists; such fills are flagged by reconciliation.

    Returns:
        Dict with order details or None on failure.
    """
    # Market-hours guard for live trading
    if not settings.paper_trading and not _is_market_hours():
        logger.warning("Blocked live order for {} — outside market hours.", symbol)
        return None

    from operations.order_rate_limiter import order_rate_limiter

    if not order_rate_limiter.acquire(is_exit=is_exit):
        logger.warning("Entry for {} refused: order rate limit reached.", symbol)
        return None

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": action.upper(),
        "exchange": exchange,
        "ordertype": order_type,
        "producttype": product_type,
        "duration": "DAY",
        "price": str(price),
        "squareoff": "0",
        "stoploss": "0",
        "quantity": str(quantity),
        "triggerprice": str(trigger_price),
    }

    if settings.paper_trading:
        with _paper_order_lock:
            return _paper_trade(
                order_params, symbol, action, quantity, price, stop_loss, target, exchange,
                strategy, confidence, reasoning,
                allow_requested_price_reference=allow_requested_price_reference,
                atr=atr,
                is_exit=is_exit,
                expected_entry_trade_ids=expected_entry_trade_ids,
                expected_paper_book=expected_paper_book,
                sector=sector,
                sector_source=sector_source,
                sector_observed_at=sector_observed_at,
                horizon=horizon, strategy_version=strategy_version, exit_policy=exit_policy,
                atr_timeframe=atr_timeframe, book=book,
            )
    else:
        return _live_trade(
            order_params, symbol, action, quantity, price, stop_loss, target,
            exchange, strategy, confidence, reasoning, atr,
            sector=sector, sector_source=sector_source,
            sector_observed_at=sector_observed_at,
        )


REFERENCE_MAX_AGE_SECONDS = 30.0


def _top_of_book_quantity(tick: dict[str, Any], side: str) -> int | None:
    """Displayed size at the best opposite level, or None when depth is unknown."""
    levels = tick.get("best_ask" if side == "BUY" else "best_bid")
    if not isinstance(levels, list) or not levels or not isinstance(levels[0], dict):
        return None
    quantity = levels[0].get("quantity")
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or quantity < 0:
        return None
    return int(quantity)


def _market_reference(symbol: str, token: str, exchange: str) -> dict[str, Any] | None:
    """Return a reference only when both exchange and receipt timestamps are fresh."""
    from broker.websocket_feed import tick_is_fresh, ws_feed

    tick = ws_feed.get_tick(token)
    if tick and tick_is_fresh(tick) and float(tick.get("ltp") or 0) > 0:
        now = datetime.now(IST)
        age = max(
            (now - datetime.fromisoformat(tick["timestamp"])).total_seconds(),
            (now - datetime.fromisoformat(tick["exchange_timestamp"])).total_seconds(),
        )
        return {
            "price": float(tick["ltp"]), "source": "ws_tick", "age_seconds": round(age, 3),
            "top_quantity": {
                "BUY": _top_of_book_quantity(tick, "BUY"),
                "SELL": _top_of_book_quantity(tick, "SELL"),
            },
        }
    return None


def simulate_paper_fill(
    action: str, order_type: str, requested_price: float, reference_price: float,
    slippage_pct: float, quantity: int = 0, available_quantity: int | None = None,
    allow_partial: bool = False,
) -> tuple[str, float | None, int]:
    """Return (outcome, fill price, filled quantity); LIMIT orders never fill beyond their limit.

    Partial fills are modelled only when ``allow_partial`` is set and the
    displayed depth is known; otherwise an order fills fully or not at all.
    """
    side = action.upper()
    slipped = reference_price * (1 + slippage_pct if side == "BUY" else 1 - slippage_pct)
    if order_type.upper() != "LIMIT":
        fill = round(slipped, 2)
    else:
        if requested_price <= 0:
            return "INVALID_PRICE", None, 0
        marketable = reference_price <= requested_price if side == "BUY" else reference_price >= requested_price
        if not marketable:
            return "NOT_MARKETABLE", None, 0
        fill = round(min(slipped, requested_price) if side == "BUY" else max(slipped, requested_price), 2)
    if allow_partial and available_quantity is not None and quantity > 0 and available_quantity < quantity:
        filled = max(0, math.floor(quantity * min(1.0, available_quantity / quantity)))
        if filled == 0:
            return "NO_DEPTH", None, 0
        return "PARTIALLY_FILLED", fill, filled
    return "FILLED", fill, quantity


def _paper_entry_limit_reason(
    symbol: str, now: datetime, *, session: Session | None = None,
) -> str | None:
    from broker.portfolio_tracker import portfolio_tracker

    if portfolio_tracker.daily_entry_count(as_of=now, session=session) >= settings.max_daily_entries:
        return f"daily entry cap {settings.max_daily_entries} reached"
    if portfolio_tracker.daily_entry_count(symbol, as_of=now, session=session) >= settings.max_daily_entries_per_symbol:
        return f"daily per-symbol entry cap {settings.max_daily_entries_per_symbol} reached"
    if settings.symbol_reentry_cooldown_minutes:
        last_exit = portfolio_tracker.last_symbol_exit_time(symbol, session=session)
        if last_exit is not None:
            if last_exit.tzinfo is not None:
                last_exit = last_exit.astimezone(IST).replace(tzinfo=None)
            elapsed = (now.replace(tzinfo=None) - last_exit).total_seconds()
            if elapsed < settings.symbol_reentry_cooldown_minutes * 60:
                return f"re-entry cooldown of {settings.symbol_reentry_cooldown_minutes} minutes active"
    return None


def _paper_trade(
    order_params: dict,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    stop_loss: float | None,
    target: float | None,
    exchange: str,
    strategy: str,
    confidence: float,
    reasoning: str,
    *,
    allow_requested_price_reference: bool = False,
    atr: float | None = None,
    is_exit: bool = False,
    expected_entry_trade_ids: list[int] | None = None,
    expected_paper_book: dict[str, dict[str, Any]] | None = None,
    sector: str | None = None,
    sector_source: str | None = None,
    sector_observed_at: str | None = None,
    horizon: str | None = None,
    strategy_version: str | None = None,
    exit_policy: str | None = None,
    atr_timeframe: str | None = None,
    book: str | None = None,
) -> dict[str, Any] | None:
    """Simulate an exchange fill against a live reference price and audit the attempt."""
    side = action.upper()
    order_type = str(order_params["ordertype"]).upper()
    requested = float(price or 0)
    product = "INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY"
    if horizon is not None and horizon_to_product(horizon) != product:
        raise ValueError(f"Horizon {horizon} cannot be carried by product {product}")
    # Entries record the strategy horizon; callers that pass only a product keep the legacy values.
    holding_period = horizon or product
    with get_db() as db:
        if db.get_bind().dialect.name != "sqlite":
            raise RuntimeError("Paper order serialization requires SQLite")
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        reference = _market_reference(symbol, order_params["symboltoken"], exchange)
        if reference is None and allow_requested_price_reference and requested > 0:
            reference = {"price": requested, "source": "requested_price", "age_seconds": None}
        if reference is None:
            outcome, fill_price, filled_quantity = "REJECTED_NO_REFERENCE", None, 0
        else:
            depth = (reference.get("top_quantity") or {}).get(side)
            outcome, fill_price, filled_quantity = simulate_paper_fill(
                side, order_type, requested, reference["price"], settings.slippage_pct,
                quantity=quantity, available_quantity=depth,
                allow_partial=settings.paper_partial_fill_enabled and not is_exit,
            )
        requested_quantity = quantity
        audit_fields = {
            "symbol": symbol, "action": side, "quantity": quantity, "order_type": order_type,
            "requested_price": requested,
            "reference_price": reference["price"] if reference else None,
            "reference_source": reference["source"] if reference else None,
            "reference_age_seconds": reference["age_seconds"] if reference else None,
            "slippage_pct": settings.slippage_pct, "fill_price": fill_price,
            "outcome": outcome, "strategy": strategy,
        }
        if outcome not in {"FILLED", "PARTIALLY_FILLED"}:
            db.add(PaperFillAudit(**audit_fields))
            logger.warning("[PAPER] {} {}x {} not filled: {}", side, quantity, symbol, outcome)
            return None
        if outcome == "PARTIALLY_FILLED":
            # Only the filled quantity becomes a position; the remainder is cancelled (IOC).
            quantity = filled_quantity
            audit_fields["quantity"] = quantity
            audit_fields["outcome"] = "FILLED"
            logger.warning(
                "[PAPER] {} {}x {} partially filled: {} of {}.", side, requested_quantity, symbol,
                quantity, requested_quantity,
            )
        now = datetime.now(IST)
        if not is_exit:
            reason = _paper_entry_limit_reason(symbol, now, session=db)
            if reason is not None:
                db.add(PaperFillAudit(**{
                    **audit_fields, "fill_price": None, "outcome": "REJECTED_ENTRY_LIMIT",
                }))
                logger.warning("[PAPER] Entry for {} rejected: {}.", symbol, reason)
                return None
            from broker.paper_ledger import position_book_identity, replay_orders, summarize_positions

            try:
                expected = position_book_identity(expected_paper_book)
                prior_orders = db.query(Trade).filter(
                    Trade.is_paper == True, Trade.status == "EXECUTED",
                ).all()
                current_book = summarize_positions(replay_orders(prior_orders))
                current = position_book_identity(current_book)
                if expected != current:
                    raise ValueError("paper position book changed since risk review")
            except (TypeError, ValueError, OverflowError) as exc:
                db.add(PaperFillAudit(**{
                    **audit_fields, "fill_price": None, "outcome": "REJECTED_STALE_BOOK",
                }))
                logger.warning("[PAPER] Entry for {} rejected: {}.", symbol, exc)
                return None
            if symbol in current_book:
                db.add(PaperFillAudit(**{
                    **audit_fields, "fill_price": None, "outcome": "EXISTING_POSITION",
                }))
                logger.warning("[PAPER] Entry for {} rejected: an open position already exists.", symbol)
                return None
        if is_exit:
            from broker.paper_ledger import replay_orders

            prior_orders = (
                db.query(Trade)
                .filter(Trade.symbol == symbol, Trade.is_paper == True, Trade.status == "EXECUTED")
                .all()
            )
            lots = replay_orders(prior_orders).open_lots.get(symbol, [])
            expected_side = "BUY" if side == "SELL" else "SELL"
            if lots and any(lot.holding_period == "BTST" and not _btst_delivered(lot, now) for lot in lots):
                # Selling shares not yet delivered risks an exchange auction; retry on a later pass.
                db.add(PaperFillAudit(**{
                    **audit_fields, "fill_price": None, "outcome": "SHORT_DELIVERY_RISK",
                }))
                logger.warning(
                    "[PAPER] Exit {}x {} deferred: BTST shares are not yet delivered.", quantity, symbol,
                )
                return None
            available_quantity = sum(lot.quantity for lot in lots)
            mixed_lots = len({lot.holding_period for lot in lots}) > 1
            if (
                not lots
                or not expected_entry_trade_ids
                or expected_entry_trade_ids != [lot.trade_id for lot in lots]
                or any(lot.side != expected_side for lot in lots)
                # Mixed legacy lots may only be flattened whole, whatever the product.
                or (not mixed_lots and horizon_to_product(lots[0].holding_period) != product)
                or (mixed_lots and quantity != available_quantity)
                or quantity > available_quantity
            ):
                db.add(PaperFillAudit(**{
                    **audit_fields, "fill_price": None, "outcome": "REJECTED_POSITION_CHANGED",
                }))
                logger.warning(
                    "[PAPER] Exit {} {}x {} rejected: current ledger position changed or is mixed.",
                    side, quantity, symbol,
                )
                return None
        trade = Trade(
            timestamp=now,
            symbol=symbol,
            exchange=exchange,
            action=side,
            quantity=quantity,
            order_type=order_type,
            price=fill_price,
            stop_loss=stop_loss,
            target=target,
            atr=atr,
            sector=sector if not is_exit else None,
            sector_source=sector_source if not is_exit else None,
            sector_observed_at=sector_observed_at if not is_exit else None,
            holding_period=holding_period,
            horizon=horizon if not is_exit else None,
            strategy_version=strategy_version if not is_exit else None,
            exit_policy=exit_policy if not is_exit else None,
            atr_timeframe=atr_timeframe if not is_exit else None,
            book=(book or "core") if not is_exit else None,
            status="EXECUTED",
            order_id=f"PAPER-{now.strftime('%Y%m%d%H%M%S%f')}",
            is_paper=True,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
        )
        db.add(trade)
        db.flush()
        db.add(PaperFillAudit(trade_id=trade.id, order_id=trade.order_id, **audit_fields))
        _record_closed_lots(db, trade, fill_price, now)
        from broker import cash_ledger
        from broker.portfolio_tracker import PortfolioTracker

        cash_ledger.record_trade_cash(db, trade, PortfolioTracker._trade_cost(trade))
        order_id = trade.order_id
        trade_id = trade.id

    result = {
        "order_id": order_id,
        "trade_id": trade_id,
        "symbol": symbol,
        "action": side,
        "quantity": quantity,
        "requested_quantity": requested_quantity,
        "partial": quantity < requested_quantity,
        "fill_price": fill_price,
        "price": fill_price,
        "reference_price": reference["price"],
        "reference_source": reference["source"],
        "stop_loss": stop_loss,
        "target": target,
        "atr": atr,
        "holding_period": holding_period,
        "horizon": horizon,
        "strategy_version": strategy_version,
        "exit_policy": exit_policy,
        "atr_timeframe": atr_timeframe,
        "strategy": strategy,
        "is_paper": True,
        "status": "EXECUTED",
    }

    logger.info(
        "[PAPER] {} {}x {} @ ₹{:.2f} (ref ₹{:.2f} via {})",
        side, quantity, symbol, fill_price, reference["price"], reference["source"],
    )
    return result


def _btst_delivered(lot: Any, now: datetime) -> bool:
    """BTST shares arrive after the next session opens (``btst_delivery_minutes_after_open``)."""
    from broker.cash_ledger import settlement_date

    bought = lot.timestamp
    day = bought.date() if isinstance(bought, datetime) else datetime.fromisoformat(str(bought)).date()
    delivery_day = datetime.fromisoformat(settlement_date(day)).date()
    delivered_at = datetime.combine(delivery_day, datetime.min.time()).replace(tzinfo=now.tzinfo) + timedelta(
        hours=9, minutes=15 + settings.btst_delivery_minutes_after_open,
    )
    return now >= delivered_at


def _record_closed_lots(db: Any, closing_order: Trade, exit_price: float, exit_at: datetime) -> None:
    """Record lots closed by ``closing_order`` using the shared FIFO replay."""
    from broker.paper_ledger import apply_order, replay_orders

    prior = (
        db.query(Trade)
        .filter(
            Trade.symbol == closing_order.symbol,
            Trade.is_paper == closing_order.is_paper,
            Trade.status == "EXECUTED",
            Trade.id != closing_order.id,
        )
        .all()
    )
    replay = replay_orders(prior)
    entries = {trade.id: trade for trade in prior}
    for close in apply_order(replay, closing_order):
        entry = entries[close.entry_id]
        db.add(ClosedTrade(
            entry_order_id=entry.id,
            exit_order_id=closing_order.id,
            symbol=entry.symbol,
            action=entry.action,
            quantity=close.quantity,
            entry_price=entry.price,
            exit_price=closing_order.price,
            entry_timestamp=entry.timestamp,
            exit_timestamp=exit_at,
            gross_pnl=close.gross_pnl,
            strategy=entry.strategy or "unknown",
            holding_period=entry.holding_period or "INTRADAY",
            is_paper=entry.is_paper,
            venue=entry.exchange or "NSE",
            currency="INR",
            exit_policy=getattr(entry, "exit_policy", None),
            book=getattr(entry, "book", None) or "core",
        ))


def _live_trade(
    order_params: dict,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    stop_loss: float | None,
    target: float | None,
    exchange: str,
    strategy: str,
    confidence: float,
    reasoning: str,
    atr: float | None = None,
    *,
    sector: str | None = None,
    sector_source: str | None = None,
    sector_observed_at: str | None = None,
    horizon: str | None = None,
    strategy_version: str | None = None,
    exit_policy: str | None = None,
    atr_timeframe: str | None = None,
    book: str | None = None,
) -> dict[str, Any] | None:
    """Execute a live order via Angel One."""
    try:
        resp = angel_client.place_order(order_params)
        order_id = resp if isinstance(resp, str) else (resp.get("data", {}).get("orderid") if isinstance(resp, dict) else None)
        if not order_id:
            raise OrderSubmissionUncertainError("Broker submission returned no order identifier")

        with get_db() as db:
            trade = Trade(
                timestamp=datetime.now(IST),
                symbol=symbol,
                exchange=exchange,
                action=action.upper(),
                quantity=quantity,
                order_type=order_params["ordertype"],
                price=price,
                stop_loss=stop_loss,
                target=target,
                atr=atr,
                sector=sector,
                sector_source=sector_source,
                sector_observed_at=sector_observed_at,
                holding_period="INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
                status="PENDING",
                order_id=str(order_id) if order_id else None,
                is_paper=False,
                strategy=strategy,
                confidence=confidence,
                reasoning=reasoning,
            )
            db.add(trade)

        logger.info("[LIVE] {} {}x {} @ ₹{:.2f} — order_id: {}", action, quantity, symbol, price, order_id)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "stop_loss": stop_loss,
            "target": target,
            "atr": atr,
            "holding_period": "INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
            "strategy": strategy,
            "is_paper": False,
            "status": "PENDING",
        }

    except BrokerRejectedError:
        logger.warning("Live order rejected before acceptance for {}.", symbol)
        return None
    except Exception as exc:
        logger.error("Live order outcome is uncertain for {} ({}).", symbol, type(exc).__name__)
        raise OrderSubmissionUncertainError(
            f"Live order for {symbol} requires reconciliation before retry"
        ) from exc


def cancel_order(order_id: str, variety: str = "NORMAL") -> bool:
    """Cancel an order by ID."""
    if settings.paper_trading:
        logger.info("[PAPER] Cancel order {}", order_id)
        with get_db() as db:
            trade = db.query(Trade).filter(Trade.order_id == order_id).first()
            if trade:
                trade.status = "CANCELLED"
        return True
    else:
        resp = angel_client.cancel_order(order_id, variety)
        return resp is not None


def get_open_orders() -> list[dict[str, Any]]:
    """Get all pending/open orders."""
    if settings.paper_trading:
        with get_db() as db:
            trades = db.query(Trade).filter(Trade.status == "PENDING", Trade.is_paper == True).all()
            return [{"order_id": t.order_id, "symbol": t.symbol, "action": t.action, "quantity": t.quantity, "price": t.price} for t in trades]
    else:
        return angel_client.get_order_book()


_OPEN_ORDER_STATUSES = {"OPEN", "PENDING", "TRIGGER PENDING", "VALIDATION PENDING", "PUT ORDER REQ RECEIVED"}


def cancel_pending_entries(positions: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Cancel resting entry orders (never exits); called when a loss halt latches.

    An order is an exit when it trades against an open position in its symbol.
    Paper entries fill or fail immediately (IOC style), so this is normally a no-op there.
    """
    if positions is None:
        from broker.portfolio_tracker import portfolio_tracker

        positions = portfolio_tracker.positions
    cancelled: list[str] = []
    for order in get_open_orders():
        status = str(order.get("orderstatus") or order.get("status") or "OPEN").upper()
        if status not in _OPEN_ORDER_STATUSES and not settings.paper_trading:
            continue
        symbol = str(order.get("tradingsymbol") or order.get("symbol") or "")
        side = str(order.get("transactiontype") or order.get("action") or "").upper()
        order_id = str(order.get("orderid") or order.get("order_id") or "")
        held = float((positions.get(symbol) or {}).get("quantity") or 0)
        is_exit = (side == "SELL" and held > 0) or (side == "BUY" and held < 0)
        if is_exit or not order_id:
            continue
        try:
            if cancel_order(order_id):
                cancelled.append(order_id)
        except Exception as exc:
            logger.error("Could not cancel pending entry {}: {}", order_id, type(exc).__name__)
    if cancelled:
        logger.warning("Cancelled {} pending entry order(s) after a safety halt.", len(cancelled))
    return cancelled
