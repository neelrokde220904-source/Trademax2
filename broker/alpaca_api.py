"""Alpaca Markets API client for US equity trading with fractional share support.

Used as secondary/fallback broker for US markets when IBKR is unavailable.
FEMA/LRS compliance enforced on every BUY order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import settings

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.info("alpaca-py not installed — Alpaca client unavailable")


@dataclass
class AlpacaOrder:
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0
    order_type: str = "market"
    limit_price: float = 0.0
    status: str = "PENDING"
    filled_qty: float = 0
    filled_avg_price: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""


class AlpacaClient:
    """Alpaca Markets REST API wrapper with fractional shares and FEMA compliance."""

    def __init__(self) -> None:
        self.api_key = getattr(settings, "alpaca_api_key", "")
        self.secret_key = getattr(settings, "alpaca_secret_key", "")
        self.paper = getattr(settings, "alpaca_paper", True)
        self._trading_client = None
        self._data_client = None

    @property
    def trading_client(self) -> Any:
        if self._trading_client is None and ALPACA_AVAILABLE and self.api_key:
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
        return self._trading_client

    @property
    def data_client(self) -> Any:
        if self._data_client is None and ALPACA_AVAILABLE and self.api_key:
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._data_client

    @property
    def available(self) -> bool:
        return ALPACA_AVAILABLE and bool(self.api_key)

    def get_account(self) -> dict[str, Any]:
        """Get account details: equity, buying power, etc."""
        if not self.trading_client:
            return {"error": "Alpaca client not configured"}
        try:
            acct = self.trading_client.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "currency": "USD",
                "status": acct.status,
                "pattern_day_trader": acct.pattern_day_trader,
            }
        except Exception as e:
            logger.error("Alpaca account fetch failed: {}", e)
            return {"error": str(e)}

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        if not self.trading_client:
            return []
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "side": p.side,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Alpaca positions failed: {}", e)
            return []

    def get_positions_strict(self) -> list[dict[str, Any]]:
        """Fetch positions without converting an API failure into an empty book."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client is not configured")
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "tradingsymbol": p.symbol,
                    "quantity": float(p.qty),
                    "average_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "ltp": float(p.current_price),
                    "side": str(p.side),
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error("Alpaca strict position fetch failed: {}", type(exc).__name__)
            raise RuntimeError("Alpaca positions unavailable") from exc

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get latest quote for a symbol."""
        if not self.data_client:
            return {"error": "Data client not configured"}
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            q = quotes.get(symbol)
            if q:
                return {
                    "symbol": symbol,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "bid_size": q.bid_size,
                    "ask_size": q.ask_size,
                    "timestamp": q.timestamp.isoformat() if q.timestamp else "",
                }
            return {"symbol": symbol, "error": "No quote available"}
        except Exception as e:
            logger.error("Alpaca quote failed for {}: {}", symbol, e)
            return {"error": str(e)}

    def get_bars(
        self, symbol: str, *, start: datetime, end: datetime, minutes: int = 15, limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch IEX-only intraday bars; data failures propagate so callers fail closed."""
        if not self.data_client:
            raise RuntimeError("Alpaca historical data client is not configured")
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                start=start,
                end=end,
                limit=limit,
                feed="iex",
            )
            response = self.data_client.get_stock_bars(request)
            return [
                {"timestamp": bar.timestamp, "close": float(bar.close)}
                for bar in response.data.get(symbol, [])
            ]
        except Exception as exc:
            logger.error("Alpaca bars failed for {}: {}", symbol, type(exc).__name__)
            raise RuntimeError(f"Alpaca bars unavailable for {symbol}") from exc

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch open Alpaca orders; callers must not treat an API failure as an empty list."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client is not configured")
        try:
            orders = self.trading_client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500),
            )
            return [
                {
                    "id": str(order.id),
                    "symbol": str(order.symbol).upper(),
                    "side": str(order.side).lower().split(".")[-1],
                    "quantity": float(order.qty or 0),
                    "filled_quantity": float(order.filled_qty or 0),
                    "limit_price": float(order.limit_price or 0),
                }
                for order in orders
            ]
        except Exception as exc:
            logger.error("Alpaca open orders failed: {}", type(exc).__name__)
            raise RuntimeError("Alpaca open orders unavailable") from exc

    def place_market_order(self, symbol: str, side: str, qty: float,
                           fractional: bool = False) -> AlpacaOrder:
        """Place a market order. Supports fractional shares."""
        return self._place_order(symbol, side, qty, "market", 0.0, fractional)

    def place_limit_order(self, symbol: str, side: str, qty: float,
                          limit_price: float, fractional: bool = False) -> AlpacaOrder:
        """Place a limit order."""
        return self._place_order(symbol, side, qty, "limit", limit_price, fractional)

    def _place_order(self, symbol: str, side: str, qty: float,
                     order_type: str, limit_price: float,
                     fractional: bool) -> AlpacaOrder:
        """Internal order placement with FEMA compliance."""
        side = side.lower()
        if side not in ("buy", "sell"):
            return AlpacaOrder(status="REJECTED", symbol=symbol, side=side, qty=qty)

        if not self.trading_client:
            return AlpacaOrder(status="NOT_CONFIGURED", symbol=symbol, side=side, qty=qty)

        # FEMA compliance on buys
        if side == "buy":
            estimated_usd = qty * limit_price if limit_price > 0 else qty * 100
            if not self._check_fema(symbol, qty, estimated_usd):
                return AlpacaOrder(status="FEMA_BLOCKED", symbol=symbol, side=side, qty=qty)

        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            if order_type == "market":
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty if not fractional else None,
                    notional=None,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
                if fractional:
                    request.qty = qty  # Alpaca handles fractional qty natively
            else:
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )

            order = self.trading_client.submit_order(request)

            return AlpacaOrder(
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price,
                status=str(order.status),
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else 0,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else "",
            )

        except Exception as e:
            logger.error("Alpaca order failed: {} {} {} — {}", side, qty, symbol, e)
            return AlpacaOrder(status="ERROR", symbol=symbol, side=side, qty=qty)

    def _check_fema(self, symbol: str, qty: float, estimated_usd: float) -> bool:
        """Check FEMA/LRS compliance."""
        try:
            from config.fema_lrs_controller import fema_controller
            result = fema_controller.pre_trade_compliance_check(
                symbol=symbol,
                quantity=qty,
                estimated_usd_value=estimated_usd,
                destination_country="US",
            )
            if not result.get("approved"):
                logger.warning("FEMA blocked Alpaca order: {} — {}", symbol,
                               result.get("reason", ""))
                return False
            return True
        except ImportError:
            return True
        except Exception as e:
            logger.error("FEMA check error: {} — blocking for safety", e)
            return False

    def cancel_order(self, order_id: str) -> bool:
        if not self.trading_client:
            return False
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error("Cancel order {} failed: {}", order_id, e)
            return False

    def cancel_all_orders(self) -> bool:
        if not self.trading_client:
            return False
        try:
            self.trading_client.cancel_orders()
            return True
        except Exception as e:
            logger.error("Cancel all orders failed: {}", e)
            return False


# Singleton
alpaca_client = AlpacaClient()
