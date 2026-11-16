"""Angel One SmartAPI wrapper — authentication, session management, and order placement."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pyotp
from loguru import logger

from operations.redaction import silence_sdk_logging
from SmartApi import SmartConnect
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from broker.base_broker import BaseBrokerAdapter, BrokerRejectedError, OrderSubmissionUncertainError
from broker.candle_rate_limiter import candle_rate_limiter


class AngelClient(BaseBrokerAdapter):
    """Thread-safe Angel One SmartAPI client with auto-TOTP and session refresh."""

    def __init__(self) -> None:
        self._api: SmartConnect | None = None
        self._jwt_token: str = ""
        self._refresh_token: str = ""
        self._feed_token: str = ""
        self._last_auth: datetime | None = None
        self._lock = threading.RLock()
        self._authenticated = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def authenticate(self) -> bool:
        """Authenticate with Angel One using API key, client ID, MPIN, and auto-generated TOTP."""
        with self._lock:
            try:
                self._api = SmartConnect(api_key=settings.angel_api_key)
                silence_sdk_logging()  # the SDK would write request headers (keys, JWT) to disk
                totp = pyotp.TOTP(settings.angel_totp_secret).now()
                data = self._api.generateSession(
                    settings.angel_client_id,
                    settings.angel_mpin,
                    totp,
                )

                if not data or data.get("status") is False:
                    self._authenticated = False
                    msg = data.get("message", "Unknown auth error") if data else "No response"
                    logger.error("Angel One auth failed: {}", msg)
                    return False

                self._jwt_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                self._feed_token = self._api.getfeedToken()
                self._last_auth = datetime.now()
                self._authenticated = True

                logger.info("Angel One authenticated. Feed token obtained.")
                return True

            except Exception as exc:
                logger.exception("Angel One authentication error: {}", exc)
                self._authenticated = False
                raise

    def refresh_session(self) -> bool:
        """Refresh the JWT token using the refresh token. Call every ~6 hours."""
        with self._lock:
            if not self._api or not self._refresh_token:
                logger.warning("Cannot refresh — no active session. Re-authenticating...")
                return self.authenticate()

            try:
                token_data = self._api.generateToken(self._refresh_token)
                if token_data and token_data.get("status"):
                    self._jwt_token = token_data["data"]["jwtToken"]
                    self._refresh_token = token_data["data"]["refreshToken"]
                    self._last_auth = datetime.now()
                    logger.info("Angel One session refreshed.")
                    return True
                else:
                    logger.warning("Token refresh failed, re-authenticating...")
                    return self.authenticate()
            except Exception as exc:
                logger.exception("Session refresh error: {}", exc)
                return self.authenticate()

    def ensure_session(self) -> None:
        """Ensure we have a valid session; refresh if >6 hours old."""
        if not self._authenticated:
            if not self.authenticate():
                raise RuntimeError("Angel One authentication was rejected")
            return
        if self._last_auth and (datetime.now() - self._last_auth) > timedelta(hours=6):
            if not self.refresh_session():
                self._authenticated = False
                raise RuntimeError("Angel One session refresh was rejected")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api(self) -> SmartConnect:
        """Return the underlying SmartConnect instance."""
        self.ensure_session()
        assert self._api is not None, "Client not authenticated"
        return self._api

    @property
    def feed_token(self) -> str:
        self.ensure_session()
        return self._feed_token

    @property
    def jwt_token(self) -> str:
        self.ensure_session()
        return self._jwt_token

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def has_credentials(self) -> bool:
        """Whether a complete non-placeholder Angel One login is configured.

        Paper trading can run without this integration, using public OHLCV data
        and simulated requested-price fills.  Callers use this guard to avoid
        slow, repeated authentication attempts when no broker is configured.
        """
        required = (
            settings.angel_api_key,
            settings.angel_client_id,
            settings.angel_mpin,
            settings.angel_totp_secret,
        )
        return all(value and not value.startswith("your_") for value in required)

    def authenticate_totp(self) -> bool:
        """Authenticate through SmartAPI using the configured TOTP secret."""
        return self.authenticate()

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_candle_data(
        self,
        exchange: str,
        symbol_token: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[list[Any]]:
        """Fetch historical OHLCV candle data.

        Args:
            exchange: 'NSE' or 'NFO'
            symbol_token: Angel One token from instrument master
            interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY, etc.
            from_date: 'YYYY-MM-DD HH:MM' format
            to_date: 'YYYY-MM-DD HH:MM' format

        Returns:
            List of candle data lists.
        """
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        api = self.api
        resp = candle_rate_limiter.call(api.getCandleData, params)
        if resp and resp.get("status"):
            return resp.get("data", [])
        logger.warning("getCandleData failed: {}", resp)
        return []

    def get_ltp(self, exchange: str, symbol: str, token: str) -> float | None:
        """Get Last Traded Price for a symbol."""
        try:
            resp = self.api.ltpData(exchange, symbol, token)
            if resp and resp.get("status"):
                return float(resp["data"]["ltp"])
        except Exception as exc:
            logger.error("LTP fetch error for {}: {}", symbol, exc)
        return None

    def get_quote(self, exchange: str, symbol: str, token: str) -> dict[str, Any] | None:
        """Get full quote data for a symbol."""
        try:
            resp = self.api.ltpData(exchange, symbol, token)
            if resp and resp.get("status"):
                return resp["data"]
        except Exception as exc:
            logger.error("Quote fetch error for {}: {}", symbol, exc)
        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Place an order via Angel One.

        Args:
            params: Order parameters dict with keys:
                variety, tradingsymbol, symboltoken, transactiontype,
                exchange, ordertype, producttype, duration, price,
                squareoff, stoploss, quantity, triggerprice
        """
        try:
            resp = self.api.placeOrder(params)
            if isinstance(resp, str) and resp:
                return {"status": True, "data": {"orderid": resp}}
            if isinstance(resp, dict) and resp.get("status") is False:
                raise BrokerRejectedError(
                    "Angel One explicitly rejected the order"
                )
            if not isinstance(resp, dict) or not (resp.get("data") or {}).get("orderid"):
                raise OrderSubmissionUncertainError("Angel One returned no verifiable order identifier")
            logger.info("Order submitted for {}; confirmation of fills is still required.", params.get("tradingsymbol"))
            return resp
        except (BrokerRejectedError, OrderSubmissionUncertainError):
            raise
        except Exception as exc:
            logger.error("Order submission outcome unknown for {} ({}).", params.get("tradingsymbol"), type(exc).__name__)
            raise OrderSubmissionUncertainError("Angel One order requires reconciliation before retry") from exc

    def modify_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Modify an existing order."""
        try:
            resp = self.api.modifyOrder(params)
            if not resp or resp.get("status") is False:
                raise RuntimeError(f"Angel One rejected the modification: {(resp or {}).get('message', 'empty response')}")
            logger.info("Order modified: {}", resp)
            return resp
        except Exception as exc:
            logger.exception("Order modification failed: {}", exc)
            return None

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict[str, Any]:
        """Cancel an order by order ID."""
        try:
            resp = self.api.cancelOrder(order_id, variety)
            if not resp or resp.get("status") is False:
                raise RuntimeError(f"Angel One rejected the cancellation: {(resp or {}).get('message', 'empty response')}")
            logger.info("Order cancelled: {} → {}", order_id, resp)
            return resp
        except Exception as exc:
            logger.exception("Order cancellation failed for {}: {}", order_id, exc)
            return None

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch current holdings."""
        try:
            resp = self.api.holding()
            if not resp or resp.get("status") is False:
                raise RuntimeError(f"Angel One holdings unavailable: {(resp or {}).get('message', 'empty response')}")
            return resp.get("data") or []
        except Exception as exc:
            logger.error("Holdings fetch error: {}", exc)
            raise

    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch current open positions."""
        try:
            resp = self.api.position()
            if not resp or resp.get("status") is False:
                raise RuntimeError(f"Angel One positions unavailable: {(resp or {}).get('message', 'empty response')}")
            return resp.get("data") or []
        except Exception as exc:
            logger.error("Positions fetch error: {}", exc)
            raise

    def get_order_book(self) -> list[dict[str, Any]]:
        """Fetch the order book or raise when broker reconciliation is unavailable."""
        try:
            resp = self.api.orderBook()
            if not resp or resp.get("status") is False:
                raise RuntimeError(
                    f"Angel One order book unavailable: {(resp or {}).get('message', 'empty response')}"
                )
            return resp.get("data") or []
        except Exception as exc:
            logger.error("Order book fetch error: {}", exc)
            raise

    # ------------------------------------------------------------------
    # Margin
    # ------------------------------------------------------------------

    def get_rms_limits(self) -> dict[str, Any] | None:
        """Fetch RMS (risk management) margin limits."""
        try:
            resp = self.api.rmsLimit()
            if resp and resp.get("status"):
                return resp.get("data")
        except Exception as exc:
            logger.error("RMS limits fetch error: {}", exc)
        return None

    def get_funds_and_margins(self) -> dict[str, Any]:
        """Return Angel RMS data or raise if the broker cannot provide it."""
        limits = self.get_rms_limits()
        if limits is None:
            raise RuntimeError("Angel One RMS limits are unavailable")
        return {"available": limits, "raw": limits}

    def place_bracket_or_gtt_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Submit a supported BO order; do not emulate OCO with unlinked orders."""
        variety = str(params.get("variety", "")).upper()
        product = str(params.get("producttype", "")).upper()
        if variety not in {"ROBO", "BO"} and product != "BO":
            raise NotImplementedError(
                "Angel One bracket support is available only through broker-approved BO orders."
            )
        order_params = dict(params)
        order_params.setdefault("variety", "ROBO")
        order_params.setdefault("producttype", "BO")
        response = self.place_order(order_params)
        if response is None:
            raise RuntimeError("Angel One returned no response for the bracket order")
        return response

    def flatten_all_intraday(self) -> list[dict[str, Any]]:
        """Close all net intraday positions with opposite-side market orders."""
        order_book = self.get_order_book()
        positions = self.api.position()
        if not positions or positions.get("status") is False:
            raise RuntimeError("Unable to reconcile Angel One positions before flattening")
        intraday_positions = [
            position for position in positions.get("data") or []
            if int(float(position.get("netqty", 0) or 0)) != 0
            and str(position.get("producttype", "")).upper() in {"INTRADAY", "MIS"}
        ]
        active_symbols = {position.get("tradingsymbol") for position in intraday_positions}
        unresolved = []
        terminal_statuses = {"COMPLETE", "COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}
        for order in order_book:
            symbol = order.get("tradingsymbol")
            status = str(order.get("orderstatus") or order.get("status") or "").upper()
            if symbol in active_symbols and status not in terminal_statuses:
                unresolved.append(f"{symbol}:{status or 'UNKNOWN'}")
        if unresolved:
            raise RuntimeError(
                "Unresolved broker orders prevent safe square-off: " + ", ".join(unresolved)
            )

        flattened: list[dict[str, Any]] = []
        for position in intraday_positions:
            quantity = int(float(position.get("netqty", 0) or 0))
            order = {
                "variety": "NORMAL",
                "tradingsymbol": position["tradingsymbol"],
                "symboltoken": position["symboltoken"],
                "transactiontype": "SELL" if quantity > 0 else "BUY",
                "exchange": position["exchange"],
                "ordertype": "MARKET",
                "producttype": position.get("producttype", "INTRADAY"),
                "duration": "DAY",
                "price": "0",
                "quantity": str(abs(quantity)),
            }
            response = self.place_order(order)
            if response is None:
                raise RuntimeError(f"Angel One returned no response flattening {position['tradingsymbol']}")
            flattened.append(response)
        return flattened

    def subscribe_ticks(
        self, exchange: str, tokens: list[str], callback: Any
    ) -> None:
        """Subscribe to Angel ticks through the shared WebSocket feed."""
        from broker.websocket_feed import tick_event_bus, ws_feed

        exchange_code = {
            "NSE": "nse_cm",
            "NFO": "nse_fo",
            "BSE": "bse_cm",
            "BFO": "bse_fo",
            "MCX": "mcx_fo",
        }.get(exchange.upper())
        if exchange_code is None:
            raise ValueError(f"Unsupported Angel One exchange: {exchange}")
        tick_event_bus.subscribe(callback)
        ws_feed.connect()
        ws_feed.subscribe(exchange_code, tokens)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def logout(self) -> None:
        """Logout and invalidate session."""
        if self._api:
            try:
                self._api.terminateSession(settings.angel_client_id)
                logger.info("Angel One session terminated.")
            except Exception as exc:
                logger.warning("Logout error: {}", exc)
        self._authenticated = False
        self._api = None


# Module-level singleton
angel_client = AngelClient()
