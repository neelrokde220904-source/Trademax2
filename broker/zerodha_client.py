"""Zerodha Kite Connect adapter with conservative order-rate limiting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from config.settings import settings
from broker.base_broker import BaseBrokerAdapter, BrokerRejectedError


class TokenBucket:
    """Thread-safe token bucket used to cap order submissions."""

    def __init__(self, rate: float = 8.0, capacity: float = 1.0) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Wait until one order token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated_at) * self.rate)
                self._updated_at = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_seconds = (1 - self._tokens) / self.rate
            time.sleep(wait_seconds)


class ZerodhaBrokerAdapter(BaseBrokerAdapter):
    """Kite Connect adapter.

    Kite Connect's supported login flow requires an interactive OAuth request
    token. TOTP alone cannot create a Kite access token through the trading API;
    this adapter therefore accepts a token created by the official login flow
    and deliberately does not automate browser login or scrape credentials.
    """

    def __init__(self, rate_limiter: TokenBucket | None = None) -> None:
        self._api: Any = None
        self._ticker: Any = None
        self._tick_unsubscribe: Callable[[], None] | None = None
        self._rate_limiter = rate_limiter or TokenBucket()

    def _client(self) -> Any:
        if self._api is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:
                raise RuntimeError("Install the optional kiteconnect dependency to use Zerodha") from exc
            if not settings.zerodha_api_key:
                raise RuntimeError("ZERODHA_API_KEY is not configured")
            self._api = KiteConnect(api_key=settings.zerodha_api_key)
            if settings.zerodha_access_token:
                self._api.set_access_token(settings.zerodha_access_token)
        return self._api

    def authenticate_totp(self) -> bool:
        """Verify the configured access token; Kite login itself is interactive OAuth."""
        if not settings.zerodha_access_token:
            raise RuntimeError(
                "Zerodha requires an access token from its official interactive OAuth login. "
                "TOTP cannot be used to mint a Kite Connect token."
            )
        self._client().profile()
        return True

    def complete_login(self, request_token: str) -> str:
        """Exchange an official Kite OAuth request token for an access token."""
        if not request_token or not settings.zerodha_api_secret:
            raise ValueError("An OAuth request token and ZERODHA_API_SECRET are required")
        session = self._client().generate_session(request_token, api_secret=settings.zerodha_api_secret)
        access_token = session.get("access_token")
        if not access_token:
            raise RuntimeError("Kite Connect did not return an access token")
        self._api.set_access_token(access_token)
        return access_token

    def get_funds_and_margins(self) -> dict[str, Any]:
        margins = self._client().margins()
        return {"available": margins, "raw": margins}

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._client().positions().get("net", []))

    def get_holdings(self) -> list[dict[str, Any]]:
        return list(self._client().holdings())

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        self._rate_limiter.acquire()
        api = self._client()
        required = ("exchange", "tradingsymbol", "transaction_type", "quantity", "order_type", "product")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing Kite order fields: {', '.join(missing)}")
        options = dict(params)
        variety = options.pop("variety", api.VARIETY_REGULAR)
        try:
            order_id = api.place_order(variety=variety, **options)
        except Exception as exc:
            if type(exc).__name__ in {"InputException", "MarginException"}:
                raise BrokerRejectedError("Kite Connect rejected the order before acceptance") from exc
            raise
        return {"order_id": str(order_id), "status": "PENDING", "broker": "zerodha"}

    def place_bracket_or_gtt_order(self, params: dict[str, Any]) -> dict[str, Any]:
        self._rate_limiter.acquire()
        api = self._client()
        kind = str(params.get("kind", "gtt_oco")).lower()
        if kind != "gtt_oco":
            raise NotImplementedError(
                "Kite bracket orders are not available; only supported GTT OCO orders are exposed."
            )
        required = ("tradingsymbol", "exchange", "trigger_values", "last_price", "orders")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing Kite GTT fields: {', '.join(missing)}")
        gtt_id = api.place_gtt(
            trigger_type=api.GTT_TYPE_OCO,
            tradingsymbol=params["tradingsymbol"],
            exchange=params["exchange"],
            trigger_values=params["trigger_values"],
            last_price=params["last_price"],
            orders=params["orders"],
        )
        return {"order_id": str(gtt_id), "status": "TRIGGER_PENDING", "broker": "zerodha"}

    def modify_order(self, params: dict[str, Any]) -> dict[str, Any]:
        self._rate_limiter.acquire()
        values = dict(params)
        order_id = values.pop("order_id", "")
        variety = values.pop("variety", self._client().VARIETY_REGULAR)
        if not order_id:
            raise ValueError("order_id is required")
        result = self._client().modify_order(variety=variety, order_id=order_id, **values)
        return {"order_id": str(result), "status": "MODIFIED", "broker": "zerodha"}

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict[str, Any]:
        if not order_id:
            raise ValueError("order_id is required")
        self._rate_limiter.acquire()
        result = self._client().cancel_order(variety=variety, order_id=order_id)
        return {"order_id": str(result), "status": "CANCELLED", "broker": "zerodha"}

    def flatten_all_intraday(self) -> list[dict[str, Any]]:
        order_book = self._client().orders()
        results = []
        intraday_positions = [
            position for position in self.get_positions()
            if int(position.get("quantity", 0) or 0)
            and str(position.get("product", "")).upper() == "MIS"
        ]
        active_symbols = {position.get("tradingsymbol") for position in intraday_positions}
        unresolved = []
        terminal_statuses = {"COMPLETE", "COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}
        for order in order_book:
            symbol = order.get("tradingsymbol")
            status = str(order.get("status") or "").upper()
            if symbol in active_symbols and status not in terminal_statuses:
                unresolved.append(f"{symbol}:{status or 'UNKNOWN'}")
        if unresolved:
            raise RuntimeError(
                "Unresolved broker orders prevent safe square-off: " + ", ".join(unresolved)
            )

        for position in intraday_positions:
            net_quantity = int(position.get("quantity", 0) or 0)
            params = {
                "exchange": position["exchange"],
                "tradingsymbol": position["tradingsymbol"],
                "transaction_type": "SELL" if net_quantity > 0 else "BUY",
                "quantity": abs(net_quantity),
                "order_type": "MARKET",
                "product": "MIS",
                "validity": "DAY",
            }
            results.append(self.place_order(params))
        return results

    def subscribe_ticks(
        self, exchange: str, tokens: list[str], callback: Callable[[dict[str, Any]], None]
    ) -> None:
        try:
            from kiteconnect import KiteTicker
        except ImportError as exc:
            raise RuntimeError("Install the optional kiteconnect dependency to use Zerodha ticks") from exc
        self._client()
        if not settings.zerodha_access_token:
            raise RuntimeError("ZERODHA_ACCESS_TOKEN is required to subscribe to ticks")
        from broker.websocket_feed import tick_event_bus

        if self._tick_unsubscribe:
            self._tick_unsubscribe()
        self._tick_unsubscribe = tick_event_bus.subscribe(callback)
        ticker = KiteTicker(settings.zerodha_api_key, settings.zerodha_access_token)
        ticker.on_ticks = lambda _ws, ticks: [
            tick_event_bus.publish("zerodha", {
                **tick,
                "token": str(tick.get("instrument_token", "")),
                "ltp": tick.get("last_price"),
                "timestamp": str(tick.get("exchange_timestamp") or tick.get("timestamp") or ""),
            })
            for tick in ticks
        ]

        def on_connect(ws: Any, _response: Any) -> None:
            tick_event_bus.set_connected("zerodha", True)
            ws.subscribe([int(token) for token in tokens])

        def on_close(_ws: Any, *_args: Any) -> None:
            tick_event_bus.set_connected("zerodha", False)

        ticker.on_connect = on_connect
        ticker.on_close = on_close
        ticker.connect(threaded=True)
        self._ticker = ticker


zerodha_client = ZerodhaBrokerAdapter()
