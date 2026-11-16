"""SmartWebSocketV2 — real-time tick data feed from Angel One."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import ssl
import math
from datetime import datetime
from typing import Any, Callable

import pytz
from loguru import logger
import websocket
import certifi
from SmartApi.smartWebSocketV2 import SmartWebSocketV2 as AngelSmartWebSocketV2

from broker.angel_client import angel_client
from operations.redaction import silence_sdk_logging
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# WebSocket mode constants
MODE_LTP = 1
MODE_QUOTE = 2
MODE_FULL = 3  # LTP + OHLC + Volume + OI + depth
TICK_MAX_AGE_SECONDS = 30.0
MAX_FUTURE_SKEW_SECONDS = 2.0


def tick_is_fresh(tick: dict[str, Any], now: datetime | None = None) -> bool:
    """Require both local receipt and exchange event time to be current."""
    now = now or datetime.now(IST)
    try:
        received = datetime.fromisoformat(tick["timestamp"])
        exchange = datetime.fromisoformat(tick["exchange_timestamp"])
        if received.tzinfo is None or exchange.tzinfo is None:
            return False
        ages = ((now - received).total_seconds(), (now - exchange).total_seconds())
        return all(-MAX_FUTURE_SKEW_SECONDS <= age <= TICK_MAX_AGE_SECONDS for age in ages)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class SmartWebSocketV2(AngelSmartWebSocketV2):
    """Keep the SDK protocol, but never disable certificate or hostname checks."""

    def connect(self) -> None:
        headers = {
            "Authorization": self.auth_token, "x-api-key": self.api_key,
            "x-client-code": self.client_code, "x-feed-token": self.feed_token,
        }
        self.wsapp = websocket.WebSocketApp(
            self.ROOT_URI, header=headers, on_open=self._on_open,
            on_error=self._on_error, on_close=self._on_close, on_data=self._on_data,
            on_ping=self._on_ping, on_pong=self._on_pong,
        )
        self.wsapp.run_forever(
            sslopt={
                "cert_reqs": ssl.CERT_REQUIRED, "check_hostname": True,
                "ca_certs": certifi.where(),
            },
            ping_interval=self.HEART_BEAT_INTERVAL,
        )


class TickEventBus:
    """Fan out normalized ticks from the preferred healthy broker feed."""

    def __init__(self, preferred_source: str = "angel") -> None:
        self._preferred_source = preferred_source
        self._connected: set[str] = set()
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.RLock()

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def set_connected(self, source: str, connected: bool) -> None:
        with self._lock:
            if connected:
                self._connected.add(source)
            else:
                self._connected.discard(source)

    @property
    def active_source(self) -> str | None:
        with self._lock:
            if self._preferred_source in self._connected:
                return self._preferred_source
            return sorted(self._connected)[0] if self._connected else None

    def publish(self, source: str, tick: dict[str, Any]) -> None:
        with self._lock:
            if source not in self._connected or self.active_source != source:
                return
            callbacks = tuple(self._callbacks)
        event = {**tick, "source": source, "received_at": datetime.now(IST).isoformat()}
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception("Normalized tick subscriber failed")


tick_event_bus = TickEventBus(
    preferred_source="zerodha" if settings.trademax_primary_broker == "zerodha" else "angel"
)


class WebSocketFeed:
    """Manages SmartWebSocketV2 connections for real-time market data."""

    def __init__(self) -> None:
        self._ws: SmartWebSocketV2 | None = None
        self._subscriptions: dict[int, dict[str, set[str]]] = {}
        self._tick_data: dict[str, dict[str, Any]] = {}  # token → latest tick
        self._callbacks: list[Callable] = []
        self._connected = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_tick_at: datetime | None = None
        self._last_exchange_at: datetime | None = None
        self._last_packet_at: datetime | None = None
        self._rejected_ticks = 0
        self._tick_count = 0
        self._connection_attempts = 0
        self._last_error: str | None = None
        self._feed_failed = False
        self._last_failure_report = 0.0

    @staticmethod
    def _market_is_open() -> bool:
        """Outages outside NSE hours are expected and never count against the circuit breaker."""
        from data.nse_calendar import is_nse_holiday

        now = datetime.now(IST)
        return (
            now.weekday() < 5 and not is_nse_holiday(now.date())
            and (9, 0) <= (now.hour, now.minute) < (15, 35)
        )

    def report_failure(self, reason: str) -> None:
        """Tell the Dead Man's Switch about an unexpected outage, at most once a minute."""
        if self._stop_event.is_set() or not self._market_is_open():
            return
        moment = time.monotonic()
        if self._feed_failed and moment - self._last_failure_report < 60:
            return
        self._feed_failed, self._last_failure_report = True, moment
        try:
            from risk.dead_mans_switch import dead_mans_switch

            dead_mans_switch.record_failure("websocket", reason)
        except Exception:
            logger.exception("Could not record the feed failure")

    def _report_recovery(self) -> None:
        if not self._feed_failed:
            return
        self._feed_failed = False
        try:
            from risk.dead_mans_switch import dead_mans_switch

            dead_mans_switch.record_success("websocket")
        except Exception:
            logger.exception("Could not record the feed recovery")

    def connect(self) -> None:
        """Start a single supervised WebSocket connection."""
        angel_client.ensure_session()
        with self._connection_lock:
            if self._thread and self._thread.is_alive():
                logger.warning("WebSocket connection is already running.")
                return
            self._stop_event.clear()
            self._ws = SmartWebSocketV2(
                angel_client.jwt_token,
                settings.angel_api_key,
                settings.angel_client_id,
                angel_client.feed_token,
                max_retry_attempt=0,
            )
            silence_sdk_logging()  # the SDK would write request headers (keys, JWT) to disk
            self._ws.input_request_dict = {}
            self._ws.on_open = self._on_open
            self._ws.on_data = self._on_data
            self._ws.on_error = self._on_error
            self._ws.on_close = self._on_close
            # The installed SDK's close callback accepts fewer arguments than
            # websocket-client supplies, and its error callback hides the cause.
            self._ws._on_close = self._on_close
            self._ws._on_error = self._on_sdk_error
            self._thread = threading.Thread(target=self._run_connection, daemon=True)
            self._thread.start()
        logger.info("WebSocket connection initiated.")

    def _run_connection(self) -> None:
        """Reconnect after the SDK connection loop exits, until explicitly stopped."""
        while not self._stop_event.is_set():
            try:
                self._connection_attempts += 1
                # The SDK otherwise bypasses our on_open callback after a subscription.
                self._ws.RESUBSCRIBE_FLAG = False
                self._ws.input_request_dict = {}
                self._ws.connect()
            except Exception:
                logger.exception("WebSocket connection loop failed")
            finally:
                self._connected = False
                tick_event_bus.set_connected("angel", False)
            if not self._stop_event.is_set():
                self.report_failure("feed disconnected")
                logger.warning("WebSocket disconnected; retrying in 5 seconds.")
                self._stop_event.wait(5)

    def _on_open(self, wsapp: Any) -> None:
        """Called when WebSocket opens."""
        if self._stop_event.is_set():
            wsapp.close()
            return
        self._connected = True
        tick_event_bus.set_connected("angel", True)
        logger.info("WebSocket connected successfully.")

        # Resubscribe to all tokens
        self._do_subscribe()

    def _on_data(self, wsapp: Any, message: dict) -> None:
        """Called on each tick — store and forward to callbacks."""
        try:
            token = str(message.get("token", ""))
            raw_price = message.get("last_traded_price")
            if not token or not isinstance(raw_price, (int, float)) or not math.isfinite(raw_price) or raw_price <= 0:
                raise ValueError("Tick requires a token and a positive finite price")
            now = datetime.now(IST)
            raw_exchange_time = message.get("exchange_timestamp")
            if (
                isinstance(raw_exchange_time, bool)
                or not isinstance(raw_exchange_time, (int, float))
                or not math.isfinite(raw_exchange_time)
            ):
                raise ValueError("Tick requires a valid exchange timestamp")
            exchange_time = datetime.fromtimestamp(raw_exchange_time / 1000, IST)
            times = {"timestamp": now.isoformat(), "exchange_timestamp": exchange_time.isoformat()}
            if not tick_is_fresh(times, now):
                raise ValueError("Tick exchange timestamp is stale or in the future")
            with self._lock:
                tick = {
                    "token": token,
                    "symbol": str(message.get("symbol", "")),
                    "ltp": raw_price / 100.0,
                    "open": message.get("open_price_of_the_day", 0) / 100.0,
                    "high": message.get("high_price_of_the_day", 0) / 100.0,
                    "low": message.get("low_price_of_the_day", 0) / 100.0,
                    "close": message.get("closed_price", 0) / 100.0,
                    "volume": message.get("volume_trade_for_the_day", 0),
                    "oi": message.get("open_interest", 0),
                    "best_bid": self._normalize_depth_prices(message.get("best_5_buy_data", [])),
                    "best_ask": self._normalize_depth_prices(message.get("best_5_sell_data", [])),
                    "upper_circuit": message.get("upper_circuit_limit", 0) / 100.0,
                    "lower_circuit": message.get("lower_circuit_limit", 0) / 100.0,
                    **times,
                }
                self._tick_data[token] = tick
                self._last_tick_at = now
                self._last_exchange_at = exchange_time
                self._last_packet_at = now
                self._tick_count += 1

            try:
                from broker.portfolio_tracker import portfolio_tracker

                portfolio_tracker.mark_token_price(token, float(tick["ltp"]))
            except Exception as exc:
                logger.error("Portfolio mark update failed for token {}: {}", token, type(exc).__name__)

            self._report_recovery()
            tick_event_bus.publish("angel", tick)
            # Fire callbacks
            for cb in self._callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    logger.error("Tick callback error: {}", e)

        except Exception as exc:
            with self._lock:
                self._last_packet_at = datetime.now(IST)
                self._rejected_ticks += 1
                rejected_count = self._rejected_ticks
            if rejected_count & (rejected_count - 1) == 0:
                logger.warning("Tick rejected ({} total): {}", rejected_count, exc)

    @staticmethod
    def _normalize_depth_prices(levels: Any) -> list[dict[str, Any]]:
        """Convert Angel's paise-denominated depth prices to rupees."""
        if not isinstance(levels, list):
            return []
        normalized = []
        for level in levels:
            if not isinstance(level, dict):
                continue
            row = dict(level)
            raw_price = row.get("price")
            if isinstance(raw_price, (int, float)) and math.isfinite(raw_price) and raw_price > 0:
                row["price"] = raw_price / 100.0
            normalized.append(row)
        return normalized

    def _on_error(self, wsapp: Any, error: Any) -> None:
        """Called on WebSocket error."""
        logger.error("WebSocket error: {}", error)
        self._last_error = type(error).__name__
        self._connected = False
        self.report_failure(f"feed error {type(error).__name__}")
        tick_event_bus.set_connected("angel", False)

    def _on_sdk_error(self, wsapp: Any, error: Any) -> None:
        """Close the failed socket so the supervisor can reconnect."""
        self._on_error(wsapp, error)
        try:
            wsapp.close()
        except Exception:
            logger.exception("Could not close failed WebSocket")

    def _on_close(self, wsapp: Any, *args: Any) -> None:
        """Record a close; the supervisor handles any reconnect."""
        logger.warning("WebSocket closed.")
        self._connected = False
        tick_event_bus.set_connected("angel", False)

    def subscribe(self, exchange: str, tokens: list[str], mode: int = MODE_FULL) -> None:
        """Subscribe to tokens for real-time data.

        Args:
            exchange: 'nse_cm', 'nse_fo', 'bse_cm', etc.
            tokens: List of Angel One tokens (strings).
            mode: 1=LTP, 2=Quote, 3=Full (default).
        """
        with self._lock:
            self._subscriptions.setdefault(mode, {}).setdefault(exchange, set()).update(tokens)

        if self._connected:
            self._do_subscribe(mode=mode)

    def _do_subscribe(self, mode: int | None = None) -> None:
        """Send subscribe request to WebSocket."""
        if not self._ws:
            return

        with self._lock:
            subscriptions = {
                subscription_mode: {exchange: sorted(tokens) for exchange, tokens in exchanges.items()}
                for subscription_mode, exchanges in self._subscriptions.items()
                if mode is None or subscription_mode == mode
            }
        for subscription_mode, exchanges in subscriptions.items():
            token_list = [
                {"exchangeType": self._exchange_type_code(exchange), "tokens": tokens}
                for exchange, tokens in exchanges.items() if tokens
            ]
            if not token_list:
                continue
            try:
                self._ws.subscribe("traderws", subscription_mode, token_list)
                logger.info("Subscribed to {} tokens in mode {}.",
                            sum(len(t["tokens"]) for t in token_list), subscription_mode)
            except Exception as exc:
                logger.exception("Subscribe error in mode {}: {}", subscription_mode, exc)

    def _exchange_type_code(self, exchange: str) -> int:
        """Map exchange string to Angel One exchange type code."""
        mapping = {
            "nse_cm": 1,
            "nse_fo": 2,
            "bse_cm": 3,
            "bse_fo": 4,
            "mcx_fo": 5,
            "ncx_fo": 7,
            "cde_fo": 13,
        }
        return mapping.get(exchange.lower(), 1)

    def get_tick(self, token: str) -> dict[str, Any] | None:
        """Return only a connected token's recently received tick."""
        with self._lock:
            tick = self._tick_data.get(token)
            if not self._connected or tick is None:
                return None
            return dict(tick) if tick_is_fresh(tick) else None

    def get_all_ticks(self) -> dict[str, dict[str, Any]]:
        """Get all latest tick data."""
        with self._lock:
            return dict(self._tick_data)

    def register_callback(self, callback: Callable) -> None:
        """Register a function to call on every tick."""
        self._callbacks.append(callback)

    def unsubscribe(self, exchange: str, tokens: list[str]) -> None:
        """Unsubscribe from tokens."""
        with self._lock:
            modes = []
            for mode, exchanges in self._subscriptions.items():
                subscribed = exchanges.get(exchange, set())
                removed = subscribed.intersection(tokens)
                if removed:
                    subscribed.difference_update(removed)
                    modes.append((mode, sorted(removed)))

        if self._ws and self._connected:
            for mode, removed in modes:
                try:
                    token_list = [{"exchangeType": self._exchange_type_code(exchange), "tokens": removed}]
                    self._ws.unsubscribe("traderws", mode, token_list)
                except Exception as exc:
                    logger.exception("Unsubscribe error in mode {}: {}", mode, exc)

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        with self._connection_lock:
            self._stop_event.set()
            if self._ws:
                try:
                    self._ws.close_connection()
                except Exception:
                    logger.exception("WebSocket close failed")
        self._connected = False
        tick_event_bus.set_connected("angel", False)
        logger.info("WebSocket disconnected.")

    def request_reconnect(self) -> None:
        """Close the current transport; the existing supervisor restores it."""
        with self._connection_lock:
            if not self._connected or not self._ws or not self._ws.wsapp:
                raise ValueError("A connected feed is required for a recovery test")
            logger.warning("Operator requested a supervised WebSocket reconnect.")
            self._ws.wsapp.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        """True while the supervised connection loop is alive (it retries drops itself)."""
        return bool(self._thread and self._thread.is_alive())

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            age = ((datetime.now(IST) - self._last_tick_at).total_seconds()
                   if self._last_tick_at else None)
            exchange_age = ((datetime.now(IST) - self._last_exchange_at).total_seconds()
                            if self._last_exchange_at else None)
            return {
                "connected": self._connected,
                "tls_verified": True,
                "last_tick_at": self._last_tick_at.isoformat() if self._last_tick_at else None,
                "tick_age_seconds": round(age, 1) if age is not None else None,
                "last_exchange_at": self._last_exchange_at.isoformat() if self._last_exchange_at else None,
                "exchange_tick_age_seconds": round(exchange_age, 1) if exchange_age is not None else None,
                "last_packet_at": self._last_packet_at.isoformat() if self._last_packet_at else None,
                "rejected_ticks": self._rejected_ticks,
                "fresh": (
                    self._connected and age is not None and exchange_age is not None
                    and -MAX_FUTURE_SKEW_SECONDS <= age <= TICK_MAX_AGE_SECONDS
                    and -MAX_FUTURE_SKEW_SECONDS <= exchange_age <= TICK_MAX_AGE_SECONDS
                ),
                "tick_count": self._tick_count,
                "tokens_seen": len(self._tick_data),
                "subscribed_tokens": len({
                    token for exchanges in self._subscriptions.values()
                    for tokens in exchanges.values() for token in tokens
                }),
                "connection_attempts": self._connection_attempts,
                "last_error_type": self._last_error,
            }


# Module-level singleton
ws_feed = WebSocketFeed()
