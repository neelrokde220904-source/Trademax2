"""Offline tests for the Angel feed connection supervisor."""

import threading
import ssl
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from broker import websocket_feed


def test_sdk_connection_requires_certificate_and_hostname_checks(monkeypatch):
    options = []
    class Socket:
        def __init__(self, *_args, **_kwargs):
            pass
        def run_forever(self, **kwargs):
            options.append(kwargs)
    monkeypatch.setattr(websocket_feed.websocket, "WebSocketApp", Socket)
    client = websocket_feed.SmartWebSocketV2("jwt", "api", "client", "feed", max_retry_attempt=0)
    client.connect()
    assert options[0]["sslopt"]["cert_reqs"] == ssl.CERT_REQUIRED
    assert options[0]["sslopt"]["check_hostname"] is True
    assert options[0]["sslopt"]["ca_certs"] == websocket_feed.certifi.where()


def test_connected_socket_is_not_fresh_without_recent_ticks():
    feed = websocket_feed.WebSocketFeed()
    feed._connected = True
    assert not feed.get_status()["fresh"]
    feed._on_data(None, {"token": "1", "last_traded_price": 12345,
                         "exchange_timestamp": int(datetime.now(websocket_feed.IST).timestamp() * 1000)})
    assert feed.get_status()["fresh"]
    assert feed.get_status()["tick_count"] == 1
    assert feed.get_tick("1")["ltp"] == 123.45
    feed._last_tick_at = datetime.now(websocket_feed.IST) - timedelta(seconds=31)
    feed._tick_data["1"]["timestamp"] = feed._last_tick_at.isoformat()
    assert not feed.get_status()["fresh"]
    assert feed.get_tick("1") is None
    feed._connected = False
    assert not feed.get_status()["fresh"]
    assert feed.get_tick("1") is None


def test_top_of_book_prices_are_normalized_from_paise_to_rupees():
    feed = websocket_feed.WebSocketFeed()
    feed._connected = True
    feed._on_data(None, {
        "token": "1",
        "last_traded_price": 10000,
        "exchange_timestamp": int(datetime.now(websocket_feed.IST).timestamp() * 1000),
        "best_5_buy_data": [{"price": 9995, "quantity": 10}],
        "best_5_sell_data": [{"price": 10005, "quantity": 12}],
    })

    tick = feed.get_tick("1")
    assert tick["best_bid"][0]["price"] == 99.95
    assert tick["best_ask"][0]["price"] == 100.05


def test_invalid_ticks_never_establish_freshness():
    feed = websocket_feed.WebSocketFeed()
    feed._connected = True
    for packet in ({}, {"token": "1", "last_traded_price": 0},
                   {"token": "1", "last_traded_price": float("nan")}):
        feed._on_data(None, packet)
    assert not feed.get_status()["fresh"]
    assert feed.get_status()["tick_count"] == 0


@pytest.mark.parametrize("offset,missing", [
    (-120, False), (10, False), (0, True),
])
def test_stale_future_or_unknown_exchange_timestamps_cannot_mark_positions(offset, missing, monkeypatch):
    from broker.portfolio_tracker import portfolio_tracker

    marks, published = [], []
    monkeypatch.setattr(portfolio_tracker, "mark_token_price", lambda *args: marks.append(args))
    monkeypatch.setattr(websocket_feed.tick_event_bus, "publish", lambda *args: published.append(args))
    feed = websocket_feed.WebSocketFeed()
    feed._connected = True
    packet = {"token": "1", "last_traded_price": 10000}
    if not missing:
        packet["exchange_timestamp"] = int((datetime.now(websocket_feed.IST) + timedelta(seconds=offset)).timestamp() * 1000)
    feed._on_data(None, packet)
    assert feed.get_tick("1") is None
    assert not feed.get_status()["fresh"]
    assert marks == [] and published == []


def test_fresh_tick_marks_position_and_is_published(monkeypatch):
    from broker.portfolio_tracker import portfolio_tracker

    marks, published, callbacks = [], [], []
    monkeypatch.setattr(portfolio_tracker, "mark_token_price", lambda *args: marks.append(args))
    monkeypatch.setattr(websocket_feed.tick_event_bus, "publish", lambda *args: published.append(args))
    feed = websocket_feed.WebSocketFeed()
    feed._connected = True
    feed.register_callback(callbacks.append)
    feed._on_data(None, {
        "token": "1", "last_traded_price": 12345,
        "exchange_timestamp": int(datetime.now(websocket_feed.IST).timestamp() * 1000),
    })
    assert marks == [("1", 123.45)]
    assert published[0][0] == "angel"
    assert published[0][1]["ltp"] == 123.45
    assert callbacks == [published[0][1]]
    assert feed.get_status()["rejected_ticks"] == 0


def test_rejected_tick_logging_is_bounded(monkeypatch):
    warnings = []
    monkeypatch.setattr(websocket_feed.logger, "warning", lambda *args: warnings.append(args))
    feed = websocket_feed.WebSocketFeed()
    for _ in range(1000):
        feed._on_data(None, {})
    assert feed.get_status()["rejected_ticks"] == 1000
    assert len(warnings) == 10


def test_reference_does_not_invent_rest_quote_freshness(monkeypatch):
    from broker import order_manager

    monkeypatch.setattr(websocket_feed.ws_feed, "get_tick", lambda _token: None)
    monkeypatch.setattr(order_manager.angel_client, "get_ltp", lambda *_args: pytest.fail("Untimed REST quote used"))
    assert order_manager._market_reference("TEST", "1", "NSE") is None


def test_recovery_test_closes_transport_without_stopping_supervisor():
    feed = websocket_feed.WebSocketFeed()
    with pytest.raises(ValueError):
        feed.request_reconnect()
    closed = []
    feed._connected = True
    feed._ws = SimpleNamespace(wsapp=SimpleNamespace(close=lambda: closed.append(True)))
    feed.request_reconnect()
    assert closed == [True]
    assert not feed._stop_event.is_set()

def test_reconnect_resubscribes_and_disconnect_stops(monkeypatch):
    ready = threading.Event()
    closed = threading.Event()
    instances = []

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_retry_attempt"] == 0
            self.RESUBSCRIBE_FLAG = False
            self.connect_calls = 0
            self.subscriptions = []
            self.unsubscriptions = []
            instances.append(self)

        def connect(self):
            self.connect_calls += 1
            # Match the SDK: its own resubscribe branch skips the user on_open.
            if not self.RESUBSCRIBE_FLAG:
                self.on_open(self)
            if self.connect_calls == 1:
                self._on_close(self, 1006, "closed")
                return
            ready.set()
            assert closed.wait(5)
            self._on_close(self, 1000, "closed")

        def subscribe(self, correlation_id, mode, tokens):
            self.subscriptions.append((correlation_id, mode, tokens))
            self.RESUBSCRIBE_FLAG = True

        def unsubscribe(self, correlation_id, mode, tokens):
            self.unsubscriptions.append((correlation_id, mode, tokens))

        def close_connection(self):
            closed.set()

        def close(self):
            closed.set()

    monkeypatch.setattr(websocket_feed, "SmartWebSocketV2", FakeSocket)
    monkeypatch.setattr(websocket_feed.angel_client, "ensure_session", lambda: None)
    feed = websocket_feed.WebSocketFeed()
    feed.subscribe("nse_cm", ["1"], mode=websocket_feed.MODE_LTP)
    feed.subscribe("nse_cm", ["2"], mode=websocket_feed.MODE_FULL)
    original_wait = feed._stop_event.wait
    monkeypatch.setattr(feed._stop_event, "wait", lambda timeout: original_wait(0))
    try:
        feed.connect()
        assert ready.wait(5)
        feed.connect()
        socket = instances[0]
        assert len(instances) == 1
        assert socket.connect_calls == 2
        assert feed.is_connected
        assert [(entry[0], entry[1]) for entry in socket.subscriptions] == [
            ("traderws", 1), ("traderws", 3), ("traderws", 1), ("traderws", 3)
        ]
        feed.unsubscribe("nse_cm", ["1", "2"])
        assert [entry[1] for entry in socket.unsubscriptions] == [1, 3]
        feed.disconnect()
        feed._thread.join(timeout=5)
        assert not feed._thread.is_alive()
        assert socket.connect_calls == 2
        assert not feed.is_connected
    finally:
        feed.disconnect()
        if feed._thread:
            feed._thread.join(timeout=5)


def test_open_after_disconnect_closes_socket():
    feed = websocket_feed.WebSocketFeed()
    feed._stop_event.set()
    closed = threading.Event()

    class Socket:
        def close(self):
            closed.set()

    feed._on_open(Socket())
    assert closed.is_set()
    assert not feed.is_connected


def test_sdk_error_closes_and_marks_disconnected():
    feed = websocket_feed.WebSocketFeed()
    closed = threading.Event()

    class Socket:
        def close(self):
            closed.set()

    feed._on_open(Socket())
    feed._on_sdk_error(Socket(), RuntimeError("feed disconnected"))
    assert closed.is_set()
    assert not feed.is_connected
