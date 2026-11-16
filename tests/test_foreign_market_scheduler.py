"""Scheduled foreign-market strategies remain paper/simulation-only and fail closed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from broker import crypto_sim
from broker.alpaca_api import AlpacaClient
from config.settings import settings
from scheduler import foreign_markets
from scheduler.foreign_markets import (
    FAST_WINDOW, SLOW_WINDOW, ForeignMarketScheduler, moving_average_signal,
)

UTC = timezone.utc


def test_moving_average_strategy_requires_complete_rising_history():
    rising = [float(index) for index in range(1, SLOW_WINDOW + 1)]
    signal, fast = moving_average_signal(rising, 60.0)
    assert signal is True and fast is not None
    assert moving_average_signal(rising[: SLOW_WINDOW - 1], 60.0) == (False, None)
    assert moving_average_signal(rising, 1.0)[0] is False
    assert moving_average_signal([0.0] * SLOW_WINDOW, 1.0) == (False, None)


def test_moving_average_signal_does_not_buy_downtrend():
    falling = [float(index) for index in range(SLOW_WINDOW, 0, -1)]
    assert moving_average_signal(falling, 1.0)[0] is False
    assert FAST_WINDOW < SLOW_WINDOW


def test_us_quote_gate_rejects_stale_or_wide_quotes():
    now = datetime.now(UTC)
    fresh = {"bid": 99.9, "ask": 100.1, "timestamp": now.isoformat()}
    assert ForeignMarketScheduler._quote_is_usable(fresh, now) == (True, "fresh quote")
    stale = {**fresh, "timestamp": (now - timedelta(minutes=5)).isoformat()}
    assert not ForeignMarketScheduler._quote_is_usable(stale, now)[0]
    wide = {**fresh, "ask": 101.0}
    assert not ForeignMarketScheduler._quote_is_usable(wide, now)[0]


def test_us_gross_exposure_includes_unfilled_entry_orders(monkeypatch):
    monkeypatch.setattr(settings, "foreign_usd_inr_rate", 90.0)
    scheduler = ForeignMarketScheduler()
    gross = scheduler._us_gross_inr(
        [{"market_value": 100.0}],
        [
            {"side": "buy", "quantity": 10, "filled_quantity": 2, "limit_price": 10},
            {"side": "sell", "quantity": 5, "filled_quantity": 0, "limit_price": 20},
        ],
    )
    assert gross == 16_200.0


def test_alpaca_data_and_pending_order_helpers_keep_risk_inputs():
    timestamp = datetime.now(UTC)

    class DataClient:
        def get_stock_bars(self, request):
            assert request.symbol_or_symbols == "SPY"
            assert request.timeframe.amount == 15
            return SimpleNamespace(data={"SPY": [SimpleNamespace(timestamp=timestamp, close=100.0)]})

    class TradingClient:
        def get_orders(self, filter):
            assert filter.limit == 500
            return [SimpleNamespace(
                id="order-1", symbol="SPY", side="OrderSide.BUY",
                qty="10", filled_qty="2", limit_price="100",
            )]

    client = AlpacaClient()
    client._data_client = DataClient()
    client._trading_client = TradingClient()
    assert client.get_bars("SPY", start=timestamp - timedelta(days=3), end=timestamp) == [
        {"timestamp": timestamp, "close": 100.0},
    ]
    assert client.get_open_orders() == [{
        "id": "order-1", "symbol": "SPY", "side": "buy", "quantity": 10.0,
        "filled_quantity": 2.0, "limit_price": 100.0,
    }]


def test_crypto_candle_fetch_requires_completed_valid_history():
    rows = [[0, 0, 0, 0, str(index + 1), 0] for index in range(SLOW_WINDOW + 1)]

    class Response:
        status_code = 200

        def json(self):
            return rows

    closes = crypto_sim.fetch_public_closes(
        "BTCUSDT", limit=SLOW_WINDOW, getter=lambda *_a, **_kw: Response(),
    )
    assert closes == [Decimal(index) for index in range(1, SLOW_WINDOW + 1)]


def test_crypto_candle_fetch_fails_closed_on_short_or_bad_data():
    class ShortResponse:
        status_code = 200

        @staticmethod
        def json():
            return []

    class BadResponse:
        status_code = 200

        @staticmethod
        def json():
            return [[0, 0, 0, 0, "not-a-price", 0] for _ in range(SLOW_WINDOW + 1)]

    assert crypto_sim.fetch_public_closes("BTCUSDT", getter=lambda *_a, **_kw: ShortResponse()) is None
    assert crypto_sim.fetch_public_closes("BTCUSDT", getter=lambda *_a, **_kw: BadResponse()) is None


def test_crypto_sim_persists_average_entry_and_loads_legacy_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    price_source = crypto_sim.PriceSource(use_public=False)
    sim = crypto_sim.CryptoSim(500, prices=price_source)
    sim.place_order("BTCUSDT", "BUY", "0.001", "MARKET", portfolio_equity_inr=100_000)
    saved = tmp_path / "crypto.json"
    sim.save(saved)

    restored = crypto_sim.CryptoSim(500, prices=price_source)
    restored.load(saved)
    assert restored.quantity("BTCUSDT") == Decimal("0.001")
    assert Decimal(restored.state.average_entry_prices["BTCUSDT"]) > 0

    saved.write_text(
        '{"cash_usdt":"500","positions":{},"trades":[],"day":"","day_start_equity_usdt":"0",'
        '"halted":false,"halt_reason":""}'
    )
    restored.load(saved)
    assert restored.state.average_entry_prices == {}


def test_foreign_scheduler_jobs_are_created_only_for_enabled_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    scheduler = ForeignMarketScheduler()
    scheduler.CRYPTO_STATE_PATH = tmp_path / "crypto.json"
    scheduler.setup()
    jobs = {job.id for job in scheduler.scheduler.get_jobs()}
    assert {"foreign_us_scan", "foreign_crypto_scan"} <= jobs
    assert scheduler.status()["us"]["paper_only"] is True
    assert scheduler.status()["crypto"]["simulated_only"] is True
    assert "5% target" in scheduler.status()["strategy"]


def test_crypto_cycle_does_not_enter_from_fixture_prices(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    monkeypatch.setattr(settings, "crypto_sim_watchlist", ["BTCUSDT"])
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    monkeypatch.setattr(foreign_markets, "fetch_public_closes", lambda *_a, **_kw: None)
    monkeypatch.setattr(foreign_markets, "fetch_public_price", lambda *_a, **_kw: None)
    scheduler = ForeignMarketScheduler()
    scheduler.CRYPTO_STATE_PATH = tmp_path / "crypto.json"
    scheduler._entry_gate = lambda: (True, "test")
    scheduler._total_equity_inr = lambda: 100_000.0
    scheduler._india_gross_inr = lambda: 0.0

    async def run():
        await scheduler._run_crypto_cycle()

    import asyncio

    asyncio.run(run())
    sim = scheduler._get_crypto_sim()
    assert sim.state.trades == []
    assert scheduler.status()["crypto"]["simulated_only"] is True
    assert scheduler._last_crypto["status"] == "completed"


def test_crypto_existing_position_exits_while_new_entries_are_paused(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    monkeypatch.setattr(settings, "crypto_sim_watchlist", ["BTCUSDT"])
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    monkeypatch.setattr(foreign_markets, "fetch_public_closes", lambda *_a, **_kw: [
        Decimal(value) for value in range(SLOW_WINDOW, 0, -1)
    ])
    monkeypatch.setattr(foreign_markets, "fetch_public_price", lambda *_a, **_kw: Decimal("65000"))
    scheduler = ForeignMarketScheduler()
    scheduler.CRYPTO_STATE_PATH = tmp_path / "crypto.json"
    scheduler._entry_gate = lambda: (False, "entries paused")
    scheduler._total_equity_inr = lambda: 100_000.0
    scheduler._india_gross_inr = lambda: 0.0
    sim = crypto_sim.CryptoSim(500, prices=crypto_sim.PriceSource(use_public=False))
    sim.place_order("BTCUSDT", "BUY", "0.001", "MARKET", portfolio_equity_inr=100_000)
    scheduler._crypto_sim = sim

    asyncio.run(scheduler._run_crypto_cycle())

    assert sim.quantity("BTCUSDT") == 0
    assert sim.state.trades[-1].side == "SELL"
    assert scheduler._last_crypto["reason"] == "entries paused"


def test_crypto_cycle_skips_market_data_when_paused_without_positions(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    monkeypatch.setattr(settings, "crypto_sim_watchlist", ["BTCUSDT"])
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)

    def unexpected_data_request(*_args, **_kwargs):
        raise AssertionError("unexpected public market-data request")

    monkeypatch.setattr(foreign_markets, "fetch_public_closes", unexpected_data_request)
    scheduler = ForeignMarketScheduler()
    scheduler.CRYPTO_STATE_PATH = tmp_path / "crypto.json"
    scheduler._entry_gate = lambda: (False, "entries paused")

    asyncio.run(scheduler._run_crypto_cycle())

    assert scheduler._last_crypto["reason"] == "entries paused"
    assert scheduler._last_crypto["symbols"]["BTCUSDT"]["status"] == "entries_paused"


def test_slow_crypto_feed_does_not_block_event_loop(monkeypatch, tmp_path):
    import asyncio
    import time

    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    monkeypatch.setattr(settings, "crypto_sim_watchlist", ["BTCUSDT"])
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)

    def slow_unavailable(*_args, **_kwargs):
        time.sleep(0.1)
        return None

    monkeypatch.setattr(foreign_markets, "fetch_public_closes", slow_unavailable)
    monkeypatch.setattr(foreign_markets, "fetch_public_price", slow_unavailable)
    scheduler = ForeignMarketScheduler()
    scheduler.CRYPTO_STATE_PATH = tmp_path / "crypto.json"
    scheduler._entry_gate = lambda: (True, "test")
    scheduler._total_equity_inr = lambda: 100_000.0
    scheduler._india_gross_inr = lambda: 0.0

    async def run():
        cycle = asyncio.create_task(scheduler._run_crypto_cycle())
        await asyncio.sleep(0.01)
        assert not cycle.done()
        await cycle

    asyncio.run(run())
    assert scheduler._last_crypto["status"] == "completed"
