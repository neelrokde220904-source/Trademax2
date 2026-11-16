"""T11/T12: US paper adapter, US calendar and crypto simulator. No network and no real keys are used."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from broker import alpaca_paper_adapter as alp
from broker import crypto_sim as cs
from broker.alpaca_api import AlpacaOrder
from broker.base_broker import BrokerRejectedError
from broker.intelligent_order_router import (
    AlgoType, Broker, IntelligentOrderRouter, RoutingDecision,
)
from config.settings import settings
from data import us_calendar as usc

NY = ZoneInfo("America/New_York")
OPEN_MOMENT = datetime(2026, 11, 16, 11, 0, tzinfo=NY)  # a Monday, regular session


class FakeAlpaca:
    def __init__(self, *, paper=True, api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets"):
        self.paper, self.api_key, self.secret_key = paper, api_key, secret_key
        self.trading_client = SimpleNamespace(_base_url=base_url)
        self.orders: list[tuple] = []

    def place_limit_order(self, symbol, side, qty, price, fractional=False):
        self.orders.append((symbol, side, qty, price, fractional))
        return AlpacaOrder(order_id="o-1", symbol=symbol, side=side, qty=qty, status="NEW")

    def get_account(self):
        return {"status": "ACTIVE", "cash": 1000.0, "buying_power": 1000.0, "equity": 1000.0}

    def get_positions(self):
        return []

    def cancel_order(self, order_id):
        return True


@pytest.fixture
def us_enabled(monkeypatch):
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", True)


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(alp, "is_open", lambda _moment: True)


# ── US calendar ──────────────────────────────────────────────────────────────

def test_us_session_follows_daylight_saving():
    # 14:30 UTC is 9:30 EDT (open) in summer but 9:30 EST needs 14:30 UTC in winter; 13:30 UTC is open only in summer.
    summer = datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc)
    winter = datetime(2026, 12, 7, 13, 30, tzinfo=timezone.utc)
    assert usc.is_open(summer) is True
    assert usc.is_open(winter) is False
    assert usc.is_open(datetime(2026, 12, 7, 14, 30, tzinfo=timezone.utc)) is True


def test_us_dst_transition_dates_keep_new_york_open():
    for day in (date(2026, 3, 9), date(2026, 11, 2)):  # the Mondays after spring-forward / fall-back
        open_, close = usc.session_bounds(day)
        assert open_.hour == 9 and open_.minute == 30 and close.hour == 16


def test_us_holidays_and_weekends_are_closed():
    assert usc.session_bounds(date(2026, 11, 26)) is None  # Thanksgiving
    assert usc.session_bounds(date(2026, 12, 25)) is None
    assert usc.session_bounds(date(2026, 11, 14)) is None  # Saturday
    assert usc.is_open(datetime(2026, 11, 26, 11, 0, tzinfo=NY)) is False


def test_us_early_closes():
    for day in (date(2026, 11, 27), date(2026, 12, 24)):
        _open, close = usc.session_bounds(day)
        assert close.hour == 13
    assert usc.is_open(datetime(2026, 11, 27, 12, 59, tzinfo=NY)) is True
    assert usc.is_open(datetime(2026, 11, 27, 13, 0, tzinfo=NY)) is False


def test_us_unknown_year_is_closed_not_open():
    assert usc.calendar_year_known(2026) is True
    assert usc.calendar_year_known(2027) is False
    assert usc.session_bounds(date(2027, 3, 1)) is None
    assert usc.is_open(datetime(2027, 3, 1, 11, 0, tzinfo=NY)) is False


def test_us_calendar_requires_aware_datetime():
    with pytest.raises(ValueError):
        usc.is_open(datetime(2026, 11, 16, 11, 0))


def test_us_next_open_skips_holiday():
    nxt = usc.next_open(datetime(2026, 11, 25, 17, 0, tzinfo=NY))
    assert nxt.date() == date(2026, 11, 27)


# ── Alpaca paper adapter ─────────────────────────────────────────────────────

def test_adapter_refuses_when_flag_off(monkeypatch):
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    with pytest.raises(alp.AlpacaPaperRefused):
        alp.AlpacaPaperAdapter(FakeAlpaca())


def test_adapter_refuses_non_paper_settings_or_client(us_enabled, monkeypatch):
    with pytest.raises(alp.AlpacaPaperRefused):
        alp.AlpacaPaperAdapter(FakeAlpaca(paper=False))
    monkeypatch.setattr(settings, "alpaca_paper", False)
    with pytest.raises(alp.AlpacaPaperRefused):
        alp.AlpacaPaperAdapter(FakeAlpaca())


def test_adapter_refuses_live_url(us_enabled):
    with pytest.raises(alp.AlpacaPaperRefused):
        alp.AlpacaPaperAdapter(FakeAlpaca(base_url="https://api.alpaca.markets"))


def test_adapter_reports_missing_keys_without_leaking(us_enabled):
    with pytest.raises(alp.AlpacaPaperRefused, match="keys_missing"):
        alp.AlpacaPaperAdapter(FakeAlpaca(api_key="", secret_key=""))


def test_readiness_states(monkeypatch):
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    assert alp.readiness(FakeAlpaca())["status"] == "disabled"
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", True)
    assert alp.readiness(FakeAlpaca(api_key=""))["status"] == "keys_missing"
    assert alp.readiness(FakeAlpaca(paper=False))["status"] == "refused"
    ready = alp.readiness(FakeAlpaca())
    assert ready["status"] == "ready_for_paper" and "IEX" in ready["data"]


def test_adapter_places_limit_orders_only(us_enabled, market_open):
    client = FakeAlpaca()
    adapter = alp.AlpacaPaperAdapter(client)
    result = adapter.place_order({"tradingsymbol": "AAPL", "transactiontype": "BUY", "quantity": 2, "price": 190.5})
    assert result["paper"] is True and result["currency"] == "USD"
    assert client.orders == [("AAPL", "buy", 2.0, 190.5, False)]
    with pytest.raises(BrokerRejectedError):
        adapter.place_order({"tradingsymbol": "AAPL", "transactiontype": "BUY", "quantity": 1, "ordertype": "MARKET", "price": 190})
    with pytest.raises(BrokerRejectedError):
        adapter.place_order({"tradingsymbol": "AAPL", "transactiontype": "BUY", "quantity": 1, "price": 0})
    with pytest.raises(BrokerRejectedError):
        adapter.place_order({"tradingsymbol": "AAPL", "transactiontype": "HOLD", "quantity": 1, "price": 10})
    assert len(client.orders) == 1


def test_adapter_rejects_orders_when_market_closed(us_enabled, monkeypatch):
    monkeypatch.setattr(alp, "is_open", lambda _moment: False)
    client = FakeAlpaca()
    with pytest.raises(BrokerRejectedError, match="closed"):
        alp.AlpacaPaperAdapter(client).place_order(
            {"tradingsymbol": "AAPL", "transactiontype": "BUY", "quantity": 1, "price": 100},
        )
    assert client.orders == []


def test_adapter_has_no_india_square_off_or_brackets(us_enabled):
    adapter = alp.AlpacaPaperAdapter(FakeAlpaca())
    assert adapter.flatten_all_intraday() == []
    with pytest.raises(BrokerRejectedError):
        adapter.place_bracket_or_gtt_order({})
    with pytest.raises(NotImplementedError):
        adapter.subscribe_ticks("NASDAQ", ["AAPL"], lambda _t: None)
    funds = adapter.get_funds_and_margins()
    assert funds["currency"] == "USD" and "IEX" in funds["data_label"]


def test_router_keeps_us_orders_unsupported_when_flag_off(monkeypatch):
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    router = IntelligentOrderRouter()
    router._alpaca_client = FakeAlpaca()
    decision = RoutingDecision("AAPL", Broker.ALPACA, AlgoType.MARKET, "test")
    report = router._execute_single(decision, "BUY", 1, 100.0)
    assert report.status == "PAPER_UNSUPPORTED"
    assert router._alpaca_client.orders == []


def test_router_sends_us_orders_through_paper_adapter_when_enabled(us_enabled, market_open, monkeypatch):
    monkeypatch.setattr(settings, "paper_trading", True)
    router = IntelligentOrderRouter()
    router._alpaca_client = FakeAlpaca()
    report = router._execute_single(RoutingDecision("AAPL", Broker.ALPACA, AlgoType.MARKET, "test"), "BUY", 1, 100.0)
    assert report.broker == "alpaca_paper" and report.order_id == "o-1"


# ── crypto simulator ─────────────────────────────────────────────────────────

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def crypto_on(monkeypatch):
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)


def make_sim(start=1000, **kwargs):
    return cs.CryptoSim(start, prices=cs.PriceSource(use_public=False), **kwargs)


def test_crypto_sim_is_disabled_by_default():
    assert settings.crypto_sim_enabled is False
    with pytest.raises(cs.CryptoSimDisabled):
        cs.CryptoSim(1000)


def test_crypto_market_buy_applies_slippage_fee_and_filters(crypto_on):
    sim = make_sim()
    fill = sim.place_order("ETHUSDT", "BUY", "0.0309999", now=NOW,
                           portfolio_equity_inr=1_000_000)
    assert fill["status"] == "FILLED" and fill["source"] == "fixture"
    assert fill["quantity"] == Decimal("0.0309")  # rounded down to the step size
    assert fill["price"] == Decimal("3201.60")  # 3200 + 5 bps, on the tick
    assert fill["fee_usdt"] == fill["quantity"] * fill["price"] * Decimal("0.001")
    assert sim.cash == Decimal("1000") - fill["quantity"] * fill["price"] - fill["fee_usdt"]
    assert "IST" in fill["time_ist"] and "static" in fill["fx"]
    assert fill["time_ist"].startswith("2026-11-14 14:30:00")  # 09:00 UTC = 14:30 IST


def test_crypto_rejects_bad_orders(crypto_on):
    sim = make_sim()
    with pytest.raises(cs.OrderRejected, match="Leverage"):
        sim.place_order("BTCUSDT", "BUY", 0.001, leverage=3.0, now=NOW)
    with pytest.raises(cs.OrderRejected, match="not tradable"):
        sim.place_order("DOGEUSDT", "BUY", 1, now=NOW)
    with pytest.raises(cs.OrderRejected, match="step size"):
        sim.place_order("BTCUSDT", "BUY", 0.000001, now=NOW)
    with pytest.raises(cs.OrderRejected, match="minimum"):
        sim.place_order("SOLUSDT", "BUY", 0.01, now=NOW, portfolio_equity_inr=1_000_000)
    with pytest.raises(cs.OrderRejected, match="no shorting"):
        sim.place_order("BTCUSDT", "SELL", 0.001, now=NOW)
    with pytest.raises(cs.OrderRejected, match="tick"):
        sim.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", "65000.005", now=NOW)


def test_crypto_limit_order_not_marketable_does_not_fill(crypto_on):
    sim = make_sim()
    result = sim.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", "60000.00", now=NOW, portfolio_equity_inr=1_000_000)
    assert result["status"] == "NOT_MARKETABLE"
    assert sim.cash == Decimal("1000") and not sim.state.trades


def test_crypto_insufficient_cash_and_caps(crypto_on):
    sim = make_sim(100)
    with pytest.raises(cs.OrderRejected, match="insufficient"):
        sim.place_order("BTCUSDT", "BUY", 0.01, now=NOW, portfolio_equity_inr=10_000_000)
    # Per-position cap: 10% of a ₹100,000 portfolio is ₹10,000, about 113 USDT at 88.
    sim = make_sim(1000)
    with pytest.raises(cs.OrderRejected, match="per-position cap"):
        sim.place_order("SOLUSDT", "BUY", 1.0, now=NOW, portfolio_equity_inr=10_000)
    # Gross cap: existing India exposure plus this entry may not pass 50% of equity.
    with pytest.raises(cs.OrderRejected, match="gross exposure"):
        sim.place_order("SOLUSDT", "BUY", 1.0, now=NOW, portfolio_equity_inr=200_000, portfolio_gross_inr=95_000)
    assert not sim.state.trades


def test_crypto_halt_is_independent_of_india_and_blocks_only_entries(crypto_on, monkeypatch):
    sim = make_sim()
    sim.place_order("ETHUSDT", "BUY", 0.1, now=NOW, portfolio_equity_inr=1_000_000)
    crash = {"ETHUSDT": Decimal("1500")}
    monkeypatch.setattr(sim.prices, "price", lambda symbol: (crash.get(symbol, cs.FIXTURE_PRICES[symbol]), "fixture"))
    sim.place_order("ETHUSDT", "SELL", 0.05, now=NOW)  # exits are always allowed and update the halt
    assert sim.state.halted is True and "daily loss" in sim.state.halt_reason
    with pytest.raises(cs.OrderRejected, match="halted"):
        sim.place_order("ETHUSDT", "BUY", 0.01, now=NOW, portfolio_equity_inr=1_000_000)
    sim.place_order("ETHUSDT", "SELL", 0.05, now=NOW)
    assert sim.quantity("ETHUSDT") == 0
    # The India halt state is untouched by a crypto halt.
    from broker.portfolio_tracker import portfolio_tracker
    assert getattr(portfolio_tracker, "daily_loss_halted", False) is False


def test_crypto_halt_clears_on_next_utc_day(crypto_on):
    sim = make_sim()
    sim.state.halted, sim.state.halt_reason = True, "x"
    sim.place_order("BTCUSDT", "BUY", 0.0005, now=datetime(2026, 11, 15, 0, 5, tzinfo=timezone.utc),
                    portfolio_equity_inr=1_000_000)
    assert sim.state.halted is False


def test_crypto_state_round_trips(crypto_on, tmp_path):
    sim = make_sim()
    sim.place_order("BTCUSDT", "BUY", 0.0005, now=NOW, portfolio_equity_inr=1_000_000)
    path = tmp_path / "crypto.json"
    sim.save(path)
    other = make_sim(5)
    other.load(path)
    assert other.state.trades == sim.state.trades and other.cash == sim.cash


def test_public_price_falls_back_to_fixture_and_labels_source():
    def failing(*_a, **_k):
        raise OSError("offline")

    assert cs.PriceSource(getter=failing).price("BTCUSDT") == (cs.FIXTURE_PRICES["BTCUSDT"], "fixture")

    class Ok:
        status_code = 200

        @staticmethod
        def json():
            return {"price": "70000.10"}

    assert cs.PriceSource(getter=lambda *_a, **_k: Ok()).price("BTCUSDT") == (Decimal("70000.10"), "public_rest")
    with pytest.raises(cs.OrderRejected):
        cs.PriceSource(use_public=False).price("UNKNOWNUSDT")


def test_crypto_market_is_24x7():
    assert cs.session_is_open(datetime(2026, 12, 25, 3, 0, tzinfo=timezone.utc)) is True


def trade(side, qty, price, symbol="BTCUSDT"):
    return cs.CryptoTrade("t", symbol, side, str(qty), str(price), "0", "TAKER")


def test_fifo_gains_split_across_lots():
    gains = cs.realised_gains([
        trade("BUY", 1, 100), trade("BUY", 1, 200), trade("SELL", 1.5, 300),
    ])
    assert len(gains) == 1
    assert gains[0]["cost"] == Decimal("200.0") and gains[0]["gain"] == Decimal("250.0")


def test_vda_tax_has_no_loss_set_off_and_shows_tds():
    est = cs.vda_tax_estimate(
        [trade("BUY", 1, 100), trade("SELL", 1, 200),  # +100
         trade("BUY", 1, 300, "ETHUSDT"), trade("SELL", 1, 250, "ETHUSDT")],  # -50, not set off
        usdt_inr=Decimal("90"),
    )
    assert est["taxable_gain_inr"] == Decimal("9000")
    assert est["tax_30pct_inr"] == Decimal("2700.0")
    assert est["losses_not_set_off_inr"] == Decimal("4500")
    assert est["tds_1pct_withholding_inr"] == Decimal("405")  # 1% of 450 USDT sale value at 90
    assert any("not tax advice" in note for note in est["notes"])
    assert "static" in est["fx"]


def test_vda_tax_zero_when_only_losses():
    est = cs.vda_tax_estimate([trade("BUY", 1, 100), trade("SELL", 1, 50)], usdt_inr=Decimal("90"))
    assert est["tax_30pct_inr"] == 0 and est["net_tax_after_tds_inr"] == 0


# ── settings, preflight and defaults ─────────────────────────────────────────

def test_foreign_markets_default_off_and_have_bounds():
    from pydantic import ValidationError

    assert settings.foreign_us_paper_enabled is False and settings.crypto_sim_enabled is False
    for field, bad in (("crypto_usdt_inr_rate", 0), ("crypto_halt_daily_loss_pct", 0.5)):
        with pytest.raises(ValidationError):
            type(settings)(**{field: bad})


def test_preflight_blocks_us_paper_without_paper_flag(monkeypatch):
    from operations.preflight import run_preflight

    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", False)
    monkeypatch.setattr(settings, "alpaca_api_key", "")
    result = run_preflight()
    assert any("ALPACA_PAPER=true" in e for e in result.errors)


def test_preflight_warns_when_us_paper_keys_missing(monkeypatch):
    from operations.preflight import run_preflight

    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", True)
    monkeypatch.setattr(settings, "alpaca_api_key", "")
    result = run_preflight()
    assert any("paper keys are missing" in w for w in result.warnings)
    assert not any("ALPACA_PAPER=true" in e for e in result.errors)


def test_env_example_documents_the_new_flags():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    for name in (
        "FOREIGN_US_PAPER_ENABLED=false", "FOREIGN_USD_INR_RATE=",
        "CRYPTO_SIM_ENABLED=false", "CRYPTO_SIM_STARTING_USDT=",
        "CRYPTO_SIM_WATCHLIST=", "CRYPTO_USDT_INR_RATE=", "CRYPTO_HALT_DAILY_LOSS_PCT=",
    ):
        assert name in text


def test_capabilities_reflect_foreign_flags(monkeypatch):
    from config.settings import settings
    from operations.capabilities import current_capabilities

    def status(name):
        return next(item["status"] for item in current_capabilities() if item["name"] == name)

    monkeypatch.setattr(settings, "foreign_us_paper_enabled", False)
    monkeypatch.setattr(settings, "crypto_sim_enabled", False)
    assert status("alpaca_us_paper") == "disabled" and status("crypto_sim") == "disabled"

    monkeypatch.setattr(settings, "foreign_us_paper_enabled", True)
    monkeypatch.setattr(settings, "alpaca_paper", True)
    monkeypatch.setattr(settings, "alpaca_api_key", "k")
    monkeypatch.setattr(settings, "alpaca_secret_key", "s")
    monkeypatch.setattr(settings, "crypto_sim_enabled", True)
    assert status("alpaca_us_paper") == "experimental" and status("crypto_sim") == "experimental"

    monkeypatch.setattr(settings, "alpaca_api_key", "")
    assert status("alpaca_us_paper") == "disabled"
