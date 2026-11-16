"""Historical fallback must never substitute an unrelated market segment."""

from datetime import datetime

import pandas as pd
import pytest


@pytest.mark.parametrize("exchange", ["NFO", "BSE", "MCX"])
@pytest.mark.parametrize("broker_result", ["missing_token", "empty", "error", "unknown_interval"])
def test_non_nse_history_never_falls_back_to_nse_equity(monkeypatch, exchange, broker_result):
    from data import market_data

    monkeypatch.setattr(market_data, "_fetch_yfinance", lambda *_args: pytest.fail("Wrong-market fallback"))
    monkeypatch.setattr(
        market_data, "get_token",
        lambda *_args: None if broker_result == "missing_token" else "token",
    )

    def candles(**_kwargs):
        if broker_result == "error":
            raise RuntimeError("broker unavailable")
        return []

    monkeypatch.setattr(market_data.angel_client, "get_candle_data", candles)
    interval = "unknown" if broker_result == "unknown_interval" else "15m"
    result = market_data.fetch_ohlcv("TEST", interval=interval, exchange=exchange)
    assert result.empty
    assert list(result.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert result.attrs["history_provenance"]["provider"] == "unavailable"
    assert result.attrs["history_provenance"]["status"] == "unavailable"
    assert result.attrs["history_provenance"]["exchange"] == exchange


def test_nse_equity_fallback_still_works(monkeypatch):
    from data import market_data

    expected = pd.DataFrame({"close": [100.0]})
    monkeypatch.setattr(market_data, "get_token", lambda *_args: None)
    monkeypatch.setattr(market_data, "_fetch_yfinance", lambda *_args: expected)
    result = market_data.fetch_ohlcv("TEST", exchange="NSE")
    assert result is expected
    provenance = result.attrs["history_provenance"]
    assert provenance["provider"] == "yahoo_finance"
    assert provenance["fallback_reason"] == "unresolved_token"
    assert provenance["adjustment"] == "unadjusted_requested"
    assert provenance["corporate_actions"] == "provider_reported_zero_unverified"
    assert provenance["returned_rows"] == 1
    assert datetime.fromisoformat(provenance["retrieved_at"]).utcoffset().total_seconds() == 0


def test_non_nse_broker_data_is_preserved_when_available(monkeypatch):
    from data import market_data

    monkeypatch.setattr(market_data, "get_token", lambda *_args: "token")
    monkeypatch.setattr(market_data, "_fetch_yfinance", lambda *_args: pytest.fail("Fallback not needed"))
    monkeypatch.setattr(
        market_data.angel_client, "get_candle_data",
        lambda **_kwargs: [["2026-06-02T09:15:00+05:30", 100, 102, 99, 101, 1000]],
    )
    result = market_data.fetch_ohlcv("TEST", exchange="NFO")
    assert len(result) == 1
    assert result.iloc[0]["close"] == 101.0
    provenance = result.attrs["history_provenance"]
    assert provenance["provider"] == "angel_one"
    assert provenance["exchange"] == "NFO"
    assert provenance["adjustment"] == "unverified"
    assert provenance["corporate_actions"] == "unverified"
    assert provenance["fallback_reason"] is None


@pytest.mark.parametrize("broker_result,reason", [
    ("empty", "empty_broker_response"),
    ("error", "broker_error"),
])
def test_fallback_records_actual_broker_failure_and_unavailable_source(monkeypatch, broker_result, reason):
    from data import market_data

    monkeypatch.setattr(market_data, "get_token", lambda *_args: "token")
    def candles(**_kwargs):
        if broker_result == "error":
            raise RuntimeError("unavailable")
        return []
    monkeypatch.setattr(market_data.angel_client, "get_candle_data", candles)
    monkeypatch.setattr(market_data, "_fetch_yfinance", lambda *_args: pd.DataFrame())
    provenance = market_data.fetch_ohlcv("TEST").attrs["history_provenance"]
    assert provenance["provider"] == "yahoo_finance"
    assert provenance["status"] == "unavailable"
    assert provenance["fallback_reason"] == reason
    assert provenance["adjustment"] == "unverified"


@pytest.mark.parametrize("event", ["dividend", "split", "missing", "invalid"])
def test_yahoo_history_rejects_unmodeled_or_unknown_corporate_actions(monkeypatch, event):
    from data import market_data

    frame = pd.DataFrame({
        "Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [1000],
        "Dividends": [0.0], "Stock Splits": [0.0],
    }, index=pd.DatetimeIndex(["2026-06-02T09:15:00+05:30"], name="Datetime"))
    if event == "dividend":
        frame.loc[:, "Dividends"] = 1.0
    elif event == "split":
        frame.loc[:, "Stock Splits"] = 2.0
    elif event == "missing":
        frame = frame.drop(columns=["Stock Splits"])
    else:
        frame.loc[:, "Dividends"] = float("nan")

    class Ticker:
        def history(self, **_kwargs):
            return frame

    monkeypatch.setattr(market_data.yf, "Ticker", lambda _symbol: Ticker())
    assert market_data._fetch_yfinance("TEST", "15m", 30).empty


def test_yahoo_history_requests_unadjusted_prices_and_corporate_actions(monkeypatch):
    from data import market_data

    calls = []
    frame = pd.DataFrame({
        "Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [1000],
        "Dividends": [0.0], "Stock Splits": [0.0],
    }, index=pd.DatetimeIndex(["2026-06-02T09:15:00+05:30"], name="Datetime"))

    class Ticker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return frame

    monkeypatch.setattr(market_data.yf, "Ticker", lambda _symbol: Ticker())
    result = market_data._fetch_yfinance("TEST", "15m", 30)
    assert len(result) == 1
    assert calls[0]["auto_adjust"] is False
    assert calls[0]["actions"] is True
