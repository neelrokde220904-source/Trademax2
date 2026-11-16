"""Durable, point-in-time research records and next-session outcomes."""

import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from tests.test_llm_runtime import llm_database


def _state():
    bars = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-06-05T09:45:00+05:30")],
        "open": [99.0], "high": [101.0], "low": [98.0], "close": [100.0],
        "volume": [1000],
    })
    return {
        "date": "2026-06-05", "watchlist": ["TEST"], "market_data": {"TEST": bars},
        "live_quotes": {}, "india_vix": 15.0, "gift_nifty": 0.0,
        "hmm_regime": {"regime": "LOW_VOL_TREND"},
        "ofi_signals": {}, "active_macro_event": {},
        "news_snapshot": {"as_of": "2026-06-05T04:30:00+00:00", "articles": [], "by_symbol": {}},
        "research_fundamental_inputs": {"TEST": {"pe_ratio": 15}},
        "research_options_inputs": {
            "NIFTY": {"chain": pd.DataFrame(), "spot": 0},
            "BANKNIFTY": {"chain": pd.DataFrame(), "spot": 0},
        },
        "research_macro_inputs": {"global_cues": {}, "usdinr": None, "crude_oil": None, "fii_dii": []},
        "research_india_inputs": {
            "fo_ban_list": [], "bulk_deals": [], "block_deals": [],
            "promoter_pledging": {}, "circuit_filters": {}, "is_expiry_day": False,
        },
        "technical_signals": {"TEST": {"signal": "BUY", "confidence": 80}},
        "sentiment_signals": {"TEST": {"signal": "HOLD", "confidence": 40}},
        "fundamental_signals": {}, "options_signals": {}, "india_signals": {},
        "risk_assessment": {"blocked": True, "reason": "VIX unavailable"},
        "trading_decisions": [], "executed_orders": [], "errors": [],
        "execution_enabled": True, "current_positions": {"PRIVATE": {"quantity": 3}},
    }


def _daily(*rows):
    return pd.DataFrame([
        {"datetime": pd.Timestamp(day, tz="Asia/Kolkata"), "close": price, "stock_splits": 0.0, "dividends": 0.0}
        for day, price in rows
    ])


def test_snapshot_rejects_unusable_fundamental_inputs(llm_database):
    from research.tracking import record_scan

    state = _state()
    state["research_fundamental_inputs"]["TEST"] = {"pe_ratio": None}
    result = record_scan(state, "bad-fundamental", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    assert result["snapshot_status"] == "incomplete"
    assert any("fundamental" in problem for problem in result["problems"])


def test_snapshot_retains_separate_history_provenance(llm_database):
    from research.tracking import load_snapshot, record_scan

    state = _state()
    state["market_data_provenance"] = {"TEST": {
        "provider": "angel_one", "symbol": "TEST", "adjustment": "unverified",
    }}
    result = record_scan(state, "history-source", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    assert result["snapshot_status"] == "complete"
    assert load_snapshot("history-source")["market_data_provenance"] == state["market_data_provenance"]


def test_legacy_complete_snapshot_and_forecasts_are_revalidated(llm_database):
    from database.db import get_db
    from database.models import ResearchForecast, ResearchSnapshot
    from research.tracking import forecast_summary, load_snapshot, record_scan, settle_forecasts

    state = _state()
    state["research_fundamental_inputs"]["TEST"] = {"pe_ratio": None}
    record_scan(state, "legacy", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    with get_db() as db:
        db.get(ResearchSnapshot, "legacy").status = "complete"
        forecast = db.query(ResearchForecast).filter(
            ResearchForecast.run_id == "legacy", ResearchForecast.direction == "BUY",
        ).one()
        forecast.status = "pending"

    with pytest.raises(ValueError, match="fundamental"):
        load_snapshot("legacy")
    result = settle_forecasts(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        loader=lambda *_args: pytest.fail("Invalid legacy capture must not fetch settlement data"),
    )
    assert result["settled"] == 0
    assert any("fundamental" in error for error in result["errors"])
    buy = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert buy["status"] == "ineligible_input"


def test_legacy_pending_forecast_reference_must_match_captured_evidence(llm_database):
    from database.db import get_db
    from database.models import ResearchForecast
    from research.tracking import forecast_summary, record_scan, settle_forecasts

    record_scan(_state(), "legacy-price", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    with get_db() as db:
        forecast = db.query(ResearchForecast).filter(
            ResearchForecast.run_id == "legacy-price", ResearchForecast.direction == "BUY",
        ).one()
        forecast.reference_price = 999

    result = settle_forecasts(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        loader=lambda *_args: pytest.fail("Changed reference must not fetch settlement data"),
    )
    assert result["settled"] == 0
    assert any("reference cannot be revalidated" in error for error in result["errors"])
    buy = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert buy["status"] == "ineligible_input"


def test_snapshot_observation_time_must_match_record(llm_database):
    from datetime import timedelta
    from database.db import get_db
    from database.models import ResearchSnapshot
    from research.tracking import load_snapshot, record_scan

    record_scan(_state(), "wrong-time", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    with get_db() as db:
        row = db.get(ResearchSnapshot, "wrong-time")
        row.observed_at += timedelta(minutes=1)
    with pytest.raises(ValueError, match="observation time does not match"):
        load_snapshot("wrong-time")


def test_forecast_observation_time_must_match_snapshot(llm_database):
    from datetime import timedelta
    from database.db import get_db
    from database.models import ResearchForecast
    from research.tracking import forecast_summary, record_scan, settle_forecasts

    record_scan(_state(), "wrong-forecast-time", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    with get_db() as db:
        forecast = db.query(ResearchForecast).filter(
            ResearchForecast.run_id == "wrong-forecast-time", ResearchForecast.direction == "BUY",
        ).one()
        forecast.observed_at += timedelta(minutes=1)
    result = settle_forecasts(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        loader=lambda *_args: pytest.fail("Changed forecast time must not fetch settlement data"),
    )
    assert result["settled"] == 0
    assert any("observation time does not match" in error for error in result["errors"])
    buy = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert buy["status"] == "ineligible_input"


def test_legacy_settled_forecasts_are_excluded_from_current_accuracy(llm_database):
    from database.db import get_db
    from database.models import ResearchForecast
    from research.tracking import forecast_summary, record_scan, settle_forecasts

    moment = datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc)
    record_scan(_state(), "old-score", observed_at=moment)
    with get_db() as db:
        old = db.query(ResearchForecast).filter(
            ResearchForecast.run_id == "old-score", ResearchForecast.direction == "BUY",
        ).one()
        old.status, old.direction_correct = "settled", True
        old.settlement_note = "Hypothetical price movement; not trade P&L"

    record_scan(_state(), "current-score", observed_at=moment)
    frames = {"^NSEI": _daily(("2026-06-08", 102)), "TEST": _daily(("2026-06-08", 99))}
    result = settle_forecasts(now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc), loader=lambda symbol, _: frames[symbol])
    assert result["settled"] == 1
    summary = forecast_summary()
    assert summary["counts"]["legacy_settled"] == 1
    assert summary["counts"]["verified_settled"] == 1
    assert summary["by_agent"]["technical"] == {"settled": 1, "correct": 0, "directional_accuracy": 0.0}
    old_row = next(row for row in summary["forecasts"] if row["run_id"] == "old-score" and row["direction"] == "BUY")
    assert old_row["evidence_validated"] is False


def test_forecast_calendar_uses_verified_special_sessions_and_rejects_unknown_years():
    from datetime import date
    from research.tracking import _next_session

    assert _next_session(date(2026, 1, 30)) == date(2026, 2, 1)
    assert _next_session(date(2026, 11, 6)) == date(2026, 11, 8)
    with pytest.raises(ValueError, match="calendar"):
        _next_session(date(2026, 12, 31))


def test_forecast_reference_ignores_stale_quote_and_forming_bar():
    from research.tracking import _reference

    observed = datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc)
    state = _state()
    state["live_quotes"] = {"TEST": {
        "ltp": 999, "timestamp": "2026-06-05T04:00:00+00:00",
        "exchange_timestamp": "2026-06-05T04:00:00+00:00",
    }}
    state["market_data"]["TEST"] = pd.DataFrame({
        "datetime": ["2026-06-05T09:45:00+05:30", "2026-06-05T10:30:00+05:30"],
        "close": [100.0, 900.0],
    })
    assert _reference(state, "TEST", observed) == (100.0, "captured_bar")


def test_forecast_reference_requires_current_completed_bar_or_fresh_quote():
    from research.tracking import _reference

    observed = datetime(2026, 6, 5, 5, 31, tzinfo=timezone.utc)
    state = _state()
    assert _reference(state, "TEST", observed) == (None, None)
    state["live_quotes"] = {"TEST": {
        "ltp": 101, "timestamp": "2026-06-05T05:30:55+00:00",
        "exchange_timestamp": "2026-06-05T05:30:54+00:00",
    }}
    assert _reference(state, "TEST", observed) == (101.0, "captured_quote")


def test_forecast_reference_rejects_untimestamped_options_spot():
    from research.tracking import _reference

    state = _state()
    state["options_signals"] = {"NIFTY": {"spot": 23000}}
    observed = datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc)
    assert _reference(state, "NIFTY", observed) == (None, None)


@pytest.mark.parametrize("observed, expected", [
    ("2026-06-05T05:00:00+00:00", (100.0, "captured_bar")),
    ("2026-06-05T05:00:01+00:00", (None, None)),
])
def test_forecast_reference_completed_bar_age_boundary(observed, expected):
    from research.tracking import _reference

    assert _reference(_state(), "TEST", datetime.fromisoformat(observed)) == expected


@pytest.mark.parametrize("quote", [
    {"ltp": 101, "timestamp": "2026-06-05T05:00:00+00:00"},
    {"ltp": 101, "timestamp": "2026-06-05T05:00:01+00:00", "exchange_timestamp": "2026-06-05T05:00:00+00:00"},
    {"ltp": 101, "timestamp": "2026-06-05T05:00:00+00:00", "exchange_timestamp": "2026-06-05T05:00:01+00:00"},
    {"ltp": 101, "timestamp": "2026-06-05T10:30:00", "exchange_timestamp": "2026-06-05T10:30:00"},
])
def test_forecast_reference_rejects_unverified_or_future_quotes(quote):
    from research.tracking import _reference

    state = _state()
    state["live_quotes"] = {"TEST": quote}
    observed = datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc)
    assert _reference(state, "TEST", observed) == (100.0, "captured_bar")


def test_missing_forecast_reference_is_not_scored(llm_database):
    from research.tracking import forecast_summary, record_scan, settle_forecasts

    state = _state()
    observed = datetime(2026, 6, 5, 5, 31, tzinfo=timezone.utc)
    record_scan(state, "stale-price", observed_at=observed)
    buy = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert buy["status"] == "missing_reference"
    result = settle_forecasts(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        loader=lambda *_args: pytest.fail("Missing reference must not be settled"),
    )
    assert result["settled"] == 0


@pytest.mark.parametrize("day, expected", [
    ("2026-06-06", None),
    ("2026-02-01", 100.0),
    ("2027-01-04", None),
])
def test_forecast_reference_requires_verified_open_session(day, expected):
    from research.tracking import _reference

    state = _state()
    state["market_data"]["TEST"]["datetime"] = [pd.Timestamp(f"{day}T09:45:00+05:30")]
    observed = datetime.fromisoformat(f"{day}T10:30:00+05:30")
    price, source = _reference(state, "TEST", observed)
    assert price == expected
    assert source == ("captured_bar" if expected is not None else None)


def test_unknown_forecast_calendar_keeps_pending_and_does_not_fetch_prices(llm_database):
    from research.tracking import record_scan, settle_forecasts, forecast_summary

    state = _state()
    state["market_data"]["TEST"]["datetime"] = [pd.Timestamp("2026-12-31T09:45:00+05:30")]
    state["news_snapshot"]["as_of"] = "2026-12-31T04:30:00+00:00"
    assert record_scan(
        state, "year-end", observed_at=datetime(2026, 12, 31, 5, 0, tzinfo=timezone.utc),
    )["snapshot_status"] == "complete"
    result = settle_forecasts(
        now=datetime(2027, 1, 4, 12, 0, tzinfo=timezone.utc),
        loader=lambda *_args: pytest.fail("Unknown calendar must not fetch settlement data"),
    )
    assert result["settled"] == 0
    assert any("calendar" in error for error in result["errors"])
    assert next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")["status"] == "pending"


@pytest.mark.parametrize("defect", ["missing_dividends", "negative_split", "duplicate", "bad_index"])
def test_forecast_does_not_settle_on_ambiguous_daily_data(llm_database, defect):
    from research.tracking import record_scan, settle_forecasts, forecast_summary

    record_scan(_state(), "scan-1", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    index = _daily(("2026-06-08", 102))
    stock = _daily(("2026-06-08", 103))
    if defect == "missing_dividends":
        stock = stock.drop(columns=["dividends"])
    elif defect == "negative_split":
        stock.loc[0, "stock_splits"] = -1
    elif defect == "duplicate":
        stock = pd.concat([stock, stock], ignore_index=True)
    else:
        index.loc[0, "close"] = float("nan")
    frames = {"^NSEI": index, "TEST": stock}
    result = settle_forecasts(
        now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
        loader=lambda symbol, _days: frames[symbol],
    )
    assert result["settled"] == 0
    assert result["errors"]
    assert next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")["status"] == "pending"


def test_snapshot_and_forecasts_are_durable_idempotent_and_account_free(llm_database):
    from research.tracking import load_snapshot, record_scan, forecast_summary

    moment = datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc)
    result = record_scan(_state(), "scan-1", observed_at=moment)
    assert result["snapshot_status"] == "complete"
    assert record_scan(_state(), "scan-1", observed_at=moment)["duplicate"]
    snapshot = load_snapshot("scan-1")
    assert "current_positions" not in snapshot
    assert snapshot["market_data"]["TEST"].iloc[0]["close"] == 100
    summary = forecast_summary()
    assert summary["counts"] == {
        "pending": 1, "unscored": 1, "verified_settled": 0, "legacy_settled": 0,
    }
    buy = next(row for row in summary["forecasts"] if row["direction"] == "BUY")
    assert buy["reference_price"] == 100
    assert buy["block_reason"] == "VIX unavailable"


def test_settlement_waits_for_next_session_and_never_uses_a_later_bar(llm_database):
    from research.tracking import record_scan, settle_forecasts, forecast_summary

    record_scan(_state(), "scan-1", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    frames = {
        "^NSEI": _daily(("2026-06-05", 100), ("2026-06-08", 102), ("2026-06-09", 103)),
        "TEST": _daily(("2026-06-05", 100), ("2026-06-09", 104)),
    }
    loader = lambda symbol, days: frames[symbol]
    early = datetime(2026, 6, 8, 8, 0, tzinfo=timezone.utc)
    assert settle_forecasts(now=early, loader=loader)["settled"] == 0
    after_close = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    assert settle_forecasts(now=after_close, loader=loader)["settled"] == 0
    pending = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert pending["status"] == "pending"
    assert "2026-06-08" in pending["settlement_note"]
    frames["TEST"] = _daily(("2026-06-08", 102), ("2026-06-09", 104))
    assert settle_forecasts(now=after_close, loader=loader)["settled"] == 1
    assert settle_forecasts(now=after_close, loader=loader)["settled"] == 0
    buy = next(row for row in forecast_summary()["forecasts"] if row["direction"] == "BUY")
    assert buy["settlement_session"] == "2026-06-08"
    assert buy["direction_correct"] is True
    assert buy["directional_return_pct"] == 2.0


def test_incomplete_snapshot_is_not_replayed(llm_database):
    from research.tracking import load_snapshot, record_scan

    state = _state()
    state["research_fundamental_inputs"] = {}
    result = record_scan(state, "incomplete", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    assert result["snapshot_status"] == "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        load_snapshot("incomplete")


def test_capture_is_persisted_even_if_execution_tracking_fails(llm_database, monkeypatch, tmp_path):
    from agents import graph as trading_graph
    from database.db import get_db
    from database.models import ResearchSnapshot
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(dict)
    workflow.add_node("scan", lambda state: {**state, **_state()})
    workflow.set_entry_point("scan")
    workflow.add_edge("scan", END)
    monkeypatch.setattr(trading_graph, "build_trading_graph", workflow.compile)
    monkeypatch.setattr(trading_graph, "__file__", str(tmp_path / "agents" / "graph.py"))

    def fail_tracking(_state):
        raise RuntimeError("execution tracking failed")

    with pytest.raises(RuntimeError, match="execution tracking failed"):
        asyncio.run(trading_graph.run_trading_pipeline(["TEST"], on_execution_complete=fail_tracking))
    with get_db() as db:
        assert db.query(ResearchSnapshot).count() == 1


def test_future_data_and_corporate_actions_are_never_scored(llm_database):
    from research.tracking import forecast_summary, record_scan, settle_forecasts

    future = _state()
    future["market_data"]["TEST"].loc[0, "datetime"] = pd.Timestamp("2026-06-05T11:00:00+05:30")
    result = record_scan(future, "future", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    assert result["snapshot_status"] == "incomplete"
    assert any("future timestamp" in item for item in result["problems"])

    mixed = _state()
    mixed["market_data"]["TEST"] = pd.concat([
        mixed["market_data"]["TEST"], future["market_data"]["TEST"],
    ], ignore_index=True)
    record_scan(mixed, "mixed", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    mixed_buy = next(row for row in forecast_summary()["forecasts"] if row["run_id"] == "mixed" and row["direction"] == "BUY")
    assert mixed_buy["status"] == "ineligible_input"

    record_scan(_state(), "valid", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    frames = {
        "^NSEI": _daily(("2026-06-08", 102)),
        "TEST": _daily(("2026-06-08", 50)),
    }
    frames["TEST"].loc[0, "stock_splits"] = 2.0
    result = settle_forecasts(
        now=datetime(2026, 6, 8, 11, 0, tzinfo=timezone.utc),
        loader=lambda symbol, days: frames[symbol],
    )
    assert result["settled"] == 0
    buy = next(row for row in forecast_summary()["forecasts"] if row["run_id"] == "valid" and row["direction"] == "BUY")
    assert buy["status"] == "pending"
    assert "Corporate action" in buy["settlement_note"]


def test_index_spot_and_debate_directions_are_recorded(llm_database):
    from research.tracking import forecast_summary, record_scan

    state = _state()
    state["options_signals"] = {"NIFTY": {"signal": "BUY", "confidence": 65, "spot": 23000}}
    state["macro_signals"] = {"overall": {"signal": "SELL", "confidence": 55}}
    state["debate_results"] = {"TEST": {"action": "SELL", "conviction_score": 72}}
    record_scan(state, "index", observed_at=datetime(2026, 6, 5, 5, 0, tzinfo=timezone.utc))
    rows = forecast_summary()["forecasts"]
    options = next(row for row in rows if row["agent"] == "options")
    macro = next(row for row in rows if row["agent"] == "macro")
    debate = next(row for row in rows if row["agent"] == "bull_bear_debate")
    assert options["reference_price"] is None and macro["reference_price"] is None
    assert options["status"] == macro["status"] == "missing_reference"
    assert debate["direction"] == "SELL"
    assert debate["confidence"] == 72


def test_daily_index_requests_use_index_tickers(monkeypatch):
    from research import tracking

    names = []

    class FakeTicker:
        def __init__(self, symbol):
            names.append(symbol)

        def history(self, **_kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(tracking.yf, "Ticker", FakeTicker)
    tracking._daily_prices("NIFTY", 10)
    tracking._daily_prices("BANKNIFTY", 10)
    tracking._daily_prices("TEST", 10)
    assert names == ["^NSEI", "^NSEBANK", "TEST.NS"]
