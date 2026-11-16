"""Verify that reviewed historical data is bound to the captured bar frame."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from agents.checkpoint_serializer import TradingCheckpointSerializer
from agents.review_context import ReviewContextError, candidate_review_identity
from operations.history_evidence import history_digest


def _state():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-06-05T09:15:00+05:30"]),
        "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1000],
    })
    provenance = {
        "provider": "angel_one", "symbol": "TEST", "exchange": "NSE", "interval": "15m",
        "status": "available", "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "returned_rows": len(frame), "bars_sha256": history_digest(frame),
    }
    return {
        "llm_profile": {"run_id": "example"},
        "market_data": {"TEST": frame},
        "market_data_provenance": {"TEST": dict(provenance)},
        "technical_signals": {"TEST": {
            "strategy": "CompositeTrend", "signal": "BUY", "history_provenance": dict(provenance),
        }},
    }


def test_history_digest_survives_checkpoint_roundtrip():
    state = _state()
    before = candidate_review_identity(state, "TEST")
    serializer = TradingCheckpointSerializer()
    restored = serializer.loads_typed(serializer.dumps_typed(state))
    assert candidate_review_identity(restored, "TEST") == before


@pytest.mark.parametrize("change", ["bar", "source", "missing_frame", "wrong_symbol", "bad_digest", "omitted"])
def test_history_evidence_changes_block_candidate_identity(change):
    state = _state()
    candidate_review_identity(state, "TEST")
    if change == "bar":
        state["market_data"]["TEST"].loc[0, "close"] = 101.5
    elif change == "source":
        state["market_data_provenance"]["TEST"]["provider"] = "yahoo_finance"
    elif change == "missing_frame":
        state["market_data"] = {}
    elif change == "wrong_symbol":
        state["technical_signals"]["TEST"]["history_provenance"]["symbol"] = "OTHER"
        state["market_data_provenance"]["TEST"]["symbol"] = "OTHER"
    elif change == "bad_digest":
        state["technical_signals"]["TEST"]["history_provenance"]["bars_sha256"] = "bad"
        state["market_data_provenance"]["TEST"]["bars_sha256"] = "bad"
    else:
        del state["technical_signals"]["TEST"]["history_provenance"]
    with pytest.raises(ReviewContextError, match="historical data"):
        candidate_review_identity(state, "TEST")


@pytest.mark.parametrize("field,value", [
    ("provider", "unknown"), ("provider", []), ("status", "unavailable"), ("exchange", "NFO"),
    ("interval", "1d"), ("returned_rows", 2), ("retrieved_at", "2026-06-05T04:00:00"),
])
def test_invalid_matching_history_metadata_blocks_review(field, value):
    state = _state()
    state["technical_signals"]["TEST"]["history_provenance"][field] = value
    state["market_data_provenance"]["TEST"][field] = value
    with pytest.raises(ReviewContextError, match="historical data"):
        candidate_review_identity(state, "TEST")


def test_missing_history_provenance_blocks_candidate_identity():
    state = _state()
    del state["technical_signals"]["TEST"]["history_provenance"]
    state["market_data_provenance"] = {}
    with pytest.raises(ReviewContextError, match="historical data provenance"):
        candidate_review_identity(state, "TEST")


@pytest.mark.parametrize("age,valid", [
    (timedelta(minutes=30), True),
    (timedelta(minutes=30, microseconds=1), False),
    (timedelta(microseconds=-1), False),
])
def test_history_capture_age_is_checked_at_review(monkeypatch, age, valid):
    from agents import review_context

    reference = datetime.now(timezone.utc)
    class FrozenTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return reference.astimezone(tz) if tz else reference.replace(tzinfo=None)
    monkeypatch.setattr(review_context, "datetime", FrozenTime)
    state = _state()
    captured = (reference - age).isoformat()
    state["technical_signals"]["TEST"]["history_provenance"]["retrieved_at"] = captured
    state["market_data_provenance"]["TEST"]["retrieved_at"] = captured
    if valid:
        assert candidate_review_identity(state, "TEST")
    else:
        with pytest.raises(ReviewContextError, match="historical data cannot be revalidated"):
            candidate_review_identity(state, "TEST")


@pytest.mark.parametrize("change", ["bar", "missing", "stale"])
def test_router_blocks_unavailable_or_changed_bars_after_review(monkeypatch, change):
    from agents import graph
    from agents import review_context
    from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state

    state = sentiment_state(["TEST"])
    captured = _state()
    state.update({
        "market_data": captured["market_data"],
        "market_data_provenance": captured["market_data_provenance"],
    })
    state["technical_signals"]["TEST"]["history_provenance"] = (
        captured["technical_signals"]["TEST"]["history_provenance"]
    )
    state["debate_results"] = {"TEST": reviewed_debate(state, "TEST", "BUY")}
    decision = reviewed_decision(state, {
        "symbol": "TEST", "action": "BUY", "quantity": 1, "price": 100,
        "holding_period": "INTRADAY",
    })
    state.update({
        "execution_enabled": True, "current_positions": {}, "watchlist": ["TEST"],
        "screener_results": {"qualified": ["TEST"], "rejected": []},
        "trading_decisions": [decision],
    })
    if change == "bar":
        state["market_data"]["TEST"].loc[0, "close"] = 101.5
    elif change == "missing":
        del state["technical_signals"]["TEST"]["history_provenance"]
        state["market_data_provenance"] = {}
    else:
        later = datetime.now(timezone.utc) + timedelta(minutes=31)
        class FrozenTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return later.astimezone(tz) if tz else later.replace(tzinfo=None)
        monkeypatch.setattr(review_context, "datetime", FrozenTime)
    monkeypatch.setattr(graph, "place_order", lambda **_kwargs: pytest.fail("Unsafe entry submitted"))
    result = graph._order_router_node(state)
    assert not result["executed_orders"]
    assert any("historical data evidence" in error.lower() for error in result["errors"])
