from datetime import datetime, timedelta, timezone

import pandas as pd

from agents.sentiment_evidence import candidate_news, sentiment_digest
from agents.review_context import candidate_review_identity, debate_review_digest, technical_review_digest
from operations.history_evidence import history_digest


def valid_sentiment(**overrides):
    return {
        "signal": "HOLD", "confidence": 50, "sentiment_score": 0,
        "key_events": [], "reasoning": "No directional signal.", **overrides,
    }


def sentiment_state(symbols):
    observed = datetime.now(timezone.utc)
    observed_text = observed.isoformat()
    article = {
        "title": "Market headline", "description": "Market context", "source": "economic_times",
        "published_at": (observed - timedelta(minutes=1)).isoformat(),
        "publication_time_status": "known", "observed_at": observed_text,
        "provider": "rss", "queried_symbol": None,
    }
    snapshot = {
        "as_of": observed_text, "source_status": {"rss:economic_times": "available"},
        "by_symbol": {symbol: [article] for symbol in symbols},
    }
    signals = {
        symbol: valid_sentiment(evidence={
            "run_id": "test-run", "symbol": symbol, "snapshot_as_of": observed_text,
            "news_digest": candidate_news(snapshot, symbol)[1],
        })
        for symbol in symbols
    }
    histories = {
        symbol: pd.DataFrame({
            "datetime": [pd.Timestamp("2026-06-05T09:15:00+05:30")],
            "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1000],
        })
        for symbol in symbols
    }
    provenance = {
        symbol: {
            "provider": "angel_one", "symbol": symbol, "exchange": "NSE", "interval": "15m",
            "status": "available", "retrieved_at": observed_text, "returned_rows": len(frame),
            "bars_sha256": history_digest(frame),
        }
        for symbol, frame in histories.items()
    }
    return {
        "news_snapshot": snapshot, "llm_profile": {"run_id": "test-run"},
        "sentiment_signals": signals,
        "market_data": histories, "market_data_provenance": provenance,
        "technical_signals": {
            symbol: {"signal": "BUY", "strategy": "CompositeTrend", "history_provenance": dict(provenance[symbol])}
            for symbol in symbols
        },
    }


def reviewed_debate(state, symbol, action):
    import hashlib

    context_json = "{}"
    return {
        "review_complete": True, "action": action, "conviction_score": 80,
        "reasoning": "Reviewed both arguments.",
        "arguments": {
            "bull": {"_review_succeeded": True, "thesis": "Bull argument."},
            "bear": {"_review_succeeded": True, "counter_thesis": "Bear argument."},
        },
        "debate_context_sha256": hashlib.sha256(context_json.encode("utf-8")).hexdigest(),
        "debate_context_json": context_json,
        "sentiment_review_digest": sentiment_digest(state["sentiment_signals"][symbol]),
        "technical_review_digest": technical_review_digest(state["technical_signals"][symbol]),
        "candidate_id": candidate_review_identity(state, symbol),
    }


def reviewed_decision(state, decision):
    symbol = decision["symbol"]
    debate = state.get("debate_results", {}).get(symbol)
    if debate is None:
        debate = reviewed_debate(state, symbol, decision["action"])
    return {
        **decision, "ai_review_passed": True,
        "sentiment_review_digest": sentiment_digest(state["sentiment_signals"][symbol]),
        "debate_review_digest": debate_review_digest(debate),
        "technical_review_digest": technical_review_digest(state["technical_signals"][symbol]),
        "candidate_id": candidate_review_identity(state, symbol),
        "debate_context_json": debate["debate_context_json"],
    }


def stub_persisted_review_verification(monkeypatch):
    """Isolate unrelated router-gate tests from the separately tested review store."""
    from operations import candidate_reviews

    monkeypatch.setattr(candidate_reviews, "verify_candidate_review", lambda *_args: None)
