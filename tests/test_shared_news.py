"""Shared news snapshots are fetched once and reach both analyst nodes."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tests.test_llm_runtime import llm_database


def test_snapshot_deduplicates_and_excludes_future_news(monkeypatch):
    from data import news_fetcher

    calls = []
    now = datetime.now(timezone.utc)
    common = {
        "title": "Market headline",
        "source": "RSS",
        "url": "https://example.test/common",
        "published_at": (now - timedelta(minutes=5)).isoformat(),
    }

    async def rss(*, status=None):
        calls.append("rss")
        return [common]

    async def news_api(query, days=3, *, status=None):
        symbol = query.split()[0]
        calls.append(f"newsapi:{symbol}")
        return [common, {
            "title": f"{symbol} update",
            "source": "NewsAPI",
            "url": f"https://example.test/{symbol}",
            "published_at": "",
        }]

    async def nse(symbol, *, status=None):
        calls.append(f"nse:{symbol}")
        return [{
            "title": "Future announcement",
            "source": "NSE",
            "url": f"https://example.test/future/{symbol}",
            "published_at": (now + timedelta(days=1)).isoformat(),
        }]

    monkeypatch.setattr(news_fetcher, "fetch_rss_feeds", rss)
    monkeypatch.setattr(news_fetcher, "fetch_news_api", news_api)
    monkeypatch.setattr(news_fetcher, "fetch_nse_announcements", nse)

    snapshot = asyncio.run(news_fetcher.fetch_news_snapshot(["AAA", "BBB"]))

    assert sorted(calls) == ["newsapi:AAA", "newsapi:BBB", "nse:AAA", "nse:BBB", "rss"]
    assert snapshot["excluded_future"] == 2
    assert len(snapshot["articles"]) == 3
    assert len(snapshot["by_symbol"]["AAA"]) == 2
    assert len(snapshot["by_symbol"]["BBB"]) == 2
    assert all(item["observed_at"] == snapshot["as_of"] for item in snapshot["articles"])
    undated = next(item for item in snapshot["articles"] if item["title"] == "AAA update")
    assert undated["publication_time_status"] == "unknown"


def test_snapshot_reports_provider_errors(monkeypatch):
    from data import news_fetcher

    async def rss(*, status=None):
        raise RuntimeError("feed unavailable")

    monkeypatch.setattr(news_fetcher, "fetch_rss_feeds", rss)
    snapshot = asyncio.run(news_fetcher.fetch_news_snapshot([]))
    assert snapshot["source_status"]["rss"] == "error"
    assert snapshot["errors"]


def test_snapshot_exposes_partial_rss_failures(monkeypatch):
    from data import news_fetcher

    async def rss(*, status=None):
        status.update(economic_times="available", moneycontrol="error")
        return [{"title": "Headline", "source": "economic_times"}]

    monkeypatch.setattr(news_fetcher, "fetch_rss_feeds", rss)
    snapshot = asyncio.run(news_fetcher.fetch_news_snapshot([]))
    assert snapshot["source_status"]["rss"] == "partial"
    assert snapshot["source_status"]["rss:moneycontrol"] == "error"
    assert "rss:moneycontrol error" in snapshot["errors"]


def test_both_analysts_receive_the_same_news(llm_database, monkeypatch):
    from agents import geopolitical_nlp_agent as geo_module
    from agents import sentiment_agent as sentiment_module

    article = {"title": "Shared market news", "source": "RSS", "description": "Context"}
    snapshot = {
        "as_of": "2026-01-02T09:00:00+00:00",
        "general": [article],
        "articles": [article],
        "by_symbol": {"TEST": [article]},
    }
    state = {"watchlist": ["TEST"], "news_snapshot": snapshot, "active_macro_event": {},
             "research_mode": True}
    sentiment = sentiment_module.SentimentAgent()
    monkeypatch.setattr(sentiment, "_llm", SimpleNamespace())
    async def no_network(*_args):
        raise AssertionError("Sentiment must use the shared snapshot")
    monkeypatch.setattr(sentiment_module, "fetch_all_news", no_network, raising=False)
    prompts = []
    monkeypatch.setattr(sentiment, "_llm_sentiment", lambda symbol, text: (
        prompts.append((symbol, text)) or {
            "signal": "HOLD", "confidence": 0, "sentiment_score": 0,
            "key_events": [], "reasoning": "No directional signal.",
        }
    ))

    geopolitical = geo_module.GeopoliticalNLPAgent()
    seen = []
    monkeypatch.setattr(geopolitical, "analyze_geopolitical_risk", lambda **kwargs: (
        seen.extend(kwargs["news_articles"]) or {"identified_risks": []}
    ))

    sentiment.run(state)
    geopolitical.run(state)
    assert "Shared market news" in prompts[0][1]
    assert seen == [article]


def test_graph_places_news_before_parallel_analysts():
    from agents.graph import build_trading_graph

    edges = {(edge.source, edge.target) for edge in build_trading_graph().get_graph().edges}
    assert ("quant_screener", "strategy_candidates") in edges
    assert ("strategy_candidates", "shared_news") in edges
    assert ("shared_news", "sentiment") in edges
    # Geopolitical analysis consumes this scan's macro output, so it follows macro.
    assert ("macro", "geopolitical_nlp") in edges
    assert ("shared_news", "geopolitical_nlp") not in edges
    assert ("macro", "bull_bear_debate") not in edges
    assert ("geopolitical_nlp", "bull_bear_debate") in edges
