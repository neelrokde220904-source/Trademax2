"""Validate the news input used for an entry candidate's sentiment review."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from data.news_fetcher import format_news_for_prompt

SNAPSHOT_MAX_AGE = timedelta(minutes=30)
ARTICLE_MAX_AGE = timedelta(days=3)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("News timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("News timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("News timestamp has no timezone")
    return timestamp.astimezone(timezone.utc)


def candidate_news(snapshot: Any, symbol: str, *, now: datetime | None = None) -> tuple[str, str]:
    """Return bounded current news; general news is not a symbol-specific claim."""
    if not isinstance(snapshot, dict):
        raise ValueError("News snapshot is unavailable")
    observed_text = snapshot.get("as_of")
    observed = _timestamp(observed_text)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None or not timedelta(0) <= reference - observed <= SNAPSHOT_MAX_AGE:
        raise ValueError("News snapshot is stale or future-dated")
    by_symbol = snapshot.get("by_symbol")
    statuses = snapshot.get("source_status")
    if not isinstance(by_symbol, dict) or not isinstance(statuses, dict):
        raise ValueError("News source outcomes are unavailable")
    articles = by_symbol.get(symbol)
    if not isinstance(articles, list):
        raise ValueError("Candidate news is unavailable")
    selected = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        provider = article.get("provider")
        source = article.get("source")
        title = article.get("title")
        if not isinstance(source, str) or not source.strip() or not isinstance(title, str) or not title.strip():
            continue
        if provider == "rss":
            source_state = statuses.get(f"rss:{source}", statuses.get("rss"))
            if article.get("queried_symbol") is not None:
                continue
        elif provider in {f"newsapi:{symbol}", f"nse:{symbol}"}:
            source_state = statuses.get(provider)
            if article.get("queried_symbol") != symbol:
                continue
        else:
            continue
        if source_state != "available" or article.get("observed_at") != observed_text:
            continue
        if article.get("publication_time_status") != "known":
            continue
        try:
            published = _timestamp(article.get("published_at"))
        except ValueError:
            continue
        if not timedelta(0) <= observed - published <= ARTICLE_MAX_AGE:
            continue
        selected.append(article)
        if len(selected) == 10:
            break
    if not selected:
        raise ValueError("No current, attributable news is available")
    text = format_news_for_prompt(selected)
    if len(text.encode("utf-8")) > 16_000:
        raise ValueError("Complete sentiment news exceeds 16000 bytes")
    provenance = [
        {key: article.get(key) for key in (
            "provider", "queried_symbol", "source", "observed_at", "published_at",
        )}
        for article in selected
    ]
    payload = json.dumps({"text": text, "provenance": provenance}, sort_keys=True, allow_nan=False)
    return text, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sentiment_digest(signal: dict[str, Any]) -> str:
    payload = json.dumps(signal, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > 16_000:
        raise ValueError("Sentiment review exceeds 16000 bytes")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
