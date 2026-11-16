"""News fetcher — NewsAPI + RSS feeds (Economic Times, Moneycontrol, NSE)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable

import feedparser
import httpx
import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# RSS feed URLs
RSS_FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
}

# NSE announcements endpoint
NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
NSE_BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


async def fetch_news_api(
    query: str, days: int = 3, *, status: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch news articles from NewsAPI.org.

    Args:
        query: Search query (e.g. 'RELIANCE NSE stock')
        days: Number of days to look back

    Returns:
        List of article dicts with title, description, source, published_at, url.
    """
    if not settings.news_api_key or settings.news_api_key.startswith("your_"):
        logger.debug("NewsAPI key not configured, skipping.")
        if status is not None:
            status["state"] = "not_configured"
        return []

    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 10,
        "apiKey": settings.news_api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        if data.get("status") == "error" or not isinstance(data.get("articles", []), list):
            raise ValueError("NewsAPI returned an invalid response")

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "source": art.get("source", {}).get("name", ""),
                "published_at": art.get("publishedAt", ""),
                "url": art.get("url", ""),
                "content": art.get("content", ""),
            })
        if status is not None:
            status["state"] = "available" if articles else "empty"
        return articles

    except Exception as exc:
        logger.error("NewsAPI fetch failed for '{}' ({})", query, type(exc).__name__)
        if status is not None:
            status["state"] = "error"
        return []


async def fetch_rss_feeds(*, status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Fetch latest articles from RSS feeds (Economic Times, Moneycontrol)."""
    articles = []

    for source, url in RSS_FEEDS.items():
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            feed = feedparser.parse(resp.text)
            if feed.bozo and not feed.entries:
                raise ValueError("RSS feed could not be parsed")
            for entry in feed.entries[:15]:
                articles.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "source": source,
                    "published_at": entry.get("published", ""),
                    "url": entry.get("link", ""),
                    "content": entry.get("summary", ""),
                })

            logger.debug("Fetched {} articles from {}.", len(feed.entries[:15]), source)
            if status is not None:
                status[source] = "partial" if feed.bozo else ("available" if feed.entries else "empty")

        except Exception as exc:
            logger.error("RSS fetch failed for {} ({})", source, type(exc).__name__)
            if status is not None:
                status[source] = "error"

    return articles


async def fetch_nse_announcements(
    symbol: str = "", *, status: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch corporate announcements from NSE India.

    Args:
        symbol: Optional filter by symbol.
    """
    try:
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with httpx.AsyncClient(timeout=15.0, headers=NSE_BASE_HEADERS) as client:
            # First hit NSE homepage to get cookies
            await client.get("https://www.nseindia.com")
            resp = await client.get(NSE_ANNOUNCEMENTS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        announcements = []
        for item in data[:20]:
            announcements.append({
                "title": item.get("desc", ""),
                "symbol": item.get("symbol", ""),
                "source": "NSE",
                "published_at": item.get("an_dt", ""),
                "url": item.get("attchmntFile", ""),
                "content": item.get("desc", ""),
            })

        if status is not None:
            status["state"] = "available" if announcements else "empty"
        return announcements

    except Exception as exc:
        logger.error("NSE announcements fetch failed ({})", type(exc).__name__)
        if status is not None:
            status["state"] = "error"
        return []


async def fetch_all_news(symbol: str = "") -> list[dict[str, Any]]:
    """Aggregate news from all sources for a symbol.

    Args:
        symbol: The stock symbol to search for.

    Returns:
        Combined list of news articles from all sources.
    """
    tasks = [
        fetch_rss_feeds(),
    ]

    if symbol:
        tasks.append(fetch_news_api(f"{symbol} NSE stock India"))
        tasks.append(fetch_nse_announcements(symbol))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    for result in results:
        if isinstance(result, list):
            all_articles.extend(result)
        elif isinstance(result, Exception):
            logger.error("News fetch error: {}", result)

    # Sort by published_at (newest first)
    all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return all_articles


def _publication_time(value: Any, source: str) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    parsed = None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            for pattern in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
                try:
                    parsed = datetime.strptime(text, pattern)
                    break
                except ValueError:
                    continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        if source.upper() != "NSE":
            return None
        parsed = IST.localize(parsed)
    return parsed.astimezone(timezone.utc)


def _deduplicate_news(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for article in articles:
        url = str(article.get("url") or "").strip()
        key = ("url", url) if url else (
            "title", str(article.get("source") or "").strip().casefold(),
            str(article.get("title") or "").strip().casefold(),
            str(article.get("published_at") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return sorted(unique, key=lambda article: article.get("published_at") or "", reverse=True)


async def fetch_news_snapshot(watchlist: list[str]) -> dict[str, Any]:
    """Fetch one shared, timestamped news snapshot for a scan."""
    semaphore = asyncio.Semaphore(4)

    async def limited(request: Awaitable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        async with semaphore:
            return await request

    symbols = list(dict.fromkeys(watchlist))
    details: dict[str, dict[str, str]] = {"rss": {}}
    sources: list[tuple[str, str | None, Awaitable[list[dict[str, Any]]]]] = [
        ("rss", None, limited(fetch_rss_feeds(status=details["rss"]))),
    ]
    for symbol in symbols:
        details[f"newsapi:{symbol}"] = {}
        details[f"nse:{symbol}"] = {}
        sources.extend([
            (f"newsapi:{symbol}", symbol, limited(fetch_news_api(
                f"{symbol} NSE stock India", status=details[f"newsapi:{symbol}"],
            ))),
            (f"nse:{symbol}", symbol, limited(fetch_nse_announcements(
                symbol, status=details[f"nse:{symbol}"],
            ))),
        ])
    results = await asyncio.gather(*(item[2] for item in sources), return_exceptions=True)
    observed = datetime.now(timezone.utc)
    observed_at = observed.isoformat()
    general: list[dict[str, Any]] = []
    specific: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    statuses: dict[str, str] = {}
    errors: list[str] = []
    excluded_future = 0

    for (name, symbol, _), result in zip(sources, results):
        if isinstance(result, BaseException):
            message = f"{name} failed ({type(result).__name__})"
            logger.error("[NewsSnapshot] {}", message)
            errors.append(message)
            statuses[name] = "error"
            continue
        states = details[name]
        if name == "rss" and states:
            statuses.update({f"rss:{source}": status for source, status in states.items()})
            statuses[name] = "partial" if any(status in {"error", "partial"} for status in states.values()) else (
                "available" if result else "empty"
            )
            for source, status in states.items():
                if status in {"error", "partial"}:
                    errors.append(f"rss:{source} {status}")
        else:
            statuses[name] = states.get("state", "available" if result else "empty_or_unavailable")
            if statuses[name] == "error":
                errors.append(f"{name} failed")
        for item in result:
            if not isinstance(item, dict):
                message = f"{name} returned a non-object article"
                logger.warning("[NewsSnapshot] {}", message)
                errors.append(message)
                continue
            article = dict(item)
            raw_timestamp = article.get("published_at", "")
            published = _publication_time(raw_timestamp, str(article.get("source") or ""))
            if published and published > observed:
                excluded_future += 1
                continue
            article["published_at"] = published.isoformat() if published else ""
            article["raw_published_at"] = str(raw_timestamp or "")
            article["publication_time_status"] = "known" if published else "unknown"
            article["observed_at"] = observed_at
            article["provider"] = name
            article["queried_symbol"] = symbol
            if symbol is None:
                general.append(article)
            else:
                specific[symbol].append(article)

    general = _deduplicate_news(general)
    by_symbol = {
        symbol: _deduplicate_news(general + specific[symbol])
        for symbol in symbols
    }
    all_articles = _deduplicate_news(
        general + [article for symbol in symbols for article in specific[symbol]]
    )
    if excluded_future:
        logger.warning("[NewsSnapshot] Excluded {} future-dated articles.", excluded_future)
    return {
        "as_of": observed_at,
        "general": general,
        "by_symbol": by_symbol,
        "articles": all_articles,
        "source_status": statuses,
        "excluded_future": excluded_future,
        "errors": errors,
    }


def format_news_for_prompt(articles: list[dict[str, Any]], max_articles: int = 10) -> str:
    """Format news articles for LLM prompt consumption."""
    if not articles:
        return "No recent news articles found."

    lines = []
    for i, art in enumerate(articles[:max_articles], 1):
        lines.append(
            f"{i}. [{art.get('source', 'Unknown')}] {art.get('title', 'No title')}\n"
            f"   {art.get('description', '')[:200]}\n"
            f"   Published: {art.get('published_at', 'Unknown')}"
        )
    return "\n\n".join(lines)
