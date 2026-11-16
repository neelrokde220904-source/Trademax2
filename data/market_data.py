"""Historical OHLCV fetcher — Angel One getCandleData + yfinance fallback."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, TypedDict

import pandas as pd
import pytz
import yfinance as yf
from loguru import logger

from broker.angel_client import angel_client
from broker.candle_rate_limiter import candle_rate_limiter
from config.instruments import get_token
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# Angel One interval constants
INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

# Max candles per request (Angel One limit)
MAX_CANDLES = {
    "ONE_MINUTE": 400,
    "THREE_MINUTE": 400,
    "FIVE_MINUTE": 400,
    "TEN_MINUTE": 400,
    "FIFTEEN_MINUTE": 400,
    "THIRTY_MINUTE": 400,
    "ONE_HOUR": 400,
    "ONE_DAY": 2000,
}


class HistoryProvenance(TypedDict):
    provider: Literal["angel_one", "yahoo_finance", "unavailable"]
    symbol: str
    exchange: str
    interval: str
    requested_days: int
    retrieved_at: str
    status: Literal["available", "unavailable"]
    adjustment: str
    corporate_actions: str
    fallback_reason: str | None
    returned_rows: int


def _record_history_provenance(
    frame: pd.DataFrame, *, provider: Literal["angel_one", "yahoo_finance", "unavailable"],
    symbol: str, exchange: str, interval: str, days: int,
    fallback_reason: str | None = None,
) -> pd.DataFrame:
    available = not frame.empty
    provenance: HistoryProvenance = {
        "provider": provider,
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "requested_days": days,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "status": "available" if available else "unavailable",
        "adjustment": "unadjusted_requested" if available and provider == "yahoo_finance" else "unverified",
        "corporate_actions": (
            "provider_reported_zero_unverified" if available and provider == "yahoo_finance"
            else "unverified"
        ),
        "fallback_reason": fallback_reason,
        "returned_rows": len(frame),
    }
    frame.attrs["history_provenance"] = provenance
    return frame


def _fallback_ohlcv(
    symbol: str, interval: str, days: int, exchange: str, reason: str,
) -> pd.DataFrame:
    if exchange != "NSE":
        logger.error(
            "Historical fallback unavailable for {}:{}; NSE equity data cannot represent this segment.",
            exchange, symbol,
        )
        frame = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return _record_history_provenance(
            frame, provider="unavailable", symbol=symbol, exchange=exchange,
            interval=interval, days=days, fallback_reason=reason,
        )
    return _record_history_provenance(
        _fetch_yfinance(symbol, interval, days), provider="yahoo_finance",
        symbol=symbol, exchange=exchange, interval=interval, days=days,
        fallback_reason=reason,
    )


def fetch_ohlcv(
    symbol: str,
    interval: str = "15m",
    days: int = 30,
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a symbol.

    Args:
        symbol: NSE trading symbol (e.g. 'RELIANCE')
        interval: '1m', '5m', '15m', '1h', '1d', etc.
        days: Number of days of history
        exchange: 'NSE' or 'NFO'

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    angel_interval = INTERVAL_MAP.get(interval)
    if not angel_interval:
        logger.warning("Unknown Angel interval '{}'; evaluating historical fallback.", interval)
        return _fallback_ohlcv(symbol, interval, days, exchange, "unsupported_interval")

    token = get_token(symbol, exchange)
    if not token:
        logger.warning("Token not found for {}:{}; evaluating historical fallback.", exchange, symbol)
        return _fallback_ohlcv(symbol, interval, days, exchange, "unresolved_token")

    try:
        now = datetime.now(IST)
        to_date = now.strftime("%Y-%m-%d %H:%M")
        from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d 09:15")

        candles = angel_client.get_candle_data(
            exchange=exchange,
            symbol_token=token,
            interval=angel_interval,
            from_date=from_date,
            to_date=to_date,
        )

        if not candles:
            logger.warning("No candle data from Angel One for {}:{}; evaluating fallback.", exchange, symbol)
            return _fallback_ohlcv(symbol, interval, days, exchange, "empty_broker_response")

        df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)

        logger.info("Fetched {} candles for {} ({}).", len(df), symbol, interval)
        return _record_history_provenance(
            df, provider="angel_one", symbol=symbol, exchange=exchange,
            interval=interval, days=days,
        )

    except Exception as exc:
        logger.exception("Angel One candle fetch failed for {}:{}: {}. Evaluating fallback.", exchange, symbol, exc)
        return _fallback_ohlcv(symbol, interval, days, exchange, "broker_error")


def _fetch_yfinance(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fallback: fetch data from yfinance (NSE symbols need .NS suffix)."""
    try:
        yf_symbol = f"{symbol}.NS"
        yf_interval = interval if interval != "15m" else "15m"

        # yfinance limits: 1m=7 days, 5m/15m/30m=60 days, 1h=730 days, 1d=unlimited
        period_map = {
            "1m": min(days, 7),
            "5m": min(days, 60),
            "15m": min(days, 60),
            "30m": min(days, 60),
            "1h": min(days, 730),
            "1d": days,
        }
        actual_days = period_map.get(interval, days)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            period=f"{actual_days}d", interval=yf_interval, auto_adjust=False, actions=True,
        )

        if df.empty:
            logger.warning("yfinance returned no data for {}.", yf_symbol)
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        action_columns = ["Dividends", "Stock Splits"]
        if not set(action_columns).issubset(df.columns):
            raise ValueError("Yahoo history lacks corporate-action fields")
        actions = df[action_columns].apply(pd.to_numeric, errors="coerce")
        if actions.isna().any().any() or (actions != 0).any().any():
            raise ValueError("Yahoo history has invalid or unmodeled corporate actions")

        df = df.reset_index()
        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})

        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info("[yfinance] Fetched {} candles for {}.", len(df), yf_symbol)
        return df

    except Exception as exc:
        logger.exception("yfinance fetch failed for {}: {}", symbol, exc)
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def fetch_multiple(
    symbols: list[str],
    interval: str = "15m",
    days: int = 30,
    exchange: str = "NSE",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols.

    Returns:
        Dict mapping symbol → DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        result[sym] = fetch_ohlcv(sym, interval=interval, days=days, exchange=exchange)
    return result


def fetch_index_data(index_token: str, interval: str = "15m", days: int = 30) -> pd.DataFrame:
    """Fetch index (Nifty50, BankNifty) OHLCV data."""
    try:
        now = datetime.now(IST)
        angel_interval = INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE")
        candles = angel_client.get_candle_data(
            exchange="NSE",
            symbol_token=index_token,
            interval=angel_interval,
            from_date=(now - timedelta(days=days)).strftime("%Y-%m-%d 09:15"),
            to_date=now.strftime("%Y-%m-%d %H:%M"),
        )
        if candles:
            df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.sort_values("datetime").reset_index(drop=True)
    except Exception as exc:
        logger.error("Index data fetch failed for token {}: {}", index_token, exc)

    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def fetch_india_vix() -> tuple[float, str]:
    """Fetch today's VIX from Angel One, then a fresh Yahoo intraday quote."""
    now = datetime.now(IST)
    token = get_token("India VIX") or settings.india_vix_token
    try:
        angel_data = fetch_index_data(token, interval="1d", days=3)
        if not angel_data.empty and {"datetime", "close"}.issubset(angel_data.columns):
            latest = angel_data.iloc[-1]
            timestamp = pd.Timestamp(latest["datetime"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(IST)
            else:
                timestamp = timestamp.tz_convert(IST)
            value = float(latest["close"])
            if timestamp.date() == now.date() and 0 < value <= 200:
                return value, "angel_one"
        logger.warning("Angel One India VIX is stale or invalid; trying Yahoo fallback.")
    except Exception as exc:
        logger.warning("Angel One India VIX failed; trying Yahoo fallback: {}", exc)

    try:
        history = yf.Ticker("^INDIAVIX").history(period="1d", interval="1m")
        if history.empty or "Close" not in history.columns:
            raise ValueError("Yahoo Finance returned no intraday India VIX quote")
        timestamp = pd.Timestamp(history.index[-1])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(IST)
        else:
            timestamp = timestamp.tz_convert(IST)
        age = now - timestamp.to_pydatetime()
        value = float(history["Close"].iloc[-1])
        if timestamp.date() != now.date() or age < timedelta(seconds=-60) or age > timedelta(minutes=20):
            raise ValueError("Yahoo Finance India VIX quote is not fresh for today's session")
        if not pd.notna(value) or not 0 < value <= 200:
            raise ValueError("Yahoo Finance returned an invalid India VIX level")
        logger.warning("Using fresh Yahoo Finance India VIX fallback (quote age {:.1f} minutes).", age.total_seconds() / 60)
        return value, "yfinance"
    except Exception as exc:
        logger.error("Fresh Yahoo Finance India VIX fallback failed: {}", exc)
        raise RuntimeError("No fresh India VIX value is available") from exc


DAILY_CANDLE_CHUNK_DAYS = 1900  # Angel One allows up to 2000 days per ONE_DAY request.
RATE_LIMIT_BACKOFF_SECONDS = (2.0, 4.0, 8.0, 16.0)
MAX_DAILY_GAP_DAYS = 10  # NSE has never been shut longer than this; larger gaps mean missing data.


def _is_rate_limited(error: Exception) -> bool:
    return "exceeding access rate" in str(error).lower()


def _daily_candle_chunk(
    index_token: str, chunk_start: date, chunk_end: date, rate_limit_backoff: tuple[float, ...],
) -> list[list[Any]]:
    """One ONE_DAY request; retries only on rate limiting and raises on any other failure."""
    params = {
        "exchange": "NSE",
        "symboltoken": index_token,
        "interval": "ONE_DAY",
        # Daily candles are stamped 00:00 IST, so bounds must span whole days.
        "fromdate": f"{chunk_start:%Y-%m-%d} 00:00",
        "todate": f"{chunk_end:%Y-%m-%d} 23:59",
    }
    for backoff in (*rate_limit_backoff, None):
        try:
            api = angel_client.api
            response = candle_rate_limiter.call(api.getCandleData, params)
            if not response or not response.get("status"):
                raise RuntimeError(
                    f"getCandleData failed for {index_token} {chunk_start}..{chunk_end}: "
                    f"{(response or {}).get('errorcode')} {(response or {}).get('message')}"
                )
            return response.get("data") or []
        except Exception as exc:
            if backoff is None or not _is_rate_limited(exc):
                raise
            logger.warning("Angel candle rate limit for {}; retrying in {:.0f}s", index_token, backoff)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def fetch_index_daily_history(
    index_token: str, start: date, end: date, *, pause_seconds: float = 1.0,
    rate_limit_backoff: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Fetch completed daily index candles in API-sized chunks.

    Raises when a request fails, nothing is returned, or the series has a gap
    longer than ``MAX_DAILY_GAP_DAYS`` so models are never trained on holes.
    """
    if end < start:
        raise ValueError("end must not be before start")
    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=DAILY_CANDLE_CHUNK_DAYS), end)
        candles = _daily_candle_chunk(index_token, chunk_start, chunk_end, rate_limit_backoff)
        if candles:
            frames.append(pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"]))
        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end and pause_seconds > 0:
            time.sleep(pause_seconds)
    if not frames:
        raise RuntimeError(f"No daily candles returned for index token {index_token} between {start} and {end}")
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    gaps = df["datetime"].diff().dt.days
    if len(df) > 1 and gaps.max() > MAX_DAILY_GAP_DAYS:
        at = df.loc[gaps.idxmax(), "datetime"]
        raise RuntimeError(
            f"Daily candles for {index_token} have a {int(gaps.max())}-day gap ending {at:%Y-%m-%d}"
        )
    return df
