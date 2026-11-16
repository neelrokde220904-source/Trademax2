"""Technical indicators using pandas-ta."""

from __future__ import annotations

import os
from pathlib import Path

# pandas-ta uses Numba's on-disk cache.  Python 3.13 cannot infer a cache
# location for some installed pandas-ta builds, so provide a writable project
# location before importing it.  This affects compilation cache only, not data.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import math

import pandas as pd
import pandas_ta as ta
from loguru import logger

from data.nse_calendar import expected_intraday_bar_starts

IST = "Asia/Kolkata"


def _reference_time(as_of: str | None) -> pd.Timestamp:
    """Return the decision time as a naive IST timestamp (now unless replaying)."""
    stamp = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz=IST)
    if stamp.tzinfo is None:
        return stamp
    return stamp.tz_convert(IST).tz_localize(None)


def _completed_bars(df: pd.DataFrame, as_of: str | None, bar_minutes: int) -> pd.DataFrame:
    """Return bars that had fully closed by the decision time, with naive IST times."""
    stamps = pd.to_datetime(df["datetime"])
    if stamps.isna().any() or bar_minutes <= 0:
        raise ValueError("Intraday bars require valid timestamps and positive duration")
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(IST).dt.tz_localize(None)
    bars = df.assign(datetime=stamps)
    duplicate_rows = bars[bars["datetime"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        values = duplicate_rows.groupby("datetime")[["open", "high", "low", "close", "volume"]]
        if values.nunique(dropna=False).gt(1).any().any():
            raise ValueError("Conflicting duplicate intraday bars")
    bars = bars.drop_duplicates("datetime").sort_values("datetime", kind="stable")
    closed = bars["datetime"] + pd.Timedelta(minutes=bar_minutes) <= _reference_time(as_of)
    return bars[closed]


def completed_intraday_bars(
    df: pd.DataFrame, as_of: str | None = None, bar_minutes: int = 15
) -> pd.DataFrame:
    """Public completed-bar view used by signal generation and numeric filters."""
    if "datetime" not in df.columns:
        raise ValueError("Intraday OHLCV data requires a datetime column")
    return _completed_bars(df, as_of, bar_minutes)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute a session-anchored VWAP aligned to the input rows."""
    if df.empty:
        return pd.Series(index=df.index, dtype=float)
    volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    if "datetime" not in df.columns:
        cumulative_volume = volume.cumsum()
        return (typical_price * volume).cumsum().div(cumulative_volume.where(cumulative_volume > 0))
    timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(IST)
    ordered = pd.DataFrame({
        "row": range(len(df)),
        "session": timestamps.dt.date.to_numpy(),
        "timestamp": timestamps.to_numpy(),
        "volume": volume.to_numpy(),
        "turnover": (typical_price * volume).to_numpy(),
    }).sort_values(["session", "timestamp", "row"], kind="stable")
    cumulative_volume = ordered["volume"].groupby(ordered["session"], sort=False).cumsum()
    cumulative_turnover = ordered["turnover"].groupby(ordered["session"], sort=False).cumsum()
    ordered["vwap"] = cumulative_turnover.div(cumulative_volume.where(cumulative_volume > 0))
    result = pd.Series(index=range(len(df)), dtype=float)
    result.iloc[ordered["row"].to_numpy(dtype=int)] = ordered["vwap"].to_numpy()
    result.index = df.index
    return result


def is_vwap_chop(
    df: pd.DataFrame,
    as_of: str | None = None,
    *,
    lookback_bars: int = 8,
    band_pct: float = 0.005,
    bar_minutes: int = 15,
) -> bool:
    """Detect an extended run of completed closes inside a narrow VWAP band."""
    bars = _completed_bars(df, as_of, bar_minutes)
    today = _reference_time(as_of).date()
    current_session = bars[bars["datetime"].dt.date == today]
    if len(current_session) < lookback_bars:
        return False
    vwap = session_vwap(current_session).tail(lookback_bars)
    current_session = current_session.tail(lookback_bars)
    valid = vwap.notna() & (vwap > 0)
    if not valid.all():
        return False
    distance = (current_session["close"].reset_index(drop=True) - vwap.reset_index(drop=True)).abs()
    return bool((distance / vwap.reset_index(drop=True) <= band_pct).all())


def is_current_session_breakout(
    df: pd.DataFrame,
    as_of: str | None = None,
    *,
    lookback_bars: int = 20,
    bar_minutes: int = 15,
) -> bool:
    """Identify a completed close beyond the prior lookback high or low."""
    bars = _completed_bars(df, as_of, bar_minutes)
    today = _reference_time(as_of).date()
    if len(bars) < lookback_bars + 1 or bars.iloc[-1]["datetime"].date() != today:
        return False
    previous = bars.iloc[-lookback_bars - 1:-1]
    latest = bars.iloc[-1]
    return bool(
        latest["close"] > previous["high"].max()
        or latest["close"] < previous["low"].min()
    )


def average_daily_turnover_inr(
    df: pd.DataFrame,
    as_of: str | None = None,
    *,
    lookback_sessions: int = 20,
    minimum_sessions: int = 10,
    minimum_bars_per_session: int = 20,
    bar_minutes: int = 15,
) -> float | None:
    """Estimate full-session traded value from complete prior sessions only."""
    bars = _completed_bars(df, as_of, bar_minutes)
    today = _reference_time(as_of).date()
    prior = bars[bars["datetime"].dt.date < today].copy()
    if prior.empty:
        return None
    close = pd.to_numeric(prior["close"], errors="coerce")
    volume = pd.to_numeric(prior["volume"], errors="coerce").clip(lower=0)
    typical = (
        pd.to_numeric(prior["high"], errors="coerce")
        + pd.to_numeric(prior["low"], errors="coerce")
        + close
    ) / 3
    prior["turnover"] = typical * volume
    grouped = prior.groupby(prior["datetime"].dt.date)
    sessions = [
        float(session["turnover"].sum())
        for _, session in grouped
        if len(session) >= minimum_bars_per_session
        and session["turnover"].notna().all()
    ][-lookback_sessions:]
    if len(sessions) < minimum_sessions:
        return None
    average = sum(sessions) / len(sessions)
    return average if math.isfinite(average) and average > 0 else None


def best_quote_spread_pct(
    quote: dict | None,
    as_of: str | None = None,
    *,
    max_age_seconds: float = 30.0,
) -> float | None:
    """Return a fresh two-sided top-of-book spread as a fraction of mid-price."""
    if not isinstance(quote, dict):
        return None
    try:
        reference = _reference_time(as_of)
        if pd.isna(reference) or not math.isfinite(max_age_seconds) or max_age_seconds < 0:
            return None
        for field in ("timestamp", "exchange_timestamp"):
            stamp = pd.Timestamp(quote.get(field))
            if pd.isna(stamp) or stamp.tzinfo is None:
                return None
            stamp = stamp.tz_convert(IST).tz_localize(None)
            age_seconds = (reference - stamp).total_seconds()
            if age_seconds < -2 or age_seconds > max_age_seconds:
                return None
    except (TypeError, ValueError, OverflowError):
        return None

    def level_price(levels: object) -> float | None:
        if not isinstance(levels, list) or not levels or not isinstance(levels[0], dict):
            return None
        try:
            price = float(levels[0].get("price"))
        except (TypeError, ValueError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    bid = level_price(quote.get("best_bid"))
    ask = level_price(quote.get("best_ask"))
    if bid is None or ask is None or ask < bid:
        return None
    midpoint = (bid + ask) / 2
    return (ask - bid) / midpoint


def daily_atr(
    df: pd.DataFrame, as_of: str | None = None, length: int = 14, bar_minutes: int = 15
) -> float | None:
    """ATR from complete prior regular sessions; reject missing or malformed recent bars."""
    if df.empty or bar_minutes <= 0 or 375 % bar_minutes:
        return None
    bars = _completed_bars(df, as_of, bar_minutes)
    today = _reference_time(as_of).date()
    prior = bars[bars["datetime"].dt.date < today]
    groups = list(prior.groupby(prior["datetime"].dt.date, sort=True))
    usable = []
    for day, session in groups:
        try:
            expected = expected_intraday_bar_starts(day, bar_minutes)
        except ValueError as exc:
            logger.warning("Daily ATR unavailable: {}", exc)
            return None
        prices = session[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        complete = (
            pd.DatetimeIndex(session["datetime"]).equals(expected)
            and prices.map(math.isfinite).all().all()
            and (prices > 0).all().all()
            and (prices["high"] >= prices[["open", "low", "close"]].max(axis=1)).all()
            and (prices["low"] <= prices[["open", "high", "close"]].min(axis=1)).all()
        )
        if not complete:
            if not usable:
                continue
            logger.warning("Daily ATR unavailable: incomplete or invalid prior session {}", day)
            return None
        usable.append(session)
    if len(usable) < length + 1:
        return None
    valid_prior = pd.concat(usable)
    sessions = valid_prior.groupby(valid_prior["datetime"].dt.date).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    atr = ta.atr(sessions["high"], sessions["low"], sessions["close"], length=length)
    value = float(atr.iloc[-1]) if atr is not None and len(atr) else math.nan
    return value if math.isfinite(value) and value > 0 else None


def session_relative_volume(
    df: pd.DataFrame, as_of: str | None = None, lookback: int = 20, bar_minutes: int = 15
) -> float | None:
    """Compare completed regular-session volume over identical time slots."""
    return _same_time_relative_volume(df, as_of, lookback, bar_minutes, False, 1)


def bar_relative_volume(
    df: pd.DataFrame, as_of: str | None = None, lookback: int = 20,
    bar_minutes: int = 15, min_samples: int = 5,
) -> float | None:
    """Compare the latest completed bar with prior sessions' matching bars."""
    return _same_time_relative_volume(df, as_of, lookback, bar_minutes, True, min_samples)


def _same_time_relative_volume(
    df: pd.DataFrame, as_of: str | None, lookback: int, bar_minutes: int,
    latest_bar_only: bool, min_samples: int,
) -> float | None:
    if (
        df.empty or bar_minutes <= 0 or 375 % bar_minutes or lookback <= 0
        or min_samples <= 0 or min_samples > lookback
    ):
        return None
    bars = _completed_bars(df, as_of, bar_minutes)
    reference = _reference_time(as_of)
    if pd.isna(reference):
        return None
    today = reference.date()
    session_open = pd.Timestamp(today) + pd.Timedelta(hours=9, minutes=15)
    count = min(
        max(int((reference - session_open).total_seconds() // (bar_minutes * 60)), 0),
        375 // bar_minutes,
    )
    if count == 0:
        return None
    dates = bars["datetime"].dt.date

    def matching_volume(day, session):
        start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
        expected = pd.date_range(start, periods=count, freq=f"{bar_minutes}min")
        window = session[
            (session["datetime"] >= start)
            & (session["datetime"] < start + pd.Timedelta(minutes=count * bar_minutes))
        ]
        if not pd.DatetimeIndex(window["datetime"]).equals(expected):
            return None
        volume = pd.to_numeric(window["volume"], errors="coerce")
        if not volume.map(math.isfinite).all() or (volume < 0).any():
            return None
        return float(volume.iloc[-1] if latest_bar_only else volume.sum())

    current_volume = matching_volume(today, bars[dates == today])
    if current_volume is None:
        return None
    baseline = [
        volume
        for day, session in bars[dates < today].groupby(dates[dates < today])
        if (volume := matching_volume(day, session)) is not None
    ][-lookback:]
    if len(baseline) < min_samples:
        return None
    average = sum(baseline) / len(baseline) if baseline else 0.0
    if not math.isfinite(average) or average <= 0:
        return None
    return current_volume / average


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical indicators on an OHLCV DataFrame.

    Expects columns: datetime, open, high, low, close, volume.
    Returns the DataFrame with indicator columns appended.
    """
    if df.empty or len(df) < 30:
        logger.warning("Insufficient data ({} rows) for indicator calculation.", len(df))
        return df

    df = df.copy()

    # RSI (14)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    # Backward-compatible aliases keep indicator consumers and historical
    # notebooks stable while the canonical names remain parameterised.
    df["rsi"] = df["rsi_14"]

    # MACD (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        df["macd"] = df["MACD_12_26_9"]

    # EMA (9, 21, 50, 200)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # Bollinger Bands (20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Supertrend (7, 3.0)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3.0)
    if st is not None:
        df = pd.concat([df, st], axis=1)

    # ADX (14)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    df["vwap"] = session_vwap(df)

    # ATR (14)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr"] = df["atr_14"]

    # OBV (On Balance Volume)
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df["close"], length=14)
    if stoch_rsi is not None:
        df = pd.concat([df, stoch_rsi], axis=1)

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Extract the latest indicator values and generate composite signals.

    Returns:
        Dict with latest values and a composite signal assessment.
    """
    if df.empty:
        return {"signal": "HOLD", "confidence": 0, "indicators": {}}

    latest = df.iloc[-1]

    indicators = {}

    # RSI
    rsi = latest.get("rsi_14")
    if pd.notna(rsi):
        indicators["rsi_14"] = round(float(rsi), 2)

    # MACD
    macd_val = latest.get("MACD_12_26_9")
    macd_signal = latest.get("MACDs_12_26_9")
    macd_hist = latest.get("MACDh_12_26_9")
    if pd.notna(macd_val):
        indicators["macd"] = round(float(macd_val), 2)
        indicators["macd_signal"] = round(float(macd_signal), 2) if pd.notna(macd_signal) else None
        indicators["macd_histogram"] = round(float(macd_hist), 2) if pd.notna(macd_hist) else None

    # EMAs
    for ema_col in ["ema_9", "ema_21", "ema_50", "ema_200"]:
        val = latest.get(ema_col)
        if pd.notna(val):
            indicators[ema_col] = round(float(val), 2)

    # Supertrend
    st_col = [c for c in df.columns if c.startswith("SUPERT_")]
    st_dir_col = [c for c in df.columns if c.startswith("SUPERTd_")]
    if st_col:
        indicators["supertrend"] = round(float(latest[st_col[0]]), 2)
    if st_dir_col:
        indicators["supertrend_direction"] = int(latest[st_dir_col[0]]) if pd.notna(latest[st_dir_col[0]]) else 0

    # ADX
    adx_val = latest.get("ADX_14")
    if pd.notna(adx_val):
        indicators["adx_14"] = round(float(adx_val), 2)

    # VWAP
    vwap = latest.get("vwap")
    if pd.notna(vwap):
        indicators["vwap"] = round(float(vwap), 2)

    # ATR
    atr = latest.get("atr_14")
    if pd.notna(atr):
        indicators["atr_14"] = round(float(atr), 2)

    # Bollinger Bands
    for bb_col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
        val = latest.get(bb_col)
        if pd.notna(val):
            indicators[bb_col.lower()] = round(float(val), 2)

    # Current price
    close = float(latest["close"])
    indicators["close"] = close

    # Compute composite signal
    signal, confidence, reasoning = _compute_composite_signal(indicators, close)

    return {
        "signal": signal,
        "composite_signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "indicators": indicators,
        # Compatibility fields are intentionally duplicated at the top level
        # for lightweight consumers; authoritative values remain in indicators.
        "rsi": indicators.get("rsi_14"),
        "macd": indicators.get("macd"),
    }


def _compute_composite_signal(indicators: dict, close: float) -> tuple[str, int, str]:
    """Generate a composite BUY/SELL/HOLD signal from indicator values."""
    bullish_count = 0
    bearish_count = 0
    reasons: list[str] = []

    rsi = indicators.get("rsi_14")
    if rsi is not None:
        if rsi < 40:
            reasons.append(f"RSI below trend-momentum zone ({rsi})")
        elif rsi > 70:
            reasons.append(f"RSI above trend-momentum zone ({rsi})")

    # VWAP
    vwap = indicators.get("vwap")
    if vwap is not None:
        if close > vwap:
            bullish_count += 1
            reasons.append("Price above VWAP")
        else:
            bearish_count += 1
            reasons.append("Price below VWAP")

    # MACD crossover
    macd_hist = indicators.get("macd_histogram")
    if macd_hist is not None:
        if macd_hist > 0:
            bullish_count += 1
            reasons.append("MACD bullish")
        else:
            bearish_count += 1
            reasons.append("MACD bearish")

    # Supertrend direction
    st_dir = indicators.get("supertrend_direction")
    if st_dir is not None:
        if st_dir == 1:
            bullish_count += 1
            reasons.append("Supertrend bullish")
        elif st_dir == -1:
            bearish_count += 1
            reasons.append("Supertrend bearish")

    # ADX strength
    adx = indicators.get("adx_14")
    if adx is not None and adx > 25:
        reasons.append(f"Strong trend (ADX={adx})")

    # EMA alignment
    ema9 = indicators.get("ema_9")
    ema21 = indicators.get("ema_21")
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            bullish_count += 1
            reasons.append("EMA9 > EMA21 (bullish)")
        else:
            bearish_count += 1
            reasons.append("EMA9 < EMA21 (bearish)")

    # Determine signal
    total = bullish_count + bearish_count
    if total == 0:
        return "HOLD", 0, "Insufficient indicator data"

    if bullish_count >= 4:
        signal = "STRONG_BUY"
        confidence = min(95, 60 + bullish_count * 8)
    elif bullish_count >= 3 and bearish_count <= 1:
        signal = "BUY"
        confidence = min(85, 50 + bullish_count * 7)
    elif bearish_count >= 4:
        signal = "STRONG_SELL"
        confidence = min(95, 60 + bearish_count * 8)
    elif bearish_count >= 3 and bullish_count <= 1:
        signal = "SELL"
        confidence = min(85, 50 + bearish_count * 7)
    else:
        signal = "HOLD"
        confidence = 30

    reasoning = " | ".join(reasons)
    return signal, confidence, reasoning
