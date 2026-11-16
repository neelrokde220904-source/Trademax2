"""One walk-forward evaluation harness for strategies and exit policies.

It reuses ``BacktestEngine`` (shared exit policies, gap fills, the Indian cost model and the
live risk sizer) and reads only already-cached or free data. Every configuration that is run
is appended to ``evaluation_runs``; nothing is ever deleted. Results inform research only: no
function here promotes, enables or reconfigures a strategy.

Scope limits, stated in every report: the replay is the deterministic technical rule only
(no AI review, sentiment, news, HMM directive or portfolio VaR gate), entries fill at the next
bar's open, and a small sample cannot support a conclusion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from backtester.engine import BacktestEngine, BacktestResult, BacktestTrade
from config.settings import PROJECT_ROOT, settings
from risk.exit_policy import get_atr_exit_policy

CACHE_DIR = PROJECT_ROOT / "research" / "data_cache"
REPORT_DIR = PROJECT_ROOT / "research" / "reports"
MIN_TRADES_FOR_CONCLUSION = 100
BOOTSTRAP_SAMPLES = 2000
IST = "Asia/Kolkata"


class EvaluationUnavailable(RuntimeError):
    """The requested evaluation cannot be run on the data that is available."""


# ── data ─────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / interval / f"{symbol}.csv"


def _read_cached(symbol: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["datetime"])
    return frame if not frame.empty else None


def _fetch_free(symbol: str, interval: str) -> pd.DataFrame | None:
    """Free public history (yfinance): about 60 days of 15m bars or 2 years of daily bars."""
    import yfinance as yf

    period = "60d" if interval == "15m" else "2y"
    raw = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC").tz_convert(IST) if index.tz is None else index.tz_convert(IST)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "datetime", index)
    if interval == "1d":
        frame["datetime"] = frame["datetime"].dt.normalize()
    return frame


def load_bars(
    symbols: list[str], interval: str, *, fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    """Cached bars first; missing symbols come from the free source and are cached."""
    if interval not in {"15m", "1d"}:
        raise ValueError("interval must be '15m' or '1d'")
    bars: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _read_cached(symbol, interval)
        if frame is None and fetch:
            try:
                frame = _fetch_free(symbol, interval)
            except Exception as exc:
                logger.warning("[Evaluation] {} {} fetch failed: {}", symbol, interval, type(exc).__name__)
                frame = None
            if frame is not None:
                path = _cache_path(symbol, interval)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
        if frame is not None and not frame.empty:
            bars[symbol] = frame
    return bars


def data_digest(bars: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for symbol in sorted(bars):
        frame = bars[symbol]
        digest.update(symbol.encode())
        digest.update(str(len(frame)).encode())
        digest.update(str(frame["datetime"].iloc[0]).encode())
        digest.update(str(frame["datetime"].iloc[-1]).encode())
        digest.update(f"{float(frame['close'].iloc[-1]):.4f}".encode())
    return digest.hexdigest()


# ── strategies ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategySpec:
    name: str
    interval: str  # bar size the strategy needs
    holding_period: str


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "CompositeTrend": StrategySpec("CompositeTrend", "15m", "INTRADAY"),
    "MomentumBreakout": StrategySpec("MomentumBreakout", "1d", "SWING"),
    "MeanReversion": StrategySpec("MeanReversion", "1d", "SWING"),
    "VCPNR7Swing": StrategySpec("VCPNR7Swing", "1d", "SWING"),
}


def _key(stamp: Any) -> pd.Timestamp:
    """Timezone-independent lookup key for a bar timestamp."""
    moment = pd.Timestamp(stamp)
    return moment.tz_localize(IST).tz_convert("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")


def _daily_atr_lookup(frame: pd.DataFrame, period: int) -> pd.Series:
    """ATR of the sessions completed *before* each date (so a bar never sees its own session)."""
    sessions = frame.assign(day=_day_series(frame["datetime"])).groupby("day").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
    )
    previous_close = sessions["close"].shift(1)
    true_range = pd.concat([
        sessions["high"] - sessions["low"],
        (sessions["high"] - previous_close).abs(), (sessions["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.shift(1)  # value for day D = ATR through D-1


def _day_series(stamps: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(stamps)
    stamps = stamps.dt.tz_localize(IST) if stamps.dt.tz is None else stamps.dt.tz_convert(IST)
    return stamps.dt.tz_localize(None).dt.normalize()


def _day_key(stamp: Any) -> pd.Timestamp:
    moment = pd.Timestamp(stamp)
    moment = moment.tz_localize(IST) if moment.tzinfo is None else moment.tz_convert(IST)
    return moment.tz_localize(None).normalize()


def _composite_trend_signals(
    symbol: str, frame: pd.DataFrame, policy_name: str,
) -> dict[pd.Timestamp, dict[str, Any]]:
    """CompositeTrend entry signals per completed bar, from one causal indicator pass."""
    from data.indicators import compute_all_indicators, get_latest_signals

    policy = get_atr_exit_policy(policy_name)
    indicators = compute_all_indicators(frame)
    atr_by_day = _daily_atr_lookup(frame, policy.atr_period)
    signals: dict[pd.Timestamp, dict[str, Any]] = {}
    for position in range(len(indicators)):
        stamp = pd.Timestamp(indicators["datetime"].iloc[position])
        row = get_latest_signals(indicators.iloc[[position]])
        action = {"BUY": "BUY", "STRONG_BUY": "BUY", "SELL": "SELL", "STRONG_SELL": "SELL"}.get(row["signal"])
        rsi = row["indicators"].get("rsi_14")
        if action is None or not isinstance(rsi, (int, float)):
            continue
        if (action == "BUY" and not 40 <= rsi <= 70) or (action == "SELL" and not 30 <= rsi <= 60):
            continue
        atr = atr_by_day.get(_day_key(stamp))
        if atr is None or not math.isfinite(atr) or atr <= 0:
            continue
        signals[_key(stamp)] = {
            "symbol": symbol, "action": action, "strategy": "CompositeTrend",
            "holding_period": "INTRADAY", "atr": float(atr), "atr_exit_policy": policy_name,
        }
    return signals


def _daily_strategy_signal(name: str, symbol: str, history: pd.DataFrame, policy_name: str):
    from strategies.mean_reversion import MeanReversion
    from strategies.momentum_breakout import MomentumBreakout
    from strategies.vcp_nr7_swing import vcp_nr7_swing

    frame = history.reset_index(drop=True) if "datetime" in history.columns else history.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if name == "MomentumBreakout":
        raw = MomentumBreakout()._analyse(symbol, frame)
    elif name == "MeanReversion":
        raw = MeanReversion()._analyse(symbol, frame)
        raw = raw if raw and raw["action"] == "BUY" else None  # long-only, as in the live sleeve
    else:
        raw = vcp_nr7_swing.scan(symbol, frame)
    if not raw:
        return []
    policy = get_atr_exit_policy(policy_name)
    close = pd.to_numeric(frame["close"])
    previous = close.shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    atr = float(true_range.ewm(alpha=1 / policy.atr_period, adjust=False, min_periods=policy.atr_period).mean().iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        return []
    return [{
        "symbol": symbol, "action": raw["action"], "strategy": name, "holding_period": raw["holding_period"],
        "atr": atr, "atr_exit_policy": policy_name,
    }]


def build_strategy_fn(
    spec: StrategySpec, policy_name: str, bars: dict[str, pd.DataFrame], counters: dict[str, int],
) -> Callable[[str, pd.DataFrame, int], list[dict[str, Any]]]:
    precomputed: dict[str, dict[pd.Timestamp, dict[str, Any]]] = {}
    if spec.name == "CompositeTrend":
        for symbol, frame in bars.items():
            precomputed[symbol] = _composite_trend_signals(symbol, frame, policy_name)

    def strategy_fn(symbol: str, history: pd.DataFrame, _index: int) -> list[dict[str, Any]]:
        if spec.name == "CompositeTrend":
            signal = precomputed.get(symbol, {}).get(_key(history.index[-1]))
            signals = [dict(signal)] if signal else []
        else:
            signals = _daily_strategy_signal(spec.name, symbol, history, policy_name)
        counters["signals"] += len(signals)
        return signals

    return strategy_fn


# ── folds, metrics, bootstrap ────────────────────────────────────────────────

def walk_forward_windows(
    sessions: list[Any], folds: int, purge_bars: int, embargo_bars: int,
) -> list[tuple[Any, Any]]:
    """Consecutive out-of-sample windows after dropping purge bars at the start and embargo at the end.

    The first fold's window starts after a warm-up equal to one fold length so indicators have history.
    """
    if folds < 1 or purge_bars < 0 or embargo_bars < 0:
        raise ValueError("folds must be >= 1 and purge/embargo bars must be >= 0")
    n = len(sessions)
    block = n // (folds + 1)
    if block <= purge_bars + embargo_bars + 1:
        raise EvaluationUnavailable(f"{n} bars are too few for {folds} folds with the purge and embargo")
    windows = []
    for fold in range(folds):
        start = block * (fold + 1) + purge_bars
        end = (block * (fold + 2) if fold < folds - 1 else n) - 1 - embargo_bars
        if start < end:
            windows.append((sessions[start], sessions[end]))
    return windows


def bootstrap_expectancy_ci(
    net_pnls: list[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 20260101,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of mean net P&L per trade (deterministic)."""
    if not net_pnls:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    values = np.asarray(net_pnls, dtype=float)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _trade_r(trade: BacktestTrade) -> float | None:
    risk = abs(trade.entry_price - trade.initial_stop) * trade.quantity if trade.initial_stop else 0.0
    return trade.net_pnl / risk if risk > 0 else None


def summarize_trades(
    trades: list[BacktestTrade], *, capital: float, exposure_bars: int, total_bars: int,
    signals: int, equity_curves: list[list[float]],
) -> dict[str, Any]:
    net = [t.net_pnl for t in trades]
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value <= 0]
    reasons: dict[str, int] = {}
    for trade in trades:
        reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1
    count = len(trades)
    r_values = [value for value in (_trade_r(t) for t in trades) if value is not None]
    low, high = bootstrap_expectancy_ci(net)
    max_drawdown = 0.0
    for curve in equity_curves:
        peak = None
        for value in curve:
            peak = value if peak is None else max(peak, value)
            if peak:
                max_drawdown = max(max_drawdown, (peak - value) / peak)
    mfe = [
        (max(t.highest_since_entry - t.entry_price, 0) if t.action == "BUY"
         else max(t.entry_price - t.lowest_since_entry, 0)) * t.quantity for t in trades
    ]
    mae = [
        (max(t.entry_price - t.lowest_since_entry, 0) if t.action == "BUY"
         else max(t.highest_since_entry - t.entry_price, 0)) * t.quantity for t in trades
    ]
    gross_loss = abs(sum(losses))
    return {
        "trade_count": count,
        "net_expectancy": float(np.mean(net)) if net else 0.0,
        "net_expectancy_r": float(np.mean(r_values)) if r_values else None,
        "expectancy_ci95": [low, high],
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
        "win_rate": len(wins) / count if count else 0.0,
        "net_pnl": float(sum(net)),
        "max_drawdown_pct": max_drawdown * 100,
        "exposure": exposure_bars / total_bars if total_bars else 0.0,
        "turnover": sum(t.entry_price * t.quantity + (t.exit_price or 0) * t.quantity for t in trades) / capital,
        "avg_mae": float(np.mean(mae)) if mae else 0.0,
        "avg_mfe": float(np.mean(mfe)) if mfe else 0.0,
        "exit_mix": {reason: n / count for reason, n in reasons.items()} if count else {},
        "fill_rate": count / signals if signals else 0.0,
        "signals": signals,
    }


def classify(metrics: dict[str, Any]) -> str:
    """'inconclusive' unless there are enough trades and the bootstrap interval excludes zero."""
    low, high = metrics["expectancy_ci95"]
    if metrics["trade_count"] < MIN_TRADES_FOR_CONCLUSION or low <= 0 <= high:
        return "inconclusive"
    return "positive" if low > 0 else "negative"


# ── orchestration ────────────────────────────────────────────────────────────

def config_digest(**parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def run_walk_forward_evaluation(
    strategy: str, policy: str, universe: list[str], start: str, end: str, capital: float,
    folds: int = 3, purge_bars: int = 8, embargo_bars: int = 8, *,
    bars: dict[str, pd.DataFrame] | None = None, fetch: bool = True, record: bool = True,
) -> dict[str, Any]:
    """Evaluate one strategy with one exit policy over consecutive out-of-sample folds."""
    spec = STRATEGY_SPECS.get(strategy)
    if spec is None:
        raise EvaluationUnavailable(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGY_SPECS)}")
    get_atr_exit_policy(policy)  # unknown policy names raise
    if capital <= 0:
        raise ValueError("capital must be positive")
    bars = bars if bars is not None else load_bars(universe, spec.interval, fetch=fetch)
    bars = {symbol: frame for symbol, frame in bars.items() if symbol in universe}
    if not bars:
        raise EvaluationUnavailable("no cached or free data is available for the universe")

    window_start, window_end = pd.Timestamp(start, tz=IST), pd.Timestamp(end, tz=IST) + pd.Timedelta(days=1)
    sliced = {}
    for symbol, frame in bars.items():
        stamps = pd.to_datetime(frame["datetime"])
        stamps = stamps.dt.tz_localize(IST) if stamps.dt.tz is None else stamps.dt.tz_convert(IST)
        keep = frame[stamps < window_end]  # keep pre-window history for indicators
        if len(keep) >= 60:
            sliced[symbol] = keep.reset_index(drop=True)
    if not sliced:
        raise EvaluationUnavailable("the data has too little history for the requested window")
    sessions = sorted({
        (ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST))
        for frame in sliced.values() for ts in pd.to_datetime(frame["datetime"])
        if (ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)) >= window_start
    })
    windows = walk_forward_windows(sessions, folds, purge_bars, embargo_bars)
    if not windows:
        raise EvaluationUnavailable("no out-of-sample window remains after purge and embargo")

    counters = {"signals": 0}
    strategy_fn = build_strategy_fn(spec, policy, sliced, counters)
    all_trades: list[BacktestTrade] = []
    curves: list[list[float]] = []
    exposure_bars = total_bars = 0
    for window_first, window_last in windows:
        engine = BacktestEngine(initial_capital=capital, intraday_rules=spec.holding_period == "INTRADAY")
        result: BacktestResult = engine.run(
            sliced, strategy_fn, start_date=str(window_first), end_date=str(window_last),
        )
        all_trades.extend(result.trades)
        curves.append(result.equity_curve)
        exposure_bars += engine.exposure_bars
        total_bars += engine.total_bars
    metrics = summarize_trades(
        all_trades, capital=capital, exposure_bars=exposure_bars, total_bars=total_bars,
        signals=counters["signals"], equity_curves=curves,
    )
    outcome = classify(metrics)
    report = {
        "strategy": strategy, "exit_policy": policy, "capital": capital, "start": start, "end": end,
        "folds": len(windows), "purge_bars": purge_bars, "embargo_bars": embargo_bars,
        "universe": sorted(sliced), "data_digest": data_digest(sliced), "outcome": outcome,
        "metrics": metrics,
        "limitations": [
            "Deterministic technical rule only: no AI review, sentiment, news, HMM directive or portfolio VaR gate.",
            "Entries fill at the next bar's open with modelled slippage and costs; no queue or partial fills.",
            f"Fewer than {MIN_TRADES_FOR_CONCLUSION} trades, or an interval spanning zero, is inconclusive.",
            "Free data history is short (about 60 sessions of 15-minute bars): one market regime.",
            "Trades overlap in time and across correlated symbols, so the bootstrap interval understates "
            "the real uncertainty; a non-zero interval here is not evidence of an edge or of its absence.",
            "This report cannot promote or enable any strategy.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report["config_digest"] = config_digest(
        strategy=strategy, policy=policy, universe=report["universe"], start=start, end=end,
        capital=capital, folds=folds, purge=purge_bars, embargo=embargo_bars,
        data=report["data_digest"],
    )
    if record:
        record_run(report)
    return report


def record_run(report: dict[str, Any]) -> int:
    from database.db import get_db
    from database.models import EvaluationRun

    with get_db() as db:
        row = EvaluationRun(
            strategy=report["strategy"], exit_policy=report["exit_policy"], capital=report["capital"],
            start_date=report["start"], end_date=report["end"], folds=report["folds"],
            purge_bars=report["purge_bars"], embargo_bars=report["embargo_bars"],
            universe_json=json.dumps(report["universe"]), data_digest=report["data_digest"],
            config_digest=report["config_digest"], trade_count=report["metrics"]["trade_count"],
            outcome=report["outcome"], metrics_json=json.dumps(report["metrics"], default=float),
        )
        db.add(row)
        db.flush()
        return int(row.id)


def write_report(reports: list[dict[str, Any]], name: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(json.dumps(reports, indent=2, default=float))
    return path


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    from database.db import get_db
    from database.models import EvaluationRun

    with get_db() as db:
        rows = db.query(EvaluationRun).order_by(EvaluationRun.id.desc()).limit(limit).all()
        return [
            {"id": r.id, "created_at": r.created_at.isoformat(), "strategy": r.strategy,
             "exit_policy": r.exit_policy, "capital": r.capital, "trades": r.trade_count,
             "outcome": r.outcome} for r in rows
        ]
