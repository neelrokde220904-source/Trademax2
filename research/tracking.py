"""Capture point-in-time research inputs and settle hypothetical forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import zlib
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy import case, func, update

from agents.checkpoint_serializer import TradingCheckpointSerializer
from config.settings import settings
from database.db import get_db
from database.models import ResearchForecast, ResearchSnapshot
from operations.scan_outcomes import scan_decision_outcomes

IST = ZoneInfo("Asia/Kolkata")
VALIDATED_SETTLEMENT_NOTE = "Hypothetical price movement; not trade P&L; capture and reference revalidated v2"
SNAPSHOT_FIELDS = (
    "date", "watchlist", "market_data", "market_data_provenance", "live_quotes", "india_vix", "gift_nifty",
    "hmm_regime", "ofi_signals", "active_macro_event", "news_snapshot",
    "research_fundamental_inputs", "research_options_inputs",
    "research_macro_inputs", "research_india_inputs",
)
AGENTS = {
    "technical": "technical_signals",
    "fundamental": "fundamental_signals",
    "sentiment": "sentiment_signals",
    "india_specific": "india_signals",
    "options": "options_signals",
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Research timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(IST)
    return stamp.tz_convert(timezone.utc)


def _snapshot_problems(payload: dict[str, Any], observed_at: datetime) -> list[str]:
    problems: list[str] = []
    watchlist = payload.get("watchlist")
    if not isinstance(watchlist, list):
        return ["watchlist was not captured"]
    market_data = payload.get("market_data")
    if not isinstance(market_data, dict) or not market_data:
        problems.append("market data was not captured")
    else:
        if not any(isinstance(frame, pd.DataFrame) and not frame.empty for frame in market_data.values()):
            problems.append("no non-empty market data was captured")
        for symbol, frame in market_data.items():
            if not isinstance(frame, pd.DataFrame):
                problems.append(f"market data for {symbol} is not a DataFrame")
                continue
            timestamps = frame["datetime"] if "datetime" in frame else frame.index
            for value in timestamps:
                stamp = _timestamp(value)
                if stamp is None or stamp.to_pydatetime() > observed_at:
                    problems.append(f"market data for {symbol} has an invalid or future timestamp")
                    break
    news = payload.get("news_snapshot")
    if not isinstance(news, dict) or not news.get("as_of"):
        problems.append("shared news snapshot was not captured")
    else:
        timestamp = _timestamp(news["as_of"])
        if timestamp is None or timestamp.to_pydatetime() > observed_at:
            problems.append("shared news snapshot has an invalid or future observation time")
        for article in news.get("articles", []):
            if not isinstance(article, dict):
                problems.append("shared news contains an invalid article")
                continue
            for name in ("observed_at", "published_at"):
                if article.get(name):
                    stamp = _timestamp(article[name])
                    if stamp is None or stamp.to_pydatetime() > observed_at:
                        problems.append(f"shared news has an invalid or future {name}")
                        break
    for symbol, quote in (payload.get("live_quotes") or {}).items():
        if isinstance(quote, dict) and quote.get("timestamp"):
            stamp = _timestamp(quote["timestamp"])
            if stamp is None or stamp.to_pydatetime() > observed_at:
                problems.append(f"quote for {symbol} has an invalid or future timestamp")
    fundamental = payload.get("research_fundamental_inputs")
    if not isinstance(fundamental, dict) or any(symbol not in fundamental for symbol in watchlist):
        problems.append("fundamental inputs were not captured for every symbol")
    else:
        from agents.fundamental_agent import validate_fundamentals

        for symbol in watchlist:
            try:
                validate_fundamentals(fundamental[symbol])
            except ValueError:
                problems.append(f"fundamental inputs for {symbol} are invalid or unavailable")
    options = payload.get("research_options_inputs")
    if not isinstance(options, dict) or any(
        not isinstance(options.get(symbol), dict)
        or not isinstance(options[symbol].get("chain"), pd.DataFrame)
        or "spot" not in options[symbol]
        for symbol in ("NIFTY", "BANKNIFTY")
    ):
        problems.append("options inputs were not captured")
    for field, names in (
        ("research_macro_inputs", ("global_cues", "usdinr", "crude_oil", "fii_dii")),
        ("research_india_inputs", (
            "fo_ban_list", "bulk_deals", "block_deals", "promoter_pledging",
            "circuit_filters", "is_expiry_day",
        )),
    ):
        captured = payload.get(field)
        if not isinstance(captured, dict) or any(name not in captured for name in names):
            problems.append(f"{field} was not captured")
    return problems


def _reference(state: dict[str, Any], symbol: str, observed_at: datetime) -> tuple[float | None, str | None]:
    from broker.websocket_feed import tick_is_fresh
    from data.indicators import completed_intraday_bars
    from data.nse_calendar import expected_intraday_bar_starts

    local_observed = pd.Timestamp(observed_at).tz_convert(IST).tz_localize(None)
    try:
        if not _is_verified_session(local_observed.date()):
            return None, None
    except ValueError as exc:
        logger.warning("[Research] Forecast reference unavailable: {}", exc)
        return None, None
    quote = state.get("live_quotes", {}).get(symbol, {})
    if isinstance(quote, dict):
        received = _timestamp(quote.get("timestamp"))
        exchange = _timestamp(quote.get("exchange_timestamp"))
        price = _positive_number(quote.get("ltp"))
        if (
            received is not None and exchange is not None and price is not None
            and received.to_pydatetime() <= observed_at and exchange.to_pydatetime() <= observed_at
            and tick_is_fresh(quote, observed_at)
        ):
            return price, "captured_quote"
    frame = state.get("market_data", {}).get(symbol)
    if isinstance(frame, pd.DataFrame) and not frame.empty and "close" in frame and "datetime" in frame:
        try:
            bars = completed_intraday_bars(frame, observed_at.isoformat())
            if not bars.empty:
                last = bars.iloc[-1]
                start = pd.Timestamp(last["datetime"])
                age = local_observed - (start + pd.Timedelta(minutes=15))
                price = _positive_number(last["close"])
                if (
                    start.date() == local_observed.date()
                    and start in expected_intraday_bar_starts(start.date(), 15)
                    and pd.Timedelta(0) <= age <= pd.Timedelta(minutes=30)
                    and price is not None
                ):
                    return price, "captured_bar"
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("[Research] Invalid forecast reference bars for {} ({})", symbol, type(exc).__name__)
    return None, None


def _block_reason(state: dict[str, Any], symbol: str, outcomes: dict[str, dict[str, Any]]) -> str | None:
    if not state.get("execution_enabled", True):
        return "Analysis-only run; no order was sent"
    risk = state.get("risk_assessment", {})
    if risk.get("blocked"):
        return str(risk.get("reason") or "Global risk block")
    assessment = risk.get(symbol, {})
    if isinstance(assessment, dict) and assessment.get("approved") is False:
        return str(assessment.get("rejection_reason") or "Risk not approved")
    outcome = outcomes.get(symbol)
    if outcome:
        return outcome.get("block_reason")
    return "No portfolio-manager decision"


def _forecast_rows(state: dict[str, Any], run_id: str, observed_at: datetime) -> list[ResearchForecast]:
    outcomes = {
        item["symbol"]: item for item in scan_decision_outcomes(
            state, analysis_only=not state.get("execution_enabled", True),
        )
    }
    rows = []
    signals = [(name, state.get(field, {})) for name, field in AGENTS.items()]
    macro = state.get("macro_signals", {}).get("overall")
    if isinstance(macro, dict):
        signals.append(("macro", {"NIFTY": macro}))
    debate = state.get("debate_results", {})
    if isinstance(debate, dict):
        signals.append(("bull_bear_debate", {
            symbol: {"signal": result.get("action"), "confidence": result.get("conviction_score")}
            for symbol, result in debate.items() if isinstance(result, dict)
        }))
    manager = {
        str(decision.get("symbol", "")): {
            "signal": decision.get("action"), "confidence": decision.get("confidence"),
        }
        for decision in state.get("trading_decisions", []) if isinstance(decision, dict)
    }
    signals.append(("portfolio_manager", manager))
    for agent, values in signals:
        if not isinstance(values, dict):
            continue
        for symbol, signal in values.items():
            if not isinstance(signal, dict) or not isinstance(symbol, str):
                continue
            direction = str(signal.get("signal") or "").upper()
            if direction not in {"BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "HOLD", "AVOID"}:
                continue
            confidence = signal.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None and not isinstance(confidence, bool) else None
            except (TypeError, ValueError, OverflowError):
                confidence = None
            if confidence is not None and not math.isfinite(confidence):
                confidence = None
            price, source = _reference(state, symbol, observed_at)
            status = "unscored" if direction in {"HOLD", "AVOID"} else (
                "pending" if price is not None else "missing_reference"
            )
            note = "No neutral range is defined" if status == "unscored" else (
                "No fresh captured reference price" if status == "missing_reference" else None
            )
            rows.append(ResearchForecast(
                id=str(uuid4()), run_id=run_id, observed_at=observed_at.replace(tzinfo=None),
                agent=agent, symbol=symbol[:50], direction=direction, confidence=confidence,
                reference_price=price, reference_source=source,
                block_reason=_block_reason(state, symbol, outcomes),
                status=status, settlement_note=note,
            ))
    return rows


def record_scan(state: dict[str, Any], run_id: str, *, observed_at: datetime | None = None) -> dict[str, Any]:
    """Persist one scan without copying real account positions or balances."""
    observed = _utc(observed_at or datetime.now(timezone.utc))
    payload = {field: state.get(field) for field in SNAPSHOT_FIELDS}
    payload["research_as_of"] = observed.isoformat()
    problems = _snapshot_problems(payload, observed)
    payload_type, raw = TradingCheckpointSerializer().dumps_typed(payload)
    compressed = zlib.compress(raw)
    digest = hashlib.sha256(payload_type.encode() + compressed).hexdigest()
    with get_db() as db:
        previous = db.get(ResearchSnapshot, run_id)
        if previous is not None:
            return {"snapshot_status": previous.status, "duplicate": True}
        db.add(ResearchSnapshot(
            id=run_id, observed_at=observed.replace(tzinfo=None),
            status="incomplete" if problems else "complete",
            problems_json=json.dumps(problems), payload_type=payload_type,
            payload=compressed, payload_sha256=digest,
        ))
        forecasts = _forecast_rows(state, run_id, observed)
        if problems:
            for forecast in forecasts:
                if forecast.status == "pending":
                    forecast.status = "ineligible_input"
                    forecast.settlement_note = "Capture incomplete; forecast cannot be scored safely"
        db.add_all(forecasts)
    return {
        "snapshot_status": "incomplete" if problems else "complete",
        "problems": problems, "forecasts": len(forecasts), "duplicate": False,
    }


def load_snapshot(run_id: str) -> dict[str, Any]:
    with get_db() as db:
        row = db.get(ResearchSnapshot, run_id)
        if row is None:
            raise ValueError(f"Research snapshot {run_id} does not exist")
        if row.status != "complete":
            raise ValueError(f"Research snapshot {run_id} is incomplete: {row.problems_json}")
        payload_type, compressed, digest = row.payload_type, row.payload, row.payload_sha256
        observed_at = row.observed_at.replace(tzinfo=timezone.utc)
    if hashlib.sha256(payload_type.encode() + compressed).hexdigest() != digest:
        raise ValueError(f"Research snapshot {run_id} failed its integrity check")
    payload = TradingCheckpointSerializer().loads_typed((payload_type, zlib.decompress(compressed)))
    if not isinstance(payload, dict):
        raise ValueError("Invalid research snapshot payload")
    try:
        captured_at = _utc(datetime.fromisoformat(payload["research_as_of"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Research snapshot {run_id} has an invalid observation time") from exc
    if captured_at != observed_at:
        raise ValueError(f"Research snapshot {run_id} observation time does not match its record")
    problems = _snapshot_problems(payload, observed_at)
    if problems:
        raise ValueError(f"Research snapshot {run_id} fails current validation: {'; '.join(problems)}")
    return payload


def list_snapshots(limit: int = 100) -> list[dict[str, Any]]:
    with get_db() as db:
        rows = db.query(ResearchSnapshot).order_by(ResearchSnapshot.observed_at.desc()).limit(
            min(max(limit, 1), 500)
        ).all()
        return [{
            "id": row.id, "observed_at": row.observed_at.isoformat() + "Z",
            "status": row.status, "problems": json.loads(row.problems_json),
        } for row in rows]


def _is_verified_session(day: date) -> bool:
    from data.nse_calendar import MUHURAT_SESSION, REGULAR_WEEKEND_SESSIONS, is_nse_holiday

    if day.year != 2026:
        raise ValueError("NSE forecast calendar is verified for 2026 only")
    return (
        (day.weekday() < 5 and not is_nse_holiday(day))
        or day in REGULAR_WEEKEND_SESSIONS
        or day == MUHURAT_SESSION
    )


def _next_session(day: date) -> date:
    if day.year != 2026:
        raise ValueError("NSE forecast calendar is verified for 2026 only")
    candidate = day + timedelta(days=1)
    while True:
        if _is_verified_session(candidate):
            return candidate
        candidate += timedelta(days=1)


def _daily_prices(symbol: str, days: int) -> pd.DataFrame:
    ticker = {"^NSEI": "^NSEI", "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(
        symbol, f"{symbol}.NS",
    )
    frame = yf.Ticker(ticker).history(
        period=f"{max(days, 10)}d", interval="1d", auto_adjust=False, actions=True, timeout=10,
    )
    if frame.empty:
        return pd.DataFrame()
    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    return frame.rename(columns={"date": "datetime"})


def _session_row(frame: pd.DataFrame, day: date) -> dict[str, Any] | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "close" not in frame:
        return None
    timestamps = frame["datetime"] if "datetime" in frame else frame.index
    matches = []
    for stamp, (_, row) in zip(timestamps, frame.iterrows()):
        parsed = _timestamp(stamp)
        if parsed is not None and parsed.tz_convert(IST).date() == day:
            matches.append(row.to_dict())
    return matches[0] if len(matches) == 1 else None


def _corporate_action_values(bar: dict[str, Any]) -> tuple[float, float] | None:
    values = []
    for key in ("stock_splits", "dividends"):
        value = bar.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            return None
        values.append(float(value))
    return values[0], values[1]


def settle_forecasts(
    *, now: datetime | None = None,
    loader: Callable[[str, int], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Score only the exact next NSE session after its close and available daily bar."""
    moment = _utc(now or datetime.now(timezone.utc))
    local_now = moment.astimezone(IST)
    loader = loader or _daily_prices
    errors: list[str] = []
    with get_db() as db:
        rows = db.query(ResearchForecast).filter(ResearchForecast.status == "pending").all()
        if not rows:
            return {"settled": 0, "pending": 0, "pending_reasons": {}, "errors": []}
        pending_rows = [{
            "id": row.id, "observed_at": row.observed_at, "symbol": row.symbol,
            "run_id": row.run_id, "reference_price": row.reference_price,
            "reference_source": row.reference_source, "direction": row.direction,
        } for row in rows]
    oldest = min(row["observed_at"].date() for row in pending_rows)
    days = max((local_now.date() - oldest).days + 10, 10)
    cache: dict[str, pd.DataFrame] = {}

    def prices(symbol: str) -> pd.DataFrame:
        if symbol not in cache:
            try:
                frame = loader(symbol, days)
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    message = f"Daily data unavailable for {symbol} (empty or invalid response)"
                    logger.warning("[Research] {}", message)
                    errors.append(message)
                    frame = pd.DataFrame()
                cache[symbol] = frame
            except Exception as exc:
                message = f"Daily data unavailable for {symbol} ({type(exc).__name__})"
                logger.warning("[Research] {}", message)
                errors.append(message)
                cache[symbol] = pd.DataFrame()
        return cache[symbol]

    changes = []
    snapshots: dict[str, dict[str, Any] | str] = {}
    references: dict[tuple[str, str], tuple[float | None, str | None]] = {}
    for row in pending_rows:
        observed = row["observed_at"].replace(tzinfo=timezone.utc).astimezone(IST)
        run_id = row["run_id"]
        if run_id not in snapshots:
            try:
                snapshots[run_id] = load_snapshot(run_id)
            except (KeyError, TypeError, ValueError, OverflowError, zlib.error) as exc:
                snapshots[run_id] = f"Capture cannot be revalidated: {exc}"
        snapshot = snapshots[run_id]
        if isinstance(snapshot, str):
            errors.append(snapshot)
            changes.append((row["id"], {"status": "ineligible_input", "settlement_note": snapshot}))
            continue
        if datetime.fromisoformat(snapshot["research_as_of"]).astimezone(timezone.utc) != observed.astimezone(timezone.utc):
            message = f"{row['symbol']}: forecast observation time does not match captured evidence"
            errors.append(message)
            changes.append((row["id"], {"status": "ineligible_input", "settlement_note": message}))
            continue
        reference_key = (run_id, row["symbol"])
        if reference_key not in references:
            references[reference_key] = _reference(snapshot, row["symbol"], observed)
        reference, source = references[reference_key]
        if (
            reference is None or reference != row["reference_price"]
            or source != row["reference_source"]
        ):
            message = f"{row['symbol']}: captured reference cannot be revalidated"
            errors.append(message)
            changes.append((row["id"], {"status": "ineligible_input", "settlement_note": message}))
            continue
        try:
            target = _next_session(observed.date())
        except ValueError as exc:
            message = str(exc)
            errors.append(message)
            changes.append((row["id"], {"settlement_note": message}))
            continue
        completed = datetime.combine(target, time(15, 30), IST)
        values: dict[str, Any] = {}
        if local_now < completed:
            values["settlement_note"] = f"Waiting for {target.isoformat()} NSE close"
        else:
            index_bar = _session_row(prices("^NSEI"), target)
            if index_bar is None or _positive_number(index_bar.get("close")) is None:
                values["settlement_note"] = f"NSE session evidence missing or invalid for {target.isoformat()}; check holiday calendar"
                errors.append(values["settlement_note"])
            else:
                bar = _session_row(prices(row["symbol"]), target)
                close = _positive_number(bar.get("close")) if bar else None
                actions = _corporate_action_values(bar) if bar else None
                if close is None:
                    values["settlement_note"] = f"Close missing or ambiguous for {target.isoformat()}"
                    errors.append(f"{row['symbol']}: {values['settlement_note']}")
                elif actions is None:
                    values["settlement_note"] = f"Corporate action data unavailable or invalid for {target.isoformat()}"
                    errors.append(f"{row['symbol']}: {values['settlement_note']}")
                elif any(actions):
                    values["settlement_note"] = f"Corporate action on {target.isoformat()}; manual review required"
                    errors.append(f"{row['symbol']}: {values['settlement_note']}")
                else:
                    change = (close / row["reference_price"] - 1) * 100
                    signed = change if row["direction"] in {"BUY", "STRONG_BUY"} else -change
                    values.update(
                        status="settled", settlement_session=target.isoformat(), settlement_price=close,
                        directional_return_pct=round(signed, 6), direction_correct=signed > 0,
                        settled_at=moment.replace(tzinfo=None),
                        settlement_note=VALIDATED_SETTLEMENT_NOTE,
                    )
        changes.append((row["id"], values))
    settled = 0
    with get_db() as db:
        for identifier, values in changes:
            updated = db.execute(update(ResearchForecast).where(
                ResearchForecast.id == identifier, ResearchForecast.status == "pending",
            ).values(**values))
            if values.get("status") == "settled":
                settled += updated.rowcount
        pending = db.query(ResearchForecast).filter(ResearchForecast.status == "pending").count()
        pending_reasons = {
            note or "Awaiting settlement": count
            for note, count in db.query(
                ResearchForecast.settlement_note, func.count(ResearchForecast.id),
            ).filter(ResearchForecast.status == "pending").group_by(ResearchForecast.settlement_note)
        }
    return {"settled": settled, "pending": pending, "pending_reasons": pending_reasons,
            "errors": list(dict.fromkeys(errors))}


def forecast_summary(limit: int = 100) -> dict[str, Any]:
    with get_db() as db:
        counts = dict(db.query(
            ResearchForecast.status, func.count(ResearchForecast.id),
        ).group_by(ResearchForecast.status).all())
        validated = (
            (ResearchForecast.status == "settled")
            & (ResearchForecast.settlement_note == VALIDATED_SETTLEMENT_NOTE)
        )
        verified_count = db.query(func.count(ResearchForecast.id)).filter(validated).scalar()
        counts["verified_settled"] = verified_count
        counts["legacy_settled"] = counts.get("settled", 0) - verified_count
        aggregates = db.query(
            ResearchForecast.agent,
            func.sum(case((validated, 1), else_=0)),
            func.sum(case((
                validated & ResearchForecast.direction_correct.is_(True), 1,
            ), else_=0)),
        ).group_by(ResearchForecast.agent).all()
        groups = {
            agent: {"settled": settled, "correct": correct,
                    "directional_accuracy": correct / settled if settled else None}
            for agent, settled, correct in aggregates
        }
        recent = db.query(ResearchForecast).order_by(ResearchForecast.observed_at.desc()).limit(
            min(max(limit, 1), 500)
        ).all()
        forecasts = [{
            "run_id": row.run_id, "observed_at": row.observed_at.isoformat() + "Z",
            "agent": row.agent, "symbol": row.symbol, "direction": row.direction,
            "confidence": row.confidence, "reference_price": row.reference_price,
            "reference_source": row.reference_source, "block_reason": row.block_reason,
            "status": row.status, "settlement_note": row.settlement_note,
            "settlement_session": row.settlement_session, "settlement_price": row.settlement_price,
            "directional_return_pct": row.directional_return_pct,
            "direction_correct": row.direction_correct,
            "evidence_validated": (
                row.status == "settled" and row.settlement_note == VALIDATED_SETTLEMENT_NOTE
            ),
        } for row in recent]
    return {"counts": counts, "by_agent": groups, "forecasts": forecasts,
            "horizon": "next completed NSE session close"}
