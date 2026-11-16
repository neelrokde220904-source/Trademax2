"""Dashboard backend — FastAPI server with SSE real-time updates."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal

import pytz
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from broker.portfolio_tracker import portfolio_tracker
from backtester.costs import calculate_indian_transaction_cost
from config.settings import PROJECT_ROOT, settings
from data.nse_calendar import is_nse_holiday
from database.db import get_db, init_db
from database.models import DailyPnL, Signal, Trade
from notifications.outbox import undelivered_critical
from operations.build_info import RUNTIME_BUILD, RUNTIME_PROCESS_ID
from operations.capabilities import current_capabilities
from risk.dead_mans_switch import dead_mans_switch
from config.llm_runtime import mode_status, select_mode, selected_profile
from agents.llm_client import usage_summary

IST = pytz.timezone("Asia/Kolkata")
LOCAL_ORIGINS = {
    f"http://{host}:{port}"
    for host in ("localhost", "127.0.0.1", "[::1]")
    for port in (3000, 3001, 5173, settings.dashboard_port)
}

SPARKLINE_POINTS = 400
SPARKLINE_MAX_TICK_AGE_SECONDS = 120
_price_history: dict[str, deque[tuple[str, float]]] = {}


def _sample_watchlist_prices(now: datetime | None = None) -> None:
    """Keep one last-traded price per minute per token for dashboard sparklines (display only)."""
    from broker.websocket_feed import ws_feed

    now = now or datetime.now(IST)
    for token, tick in ws_feed.get_all_ticks().items():
        try:
            ltp = float(tick.get("ltp") or 0)
            stamp = datetime.fromisoformat(tick["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if ltp <= 0 or not 0 <= (now - stamp).total_seconds() <= SPARKLINE_MAX_TICK_AGE_SECONDS:
            continue
        minute = stamp.astimezone(IST).strftime("%Y-%m-%dT%H:%M")
        series = _price_history.setdefault(token, deque(maxlen=SPARKLINE_POINTS))
        if series and series[-1][0][:10] != minute[:10]:
            series.clear()
        if series and series[-1][0] == minute:
            series[-1] = (minute, ltp)
        elif not series or series[-1][0] < minute:
            series.append((minute, ltp))


def _market_session(now: datetime | None = None) -> dict[str, Any]:
    """NSE cash-market session state in IST, using the configured holiday calendar."""
    now = now or datetime.now(IST)
    minutes = now.hour * 60 + now.minute
    if now.weekday() >= 5:
        state, label = "weekend", "Closed — weekend"
    elif is_nse_holiday(now.date()):
        state, label = "holiday", "Closed — NSE holiday"
    elif minutes < 9 * 60:
        state, label = "pre_market", "Closed — opens 09:15 IST"
    elif minutes < 9 * 60 + 15:
        state, label = "pre_open", "Pre-open auction"
    elif minutes < 15 * 60 + 30:
        state, label = "open", "Market open"
    else:
        state, label = "closed", "Closed for the day"
    return {
        "state": state,
        "label": label,
        "is_open": state == "open",
        "entry_window_open": state == "open" and 9 * 60 + 30 <= minutes < 14 * 60 + 45,
    }


async def _dashboard_heartbeat() -> None:
    """Publish a compact status event for connected dashboards every five seconds."""
    while True:
        try:
            _sample_watchlist_prices()
        except Exception as exc:
            logger.debug("Dashboard price sampling skipped: {}", type(exc).__name__)
        await broadcast_event("system_status", _system_status())
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start and cleanly cancel the dashboard's non-trading status publisher."""
    init_db()
    selected_profile()
    task = asyncio.create_task(_dashboard_heartbeat())
    try:
        yield
    finally:
        analysis = getattr(app.state, "analysis_task", None)
        if analysis and not analysis.done():
            analysis.cancel()
            try:
                await analysis
            except asyncio.CancelledError:
                pass
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="India AI Trader", version="1.1.0", lifespan=lifespan)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(LOCAL_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])


# ---- MODELS ----

class TradeResponse(BaseModel):
    id: int
    symbol: str
    action: str
    price: float
    quantity: int
    strategy: str | None = None
    status: str
    pnl: float | None = None
    created_at: str


class SignalResponse(BaseModel):
    id: int
    symbol: str
    signal: str
    confidence: float | None = None
    strategy: str | None = None
    reasoning: str | None = None
    created_at: str
    outcome: str | None = None
    block_reason: str | None = None
    quantity: int | None = None
    price: float | None = None
    analysis_only: bool | None = None
    review_contract: dict[str, Any] | None = None
    holding_period: str | None = None


class PortfolioResponse(BaseModel):
    positions: list[dict[str, Any]]
    available_capital: float
    total_value: float
    daily_pnl: float
    gross_daily_pnl: float
    transaction_costs: float
    position_count: int


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "halt", "reset_halt", "reset_drawdown"]
    reviewed: bool = False


class CashFlowStatementImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csv_text: str = Field(min_length=1, max_length=1_000_000)
    reviewed: bool = Field(strict=True)


class LLMModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["claude", "economy"]
    expected_revision: int = Field(ge=0, strict=True)


class ResearchComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_ids: list[str] = Field(min_length=2, max_length=10)
    acknowledge_model_cost: bool = Field(strict=True)


def _require_local_control(request: Request) -> None:
    if (not request.client or request.client.host not in ("127.0.0.1", "::1")
            or request.headers.get("origin") not in LOCAL_ORIGINS
            or request.headers.get("x-dashboard-control") != "1"
            or request.headers.get("content-type", "").split(";")[0] != "application/json"):
        raise HTTPException(status_code=403, detail="Dashboard control requires a trusted local origin.")


# ---- SSE EVENT STREAM ----

_sse_clients: list[asyncio.Queue] = []


def _system_status() -> dict[str, Any]:
    """Return UI-safe operational state without exposing credentials or secrets."""
    from broker.angel_client import angel_client
    from scheduler.market_scheduler import market_scheduler
    from config.universe import active_universe_status

    dms = dead_mans_switch.get_status()
    portfolio = portfolio_tracker.snapshot()
    return {
        "mode": "PAPER" if settings.paper_trading else "LIVE",
        "paper_trading": settings.paper_trading,
        "runtime_build": RUNTIME_BUILD,
        "process_id": RUNTIME_PROCESS_ID,
        "universe": active_universe_status(),
        "scheduler_running": market_scheduler.is_running,
        "trading_paused": market_scheduler.is_paused,
        "broker_configured": angel_client.has_credentials,
        "broker_authenticated": angel_client.is_authenticated,
        "dead_mans_switch": dms,
        "portfolio_risk": {
            "daily_loss_pct": portfolio_tracker.daily_loss_pct,
            "daily_loss_limit_pct": settings.max_daily_loss_pct,
            "daily_loss_halted": portfolio_tracker.has_hit_daily_loss_limit(),
            "daily_baseline_valid": portfolio["daily_loss_baseline_valid"],
            "max_daily_entries": settings.max_daily_entries,
            "max_daily_entries_per_symbol": settings.max_daily_entries_per_symbol,
            "symbol_reentry_cooldown_minutes": settings.symbol_reentry_cooldown_minutes,
            "portfolio_drawdown_pct": portfolio_tracker.portfolio_drawdown_pct,
            "portfolio_drawdown_limit_pct": settings.max_portfolio_drawdown_pct,
            "portfolio_drawdown_halted": portfolio_tracker.has_hit_portfolio_drawdown_limit(),
            "high_water_mark": portfolio["portfolio_high_water_mark"],
            "drawdown_equity": portfolio["drawdown_equity"],
            "net_external_cash_flows": portfolio["net_external_cash_flows"],
            "cash_flow_reconciliation_pending": portfolio_tracker.cash_flow_reconciliation_pending,
            "max_portfolio_var_pct": settings.max_portfolio_var_pct,
            "max_sector_exposure_pct": settings.max_sector_exposure_pct,
        },
        "scan_in_progress": market_scheduler.scan_in_progress,
        "last_scan": market_scheduler.last_scan,
        "scan_status": market_scheduler.scan_status(),
        "controls_persistent": True,
        "market_session": _market_session(),
        "llm": mode_status(),
        "timestamp": datetime.now(IST).isoformat(),
    }


async def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now(IST).isoformat()})
    for queue in list(_sse_clients):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def sse_generator() -> AsyncGenerator[str, None]:
    """SSE event stream generator."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_clients.append(queue)
    try:
        while True:
            data = await queue.get()
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        _sse_clients.remove(queue)


# ---- ROUTES ----

@app.get("/")
async def root():
    index = PROJECT_ROOT / "dashboard" / "frontend" / "dist" / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Dashboard is not built. Run npm run build in dashboard/frontend.")
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def get_health():
    """Expose trading-system health for the dashboard's persistent status bar."""
    return _system_status()


@app.get("/api/portfolio", response_model=PortfolioResponse)
async def get_portfolio():
    """Get current portfolio state."""
    positions = [
        {
            "symbol": k,
            "quantity": v["quantity"],
            "avg_price": v["avg_price"],
            "ltp": v.get("ltp", 0),
            "pnl": v.get("pnl", 0),
            "product": v.get("product", ""),
        }
        for k, v in portfolio_tracker.positions.items()
    ]
    return PortfolioResponse(
        positions=positions,
        available_capital=portfolio_tracker.available_capital,
        total_value=portfolio_tracker.total_value,
        daily_pnl=portfolio_tracker.daily_pnl,
        gross_daily_pnl=portfolio_tracker.gross_daily_pnl,
        transaction_costs=portfolio_tracker.daily_transaction_costs,
        position_count=portfolio_tracker.position_count,
    )


@app.get("/api/trades")
async def get_trades(limit: int = 50, offset: int = 0):
    """Get recent trades."""
    with get_db() as db:
        trades = (
            db.query(Trade)
            .order_by(Trade.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                action=t.action,
                price=t.price,
                quantity=t.quantity,
                strategy=t.strategy,
                status=t.status,
                pnl=t.pnl,
                created_at=t.created_at.isoformat() if t.created_at else "",
            )
            for t in trades
        ]


@app.get("/api/signals")
async def get_signals(limit: int = 50, offset: int = 0):
    """Get recent signals."""
    with get_db() as db:
        signals = (
            db.query(Signal)
            .order_by(Signal.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_signal_response(s) for s in signals]


def _signal_response(s: Signal) -> SignalResponse:
    try:
        details = json.loads(s.data_json) if s.data_json else {}
    except (TypeError, ValueError):
        details = {}
    if not isinstance(details, dict):
        details = {}
    try:
        quantity = int(details["quantity"]) if details.get("quantity") is not None else None
    except (TypeError, ValueError):
        quantity = None
    price = details.get("price")
    return SignalResponse(
        id=s.id,
        symbol=s.symbol,
        signal=s.signal,
        confidence=s.confidence,
        strategy=s.agent_name,
        reasoning=s.reasoning,
        created_at=s.created_at.isoformat() if s.created_at else "",
        outcome=details.get("outcome"),
        block_reason=details.get("block_reason"),
        quantity=quantity,
        price=price if isinstance(price, (int, float)) else None,
        analysis_only=details.get("analysis_only"),
        review_contract=(
            details.get("review_contract") if isinstance(details.get("review_contract"), dict) else None
        ),
        holding_period=details.get("holding_period") if isinstance(details.get("holding_period"), str) else None,
    )


@app.get("/api/pnl/daily")
async def get_daily_pnl(days: int = 30):
    """Get daily PnL history."""
    with get_db() as db:
        history = db.query(DailyPnL).order_by(DailyPnL.date.asc()).all()
        records = history[-max(days, 0):] if days > 0 else []
        if settings.paper_trading:
            costs_by_date, equity_by_date = _paper_net_equity_history(db, history)
        else:
            costs_by_date, equity_by_date = {}, {}
        return [
            {
                "date": r.date,
                "gross_pnl": r.realised_pnl + r.unrealised_pnl,
                "transaction_costs": round(costs_by_date.get(r.date, 0.0), 2),
                "pnl": r.realised_pnl + r.unrealised_pnl - costs_by_date.get(r.date, 0.0),
                "portfolio_value": equity_by_date.get(r.date, r.portfolio_value),
                "trades": r.trades_count,
                "wins": r.win_count,
                "losses": r.loss_count,
            } for r in reversed(records)
        ]


@app.get("/api/equity-curve")
async def get_equity_curve():
    """Get equity curve from daily PnL history."""
    with get_db() as db:
        records = db.query(DailyPnL).order_by(DailyPnL.date.asc()).all()
        if not settings.paper_trading:
            return [{"date": row.date, "value": row.portfolio_value} for row in records]
        _, equity_by_date = _paper_net_equity_history(db, records)
        return [{"date": row.date, "value": equity_by_date[row.date]} for row in records]


def _paper_net_equity_history(db, records: list[DailyPnL]) -> tuple[dict[str, float], dict[str, float]]:
    costs_by_date = _transaction_costs_by_date(db, records)
    equity_by_date: dict[str, float] = {}
    equity = settings.initial_capital
    for row in records:
        equity += row.realised_pnl + row.unrealised_pnl - costs_by_date.get(row.date, 0.0)
        equity_by_date[row.date] = round(equity, 2)
    return costs_by_date, equity_by_date


def _transaction_costs_by_date(db, records: list[DailyPnL]) -> dict[str, float]:
    costs_by_date: dict[str, float] = {}
    if not records:
        return costs_by_date
    start = datetime.strptime(min(row.date for row in records), "%Y-%m-%d")
    end = datetime.strptime(max(row.date for row in records), "%Y-%m-%d") + timedelta(days=1)
    fills = (
        db.query(Trade)
        .filter(
            Trade.status == "EXECUTED",
            Trade.is_paper == settings.paper_trading,
            Trade.timestamp >= start,
            Trade.timestamp < end,
        )
        .all()
    )
    for trade in fills:
        date_key = trade.timestamp.strftime("%Y-%m-%d")
        costs_by_date[date_key] = costs_by_date.get(date_key, 0.0) + (
            calculate_indian_transaction_cost(
                price=float(trade.price),
                quantity=int(trade.quantity),
                is_buy=str(trade.action).upper() == "BUY",
                is_intraday=str(trade.holding_period or "INTRADAY").upper() == "INTRADAY",
            )
        )
    return costs_by_date


@app.get("/api/agents/status")
async def get_agent_status():
    """Return a transparent readiness view, not fabricated running state."""
    agents = [
        "MarketDataAgent", "TechnicalAgent", "FundamentalAgent",
        "SentimentAgent", "OptionsAgent", "MacroAgent",
        "IndiaSpecificAgent", "RiskManagerAgent", "PortfolioManagerAgent",
    ]
    scheduler_status = "paused" if _system_status()["trading_paused"] else "ready"
    return [{"name": a, "status": scheduler_status, "last_run": None} for a in agents]


@app.get("/api/stream")
async def sse_endpoint():
    """SSE endpoint for real-time updates."""
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/control")
async def control(req: ControlRequest, request: Request):
    """Pause or resume scan scheduling; mode changes remain configuration-only."""
    from scheduler.market_scheduler import market_scheduler

    _require_local_control(request)
    if req.action in {"resume", "reset_halt", "reset_drawdown"}:
        if not settings.paper_trading:
            raise HTTPException(status_code=409, detail="Dashboard recovery is paper-only.")
        if not req.reviewed:
            raise HTTPException(status_code=409, detail="Confirm operator review before recovery.")
        if market_scheduler.scan_in_progress:
            raise HTTPException(status_code=409, detail="Wait for the active scan to finish.")
    if req.action == "pause":
        market_scheduler.pause_trading()
    elif req.action == "resume":
        try:
            market_scheduler.resume_trading()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    elif req.action == "halt":
        market_scheduler.pause_trading()
        dead_mans_switch.force_halt("Operator emergency halt")
    elif req.action == "reset_halt":
        if not market_scheduler.is_paused:
            raise HTTPException(status_code=409, detail="Pause entries before clearing the circuit breaker.")
        dead_mans_switch.resume_trading(reason="Local operator review; entries remain paused")
    elif req.action == "reset_drawdown":
        if not market_scheduler.is_paused:
            raise HTTPException(status_code=409, detail="Pause entries before resetting the drawdown latch.")
        if portfolio_tracker.cash_flow_reconciliation_pending:
            raise HTTPException(status_code=409, detail="Reconcile external cash flows before resetting drawdown.")
        if not portfolio_tracker.has_hit_portfolio_drawdown_limit():
            raise HTTPException(status_code=409, detail="The portfolio drawdown latch is not active.")
        portfolio_tracker.reset_portfolio_drawdown_halt()

    status = _system_status()
    await broadcast_event("control", {"action": req.action, "status": status})
    return {"status": "ok", "action": req.action, "system": status}


def _sleeve_status() -> list[dict[str, Any]]:
    """Which strategy sleeves are enabled; every one ships disabled."""
    from strategies.contracts import STRATEGY_CONTRACTS

    return [
        {
            "name": contract.name, "enabled": contract.is_enabled(), "flag": contract.enabled_setting,
            "horizons": list(contract.horizons), "book": contract.book or "core",
            "exit_policy": contract.default_exit_policy, "segment": contract.segment,
        }
        for contract in STRATEGY_CONTRACTS.values()
    ]


def _llm_budget_status() -> dict[str, Any]:
    from agents.llm_client import llm_budget_exceeded

    try:
        exceeded, summary = llm_budget_exceeded()
        return {**summary, "exceeded": exceeded}
    except Exception as exc:
        return {"exceeded": True, "error": type(exc).__name__, "budget_inr": settings.llm_daily_budget_inr}


@app.get("/api/operations")
async def operations_status():
    from broker.websocket_feed import ws_feed
    from notifications.telegram_notifier import telegram_notifier
    from scheduler.market_scheduler import market_scheduler
    from scheduler.foreign_markets import foreign_market_scheduler
    from config.instruments import get_token
    from risk.hmm_regime_classifier import hmm_classifier
    from config.universe import (
        UniverseUnavailableError, active_universe_status, get_active_watchlist,
    )

    reconciliation = market_scheduler.last_reconciliation
    universe = active_universe_status()
    try:
        active_watchlist = get_active_watchlist()
    except UniverseUnavailableError:
        active_watchlist = []
    return {
        "feed": ws_feed.get_status(),
        "hmm": {
            **hmm_classifier.status(),
            "scheduler_regime": market_scheduler.last_regime,
            "scheduler_vix": market_scheduler.last_vix,
        },
        "reconciliation": None if reconciliation is None else {
            key: reconciliation.get(key) for key in ("status", "checked_at", "summary")
        } | {"issue_count": len(reconciliation.get("issues", []))},
        "telegram": telegram_notifier.get_status(),
        "portfolio_risk": _system_status()["portfolio_risk"],
        "protection": market_scheduler.protection_status(),
        "universe": universe,
        "scan": {
            "in_progress": market_scheduler.scan_in_progress, "last": market_scheduler.last_scan,
            **market_scheduler.scan_status(),
        },
        "capabilities": current_capabilities(),
        "foreign_markets": foreign_market_scheduler.status(),
        "sleeves": _sleeve_status(),
        "llm_budget": _llm_budget_status(),
        "undelivered_critical_alerts": undelivered_critical(),
        "instruments": {
            "watchlist_count": len(active_watchlist),
            "unresolved": [symbol for symbol in active_watchlist if not get_token(symbol)],
        },
        "jobs": [
            {"id": job.id, "name": job.name,
             "next_run": job.next_run_time.isoformat() if getattr(job, "next_run_time", None) else None}
            for job in market_scheduler.scheduler.get_jobs()
        ],
        "limitations": [
            "Paper evaluation only; not production-certified.",
            "Full Time Machine gate is not enforced for paper entries.",
            "Paper fills are simulated against the live LTP: no queue position; partial fills only when PAPER_PARTIAL_FILL_ENABLED and depth is known.",
            "A marketable stop exit can still be delayed by a broker/API outage or a price gap.",
            "Ticks trigger stop/target exits; trailing ratchets and time exits run on the 5-minute check.",
            "New live entries are blocked: the live order lifecycle is not validated.",
            "A connected market feed is not proof of fresh data.",
        ],
    }


@app.get("/api/market/watchlist")
async def market_watchlist():
    """Latest feed prices for the watchlist with intraday sparklines; read-only and display-only."""
    from broker.websocket_feed import tick_is_fresh, ws_feed
    from config.instruments import get_token
    from config.universe import UniverseUnavailableError, get_active_watchlist

    try:
        watchlist = get_active_watchlist()
    except UniverseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    now = datetime.now(IST)
    ticks = ws_feed.get_all_ticks()
    feed = ws_feed.get_status()
    rows = []
    for symbol in watchlist:
        token = get_token(symbol)
        tick = ticks.get(token) if token else None
        row: dict[str, Any] = {"symbol": symbol, "resolved": bool(token), "has_tick": False}
        if tick:
            try:
                age = round((now - datetime.fromisoformat(tick["timestamp"])).total_seconds(), 1)
            except (KeyError, TypeError, ValueError):
                age = None
            ltp = float(tick.get("ltp") or 0) or None
            previous = float(tick.get("close") or 0) or None
            row.update(
                has_tick=ltp is not None,
                ltp=ltp,
                previous_close=previous,
                change_pct=round((ltp / previous - 1) * 100, 2) if ltp and previous else None,
                open=float(tick.get("open") or 0) or None,
                high=float(tick.get("high") or 0) or None,
                low=float(tick.get("low") or 0) or None,
                volume=tick.get("volume"),
                tick_age_seconds=age,
                fresh=bool(feed.get("connected")) and tick_is_fresh(tick, now),
            )
        series = _price_history.get(token) if token else None
        row["sparkline"] = [{"t": minute[11:], "p": price} for minute, price in series] if series else []
        rows.append(row)
    return {
        "as_of": now.isoformat(),
        "connected": feed.get("connected"),
        "fresh": feed.get("fresh"),
        "market_session": _market_session(now),
        "symbols": rows,
    }


@app.get("/api/operations/reconciliation")
async def paper_reconciliation():
    """Re-sync the paper tracker from the ledger and cross-check fills, lots, positions and stops."""
    if not settings.paper_trading:
        raise HTTPException(status_code=409, detail="Paper reconciliation is only available in paper mode.")
    from scheduler.market_scheduler import market_scheduler

    return market_scheduler.reconcile_paper()


@app.post("/api/operations/cash-flows/angelone-statement")
async def import_angelone_cash_flow_statement(
    payload: CashFlowStatementImportRequest, request: Request,
):
    """Import a reviewed normalized broker statement while paper entries are paused."""
    from scheduler.market_scheduler import market_scheduler
    from operations.cash_flow_statement import parse_angelone_cash_flow_statement

    _require_local_control(request)
    if not settings.paper_trading:
        raise HTTPException(status_code=409, detail="Cash-flow statement import is paper-only.")
    if not market_scheduler.is_paused or market_scheduler.scan_in_progress:
        raise HTTPException(status_code=409, detail="Pause entries and wait for scans to finish before importing.")
    if not payload.reviewed:
        raise HTTPException(status_code=400, detail="Confirm that this is an official broker statement containing completed transfers.")
    try:
        digest, parsed_entries = parse_angelone_cash_flow_statement(payload.csv_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    entries = [
        (entry.reference, entry.amount, entry.occurred_at)
        for entry in parsed_entries
    ]
    try:
        result = portfolio_tracker.import_angelone_cash_flow_statement(digest, entries)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    portfolio_tracker.sync_from_broker()
    marks_fresh = portfolio_tracker.update_unrealised_pnl()
    pending = portfolio_tracker.cash_flow_reconciliation_pending
    logger.info(
        "Imported Angel One cash-flow statement {}: {} new, {} duplicate transfers; reconciliation pending={}",
        digest, result["imported"], result["duplicates"], pending,
    )
    return {
        **result,
        "status": "recorded",
        "reconciliation_pending": pending,
        "portfolio_marks_fresh": marks_fresh,
        "message": (
            "Entries remain blocked until fresh marks are available and the ledger generation reconciles."
            if pending else "Statement rows were already recorded and reconciled."
        ),
    }


@app.post("/api/operations/telegram-test")
async def telegram_test(request: Request):
    _require_local_control(request)
    if not settings.paper_trading:
        raise HTTPException(status_code=409, detail="Operational tests are paper-only.")
    from notifications.telegram_notifier import telegram_notifier

    if not await telegram_notifier.test_delivery():
        raise HTTPException(status_code=409, detail=(
            "Telegram delivery failed: " + telegram_notifier.get_status()["delivery_status"]
            + ". Follow private setup in OPERATIONS.md; never paste credentials here."
        ))
    return telegram_notifier.get_status()


@app.post("/api/operations/regime")
async def classify_regime(request: Request):
    """Classify the latest completed session with the fitted HMM; read-only, places no orders."""
    _require_local_control(request)
    from risk.hmm_regime_classifier import entry_regime_block_reason, hmm_classifier

    if not hmm_classifier.is_fitted:
        raise HTTPException(status_code=409, detail=f"HMM unavailable: {hmm_classifier.load_error}")
    regime = await asyncio.to_thread(hmm_classifier.classify_latest)
    if regime.get("degraded"):
        raise HTTPException(status_code=409, detail=f"HMM classification degraded: {regime.get('reason')}")
    return {
        key: regime.get(key)
        for key in ("regime", "confidence", "panic_risk", "as_of", "model_stale")
    } | {
        "entry_block_reason": entry_regime_block_reason(regime),
        "description": (regime.get("action_directive") or {}).get("description"),
    }


_feed_control_lock = asyncio.Lock()


@app.post("/api/operations/analysis")
async def supervised_analysis(request: Request):
    _require_local_control(request)
    from scheduler.market_scheduler import market_scheduler
    from config.instruments import get_token
    from config.universe import UniverseUnavailableError, get_active_watchlist
    if not settings.paper_trading or not market_scheduler.is_paused:
        raise HTTPException(status_code=409, detail="Supervised analysis requires paused paper entries.")
    existing = getattr(app.state, "analysis_task", None)
    if market_scheduler.scan_in_progress or (existing and not existing.done()):
        raise HTTPException(status_code=409, detail="A scan is already in progress.")
    try:
        watchlist = get_active_watchlist()
    except UniverseUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    symbol = next((symbol for symbol in watchlist if get_token(symbol)), None)
    if not symbol:
        raise HTTPException(status_code=409, detail="No watchlist token available for supervised analysis.")
    app.state.analysis_task = asyncio.create_task(
        market_scheduler._run_scan(analysis_only=True, watchlist=[symbol])
    )
    return {"status": "started", "analysis_only": True, "symbol": symbol}


@app.post("/api/operations/feed-connect")
async def feed_connect(request: Request):
    _require_local_control(request)
    from broker.angel_client import angel_client
    from broker.websocket_feed import ws_feed
    from scheduler.market_scheduler import market_scheduler

    if not settings.paper_trading or not market_scheduler.is_paused:
        raise HTTPException(status_code=409, detail="Pause paper entries before testing the feed.")
    if not angel_client.has_credentials:
        raise HTTPException(status_code=409, detail="Angel One credentials are not configured.")
    if _feed_control_lock.locked():
        raise HTTPException(status_code=409, detail="Feed connection is already in progress.")
    async with _feed_control_lock:
        try:
            await market_scheduler.connect_market_feed()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Feed start failed ({})", type(exc).__name__)
            raise HTTPException(status_code=502, detail=f"Feed start failed ({type(exc).__name__}).") from exc
    return ws_feed.get_status()


@app.post("/api/operations/feed-reconnect")
async def feed_reconnect(request: Request):
    _require_local_control(request)
    from broker.websocket_feed import ws_feed
    from scheduler.market_scheduler import market_scheduler
    if not settings.paper_trading or not market_scheduler.is_paused:
        raise HTTPException(status_code=409, detail="Recovery tests require paused paper entries.")
    try:
        await asyncio.to_thread(ws_feed.request_reconnect)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ws_feed.get_status()


@app.get("/api/config")
async def get_config():
    """Get current (safe) config values."""
    from config.universe import (
        UniverseUnavailableError, active_universe_status, get_active_watchlist,
    )

    universe = active_universe_status()
    try:
        watchlist = get_active_watchlist()
    except UniverseUnavailableError:
        watchlist = []
    return {
        "paper_trading": settings.paper_trading,
        "initial_capital": settings.initial_capital,
        "max_positions": settings.max_positions,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "max_portfolio_drawdown_pct": settings.max_portfolio_drawdown_pct,
        "max_daily_entries": settings.max_daily_entries,
        "max_daily_entries_per_symbol": settings.max_daily_entries_per_symbol,
        "symbol_reentry_cooldown_minutes": settings.symbol_reentry_cooldown_minutes,
        "max_capital_per_trade_pct": settings.max_capital_per_trade_pct,
        "trading_universe_mode": settings.trading_universe_mode,
        "universe_status": universe,
        "watchlist": watchlist,
        "dashboard_port": settings.dashboard_port,
    }


@app.get("/api/llm-mode")
def get_llm_mode():
    try:
        return mode_status()
    except (SQLAlchemyError, ValueError):
        raise HTTPException(status_code=503, detail="Model selection is unavailable.") from None


@app.put("/api/llm-mode")
async def set_llm_mode(body: LLMModeRequest, request: Request):
    _require_local_control(request)
    try:
        select_mode(body.mode, body.expected_revision)
        result = mode_status()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Model selection could not be saved; refresh to confirm.") from None
    await broadcast_event("llm_mode", result)
    return result


@app.get("/api/llm-usage")
def get_llm_usage():
    try:
        return usage_summary()
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Usage accounting is unavailable.") from None


@app.get("/api/research")
def get_research():
    from research.comparison import list_comparisons
    from research.tracking import forecast_summary, list_snapshots
    try:
        return {
            "snapshots": list_snapshots(), "forecast_summary": forecast_summary(),
            "comparisons": list_comparisons(limit=5),
            "job": getattr(app.state, "research_status", {"status": "idle"}),
            "paper_trading": settings.paper_trading,
        }
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Research results are unavailable.") from None


_research_control_lock = asyncio.Lock()


@app.post("/api/research/compare")
async def start_research_comparison(body: ResearchComparisonRequest, request: Request):
    _require_local_control(request)
    if not body.acknowledge_model_cost:
        raise HTTPException(status_code=400, detail="Confirm that model API charges may apply.")
    from research.comparison import run_comparison, validate_comparison
    async with _research_control_lock:
        existing = getattr(app.state, "research_task", None)
        if existing and not existing.done():
            raise HTTPException(status_code=409, detail="A paper comparison is already running.")
        try:
            await asyncio.to_thread(validate_comparison, body.snapshot_ids)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

        async def execute():
            try:
                result = await asyncio.to_thread(run_comparison, body.snapshot_ids)
                app.state.research_status = {
                    "status": result["status"], "id": result["id"],
                    "error": result.get("error"),
                }
            except Exception as exc:
                logger.error("[Research] Comparison failed ({})", type(exc).__name__)
                app.state.research_status = {
                    "status": "failed", "error": f"Comparison failed ({type(exc).__name__})",
                }

        app.state.research_status = {"status": "running"}
        app.state.research_task = asyncio.create_task(execute())
    return {"status": "started", "paper_only": True}


@app.post("/api/research/settle")
async def settle_research(request: Request):
    _require_local_control(request)
    if not settings.paper_trading:
        raise HTTPException(status_code=409, detail="Forecast settlement requires paper-trading mode.")
    from research.tracking import settle_forecasts
    try:
        return await asyncio.to_thread(settle_forecasts)
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Forecast settlement could not be saved.") from None


_assets = PROJECT_ROOT / "dashboard" / "frontend" / "dist" / "assets"
if _assets.exists():
    app.mount("/assets", StaticFiles(directory=_assets), name="dashboard-assets")
