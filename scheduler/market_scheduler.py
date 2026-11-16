"""Market Scheduler — APScheduler-based IST market-hours orchestration."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from agents.base_agent import TradingState
from agents.graph import run_trading_pipeline
from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from broker.websocket_feed import tick_event_bus, ws_feed
from config.settings import settings
from data.instrument_master import download_instrument_master
from data.nse_calendar import is_nse_holiday
from notifications.telegram_notifier import telegram_notifier
from risk.dead_mans_switch import dead_mans_switch
from risk.stop_loss_manager import stop_loss_manager
from operations.control_state import ControlStateStore, control_state
from operations.scan_outcomes import (
    record_scan_decisions, scan_decision_outcomes, summarize_outcomes,
    summarize_review_rejections, summarize_risk_rejections,
)

IST = pytz.timezone("Asia/Kolkata")

# After a tick-triggered attempt that did not close the position, wait before the next one.
TICK_EXIT_RETRY_SECONDS = 5.0
MAX_WS_TOKENS = 50  # Angel One allows max 50 per connection
FILLED_STATUSES = {"EXECUTED", "FILLED", "COMPLETE", "COMPLETED"}


class MarketScheduler:
    """Schedules all trading activities around NSE market hours.

    Daily Schedule (IST):
    08:00 — Pre-market: download instrument master
    08:55 — Authenticate configured brokers and reconcile positions
    09:00 — Connect WebSocket feed
    09:00 — Slow analysis mandate (order execution disabled)
    09:30 — First executable scan (after 15-min no-trade window)
    10:00-14:30 — Adaptive scans (30m/60m/120m based on HMM regime)
    14:30 — Last scan before the intraday entry cutoff
    14:45 — Stop new intraday entries
    15:15 — Intraday square-off check
    15:45 — Post-market: settlement, reflection, and daily PnL snapshot
    16:00 — Send daily summary via Telegram
    18:00 — End-of-day cleanup
    20:00-06:00 — Overnight global cue monitoring (every 2 hours)
    """

    def __init__(self, store: ControlStateStore | None = None) -> None:
        self._store = store
        # Run briefly-late jobs (event-loop stalls, short host sleep) once instead of dropping them.
        self.scheduler = AsyncIOScheduler(
            timezone=IST, job_defaults={"misfire_grace_time": 300, "coalesce": True},
        )
        self._running = False
        self._trading_paused = store.paused() if store else True
        self.scan_in_progress = False
        self.last_scan: dict | None = None
        self._exit_lock = asyncio.Lock()
        self._squareoff_date: date | None = None
        self.last_regime: str | None = None
        self.last_vix: float | None = None
        self.last_reconciliation: dict | None = None
        self.last_protection_check: dict[str, Any] | None = None
        self.unsubscribed_positions: list[str] = []
        self._exit_outcomes: dict[str, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tick_unsubscribe: Callable[[], None] | None = None
        self._tick_guard = threading.Lock()
        self._tick_exits_in_flight: set[str] = set()
        self._tick_retry_after: dict[str, float] = {}
        self._token_symbols: tuple[tuple[str, ...], dict[str, str]] = ((), {})
        self.last_skip: dict[str, Any] | None = None
        self.last_successful_analysis: dict[str, Any] | None = None
        self._exit_reprice_attempts: dict[str, int] = {}
        self._load_scan_status()

    # ---- SCAN STATUS (persisted so a restart does not erase it) ----

    SCAN_JOB_IDS = ("scan_0930", "adaptive_scan", "scan_1430")

    def _load_scan_status(self) -> None:
        if self._store is None:
            return
        try:
            saved = self._store.read("scan_status")
        except Exception as exc:
            logger.error("[Scheduler] Could not load scan status ({})", type(exc).__name__)
            return
        if isinstance(saved, dict):
            self.last_scan = saved.get("last_scan") if isinstance(saved.get("last_scan"), dict) else None
            self.last_successful_analysis = (
                saved.get("last_successful_analysis")
                if isinstance(saved.get("last_successful_analysis"), dict) else None
            )
            self.last_skip = saved.get("last_skip") if isinstance(saved.get("last_skip"), dict) else None

    def _persist_scan_status(self) -> None:
        if self._store is None:
            return
        try:
            self._store.write("scan_status", {
                "last_scan": self.last_scan,
                "last_successful_analysis": self.last_successful_analysis,
                "last_skip": self.last_skip,
                "next_trigger": self.next_scan_trigger(),
                "next_eligible_reason": self.scan_eligibility()["reason"],
            })
        except Exception as exc:
            logger.error("[Scheduler] Could not persist scan status ({})", type(exc).__name__)

    def _record_skip(self, reason: str, detail: str = "") -> None:
        self.last_skip = {"at": datetime.now(IST).isoformat(), "reason": reason, "detail": detail}
        self._persist_scan_status()

    def _set_last_scan(self, scan: dict[str, Any]) -> None:
        self.last_scan = scan
        if scan.get("status") == "completed":
            self.last_successful_analysis = scan
        self._persist_scan_status()

    def next_scan_trigger(self) -> str | None:
        """Time of the next scan job, regardless of whether that scan would be allowed to run."""
        times = []
        for job_id in self.SCAN_JOB_IDS:
            job = self.scheduler.get_job(job_id) if self._running else None
            next_run = getattr(job, "next_run_time", None) if job else None
            if next_run is not None:
                times.append(next_run)
        return min(times).isoformat() if times else None

    def scan_eligibility(self) -> dict[str, Any]:
        """Would a trading scan run now, and if not, why not (entries-only reasons)."""
        try:
            if self._trading_paused:
                return {"will_run": False, "reason": "entries are paused"}
            if self.scan_in_progress:
                return {"will_run": False, "reason": "a scan is already in progress"}
            allowed, reason = dead_mans_switch.check_allowed()
            if not allowed:
                return {"will_run": False, "reason": f"circuit breaker: {reason}"}
            if portfolio_tracker.has_hit_portfolio_drawdown_limit():
                return {"will_run": False, "reason": "portfolio drawdown limit is active"}
            if portfolio_tracker.has_hit_daily_loss_limit():
                return {"will_run": False, "reason": "daily loss halt (or unavailable baseline) is active"}
            now = datetime.now(IST)
            if now.weekday() >= 5 or is_nse_holiday(now.date()):
                return {"will_run": False, "reason": "market holiday or weekend"}
            regime = self.last_regime
            if regime == "SYSTEMIC_PANIC" or (self.last_vix is not None and self.last_vix > 28):
                return {"will_run": False, "reason": "systemic panic regime or VIX above 28"}
        except Exception as exc:
            return {"will_run": False, "reason": f"eligibility unknown ({type(exc).__name__})"}
        return {"will_run": True, "reason": "all entry gates are open"}

    def scan_status(self) -> dict[str, Any]:
        eligibility = self.scan_eligibility()
        return {
            "last_scan": self.last_scan,
            "last_successful_analysis": self.last_successful_analysis,
            "last_skip": self.last_skip,
            "next_trigger": self.next_scan_trigger(),
            "will_run": eligibility["will_run"],
            "next_eligible_reason": eligibility["reason"],
        }

    @property
    def is_running(self) -> bool:
        """Whether APScheduler is running (used by the read-only dashboard health API)."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Whether new trading scans are manually paused while exit monitoring continues."""
        return self._trading_paused

    def pause_trading(self) -> None:
        """Block new scan cycles without disabling stop-loss and time-exit jobs."""
        self._trading_paused = True
        if self._store:
            self._store.set_paused(True)
        logger.warning("[Scheduler] New entries paused by dashboard control.")

    def resume_trading(self) -> None:
        """Allow subsequent scheduled scan cycles after a manual review."""
        if dead_mans_switch.is_halted:
            raise ValueError("Clear the circuit breaker after review before resuming entries")
        if portfolio_tracker.has_hit_portfolio_drawdown_limit():
            raise ValueError("Reset the portfolio drawdown latch after review before resuming entries")
        if settings.paper_trading:
            report = self.reconcile_paper()
            if report["status"] == "fail":
                failures = [issue["code"] for issue in report["issues"] if issue["severity"] == "fail"]
                raise ValueError(f"Paper reconciliation is failing ({', '.join(sorted(set(failures)))}); review before resuming")
        if self._store:
            self._store.set_paused(False)
        self._trading_paused = False
        logger.info("[Scheduler] New entries resumed by dashboard control.")

    def reconcile_paper(self) -> dict:
        """Sync the tracker from the paper ledger and cross-check fills, lots, positions and stops."""
        from operations.paper_reconciliation import reconcile_live_state

        self.last_reconciliation = reconcile_live_state()
        return self.last_reconciliation

    def restore_paper_state(self) -> list[str]:
        """Rebuild positions and stops from the ledger at startup; returns symbols lacking a valid stop."""
        portfolio_tracker.sync_from_broker()
        return stop_loss_manager.restore_from_positions(portfolio_tracker.positions)

    def _track_executed_trades(self, state: TradingState) -> None:
        trades = state.get("executed_orders", [])
        for trade in trades:
            if not trade.get("is_exit"):
                try:
                    stop_loss_manager.register_trade(trade)
                except ValueError as exc:
                    # Protection must never be skipped because metadata is missing.
                    logger.critical("[Scheduler] {}; registering with legacy trailing rules.", exc)
                    stop_loss_manager.register_trade(trade, restoring=True)
        if trades and settings.paper_trading:
            portfolio_tracker.sync_from_broker()
            for trade in trades:
                symbol = trade.get("symbol")
                if not trade.get("is_exit") and symbol in portfolio_tracker.positions:
                    stop_loss_manager.bind_entry_identity(symbol, portfolio_tracker.positions[symbol])
        for trade in trades:
            if trade.get("is_exit") and trade.get("symbol") not in portfolio_tracker.positions:
                stop_loss_manager.unregister_trade(trade.get("symbol"))

    def setup(self) -> None:
        """Configure all scheduled jobs."""
        # Only run Mon-Fri (day_of_week="mon-fri")
        dow = "mon-fri"

        # --- PRE-MARKET 08:00 ---
        self.scheduler.add_job(
            self._pre_market,
            CronTrigger(hour=8, minute=0, day_of_week=dow, timezone=IST),
            id="pre_market",
            name="Pre-Market Setup",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._authenticate_and_reconcile,
            CronTrigger(hour=8, minute=55, day_of_week=dow, timezone=IST),
            id="broker_auth",
            name="Broker Authentication and Reconciliation",
            replace_existing=True,
        )

        # --- WEBSOCKET CONNECT 09:00 ---
        self.scheduler.add_job(
            self._connect_ws,
            CronTrigger(hour=9, minute=0, day_of_week=dow, timezone=IST),
            id="ws_connect",
            name="WebSocket Connect",
            replace_existing=True,
        )

        # --- FIRST SCAN 09:30 ---
        self.scheduler.add_job(
            self._run_scan,
            CronTrigger(hour=9, minute=30, day_of_week=dow, timezone=IST),
            id="scan_0930",
            name="First Scan 09:30",
            replace_existing=True,
        )

        # --- ADAPTIVE SCANS 10:00-14:45 ---
        # Instead of fixed hourly scans, run every 30 min and let _adaptive_scan
        # decide whether to actually scan based on the current HMM regime / VIX.
        #   HIGH_VOL_MEAN_REVERT → scan every 30 min  (all triggers fire)
        #   NORMAL_TRENDING      → scan every 60 min  (skip :30 triggers)
        #   LOW_VOL_TREND        → scan every 120 min (skip :30 and odd-hour triggers)
        #   SYSTEMIC_PANIC       → skip all scans     (DMS will also block)
        self.scheduler.add_job(
            self._adaptive_scan,
            CronTrigger(minute="0,30", hour="10-14", day_of_week=dow, timezone=IST),
            id="adaptive_scan",
            name="Adaptive Scan 10:00-14:30",
            replace_existing=True,
        )

        # --- LAST SCAN 14:45 (always runs regardless of regime) ---
        self.scheduler.add_job(
            self._run_premarket_mandate,
            CronTrigger(hour=9, minute=0, day_of_week=dow, timezone=IST),
            id="premarket_mandate",
            name="Pre-Market Analysis Mandate",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._run_scan,
            CronTrigger(hour=14, minute=30, day_of_week=dow, timezone=IST),
            id="scan_1430",
            name="Last Scan 14:30",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._pause_intraday_entries,
            CronTrigger(hour=14, minute=45, day_of_week=dow, timezone=IST),
            id="pause_intraday_entries",
            name="Stop New Intraday Entries",
            replace_existing=True,
        )

        # Restarts after 09:00 would otherwise leave the feed down for the rest of the day.
        self.scheduler.add_job(
            self._ensure_ws,
            CronTrigger(minute="*/2", hour="9-15", day_of_week=dow, timezone=IST),
            id="ws_watchdog",
            name="WebSocket Watchdog (2m)",
            replace_existing=True,
        )

        # --- SL/TARGET CHECK every 5 min during market hours ---
        self.scheduler.add_job(
            self._check_stops,
            CronTrigger(minute="*/5", hour="9-15", day_of_week=dow, timezone=IST),
            id="check_stops",
            name="Stop-Loss Check (5m)",
            replace_existing=True,
        )

        # --- INTRADAY SQUARE-OFF 15:15 ---
        self.scheduler.add_job(
            self._intraday_squareoff,
            CronTrigger(hour=15, minute=15, day_of_week=dow, timezone=IST),
            id="squareoff",
            name="Intraday Square-Off 15:15",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._intraday_squareoff,
            CronTrigger(hour=15, minute="16-30", day_of_week=dow, timezone=IST),
            id="squareoff_retry",
            name="Retry Unresolved Intraday Square-Off",
            replace_existing=True,
        )

        # --- POST-MARKET 15:45 ---
        self.scheduler.add_job(
            self._post_market,
            CronTrigger(hour=15, minute=45, day_of_week=dow, timezone=IST),
            id="post_market",
            name="Post-Market 15:45",
            replace_existing=True,
        )

        # --- DAILY SUMMARY 16:00 ---
        self.scheduler.add_job(
            self._daily_summary,
            CronTrigger(hour=16, minute=0, day_of_week=dow, timezone=IST),
            id="daily_summary",
            name="Daily Summary 16:00",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._settle_research_forecasts,
            CronTrigger(hour=16, minute=15, day_of_week=dow, timezone=IST),
            id="research_forecasts",
            name="Research Forecast Settlement 16:15",
            replace_existing=True,
        )

        # --- END-OF-DAY 18:00 ---
        self.scheduler.add_job(
            self._end_of_day,
            CronTrigger(hour=18, minute=0, day_of_week=dow, timezone=IST),
            id="eod_cleanup",
            name="End-of-Day Cleanup 18:00",
            replace_existing=True,
        )

        # --- OVERNIGHT MONITOR 20:00, 22:00, 00:00, 02:00, 06:00 ---
        self.scheduler.add_job(
            self._overnight_check,
            CronTrigger(hour="20,22,0,2,6", minute=0, day_of_week=dow, timezone=IST),
            id="overnight_monitor",
            name="Overnight Position Monitor",
            replace_existing=True,
        )

        # Undelivered alerts are retried with backoff (attempt limits live in the outbox).
        self.scheduler.add_job(
            self._retry_alert_outbox,
            IntervalTrigger(seconds=60, timezone=IST),
            id="alert_outbox_retry",
            name="Alert Outbox Retry (60s)",
            replace_existing=True,
        )

        logger.info("[Scheduler] All jobs configured (including overnight monitor).")

    async def _retry_alert_outbox(self) -> None:
        try:
            await telegram_notifier.retry_outbox()
        except Exception as exc:
            logger.error("[Scheduler] Alert outbox retry failed ({})", type(exc).__name__)

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        self.scheduler.start()
        self._running = True
        self._subscribe_protective_ticks()
        logger.info("[Scheduler] Started.")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._tick_unsubscribe is not None:
            self._tick_unsubscribe()
            self._tick_unsubscribe = None
        if not self._running:
            return
        self.scheduler.shutdown(wait=False)
        self._running = False
        logger.info("[Scheduler] Stopped.")

    # ---- TICK-DRIVEN PROTECTION ----

    def _subscribe_protective_ticks(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[Scheduler] No running event loop; protective exits use the 5-minute check only.")
            return
        if self._tick_unsubscribe is None:
            self._tick_unsubscribe = tick_event_bus.subscribe(self._on_protective_tick)

    def _symbol_for_token(self, token: str) -> str | None:
        from config.instruments import get_token

        tracked = stop_loss_manager.tracked_symbols()
        cached_for, mapping = self._token_symbols
        if cached_for != tracked:
            mapping = {}
            for symbol in tracked:
                symbol_token = get_token(symbol)
                if symbol_token:
                    mapping[str(symbol_token)] = symbol
            self._token_symbols = (tracked, mapping)
        return mapping.get(token)

    def _on_protective_tick(self, tick: dict[str, Any]) -> None:
        """Feed-thread callback: detect a stop/target crossing and hand it to the event loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        symbol = self._symbol_for_token(str(tick.get("token") or ""))
        if symbol is None or stop_loss_manager.price_trigger(symbol, tick.get("ltp")) is None:
            return
        with self._tick_guard:
            if (
                symbol in self._tick_exits_in_flight
                or time.monotonic() < self._tick_retry_after.get(symbol, 0.0)
            ):
                return
            self._tick_exits_in_flight.add(symbol)
        try:
            asyncio.run_coroutine_threadsafe(self._run_tick_exit(symbol), loop)
        except RuntimeError:
            with self._tick_guard:
                self._tick_exits_in_flight.discard(symbol)

    async def _run_tick_exit(self, symbol: str) -> None:
        """Re-check the crossing against a fresh reference under the exit lock, then exit."""
        from broker.order_manager import _market_reference
        from config.instruments import get_token

        attempted = False
        try:
            async with self._exit_lock:
                if symbol in stop_loss_manager.pending_exits():
                    return
                token = get_token(symbol)
                reference = _market_reference(symbol, token, "NSE") if token else None
                if reference is None:
                    return
                exit_info = stop_loss_manager.price_trigger(symbol, reference["price"])
                if exit_info is None:
                    return
                attempted = True
                await self._execute_exit_signal(exit_info, source="tick")
        except Exception as exc:
            logger.exception("[Scheduler] Tick-triggered exit for {} failed.", symbol)
            self._record_exit_outcome(symbol, "tick", "ERROR", type(exc).__name__)
        finally:
            with self._tick_guard:
                self._tick_exits_in_flight.discard(symbol)
                if attempted:
                    self._tick_retry_after[symbol] = time.monotonic() + TICK_EXIT_RETRY_SECONDS

    def _record_exit_outcome(
        self, symbol: str, source: str, outcome: str, detail: str | None = None,
    ) -> None:
        self._exit_outcomes[symbol] = {
            "at": datetime.now(IST).isoformat(), "source": source,
            "outcome": outcome, "detail": detail,
        }

    def _record_protection_check(self, error: str | None) -> None:
        self.last_protection_check = {
            "at": datetime.now(IST).isoformat(), "error": error,
            **(stop_loss_manager.last_check() or {}),
        }
        try:
            if error:
                dead_mans_switch.record_failure("protective_exits", error)
            else:
                dead_mans_switch.record_success("protective_exits")
        except Exception:
            logger.exception("[Scheduler] Could not record protective-check health.")

    def protection_status(self) -> dict[str, Any]:
        """Read-only view of how each open position is currently protected."""
        from risk.stop_loss_manager import mark_age_seconds, mark_is_fresh

        now = datetime.now(IST)
        tracked = stop_loss_manager.snapshot()
        pending = stop_loss_manager.pending_exits()
        positions = []
        for symbol, pos in sorted(portfolio_tracker.positions.items()):
            stop = tracked.get(symbol)
            age = mark_age_seconds(pos, now)
            marker = pending.get(symbol)
            positions.append({
                "symbol": symbol,
                "quantity": pos.get("quantity"),
                "holding_period": pos.get("holding_period") or pos.get("product"),
                "tracked": stop is not None,
                "stop_loss": stop.get("stop_loss") if stop else None,
                "target": stop.get("target") if stop else None,
                "mark_price": pos.get("ltp"),
                "mark_source": pos.get("mark_source"),
                "mark_age_seconds": round(age, 1) if age is not None else None,
                "mark_fresh": mark_is_fresh(pos, now),
                "pending_exit": None if not marker else (
                    "unresolved_intent" if marker.startswith("UNRESOLVED:") else "submitted_order"
                ),
                "last_exit_attempt": self._exit_outcomes.get(symbol),
            })
        return {
            "tick_triggers_active": self._tick_unsubscribe is not None,
            "tick_retry_backoff_seconds": TICK_EXIT_RETRY_SECONDS,
            "last_check": self.last_protection_check,
            "unsubscribed_positions": list(self.unsubscribed_positions),
            "positions": positions,
        }

    # ---- JOB IMPLEMENTATIONS ----

    async def _settle_research_forecasts(self) -> None:
        if not settings.paper_trading:
            return
        try:
            from research.tracking import settle_forecasts
            result = await asyncio.to_thread(settle_forecasts)
            logger.info("[Research] Settled {} forecasts; {} pending.", result["settled"], result["pending"])
            if result["errors"]:
                logger.warning("[Research] Settlement data errors: {}", result["errors"])
        except Exception as exc:
            logger.error("[Research] Forecast settlement failed: {}", type(exc).__name__)

    async def _pre_market(self) -> None:
        """Download the instrument master before the 08:55 reconciliation."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if is_nse_holiday(today):
            logger.info("[Scheduler] NSE holiday today ({}) — skipping.", today)
            self._record_skip("holiday", today)
            return

        logger.info("[Scheduler] ===== PRE-MARKET START =====")
        try:
            if settings.trading_universe_mode == "nifty50":
                from config.universe import refresh_nifty50_membership

                membership = await asyncio.to_thread(refresh_nifty50_membership)
                logger.info(
                    "[Scheduler] Nifty 50 membership verified: {} symbols observed {}.",
                    len(membership.symbols), membership.observed_on,
                )
            # Download fresh instrument master
            await download_instrument_master()

            logger.info("[Scheduler] Pre-market setup complete.")
        except Exception as exc:
            logger.error("[Scheduler] Pre-market failed: {}", exc)
            await telegram_notifier.notify_error(f"Pre-market setup failed: {exc}")

    async def _authenticate_and_reconcile(self) -> None:
        """Authenticate configured brokers and sync positions before the open."""
        try:
            if settings.trademax_broker_mode in {"angel_only", "dual_active_failover"}:
                if angel_client.has_credentials:
                    angel_client.authenticate_totp()
                elif not settings.paper_trading:
                    raise RuntimeError("Angel One credentials are required by the configured broker mode")
            if settings.trademax_broker_mode in {"zerodha_only", "dual_active_failover"}:
                from broker.zerodha_client import zerodha_client

                if settings.zerodha_access_token:
                    zerodha_client.authenticate_totp()
                elif not settings.paper_trading:
                    raise RuntimeError(
                        "A daily Kite Connect access token from official OAuth login is required"
                    )
            if not settings.paper_trading:
                portfolio_tracker.sync_from_broker()
            logger.info("[Scheduler] Broker authentication/reconciliation complete.")
        except Exception as exc:
            logger.error("[Scheduler] Broker readiness failed: {}", exc)
            dead_mans_switch.record_failure("broker_api", str(exc))
            await telegram_notifier.notify_error(f"Broker readiness failed: {exc}")

    async def _run_premarket_mandate(self) -> None:
        """Run slow pre-market analysis without passing decisions to brokers."""
        try:
            from config.universe import get_active_watchlist

            mandate = await run_trading_pipeline(get_active_watchlist(), execute_orders=False)
            logger.info(
                "[Scheduler] Pre-market mandate created: {} decisions, {} errors.",
                len(mandate.get("trading_decisions", [])), len(mandate.get("errors", [])),
            )
        except Exception as exc:
            logger.error("[Scheduler] Pre-market mandate failed: {}", exc)
            dead_mans_switch.record_failure("langgraph", str(exc))

    async def _pause_intraday_entries(self) -> None:
        """Record the cutoff; graph-level order routing also enforces it."""
        logger.info("[Scheduler] New intraday entries are blocked from 14:45 IST.")

    async def _connect_ws(self) -> None:
        """Connect WebSocket at 09:00 IST; held positions are subscribed before the watchlist."""
        if not angel_client.has_credentials:
            logger.info("[Scheduler] WebSocket skipped: brokerless paper mode.")
            return
        try:
            await self.connect_market_feed()
        except Exception as exc:
            logger.error("[Scheduler] WebSocket connect failed: {}", exc)

    async def connect_market_feed(self) -> None:
        """Shared scheduled/manual connection path; failures propagate to the caller."""
        from config.instruments import get_token
        from config.universe import UniverseUnavailableError, get_active_watchlist

        held = list(portfolio_tracker.positions)
        try:
            watchlist = get_active_watchlist()
        except UniverseUnavailableError as exc:
            logger.error("[Scheduler] Watchlist unavailable ({}); subscribing held positions only.",
                         type(exc).__name__)
            watchlist = []
        token_symbols: dict[str, str] = {}
        for symbol in [*held, *watchlist]:
            token = get_token(symbol)
            if token and token not in token_symbols:
                token_symbols[token] = symbol
        tokens = list(token_symbols)[:MAX_WS_TOKENS]
        covered = {token_symbols[token] for token in tokens}
        self.unsubscribed_positions = sorted(symbol for symbol in held if symbol not in covered)
        if self.unsubscribed_positions:
            reason = (
                "No live feed for held positions "
                f"{', '.join(self.unsubscribed_positions)}; their protective exits cannot be priced"
            )
            logger.critical("[Scheduler] {}.", reason)
            await telegram_notifier.notify_error(reason)
        if not tokens:
            raise ValueError("Instrument tokens unavailable; download the instrument master first.")
        await asyncio.to_thread(ws_feed.connect)
        ws_feed.subscribe("nse_cm", tokens)
        logger.info("[Scheduler] WebSocket connected with {} tokens ({} held positions).",
                    len(tokens), len(held))

    async def _ensure_ws(self) -> None:
        """Start the feed during market hours if its connection loop is not running."""
        now = datetime.now(IST)
        if (is_nse_holiday(now.date())
                or not (9 * 60 <= now.hour * 60 + now.minute < 15 * 60 + 30)):
            return
        if ws_feed.is_running:
            self._check_feed_staleness()
            return
        logger.warning("[Scheduler] WebSocket feed not running during market hours; connecting.")
        await self._connect_ws()

    FEED_STALE_SECONDS = 120.0

    @staticmethod
    def _check_feed_staleness() -> None:
        """A connected feed that stopped delivering fresh ticks counts as a feed failure."""
        try:
            age = ws_feed.get_status().get("tick_age_seconds")
        except Exception:
            return
        if isinstance(age, (int, float)) and age > MarketScheduler.FEED_STALE_SECONDS:
            ws_feed.report_failure(f"feed stale for {age:.0f}s")

    async def _run_scan(
        self, *, analysis_only: bool = False, watchlist: list[str] | None = None,
    ) -> None:
        """Run the full AI trading pipeline."""
        if analysis_only and (not settings.paper_trading or not self._trading_paused):
            raise ValueError("Supervised analysis requires paused paper entries")
        if self._trading_paused and not analysis_only:
            logger.warning("[Scheduler] Scan skipped: new entries are manually paused.")
            self._record_skip("paused", "new entries are manually paused")
            return
        try:
            from config.universe import UniverseUnavailableError, get_active_watchlist

            active_watchlist = watchlist if watchlist is not None else get_active_watchlist()
        except UniverseUnavailableError as exc:
            logger.error("[Scheduler] Scan blocked: active trading universe unavailable ({})", type(exc).__name__)
            dead_mans_switch.record_failure("market_data", str(exc))
            self._record_skip("universe_unavailable", type(exc).__name__)
            await telegram_notifier.notify_error(
                f"Trading universe unavailable; new entries blocked ({type(exc).__name__}).",
                dedup_key="universe_unavailable",
            )
            return
        # Dead Man's Switch pre-check
        allowed, reason = dead_mans_switch.check_allowed()
        if not allowed and not analysis_only:
            logger.warning("[Scheduler] Scan blocked by DMS: {}", reason)
            self._record_skip("circuit_breaker", str(reason))
            return

        if self.scan_in_progress:
            logger.warning("[Scheduler] Scan skipped: another scan is in progress.")
            self._record_skip("scan_in_progress", "another scan is in progress")
            return
        if settings.paper_trading and not analysis_only:
            try:
                report = self.reconcile_paper()
            except Exception as exc:
                report = {"status": "fail", "issues": [{"severity": "fail", "code": "reconciliation_error", "detail": str(exc)}]}
            if report["status"] == "fail":
                codes = sorted({issue["code"] for issue in report["issues"] if issue["severity"] == "fail"})
                self.pause_trading()
                logger.critical("[Scheduler] Scan blocked and entries paused: paper reconciliation failed {}", codes)
                self._record_skip("reconciliation", ", ".join(codes))
                await telegram_notifier.notify_error(
                    f"Paper reconciliation failed ({', '.join(codes)}); entries paused for review.",
                    dedup_key="paper_reconciliation_failed",
                )
                return
            if portfolio_tracker.has_hit_portfolio_drawdown_limit():
                self.pause_trading()
                logger.critical("[Scheduler] Scan blocked: persistent portfolio drawdown limit is active.")
                self._record_skip("drawdown", "persistent portfolio drawdown limit")
                await telegram_notifier.notify_error(
                    "Portfolio drawdown limit reached; new entries are paused for operator review.",
                    dedup_key="portfolio_drawdown_halt",
                )
                return
        if not analysis_only:
            if portfolio_tracker.has_hit_daily_loss_limit():
                # Paid analysis cannot lead to an entry while the halt lasts; exits run as separate jobs.
                logger.warning("[Scheduler] Scan skipped: daily loss halt active")
                self._record_skip("daily_halt", "daily loss halt or unavailable baseline")
                return
            from agents.llm_client import llm_budget_exceeded

            try:
                over_budget, spend = await asyncio.to_thread(llm_budget_exceeded)
            except Exception as exc:
                over_budget, spend = True, {"error": type(exc).__name__}
            if over_budget:
                logger.warning("[Scheduler] Scan skipped: LLM daily budget exhausted or unmeasurable: {}", spend)
                self._record_skip("llm_budget", str(spend))
                await telegram_notifier.notify_risk_alert(
                    "Daily LLM budget reached or unmeasurable; paid scans are skipped until tomorrow. "
                    "Exits continue to run.",
                    dedup_key=f"llm_budget:{datetime.now(IST).strftime('%Y-%m-%d')}",
                )
                return
        self.scan_in_progress = True
        started = datetime.now(IST).isoformat()
        logger.info("[Scheduler] Running trading pipeline scan...")
        try:
            result = await run_trading_pipeline(
                active_watchlist, execute_orders=not analysis_only,
                on_execution_complete=self._track_executed_trades,
            )
            trades = result.get("executed_orders", [])
            logger.info("[Scheduler] Scan complete: {} orders executed.", len(trades))
            regime = result.get("hmm_regime") or {}
            self.last_regime = regime.get("regime") if not regime.get("degraded", True) else None
            vix = result.get("india_vix")
            self.last_vix = float(vix) if isinstance(vix, (int, float)) and vix > 0 else None

            degraded = self._scan_is_degraded(result)
            if not analysis_only and not degraded:
                dead_mans_switch.record_success("langgraph")
            outcomes = scan_decision_outcomes(result, analysis_only=analysis_only)
            record_scan_decisions(outcomes, scan_started_at=started, analysis_only=analysis_only)
            self._set_last_scan({
                "started_at": started, "finished_at": datetime.now(IST).isoformat(),
                "status": "degraded" if degraded else "completed",
                "orders": len(trades), "errors": len(result.get("errors", [])),
                "analysis_only": analysis_only,
                "india_vix": result.get("india_vix"),
                "hmm_regime": regime.get("regime"),
                "risk_block_reasons": summarize_risk_rejections(result),
                "review_block_reasons": summarize_review_rejections(result),
                **summarize_outcomes(outcomes),
            })

            for trade in trades:
                await telegram_notifier.notify_trade_executed(trade)

        except Exception as exc:
            self._set_last_scan({
                "started_at": started, "finished_at": datetime.now(IST).isoformat(),
                "status": "failed", "error_type": type(exc).__name__,
                "analysis_only": analysis_only,
            })
            logger.error("[Scheduler] Scan failed: {}", exc)
            dead_mans_switch.record_failure("langgraph", str(exc))
            await telegram_notifier.notify_error(f"Scan failed: {exc}")
        finally:
            self.scan_in_progress = False

    @staticmethod
    def _scan_is_degraded(result: dict[str, Any]) -> bool:
        """Router and risk rejections are valid outcomes; only other errors degrade a scan."""
        return any(
            isinstance(error, str) and not error.startswith("[OrderRouter]")
            for error in result.get("errors") or []
        )

    async def _adaptive_scan(self) -> None:
        """Regime-aware scan gate — decides whether the current 30-min slot should run."""
        now = datetime.now(IST)
        minute = now.minute
        hour = now.hour

        # Regime/VIX from the last completed scan; unknown values use the normal hourly cadence.
        regime = self.last_regime or "NORMAL_TRENDING"
        vix = self.last_vix

        # SYSTEMIC_PANIC → block all scans (the router also blocks entries, belt-and-suspenders)
        if regime == "SYSTEMIC_PANIC" or (vix is not None and vix > 28):
            logger.info(
                "[Scheduler] Adaptive scan SKIPPED — regime={}, VIX={}",
                regime, vix,
            )
            self._record_skip("regime_panic", f"regime={regime}, VIX={vix}")
            return

        # HIGH_VOL_MEAN_REVERT → every 30 min (all triggers fire)
        if regime == "HIGH_VOL_MEAN_REVERT":
            logger.info("[Scheduler] HIGH_VOL regime — scanning (every 30 min).")
            await self._run_scan()
            return

        # LOW_VOL_TREND → every 2 hours (only even-hour :00 triggers)
        if regime == "LOW_VOL_TREND":
            if minute == 0 and hour % 2 == 0:
                logger.info("[Scheduler] LOW_VOL regime — scanning (every 2 hours).")
                await self._run_scan()
            else:
                logger.debug("[Scheduler] LOW_VOL regime — skipping slot {:02d}:{:02d}.", hour, minute)
                self._record_skip("regime_slot", f"LOW_VOL_TREND skips {hour:02d}:{minute:02d}")
            return

        # NORMAL_TRENDING (default) → every 60 min (only :00 triggers)
        if minute == 0:
            logger.info("[Scheduler] NORMAL regime — scanning (every 60 min).")
            await self._run_scan()
        else:
            logger.debug("[Scheduler] NORMAL regime — skipping :30 slot.")
            self._record_skip("regime_slot", "NORMAL_TRENDING skips the :30 slot")

    async def _check_stops(self) -> None:
        """Check stop-losses, targets and time exits every 5 minutes (fallback to tick triggers)."""
        async with self._exit_lock:
            try:
                await self._reconcile_pending_exits()
                portfolio_tracker.update_unrealised_pnl()
                exits = stop_loss_manager.check_all_positions()
            except Exception as exc:
                logger.exception("[Scheduler] Stop check failed: {}", type(exc).__name__)
                self._record_protection_check(type(exc).__name__)
                return
            failure: str | None = None
            for exit_info in exits:
                try:
                    await self._execute_exit_signal(exit_info, source="schedule")
                except Exception as exc:
                    failure = failure or type(exc).__name__
                    logger.exception("[Scheduler] Protective exit for {} failed.", exit_info.get("symbol"))
                    self._record_exit_outcome(exit_info["symbol"], "schedule", "ERROR", type(exc).__name__)
            self._record_protection_check(failure)

    async def _execute_exit_signal(self, exit_info: dict[str, Any], *, source: str) -> str:
        """Submit one protective exit and settle its outcome; caller must hold the exit lock."""
        symbol = exit_info["symbol"]
        pos = portfolio_tracker.positions.get(symbol, {})
        if not pos:
            return "NO_POSITION"
        logger.info("[Scheduler] Exit signal ({}): {}", source, exit_info)
        result = self._submit_protective_exit(symbol, exit_info["exit_action"], pos)
        outcome = self._settle_exit_result(symbol, result)
        self._record_exit_outcome(symbol, source, outcome, exit_info.get("exit_type"))
        if outcome == "CLOSED":
            self._exit_reprice_attempts.pop(symbol, None)
            await telegram_notifier.notify_trade_closed({
                "symbol": symbol,
                "exit_price": pos.get("ltp", 0),
                "entry_price": pos.get("avg_price", 0),
                "pnl": pos.get("pnl", 0),
                "pnl_pct": (pos.get("pnl", 0) / (pos.get("avg_price", 1) * abs(pos.get("quantity", 1)))) * 100,
                "exit_reason": exit_info["reason"],
            })
        return outcome

    @staticmethod
    def _settle_exit_result(symbol: str, result: dict | None) -> str:
        if not result:
            return "UNRESOLVED" if symbol in stop_loss_manager.pending_exits() else "NOT_FILLED"
        status = str(result.get("status") or "").upper()
        if status in {"PENDING", "SUBMITTED", "PARTIAL"}:
            order_id = result.get("order_id")
            if order_id:
                stop_loss_manager.mark_exit_pending(symbol, str(order_id))
                return "SUBMITTED"
            dead_mans_switch.force_halt(f"Exit submission for {symbol} has no order id to reconcile")
            return "UNRESOLVED"
        if status not in FILLED_STATUSES:
            dead_mans_switch.force_halt(f"Exit for {symbol} has unconfirmed status {status or 'UNKNOWN'}")
            return "UNRESOLVED"
        portfolio_tracker.sync_from_broker()
        if symbol not in portfolio_tracker.positions:
            stop_loss_manager.unregister_trade(symbol)
            return "CLOSED"
        logger.warning(
            "[Scheduler] Exit filled only partially for {}; remaining position stays protected.", symbol,
        )
        return "PARTIALLY_FILLED"

    async def _reconcile_pending_exits(self) -> None:
        """Resolve recorded exits before allowing another order for that position."""
        pending = stop_loss_manager.pending_exits()
        if not pending:
            return
        if settings.paper_trading:
            await self._resolve_paper_exit_intents(pending)
            return
        for symbol, order_id in list(pending.items()):
            if order_id.startswith("UNRESOLVED:"):
                dead_mans_switch.force_halt(
                    f"Protective exit for {symbol} has an uncertain submission; operator reconciliation required"
                )
                pending.pop(symbol)
        if not pending:
            return
        try:
            orders = await asyncio.to_thread(angel_client.get_order_book)
        except Exception as exc:
            dead_mans_switch.force_halt(
                f"Could not reconcile pending protective exits ({type(exc).__name__})"
            )
            return

        by_id = {
            str(order.get("orderid") or order.get("order_id") or ""): order
            for order in orders
        }
        terminal = {"COMPLETE", "COMPLETED", "FILLED", "REJECTED", "CANCELLED", "CANCELED"}
        for symbol, order_id in pending.items():
            order = by_id.get(order_id)
            if order is None:
                dead_mans_switch.force_halt(
                    f"Pending protective exit for {symbol} is absent from broker order book"
                )
                continue
            status = str(order.get("orderstatus") or order.get("status") or "").upper()
            if status not in terminal:
                continue
            try:
                await asyncio.to_thread(portfolio_tracker.sync_from_broker)
            except Exception as exc:
                dead_mans_switch.force_halt(
                    f"Could not verify terminal exit for {symbol} ({type(exc).__name__})"
                )
                continue
            if symbol not in portfolio_tracker.positions:
                stop_loss_manager.unregister_trade(symbol)
                continue
            stop_loss_manager.clear_exit_pending(symbol)

    async def _resolve_paper_exit_intents(self, pending: dict[str, str]) -> None:
        """Paper fills commit atomically, so the ledger alone settles a recorded exit intent.

        Protection resumes only for remaining lots from the recorded entry identity;
        an entry halt raised by the original failure stays in place for operator review.
        """
        try:
            await asyncio.to_thread(portfolio_tracker.sync_from_broker)
        except Exception as exc:
            dead_mans_switch.force_halt(
                f"Could not verify paper exit intents against the ledger ({type(exc).__name__})"
            )
            return
        tracked = stop_loss_manager.snapshot()
        for symbol, marker in pending.items():
            position = portfolio_tracker.positions.get(symbol) or {}
            quantity = int(position.get("quantity") or 0)
            kind = "unresolved_intent" if marker.startswith("UNRESOLVED:") else "submitted_order"
            if quantity == 0:
                stop_loss_manager.unregister_trade(symbol)
                logger.warning("[Scheduler] Paper exit intent for {} settled: ledger is flat.", symbol)
                self._record_exit_outcome(symbol, "ledger", "CLOSED", kind)
                continue
            protection = tracked.get(symbol, {})
            if ("BUY" if quantity > 0 else "SELL") != protection.get("action"):
                dead_mans_switch.force_halt(
                    f"Paper position for {symbol} changed side while an exit was pending; operator review required"
                )
                self._record_exit_outcome(symbol, "ledger", "SIDE_MISMATCH", kind)
                continue
            original_ids = protection.get("entry_trade_ids") or []
            remaining_ids = position.get("entry_trade_ids") or []
            if (
                not original_ids or not remaining_ids
                or any(entry_id not in original_ids for entry_id in remaining_ids)
                or position.get("holding_period") != protection.get("holding_period")
            ):
                dead_mans_switch.force_halt(
                    f"Paper position for {symbol} has an unverifiable or changed entry identity "
                    "while an exit was pending; operator review required"
                )
                self._record_exit_outcome(symbol, "ledger", "IDENTITY_MISMATCH", kind)
                continue
            stop_loss_manager.bind_entry_identity(symbol, position)
            stop_loss_manager.clear_exit_pending(symbol)
            logger.warning(
                "[Scheduler] Paper exit intent for {} settled from the ledger ({} open); protection resumed.",
                symbol, quantity,
            )
            self._record_exit_outcome(symbol, "ledger", "PROTECTION_RESUMED", kind)

    async def _intraday_squareoff(self) -> None:
        """Square off all intraday positions at 15:15 IST."""
        async with self._exit_lock:
            today = datetime.now(IST).date()
            if self._squareoff_date == today:
                return
            logger.info("[Scheduler] Intraday square-off check...")
            if settings.paper_trading:
                await self._reconcile_pending_exits()
                # Flatten every intraday position from the ledger, including untracked or stop-hit ones.
                portfolio_tracker.sync_from_broker()
                portfolio_tracker.update_unrealised_pnl()
                failed: list[str] = []
                for symbol, pos in list(portfolio_tracker.positions.items()):
                    if str(pos.get("holding_period", "INTRADAY")).upper() != "INTRADAY":
                        continue
                    action = "SELL" if int(pos.get("quantity", 0)) > 0 else "BUY"
                    result = self._submit_protective_exit(symbol, action, pos)
                    if result and str(result.get("status") or "").upper() in {
                        "EXECUTED", "FILLED", "COMPLETE", "COMPLETED",
                    }:
                        logger.info("[Scheduler] Paper intraday square-off filled for {}.", symbol)
                    else:
                        failed.append(symbol)
                portfolio_tracker.sync_from_broker()
                remaining = [
                    symbol for symbol, pos in portfolio_tracker.positions.items()
                    if str(pos.get("holding_period", "INTRADAY")).upper() == "INTRADAY"
                ]
                if not failed and not remaining:
                    self._squareoff_date = today
                else:
                    unresolved = sorted(set(failed) | set(remaining))
                    reason = f"Paper intraday square-off unresolved for {', '.join(unresolved)}"
                    logger.critical("[Scheduler] {}. Manual review required.", reason)
                    self.pause_trading()
                    await telegram_notifier.notify_error(reason)
                return

            flatteners = []
            failures: list[str] = []
            if settings.trademax_broker_mode in {"angel_only", "dual_active_failover"}:
                flatteners.append(("Angel One", angel_client.flatten_all_intraday))
            if settings.trademax_broker_mode in {"zerodha_only", "dual_active_failover"}:
                from broker.zerodha_client import zerodha_client

                flatteners.append(("Zerodha", zerodha_client.flatten_all_intraday))
            for broker_name, flatten in flatteners:
                try:
                    orders = flatten()
                    logger.warning(
                        "[Scheduler] {} square-off submitted {} orders; reconcile fills.",
                        broker_name, len(orders),
                    )
                except Exception as exc:
                    reason = f"{broker_name} square-off failed: {type(exc).__name__}"
                    logger.critical("[Scheduler] {}. Manual position reconciliation is required.", reason)
                    dead_mans_switch.force_halt(reason)
                    failures.append(broker_name)
            try:
                portfolio_tracker.sync_from_broker()
                remaining = [
                    symbol for symbol, pos in portfolio_tracker.positions.items()
                    if str(pos.get("holding_period", pos.get("product", "INTRADAY"))).upper()
                    == "INTRADAY"
                ]
            except Exception as exc:
                logger.exception("[Scheduler] Could not verify square-off positions: {}", type(exc).__name__)
                failures.append("position verification")
                remaining = ["unknown"]
            if not failures and not remaining:
                self._squareoff_date = today
            elif remaining:
                reason = f"Intraday square-off remains unresolved for {', '.join(remaining)}"
                logger.critical("[Scheduler] {}. Reconciliation and protective monitoring continue.", reason)
                dead_mans_switch.force_halt(reason)

    def _submit_protective_exit(self, symbol: str, action: str, position: dict) -> dict | None:
        from broker.base_broker import OrderSubmissionUncertainError

        if symbol in stop_loss_manager.pending_exits():
            logger.warning("[Scheduler] Exit for {} awaits reconciliation; not resubmitting.", symbol)
            return None
        if symbol not in stop_loss_manager.snapshot():
            stop_loss_manager.register_trade({
                "symbol": symbol, "action": "BUY" if action == "SELL" else "SELL",
                "price": position.get("avg_price"), "stop_loss": position.get("stop_loss"),
                "target": position.get("target"), "atr": position.get("atr"),
                "holding_period": position.get("holding_period", "INTRADAY"),
                "entry_time": position.get("entry_time"),
                "entry_trade_ids": position.get("entry_trade_ids"),
                "exit_policy": position.get("exit_policy"),
                "atr_timeframe": position.get("atr_timeframe"),
            }, restoring=True)
        stop_loss_manager.begin_exit(symbol)
        attempts = self._exit_reprice_attempts.get(symbol, 0)
        # Each unfilled attempt widens the collar by one base step, up to the configured cap.
        collar_multiple = min(attempts + 1, settings.protective_exit_max_reprice_steps)
        try:
            result = self._place_exit_order(
                symbol, action, {**position, "_collar_multiple": collar_multiple},
            )
        except OrderSubmissionUncertainError:
            dead_mans_switch.force_halt(
                f"Protective exit for {symbol} has an uncertain submission; operator reconciliation required"
            )
            logger.exception("[Scheduler] Protective exit submission for {} is uncertain.", symbol)
            return None
        except Exception as exc:
            dead_mans_switch.force_halt(
                f"Protective exit for {symbol} failed with unresolved intent ({type(exc).__name__})"
            )
            logger.exception("[Scheduler] Protective exit for {} failed; reconcile before retry.", symbol)
            return None
        if result is None:
            stop_loss_manager.clear_exit_pending(symbol)
            self._exit_reprice_attempts[symbol] = attempts + 1
            if attempts + 1 >= settings.protective_exit_max_reprice_steps:
                logger.critical(
                    "[Scheduler] Protective exit for {} is still not filled at the widest collar "
                    "({:.2%}); monitoring continues.",
                    symbol, settings.protective_exit_collar_pct * collar_multiple,
                )
                self._fire_alert(
                    f"Protective exit for {symbol} has not filled at the widest allowed price collar. "
                    "The position stays protected and monitored; review it.",
                    dedup_key=f"exit_unfilled:{symbol}",
                )
            return None
        status = str(result.get("status") or "").upper()
        if status in {"EXECUTED", "FILLED", "COMPLETE", "COMPLETED"}:
            self._exit_reprice_attempts.pop(symbol, None)
            stop_loss_manager.clear_exit_pending(symbol)
        elif result.get("order_id"):
            stop_loss_manager.mark_exit_pending(symbol, str(result["order_id"]))
        else:
            dead_mans_switch.force_halt(f"Protective exit for {symbol} has no order id to reconcile")
        return result

    def _fire_alert(self, text: str, *, dedup_key: str, severity: str = "critical") -> None:
        """Send an alert from synchronous code without ever blocking or raising."""
        coroutine = telegram_notifier.notify_error(text, dedup_key=dedup_key, severity=severity)
        try:
            asyncio.get_running_loop().create_task(coroutine)
        except RuntimeError:
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            else:
                coroutine.close()

    @staticmethod
    def _place_exit_order(symbol: str, action: str, position: dict) -> dict | None:
        """Close a tracked position using the normal synchronous order-manager API.

        Exit calls deliberately bypass strategy selection but retain the same broker,
        paper/live mode, database audit trail, and market-hours guard as entries.
        """
        from broker.order_manager import _market_reference, place_order
        from config.instruments import get_instrument, get_token
        from risk.exit_policy import collared_exit_limit

        token = get_token(symbol)
        quantity = abs(int(position.get("quantity", 0)))
        if not token or quantity <= 0:
            logger.error("[Scheduler] Cannot exit {}: missing token or quantity.", symbol)
            return None
        reference = _market_reference(symbol, token, "NSE")
        if reference is None:
            logger.error("[Scheduler] Cannot exit {}: no fresh price reference.", symbol)
            return None
        instrument = get_instrument(symbol, "NSE")
        try:
            # Angel's instrument master quotes tick sizes in paise.
            tick_size = float(instrument["tick_size"]) / 100 if instrument else 0
            limit_price = collared_exit_limit(
                float(reference["price"]), action, tick_size,
                settings.protective_exit_collar_pct * max(1, int(position.get("_collar_multiple", 1))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("[Scheduler] Cannot price protective exit for {}: {}", symbol, exc)
            return None

        return place_order(
            symbol=symbol,
            token=token,
            action=action,
            quantity=quantity,
            order_type="LIMIT",
            price=limit_price,
            exchange="NSE",
            product_type=(
                "INTRADAY" if str(position.get("holding_period", "INTRADAY")).upper() == "INTRADAY"
                else "DELIVERY"
            ),
            strategy="risk_exit",
            reasoning="Automated stop-loss, target, or time-based exit",
            allow_requested_price_reference=False,
            is_exit=True,
            expected_entry_trade_ids=position.get("entry_trade_ids"),
        )

    async def _post_market(self) -> None:
        """Post-market tasks and settlement at 15:45 IST."""
        logger.info("[Scheduler] ===== POST-MARKET =====")
        try:
            ws_feed.disconnect()
            portfolio_tracker.sync_from_broker()
            portfolio_tracker.save_daily_snapshot()
            from agents.episodic_reflection_agent import episodic_reflection

            settlement = await asyncio.to_thread(episodic_reflection.settle_all_pending)
            logger.info("[Scheduler] Settled {} closed trades.", settlement["settled"])
            if settings.paper_trading:
                report = self.reconcile_paper()
                logger.info("[Scheduler] Paper reconciliation: {} ({})", report["status"], report["summary"])
                if report["status"] != "pass":
                    codes = sorted({issue["code"] for issue in report["issues"]})
                    await telegram_notifier.notify_error(
                        f"Post-market paper reconciliation {report['status'].upper()}: {', '.join(codes)}"
                    )
            logger.info("[Scheduler] Post-market tasks complete.")
        except Exception as exc:
            logger.error("[Scheduler] Post-market failed: {}", exc)

    async def _daily_summary(self) -> None:
        """Send daily summary at 16:00 IST."""
        try:
            summary = {
                "pnl": portfolio_tracker.daily_pnl,
                "gross_pnl": portfolio_tracker.gross_daily_pnl,
                "transaction_costs": portfolio_tracker.daily_transaction_costs,
                "trades": portfolio_tracker._trades_today,
                "wins": portfolio_tracker._wins_today,
                "losses": portfolio_tracker._losses_today,
                "portfolio_value": portfolio_tracker.total_value,
            }
            await telegram_notifier.notify_daily_summary(summary)
        except Exception as exc:
            logger.error("[Scheduler] Daily summary failed: {}", exc)

    async def _end_of_day(self) -> None:
        """End-of-day cleanup at 18:00 IST."""
        logger.info("[Scheduler] End-of-day cleanup.")
        try:
            angel_client.logout()
        except Exception:
            pass

    async def _overnight_check(self) -> None:
        """Check global overnight risks for open positions."""
        try:
            from scheduler.overnight_monitor import overnight_monitor

            positions = portfolio_tracker.positions
            if not positions:
                logger.debug("[Scheduler] No open positions — skipping overnight check.")
                return

            result = await overnight_monitor.check_overnight_risks(positions)
            alerts = result.get("alerts", [])
            if alerts:
                logger.warning("[Scheduler] Overnight alerts: {}", len(alerts))
            else:
                logger.debug("[Scheduler] Overnight check clear — no threshold breaches.")
        except Exception as exc:
            logger.error("[Scheduler] Overnight check failed: {}", exc)


# Singleton
market_scheduler = MarketScheduler(store=control_state)
