"""India AI Trader — main entry point.

Usage:
    python main.py              # Run the full trading system (scheduler mode)
    python main.py --scan       # Run a single scan cycle
    python main.py --backtest   # Run backtest with default strategy
    python main.py --dashboard  # Start dashboard only
    python main.py --train-hmm  # Fit the HMM regime model from Angel One daily candles
    python main.py --reconcile-paper  # Check the paper ledger (exit 1 on failure)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import uvicorn
from loguru import logger

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from database.db import init_db
from operations.process_lock import trading_process_lock
from operations.redaction import loguru_patcher, silence_sdk_logging
from risk.dead_mans_switch import dead_mans_switch

GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 5


def setup_logging(*, file_logging: bool = True) -> None:
    """Configure loguru logging."""
    logger.remove()
    # Mask credentials in every message, and never dump local variables into tracebacks.
    logger.configure(patcher=loguru_patcher)
    silence_sdk_logging()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        diagnose=False, backtrace=False,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
    )
    if file_logging:
        logger.add(
            "logs/trader_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
            diagnose=False, backtrace=False,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function} — {message}",
        )


async def run_scheduler_mode(*, start_paused: bool = False) -> None:
    """Run the full scheduled trading system."""
    with trading_process_lock():
        await _run_scheduler_mode_locked(start_paused=start_paused)


async def _run_scheduler_mode_locked(*, start_paused: bool = False) -> None:
    from scheduler.market_scheduler import market_scheduler
    from scheduler.foreign_markets import foreign_market_scheduler

    if start_paused:
        if not settings.paper_trading:
            raise RuntimeError("Controlled recovery requires PAPER_TRADING=true")
        market_scheduler.pause_trading()
        dead_mans_switch.force_halt("Controlled paper recovery; operator review required")

    if settings.paper_trading:
        from broker.portfolio_tracker import portfolio_tracker

        unprotected = market_scheduler.restore_paper_state()
        report = market_scheduler.reconcile_paper()
        logger.info(
            "Paper ledger restored: {} open positions; reconciliation {}",
            len(report["positions"]), report["status"].upper(),
        )
        if unprotected or report["status"] == "fail":
            market_scheduler.pause_trading()
            logger.critical(
                "Entries paused at startup: unprotected={} reconciliation={}",
                unprotected, [issue["code"] for issue in report["issues"] if issue["severity"] == "fail"],
            )
        elif portfolio_tracker.has_hit_portfolio_drawdown_limit():
            market_scheduler.pause_trading()
            logger.critical(
                "Entries paused at startup: persistent portfolio drawdown limit requires operator review."
            )

    logger.info("=" * 60)
    logger.info("  INDIA AI TRADER — Starting")
    logger.info("  Mode: {}", "PAPER TRADING" if settings.paper_trading else "LIVE TRADING")
    logger.info("  Capital: ₹{:,.2f}", settings.initial_capital)
    logger.info("  Universe mode: {} (membership checked before scans)", settings.trading_universe_mode)
    logger.info("=" * 60)

    if not settings.paper_trading:
        logger.warning("⚠️  LIVE TRADING MODE — real money at risk!")

    # Setup and start scheduler
    market_scheduler.setup()
    market_scheduler.start()
    foreign_market_scheduler.start()

    logger.info("Dead Man's Switch active (thresholds: {})", dead_mans_switch._thresholds)

    # Start dashboard in background
    config = uvicorn.Config(
        "dashboard.backend.app:app",
        host="127.0.0.1",
        port=settings.dashboard_port,
        log_level="warning",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
    )
    server = uvicorn.Server(config)

    logger.info("Dashboard: http://localhost:{}", settings.dashboard_port)
    logger.info("Scheduler running. Press Ctrl+C to stop.")

    try:
        await server.serve()
    finally:
        logger.info("Shutting down...")
        foreign_market_scheduler.stop()
        market_scheduler.stop()
        from broker.websocket_feed import ws_feed
        ws_feed.disconnect()


async def run_single_scan() -> None:
    """Run a single scan cycle (for testing)."""
    with trading_process_lock():
        await _run_single_scan_locked()


async def _run_single_scan_locked() -> None:
    from agents.graph import run_trading_pipeline
    from broker.angel_client import angel_client
    from broker.portfolio_tracker import portfolio_tracker
    from data.instrument_master import download_instrument_master
    from config.universe import get_active_watchlist

    logger.info("Running single scan cycle...")

    await download_instrument_master()
    angel_client.authenticate()
    portfolio_tracker.sync_from_broker()

    result = await run_trading_pipeline(get_active_watchlist())

    executed = result.get("executed_orders", [])
    logger.info("Scan complete: {} orders executed.", len(executed))
    for order in executed:
        logger.info("  {} {} @ ₹{:.2f} × {}", order.get("action"), order.get("symbol"), order.get("price", 0), order.get("quantity", 0))


def run_backtest() -> None:
    """Run a daily-strategy walk-forward evaluation."""
    from backtester.engine import BacktestEngine, BacktestResult
    from backtester.report_generator import generate_report
    from data.market_data import fetch_ohlcv
    from strategies.momentum_breakout import MomentumBreakout

    logger.info("Running MomentumBreakout daily walk-forward evaluation...")

    engine = BacktestEngine()
    strategy = MomentumBreakout()

    import asyncio as aio

    async def _fetch():
        data = {}
        for symbol in settings.default_watchlist[:5]:  # limit for speed
            try:
                df = fetch_ohlcv(symbol, "1d", days=3650)
                if df is not None and not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning("Failed to fetch {}: {}", symbol, e)
        return data

    data = aio.run(_fetch())

    if not data:
        logger.error("No data fetched — cannot run backtest.")
        return

    def strategy_fn(symbol, df, bar_index):
        if bar_index < strategy.lookback_52w - 1:
            return []
        signal = strategy._analyse(symbol, df)
        return [signal] if signal else []

    walk_forward = engine.run_walk_forward(
        data,
        strategy_fn,
        train_days=756,
        test_days=252,
        step_days=252,
    )
    if not walk_forward.folds:
        logger.error("Insufficient daily history for walk-forward evaluation.")
        return

    result = BacktestResult(
        trades=walk_forward.combined_trades,
        equity_curve=walk_forward.combined_equity,
        dates=[date for fold in walk_forward.folds for date in fold.dates],
        initial_capital=walk_forward.initial_capital,
        final_capital=walk_forward.final_capital,
    )
    report = generate_report(result, output_dir="backtest_results")

    logger.info(
        "Walk-forward complete: {} OOS folds, {} trades, net P&L ₹{:,.2f}, return {:.2f}%. "
        "See backtest_results/ for the report.",
        walk_forward.num_folds,
        len(walk_forward.combined_trades),
        report["summary"]["net_pnl"],
        walk_forward.oos_return_pct,
    )


def run_dashboard() -> None:
    """Start only the dashboard."""
    logger.info("Starting dashboard on port {}...", settings.dashboard_port)
    uvicorn.run(
        "dashboard.backend.app:app",
        host="127.0.0.1",
        port=settings.dashboard_port,
        reload=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
    )


def run_preflight() -> int:
    """Print non-mutating readiness diagnostics and return a shell-friendly status."""
    from operations.preflight import run_preflight as check

    result = check()
    logger.info("Preflight: {}", "PASS" if result.passed else "FAIL")
    for warning in result.warnings:
        logger.warning("Preflight warning: {}", warning)
    for error in result.errors:
        logger.error("Preflight error: {}", error)
    return 0 if result.passed else 1


def run_train_hmm(start: date | None) -> int:
    """Fit, validate and save the HMM regime model; prints a JSON report."""
    from risk.hmm_training import DEFAULT_TRAINING_START, train_hmm_from_angel

    try:
        report = train_hmm_from_angel(start=start or DEFAULT_TRAINING_START)
    except Exception as exc:
        logger.error("HMM training failed: {}", exc)
        return 1
    print(json.dumps({key: report[key] for key in (
        "model_path", "trained_at", "data_start", "data_end", "n_observations",
        "log_likelihood_per_observation", "state_profiles", "validation", "latest",
    )}, indent=2, default=str))
    return 0


def run_reconcile_paper() -> int:
    """Reconcile the persisted paper ledger (fills, lots, positions); exit 1 on failure."""
    from broker.portfolio_tracker import portfolio_tracker
    from operations.paper_reconciliation import reconcile_paper

    portfolio_tracker.sync_from_broker()
    report = reconcile_paper(tracker_positions=portfolio_tracker.positions)
    print(json.dumps(report, indent=2, default=str))
    return 1 if report["status"] == "fail" else 0


def run_evaluation(args: argparse.Namespace) -> int:
    """Walk-forward evaluation on cached or free data; research only, changes no setting."""
    from config.universe import get_active_watchlist
    from research.evaluation import EvaluationUnavailable, run_walk_forward_evaluation, write_report

    universe = get_active_watchlist()[: args.eval_symbols]
    reports = []
    try:
        for capital in (10_000.0, 100_000.0):
            reports.append(run_walk_forward_evaluation(
                args.evaluate, args.policy, universe, args.date_from, args.date_to, capital,
            ))
    except (EvaluationUnavailable, ValueError) as exc:
        logger.error("Evaluation rejected: {}", exc)
        return 1
    name = f"evaluation_{args.evaluate}_{args.policy}_{args.date_from}_{args.date_to}.json"
    path = write_report(reports, name)
    print(json.dumps([
        {key: report[key] for key in ("strategy", "exit_policy", "capital", "outcome")}
        | {"trades": report["metrics"]["trade_count"], "report": str(path)}
        for report in reports
    ], indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="India AI Trader")
    parser.add_argument("--scan", action="store_true", help="Run a single scan cycle")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard only")
    parser.add_argument("--preflight", action="store_true", help="Validate configuration without contacting brokers or placing orders")
    parser.add_argument("--start-paused", action="store_true", help="Start paper scheduler with durable entry pause and circuit breaker halt")
    parser.add_argument("--train-hmm", action="store_true", help="Fit and save the HMM regime model from Angel One daily candles (no orders)")
    parser.add_argument("--hmm-start", type=date.fromisoformat, default=None, help="First training date for --train-hmm (YYYY-MM-DD; default 2017-10-01, after a gap in Angel's Nifty history)")
    parser.add_argument("--reconcile-paper", action="store_true", help="Reconcile the paper ledger and exit non-zero on failure")
    parser.add_argument("--research-snapshots", action="store_true", help="List captured paper-research snapshots")
    parser.add_argument("--forecasts", action="store_true", help="Show hypothetical next-session forecast outcomes")
    parser.add_argument("--settle-forecasts", action="store_true", help="Fetch completed daily closes and settle pending forecasts")
    parser.add_argument("--paper-compare", nargs="+", metavar="SNAPSHOT_ID", help="Compare captured scans in isolated paper portfolios (2–10 IDs)")
    parser.add_argument("--atr-exit-compare", nargs="+", metavar="SNAPSHOT_ID", help="Compare baseline and 2x/4x daily ATR exits using captured rules-only inputs (no model charges)")
    parser.add_argument("--evaluate", metavar="STRATEGY", help="Walk-forward evaluation of a strategy on cached/free data (research only; changes no setting)")
    parser.add_argument("--policy", default="daily_atr_baseline_v1", help="Exit policy name for --evaluate")
    parser.add_argument("--from", dest="date_from", type=str, help="Evaluation start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", type=str, help="Evaluation end date (YYYY-MM-DD)")
    parser.add_argument("--eval-symbols", type=int, default=20, help="Number of watchlist symbols to evaluate (default 20)")
    parser.add_argument("--confirm-research-cost", action="store_true", help="Confirm model API charges for --paper-compare")
    args = parser.parse_args()

    commands = [args.scan, args.backtest, args.dashboard, args.preflight, args.train_hmm,
                args.reconcile_paper, args.research_snapshots, args.forecasts,
                args.settle_forecasts, args.paper_compare, args.atr_exit_compare, args.evaluate]
    if sum(bool(command) for command in commands) > 1:
        parser.error("choose only one command")
    if args.evaluate and not (args.date_from and args.date_to):
        parser.error("--evaluate requires --from and --to (YYYY-MM-DD)")
    if args.start_paused and any(commands):
        parser.error("--start-paused is only supported in scheduler mode")
    if args.paper_compare and not args.confirm_research_cost:
        parser.error("--paper-compare requires --confirm-research-cost; model API charges may apply")
    if args.confirm_research_cost and not args.paper_compare:
        parser.error("--confirm-research-cost requires --paper-compare")
    setup_logging(file_logging=not args.preflight)

    if args.preflight:
        raise SystemExit(run_preflight())

    # Every other command writes database state or needs the trade ledger.
    init_db()

    if args.research_snapshots:
        from research.tracking import list_snapshots
        print(json.dumps(list_snapshots(), indent=2))
        return
    if args.forecasts:
        from research.tracking import forecast_summary
        print(json.dumps(forecast_summary(), indent=2))
        return
    if args.settle_forecasts:
        if not settings.paper_trading:
            parser.error("--settle-forecasts requires paper-trading mode")
        from research.tracking import settle_forecasts
        result = settle_forecasts()
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if result["errors"] else 0)
    if args.paper_compare:
        from research.comparison import run_comparison
        try:
            result = run_comparison(args.paper_compare)
        except ValueError as exc:
            logger.error("Paper comparison rejected: {}", exc)
            raise SystemExit(1) from None
        print(json.dumps(result, indent=2, default=str))
        raise SystemExit(0 if result["status"] == "completed" else 1)
    if args.atr_exit_compare:
        from research.atr_exit_comparison import run_atr_exit_comparison
        try:
            result = run_atr_exit_comparison(args.atr_exit_compare)
        except ValueError as exc:
            logger.error("ATR exit comparison rejected: {}", exc)
            raise SystemExit(1) from None
        print(json.dumps(result, indent=2, default=str))
        return

    if args.evaluate:
        raise SystemExit(run_evaluation(args))
    if args.train_hmm:
        raise SystemExit(run_train_hmm(args.hmm_start))
    if args.reconcile_paper:
        raise SystemExit(run_reconcile_paper())
    if args.scan:
        asyncio.run(run_single_scan())
    elif args.backtest:
        run_backtest()
    elif args.dashboard:
        run_dashboard()
    else:
        asyncio.run(run_scheduler_mode(start_paused=args.start_paused))


if __name__ == "__main__":
    main()
