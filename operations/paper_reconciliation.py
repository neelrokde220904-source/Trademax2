"""Paper-ledger reconciliation: proves fills, lots, positions and stops agree before entries resume."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pytz

from broker.order_manager import REFERENCE_MAX_AGE_SECONDS, simulate_paper_fill
from broker.paper_ledger import replay_orders, summarize_positions
from database.db import get_db
from database.models import ClosedTrade, PaperFillAudit, Trade
from risk.stop_loss_manager import stop_on_protective_side

IST = pytz.timezone("Asia/Kolkata")
PRICE_TOLERANCE = 0.005
INTRADAY_FLAT_BY = (15, 20)


def _issue(severity: str, code: str, detail: str, symbol: str | None = None) -> dict[str, Any]:
    return {"severity": severity, "code": code, "symbol": symbol, "detail": detail}


def _close_key(entry_id: int, exit_id: int, quantity: int, pnl: float) -> tuple[int, int, int, float]:
    return int(entry_id), int(exit_id), int(quantity), round(float(pnl), 2)


def reconcile_paper(
    *,
    tracker_positions: dict[str, dict[str, Any]] | None = None,
    stop_snapshot: dict[str, dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Cross-check the paper ledger. ``fail`` means automatic entries must stay paused."""
    now = now or datetime.now(IST)
    today = now.date()
    issues: list[dict[str, Any]] = []

    with get_db() as db:
        trades = db.query(Trade).filter(Trade.is_paper == True).all()
        executed = [trade for trade in trades if trade.status == "EXECUTED"]
        pending = [trade for trade in trades if trade.status != "EXECUTED"]
        closed_rows = db.query(ClosedTrade).filter(ClosedTrade.is_paper == True).all()
        audits = db.query(PaperFillAudit).all()
        closed_keys = Counter(
            _close_key(row.entry_order_id, row.exit_order_id, row.quantity, row.gross_pnl)
            for row in closed_rows
        )
        realised = sum(row.gross_pnl for row in closed_rows)
        replay = replay_orders(executed)
        audits_by_trade = {audit.trade_id: audit for audit in audits if audit.trade_id is not None}
        outcome_counts = Counter(audit.outcome for audit in audits)

        replay_keys = Counter(
            _close_key(close.entry_id, close.exit_id, close.quantity, close.gross_pnl)
            for close in replay.closed
        )
        if replay_keys != closed_keys:
            missing = sum((replay_keys - closed_keys).values())
            extra = sum((closed_keys - replay_keys).values())
            issues.append(_issue(
                "fail", "closed_lots_mismatch",
                f"closed-lot table differs from FIFO replay ({missing} missing, {extra} unexpected)",
            ))

        slippage_bps: list[float] = []
        for trade in executed:
            audit = audits_by_trade.get(trade.id)
            if audit is None:
                issues.append(_issue(
                    "fail", "missing_fill_audit",
                    f"executed paper trade #{trade.id} has no fill audit", trade.symbol,
                ))
                continue
            if audit.outcome != "FILLED" or audit.fill_price is None:
                issues.append(_issue(
                    "fail", "audit_outcome_mismatch",
                    f"trade #{trade.id} is executed but audit outcome is {audit.outcome}", trade.symbol,
                ))
                continue
            if abs(float(trade.price) - float(audit.fill_price)) > PRICE_TOLERANCE:
                issues.append(_issue(
                    "fail", "trade_price_mismatch",
                    f"trade #{trade.id} price {trade.price} != audited fill {audit.fill_price}", trade.symbol,
                ))
            outcome, expected, _filled = simulate_paper_fill(
                audit.action, audit.order_type, audit.requested_price,
                float(audit.reference_price or 0), audit.slippage_pct,
            )
            if outcome != "FILLED" or expected is None or abs(expected - float(audit.fill_price)) > PRICE_TOLERANCE:
                issues.append(_issue(
                    "fail", "fill_price_mismatch",
                    f"trade #{trade.id} audited fill {audit.fill_price} does not match simulator ({outcome}, {expected})",
                    trade.symbol,
                ))
            if audit.reference_source == "requested_price":
                issues.append(_issue(
                    "warn", "requested_price_reference",
                    f"trade #{trade.id} exit filled at requested price without a live reference", trade.symbol,
                ))
            elif (audit.reference_age_seconds or 0) > REFERENCE_MAX_AGE_SECONDS:
                issues.append(_issue(
                    "warn", "stale_reference",
                    f"trade #{trade.id} reference was {audit.reference_age_seconds:.0f}s old", trade.symbol,
                ))
            if audit.reference_price:
                sign = 1 if audit.action == "BUY" else -1
                slippage_bps.append(sign * (audit.fill_price - audit.reference_price) / audit.reference_price * 1e4)

        for trade in pending:
            issues.append(_issue(
                "warn", "pending_paper_order", f"paper trade #{trade.id} has status {trade.status}", trade.symbol,
            ))
        executed_count, pending_count, closed_count = len(executed), len(pending), len(closed_rows)

    positions = summarize_positions(replay)
    flat_hour, flat_minute = INTRADAY_FLAT_BY
    after_squareoff = (now.hour, now.minute) >= (flat_hour, flat_minute)
    for symbol, lots in replay.open_lots.items():
        for lot in lots:
            if lot.holding_period != "INTRADAY":
                continue
            lot_day = lot.timestamp.date()
            if lot_day < today:
                issues.append(_issue(
                    "fail", "intraday_carried_overnight",
                    f"intraday lot from trade #{lot.trade_id} ({lot_day}) is still open", symbol,
                ))
            elif after_squareoff:
                issues.append(_issue(
                    "fail", "intraday_open_after_squareoff",
                    f"intraday lot from trade #{lot.trade_id} is open after "
                    f"{flat_hour:02d}:{flat_minute:02d} IST", symbol,
                ))

    for symbol, position in positions.items():
        action = "BUY" if position["quantity"] > 0 else "SELL"
        if not stop_on_protective_side(action, position["avg_price"], position["stop_loss"]):
            issues.append(_issue(
                "fail", "invalid_ledger_stop",
                f"ledger stop {position['stop_loss']} is missing or on the wrong side of "
                f"entry {position['avg_price']:.2f}", symbol,
            ))

    if tracker_positions is not None:
        for symbol in sorted(set(positions) | set(tracker_positions)):
            ledger = positions.get(symbol)
            tracked = tracker_positions.get(symbol)
            if ledger is None or tracked is None:
                issues.append(_issue(
                    "fail", "tracker_position_mismatch",
                    f"ledger={'open' if ledger else 'flat'} tracker={'open' if tracked else 'flat'}", symbol,
                ))
            elif int(tracked.get("quantity", 0)) != ledger["quantity"] or abs(
                float(tracked.get("avg_price", 0)) - ledger["avg_price"]
            ) > PRICE_TOLERANCE:
                issues.append(_issue(
                    "fail", "tracker_position_mismatch",
                    f"tracker {tracked.get('quantity')}@{tracked.get('avg_price')} != "
                    f"ledger {ledger['quantity']}@{ledger['avg_price']:.2f}", symbol,
                ))

    if stop_snapshot is not None:
        for symbol, position in positions.items():
            action = "BUY" if position["quantity"] > 0 else "SELL"
            stop = stop_snapshot.get(symbol)
            if stop is None:
                issues.append(_issue("fail", "missing_stop", "open position has no tracked stop", symbol))
            elif stop.get("action") != action or float(stop.get("stop_loss") or 0) <= 0:
                issues.append(_issue(
                    "fail", "stop_side_mismatch",
                    f"tracked stop is {stop.get('action')} @ {stop.get('stop_loss')} for a {action} position",
                    symbol,
                ))
        for symbol in sorted(set(stop_snapshot) - set(positions)):
            issues.append(_issue("warn", "orphan_stop", "stop tracked for a flat symbol", symbol))

    severities = {issue["severity"] for issue in issues}
    status = "fail" if "fail" in severities else "warn" if "warn" in severities else "pass"
    return {
        "status": status,
        "checked_at": now.isoformat(),
        "issues": issues,
        "summary": {
            "executed_orders": executed_count,
            "pending_orders": pending_count,
            "closed_lots": closed_count,
            "open_positions": len(positions),
            "realised_pnl": round(realised, 2),
            "fill_outcomes": dict(outcome_counts),
            "avg_slippage_bps": round(sum(slippage_bps) / len(slippage_bps), 2) if slippage_bps else None,
        },
        "positions": {
            symbol: {
                "quantity": position["quantity"],
                "avg_price": round(position["avg_price"], 2),
                "stop_loss": position["stop_loss"],
                "target": position["target"],
                "holding_period": position["holding_period"],
            }
            for symbol, position in positions.items()
        },
    }


def reconcile_live_state() -> dict[str, Any]:
    """Sync the in-process tracker from the ledger, then reconcile it and the stop book."""
    from broker.portfolio_tracker import portfolio_tracker
    from risk.stop_loss_manager import stop_loss_manager

    portfolio_tracker.sync_from_broker()
    return reconcile_paper(
        tracker_positions=portfolio_tracker.positions,
        stop_snapshot=stop_loss_manager.snapshot(),
    )
