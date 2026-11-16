"""Broker-free, rules-only ATR exit comparisons using captured scan inputs."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from uuid import uuid4

from agents.technical_agent import technical_agent
from config.settings import settings
from database.db import get_db
from database.models import ResearchComparison
from research.comparison import _PaperBook, _positive
from research.tracking import load_snapshot
from risk.exit_policy import BASELINE_ATR_POLICY, RESEARCH_ATR_POLICY


def run_atr_exit_comparison(snapshot_ids: list[str]) -> dict:
    if not settings.paper_trading:
        raise ValueError("ATR exit comparison requires paper-trading mode")
    if not 2 <= len(snapshot_ids) <= 10 or len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("Select 2-10 unique captured snapshot IDs")
    snapshots = [load_snapshot(identifier) for identifier in snapshot_ids]
    snapshots.sort(key=lambda item: datetime.fromisoformat(item["research_as_of"]))
    policies = (BASELINE_ATR_POLICY, RESEARCH_ATR_POLICY)
    books = {policy.name: _PaperBook() for policy in policies}
    for snapshot in snapshots:
        signals = technical_agent({
            **deepcopy(snapshot), "research_mode": True, "execution_enabled": False,
        })
        if signals.get("errors"):
            raise ValueError(f"Captured technical analysis failed: {signals['errors']}")
        for policy in policies:
            book = books[policy.name]
            book.advance(snapshot)
            state = book.state(snapshot)
            state["technical_signals"] = deepcopy(signals.get("technical_signals", {}))
            state["risk_assessment"] = {"blocked": book.loss_halted}
            state["india_signals"] = {"fo_ban_list": []}
            state["trading_decisions"] = []
            for symbol, technical in state["technical_signals"].items():
                signal = str(technical.get("signal") or "").upper()
                side = "BUY" if signal in {"BUY", "STRONG_BUY"} else (
                    "SELL" if signal in {"SELL", "STRONG_SELL"} else ""
                )
                if not side:
                    continue
                indicators = technical.get("indicators", {})
                atr = _positive(indicators.get("daily_atr_14"))
                price = _positive(state.get("live_quotes", {}).get(symbol, {}).get("ltp"))
                price = price or _positive(indicators.get("close"))
                if atr is None or price is None:
                    book.audit.append({"symbol": symbol, "status": "missing_daily_atr_or_price"})
                    continue
                stop, target = policy.levels(price, atr, side)
                technical.update(
                    atr_exit_policy=policy.name, stop_loss=stop, target=target,
                    holding_period="INTRADAY",
                )
                state["risk_assessment"][symbol] = {
                    "approved": True, "deterministic_action": side,
                    "stop_loss": stop, "target": target, "vix_factor": 1.0,
                    "holding_period": "INTRADAY",
                }
                state["trading_decisions"].append({
                    "symbol": symbol, "action": side, "quantity": 1,
                    "price": price, "stop_loss": stop, "target": target,
                    "order_type": "LIMIT", "holding_period": "INTRADAY",
                    "strategy": "CompositeTrend_rules_exit_study",
                })
            book.submit(state)
    result = {
        "id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "atr_exit_rules_comparison", "status": "completed",
        "snapshot_ids": snapshot_ids, "verdict": "inconclusive_not_for_promotion",
        "variants": [{
            **books[policy.name].report(policy.name),
            "exit_policy": {
                "name": policy.name, "stop_atr": policy.stop_multiplier,
                "target_atr": policy.target_multiplier,
                "trail_atr": policy.trailing_multiplier, "atr_timeframe": policy.atr_timeframe,
            },
        } for policy in policies],
        "limitations": [
            "Rules-only diagnostic, not the production AI-reviewed portfolio; no model calls or broker orders.",
            "Both books receive identical captured technical candidates, with independent risk-sized inventory.",
            "ATR is frozen from completed daily sessions at signal time; trails update after completed 15-minute bars.",
            "Captured next-bar LIMIT fills are simulated without broker queue, latency or market impact.",
            "Transaction costs, slippage, gaps, entry caps and paper daily loss gates are included.",
            "A gross 2:1 target may fail the unchanged minimum NET 2:1 risk gate after costs.",
            "No promotion: additional chronological out-of-sample data, portfolio constraints and stress tests are required.",
        ],
    }
    with get_db() as db:
        db.add(ResearchComparison(
            id=result["id"], created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status=result["status"], results_json=json.dumps(result, default=str),
        ))
    return result
