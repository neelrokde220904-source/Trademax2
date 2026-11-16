"""Regression tests for provider routing, broker safety, and deterministic gates."""

from __future__ import annotations

import asyncio
import sqlite3
import time as real_time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from alpha_library.pattern_accuracy import AccuracyTrials
from broker.base_broker import BrokerRejectedError, OrderSubmissionUncertainError
from broker.dual_broker_router import DualBrokerExecutionRouter
from broker.zerodha_client import TokenBucket, ZerodhaBrokerAdapter
from config import llm_factory
from config.settings import settings
from data.point_in_time_guard import filter_published_as_of
from database.hash_ledger import HashChainedLedger
from models.signal_gate.strategy_selector import select_strategies
from risk.dead_mans_switch import DeadMansSwitch
from risk.timemachine_validator import TimeMachineValidator


def test_llm_factory_routes_tiers_to_independent_providers(monkeypatch):
    class FakeChatModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(settings, "trademax_llm_provider", "auto")
    monkeypatch.setattr(settings, "trademax_quick_think_provider", "openai")
    monkeypatch.setattr(settings, "trademax_quick_think_llm", "quick-test")
    monkeypatch.setattr(settings, "trademax_deep_think_provider", "deepseek")
    monkeypatch.setattr(settings, "trademax_deep_think_llm", "deep-test")
    monkeypatch.setattr(settings, "openai_api_key", "openai-test")
    monkeypatch.setattr(settings, "deepseek_api_key", "deepseek-test")
    monkeypatch.setattr(llm_factory, "_load_class", lambda *_args: FakeChatModel)

    quick = llm_factory.create_llm("quick", temperature=0)
    deep = llm_factory.create_llm("deep", temperature=0.1)

    assert quick.kwargs["model"] == "quick-test"
    assert "base_url" not in quick.kwargs
    assert deep.kwargs["model"] == "deep-test"
    assert deep.kwargs["base_url"] == "https://api.deepseek.com"
    assert quick.kwargs["max_retries"] == settings.trademax_llm_retry_budget


def test_ollama_openai_compatible_endpoint_uses_chat_openai(monkeypatch):
    class FakeChatModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(settings, "trademax_llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_factory, "_load_class", lambda *_args: FakeChatModel)

    model = llm_factory.create_llm("quick")

    assert model.kwargs["base_url"] == "http://localhost:11434/v1"
    assert model.kwargs["api_key"] == "ollama"


class FakeBroker:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.orders = []

    def place_order(self, params):
        self.orders.append(params)
        if self.error:
            raise self.error
        return self.response


def test_dual_broker_fails_over_only_after_confirmed_rejection():
    primary = FakeBroker(error=BrokerRejectedError("margin rejected"))
    secondary = FakeBroker(response={"status": "PENDING", "order_id": "secondary-1"})

    result = DualBrokerExecutionRouter().execute(
        primary=primary, secondary=secondary,
        primary_order={"quantity": 3}, secondary_order={"quantity": 3},
    )

    assert len(primary.orders) == len(secondary.orders) == 1
    assert result["order_id"] == "secondary-1"
    assert result["routed_broker"] == "secondary"


def test_dual_broker_does_not_retry_ambiguous_submission():
    primary = FakeBroker(error=TimeoutError("request timed out"))
    secondary = FakeBroker(response={"status": "PENDING"})

    with pytest.raises(OrderSubmissionUncertainError):
        DualBrokerExecutionRouter().execute(
            primary=primary, secondary=secondary,
            primary_order={"quantity": 1}, secondary_order={"quantity": 1},
        )
    assert len(primary.orders) == 1
    assert secondary.orders == []


def test_critical_squareoff_failure_forces_dead_mans_switch():
    switch = DeadMansSwitch()

    assert switch.force_halt("broker square-off failed")
    assert switch.is_halted
    assert "broker square-off failed" in switch.halt_reason
    assert not switch.force_halt("another failure")


def test_live_position_exit_bypasses_entry_evidence_and_cutoff(monkeypatch):
    from agents import graph

    calls = []
    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(settings, "max_positions", 0)
    monkeypatch.setattr(graph, "get_token", lambda _symbol: "token")
    monkeypatch.setattr(
        graph, "place_order",
        lambda **kwargs: calls.append(kwargs) or {"symbol": "TEST", "status": "EXECUTED"},
    )
    state = {
        "execution_enabled": True,
        "trading_decisions": [{
            "symbol": "TEST", "action": "SELL", "quantity": 1, "price": 100,
            "holding_period": "INTRADAY",
        }],
        "current_positions": {"TEST": {"quantity": 2}},
    }

    result = graph._order_router_node(state)

    assert len(calls) == 1
    assert calls[0]["action"] == "SELL"
    assert calls[0]["allow_requested_price_reference"] is False
    assert result["executed_orders"]
    assert not result["errors"]


def test_live_squareoff_uses_broker_flatten_once_without_strategy_exit(monkeypatch):
    from scheduler import market_scheduler

    calls = []
    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(
        market_scheduler.stop_loss_manager, "check_all_positions",
        lambda: [{"symbol": "TEST", "exit_type": "TIME_EXIT", "exit_action": "SELL"}],
    )
    monkeypatch.setattr(
        market_scheduler.angel_client, "flatten_all_intraday",
        lambda: calls.append("flatten") or [{"status": "PENDING"}],
    )
    monkeypatch.setattr(
        market_scheduler.MarketScheduler, "_place_exit_order",
        staticmethod(lambda *_args: calls.append("exit")),
    )

    asyncio.run(market_scheduler.MarketScheduler()._intraday_squareoff())

    assert calls == ["flatten"]


def test_zerodha_token_bucket_enforces_default_rate(monkeypatch):
    clock = [0.0]
    waits = []
    monkeypatch.setattr("broker.zerodha_client.time.monotonic", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("broker.zerodha_client.time.sleep", sleep)
    bucket = TokenBucket()
    bucket.acquire()
    bucket.acquire()

    assert bucket.rate == 8
    assert waits == pytest.approx([1 / 8])


def test_zerodha_squareoff_fails_closed_when_order_is_unresolved():
    class FakeKite:
        def orders(self):
            return [{"tradingsymbol": "TEST", "product": "MIS", "status": "OPEN"}]

        def positions(self):
            return {"net": [{
                "tradingsymbol": "TEST", "exchange": "NSE",
                "product": "MIS", "quantity": 2,
            }]}

        def place_order(self, **_kwargs):
            pytest.fail("Must not duplicate an unresolved order")

    adapter = ZerodhaBrokerAdapter()
    adapter._api = FakeKite()

    with pytest.raises(RuntimeError, match="Unresolved broker orders"):
        adapter.flatten_all_intraday()


def test_paper_trade_closes_and_settles_once_with_injected_benchmark(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from agents import episodic_reflection_agent
    from broker import order_manager
    from database import db, hash_ledger as ledger_module
    from database.models import Base, ClosedTrade, StrategyAlphaWeight, Trade, TradeReflectionSettlement

    engine = create_engine(
        f"sqlite:///{tmp_path / 'paper.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    ledger = HashChainedLedger(tmp_path / "audit.sqlite")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(ledger_module, "hash_ledger", ledger)
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(
        episodic_reflection_agent.episodic_reflection,
        "store_completed_trade", lambda _trade: None,
    )

    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    references = iter([100.0, 101.0])
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": next(references), "source": "ws_tick", "age_seconds": 1.0},
    )
    entry = order_manager.place_order(
        "TEST", "token", "BUY", 2, price=100, product_type="INTRADAY",
        strategy="test_strategy", expected_paper_book={},
    )
    assert entry is not None
    entry_id = entry["trade_id"]
    assert isinstance(entry_id, int)
    with db.get_db() as database:
        assert database.query(Trade).filter_by(order_id=entry["order_id"]).one().id == entry_id
    exit_order = order_manager.place_order(
        "TEST", "token", "SELL", 2, price=101, product_type="INTRADAY",
        strategy="test_strategy", is_exit=True, expected_entry_trade_ids=[entry_id],
    )
    assert exit_order is not None
    settlement = episodic_reflection_agent.episodic_reflection.settle_all_pending(
        benchmark_return_provider=lambda _entry, _exit: 0.25
    )

    assert settlement["settled"] == 1
    with db.get_db() as database:
        assert database.query(ClosedTrade).count() == 1
        assert database.query(TradeReflectionSettlement).count() == 1
        assert database.get(StrategyAlphaWeight, "test_strategy").samples == 1
    assert [entry["event_type"] for entry in ledger.entries()] == ["trade_settlement"]
    assert episodic_reflection_agent.episodic_reflection.settle_all_pending(
        benchmark_return_provider=lambda _entry, _exit: 0.25
    )["settled"] == 0
    ledger.close()


def _sample_bars(count: int = 4_500, step_return: float = 0.02) -> pd.DataFrame:
    close = 100 * (1 + step_return) ** np.arange(count)
    index = pd.date_range("2026-01-01 09:15", periods=count, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({
        "open": close / (1 + step_return),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    }, index=index)


def test_time_machine_10000_path_ev_and_latency():
    validator = TimeMachineValidator(paths=10_000, min_quality_score=0.5, random_seed=11)
    bars = _sample_bars()

    started = real_time.perf_counter()
    result = validator.validate(
        bars,
        entry_price=100,
        target_price=102,
        stop_price=98,
        quantity=10,
        historical_setup_returns=[0.02, 0.03],
        accuracy_score=0.8,
        remaining_time_buckets=[9 * 60 + 30],
    )
    elapsed = real_time.perf_counter() - started

    assert result.paths == 10_000
    assert result.prob_target_first == 1
    assert result.prob_target_first + result.prob_stop_first + result.prob_open_at_close == pytest.approx(1)
    assert result.ev_net > 0
    assert result.approved
    assert elapsed < 0.05


def test_time_machine_rejects_small_trade_when_minimum_cost_exceeds_edge():
    validator = TimeMachineValidator(paths=100, random_seed=11)
    result = validator.validate(
        _sample_bars(),
        entry_price=100,
        target_price=102,
        stop_price=98,
        quantity=1,
        historical_setup_returns=[0.02, 0.03],
        accuracy_score=0.8,
        remaining_time_buckets=[9 * 60 + 30],
    )
    assert result.ev_net < 0
    assert not result.approved
    assert result.reason == "Net expected value is not positive"


def test_time_machine_blocks_missing_exact_setup_history():
    validator = TimeMachineValidator(paths=100, random_seed=3)
    result = validator.validate(
        _sample_bars(50), entry_price=100, target_price=102, stop_price=98,
        quantity=1, accuracy_score=1,
    )
    assert not result.approved
    assert "No exact-setup" in result.reason


def test_accuracy_trials_forward_scoring_and_horizon():
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    score = AccuracyTrials(candles_ahead_qty=2).score(close, [0, 3, 8])
    assert score.observations == 2
    assert score.success_rate == 1
    assert score.end_close_return > 0
    assert AccuracyTrials(2).GetAverPercentPatternAccuracy(close, [0, 3, 8]) == 1


def test_point_in_time_guard_fails_closed_and_filters_future_rows():
    frame = pd.DataFrame({
        "published_at": ["2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z"],
        "headline": ["available", "future"],
    })
    filtered = filter_published_as_of(frame, datetime.fromisoformat("2026-01-01T11:00:00+00:00"))
    assert filtered["headline"].tolist() == ["available"]
    with pytest.raises(ValueError, match="valid publication timestamp"):
        filter_published_as_of(
            pd.DataFrame({"published_at": [None]}), datetime(2026, 1, 1)
        )


def test_hash_ledger_detects_payload_tampering(tmp_path):
    ledger_path = tmp_path / "audit.sqlite"
    ledger = HashChainedLedger(ledger_path)
    first = ledger.append("decision", {"symbol": "TEST", "quantity": 2})
    ledger.append("risk_check", {"approved": True})
    assert first["sequence"] == 1
    assert ledger.verify()
    ledger._connection.execute(
        "UPDATE hash_ledger SET payload_json = ? WHERE sequence = 1",
        ('{"quantity":99,"symbol":"TEST"}',),
    )
    ledger._connection.commit()
    assert not ledger.verify()
    ledger.close()


@pytest.mark.parametrize(
    "values,expected",
    [
        ((30, 30, 18, 0.019, ""), "trend"),
        ((10, 70, 18, 0.01, ""), "range"),
        ((30, 30, 18, 0.02, ""), "loss_halt"),
        ((30, 30, 28, 0.01, ""), "halt"),
        ((30, 30, 18, 0.01, "CRISIS"), "halt"),
    ],
)
def test_strategy_selector_regime_rules(values, expected):
    adx, chop, vix, drawdown, regime = values
    selected = select_strategies(
        adx=adx, choppiness=chop, india_vix=vix,
        daily_drawdown=drawdown, hmm_regime=regime,
    )
    if expected == "halt":
        assert selected.halt_new_entries
        assert selected.flatten_intraday
    elif expected == "loss_halt":
        assert selected.halt_new_entries
        assert not selected.flatten_intraday
        assert not selected.tighten_swing_stops
    elif expected == "trend":
        assert "orb_intraday" in selected.active_strategies
    else:
        assert "pairs_trading" in selected.active_strategies


def test_strategy_selector_uses_configured_daily_loss_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_daily_loss_pct", 0.015)
    selected = select_strategies(adx=30, choppiness=30, india_vix=18, daily_drawdown=0.015)
    assert selected.halt_new_entries and not selected.flatten_intraday
