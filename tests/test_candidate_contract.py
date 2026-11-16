"""T06: typed candidate contract, strategy registry, horizon/product separation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from broker import order_manager
from broker.paper_ledger import replay_orders, summarize_positions
from config.settings import settings
from database import db
from database.models import Base, Trade
from strategies.candidate import (
    AssetId, CandidateInvalid, TradeCandidate, horizon_to_product, normalize_horizon,
    round_to_tick,
)

NOW = datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc)


def make_candidate(**overrides) -> TradeCandidate:
    base = dict(
        asset=AssetId("NSE", "CASH", "INR", "TCS"), candidate_id="c1", strategy="CompositeTrend",
        strategy_version="composite-trend-v1", side="BUY", horizon="INTRADAY", product="INTRADAY",
        signal_time=NOW, expires_at=NOW + timedelta(minutes=30), tick_size=Decimal("0.05"), lot_size=1,
        entry_price=Decimal("100.00"), stop_loss=Decimal("98.50"), target=Decimal("103.75"),
        exit_policy="daily_atr_baseline_v1", atr_value=Decimal("1.5"),
        atr_timeframe="DAILY_14_COMPLETED_SESSIONS", thesis="trend", review_contract_version="review-v2",
    )
    base.update(overrides)
    return TradeCandidate(**base)


def test_a_well_formed_candidate_validates():
    assert make_candidate().validate(quantity=10, now=NOW) is not None
    assert make_candidate(
        side="SELL", stop_loss=Decimal("101.50"), target=Decimal("96.25"),
    ).validate(now=NOW)


@pytest.mark.parametrize("overrides,message", [
    ({"stop_loss": Decimal("101.00")}, "BUY requires"),
    ({"target": Decimal("99.00")}, "BUY requires"),
    ({"side": "SELL"}, "SELL requires"),
    ({"entry_price": Decimal("100.03")}, "not a multiple of tick size"),
    ({"stop_loss": Decimal("98.52")}, "not a multiple of tick size"),
    ({"target": Decimal("103.77")}, "not a multiple of tick size"),
    ({"expires_at": NOW - timedelta(seconds=1), "signal_time": NOW - timedelta(minutes=31)}, "expired"),
    ({"exit_policy": "made_up_policy"}, "Unknown ATR exit policy"),
    ({"product": "DELIVERY"}, "does not match horizon"),
    ({"horizon": "WEEKLY"}, "Unknown horizon"),
    ({"atr_timeframe": ""}, "atr_timeframe is required"),
    ({"atr_value": Decimal("0")}, "atr_value"),
    ({"lot_size": 0}, "lot_size"),
    ({"signal_time": datetime(2026, 11, 16, 5, 0)}, "timezone-aware"),
    ({"expires_at": NOW}, "expired"),
])
def test_candidate_fails_closed(overrides, message):
    with pytest.raises(CandidateInvalid, match=message):
        make_candidate(**overrides).validate(quantity=10, now=NOW)


def test_quantity_must_be_a_lot_multiple():
    candidate = make_candidate(lot_size=75)
    candidate.validate(quantity=150, now=NOW)
    with pytest.raises(CandidateInvalid, match="lot size"):
        candidate.validate(quantity=100, now=NOW)
    with pytest.raises(CandidateInvalid, match="positive integer"):
        candidate.validate(quantity=0, now=NOW)


def test_asset_identity_rejects_unknown_values():
    for kwargs in (
        dict(venue="MARS", segment="CASH", currency="INR", symbol="X"),
        dict(venue="NSE", segment="FUT", currency="INR", symbol="X"),
        dict(venue="NSE", segment="CASH", currency="EUR", symbol="X"),
        dict(venue="NSE", segment="CASH", currency="INR", symbol=" "),
    ):
        with pytest.raises(CandidateInvalid):
            AssetId(**kwargs)


def test_horizon_and_product_are_separate_concepts():
    assert horizon_to_product("INTRADAY") == "INTRADAY"
    for horizon in ("SWING", "BTST", "POSITIONAL"):
        assert horizon_to_product(horizon) == "DELIVERY"
    assert horizon_to_product("DELIVERY") == "DELIVERY"  # legacy rows only
    with pytest.raises(CandidateInvalid):
        horizon_to_product("MONTHLY")
    assert normalize_horizon("SWING") == "SWING"
    with pytest.raises(CandidateInvalid):
        normalize_horizon("DELIVERY")  # not valid for new decisions
    assert normalize_horizon("DELIVERY", legacy_rows=True) == "SWING"


def test_tick_rounding_is_decimal_exact():
    assert round_to_tick("100.0300000001", "0.05", ROUND_CEILING) == Decimal("100.05")
    assert round_to_tick(98.52, "0.05", ROUND_FLOOR) == Decimal("98.50")
    assert round_to_tick(0.1 + 0.2, "0.05", ROUND_FLOOR) == Decimal("0.30")


# ── registry ──────────────────────────────────────────────────────────────────

def test_registry_keeps_composite_trend_unchanged_and_fails_closed_for_unknowns():
    from strategies.contracts import (
        COMPOSITE_TREND_REVIEW, STRATEGY_CONTRACTS, require_strategy_contract,
        require_strategy_review_contract,
    )

    contract = STRATEGY_CONTRACTS["CompositeTrend"]
    assert contract.horizons == ("INTRADAY",) and contract.asset_venues == ("NSE",)
    assert contract.review is COMPOSITE_TREND_REVIEW and contract.review.version == "review-v2"
    assert contract.enabled_setting is None and contract.default_exit_policy == "daily_atr_baseline_v1"
    assert require_strategy_review_contract({"strategy": "CompositeTrend"}) is COMPOSITE_TREND_REVIEW
    for unknown in ("Mystery", None, 5):
        with pytest.raises(ValueError):
            require_strategy_contract(unknown)
        with pytest.raises(ValueError):
            require_strategy_review_contract({"strategy": unknown})
    with pytest.raises(ValueError):
        require_strategy_review_contract(None)


def test_a_disabled_contract_cannot_enter(monkeypatch):
    from strategies import contracts

    sleeve = contracts.StrategyContract(
        name="Sleeve", version="v1", horizons=("SWING",), asset_venues=("NSE",),
        review=contracts.COMPOSITE_TREND_REVIEW, enabled_setting="paper_partial_fill_enabled",
        default_exit_policy="daily_atr_baseline_v1",
    )
    monkeypatch.setitem(contracts.STRATEGY_CONTRACTS, "Sleeve", sleeve)
    with pytest.raises(ValueError, match="disabled"):
        contracts.require_strategy_contract("Sleeve")
    monkeypatch.setattr(settings, "paper_partial_fill_enabled", True)
    assert contracts.require_strategy_contract("Sleeve") is sleeve


# ── router ────────────────────────────────────────────────────────────────────

def _router_state(strategy="CompositeTrend", holding_period="INTRADAY"):
    from agents.graph import _order_router_node  # noqa: F401  (import check)
    return None


def test_entry_validation_accepts_only_contract_horizons():
    from agents.graph import _validate_entry_decision

    base = {"price": 100, "stop_loss": 95, "target": 110}
    assert _validate_entry_decision({**base, "holding_period": "INTRADAY"}, "BUY") is None
    assert _validate_entry_decision({**base, "holding_period": "SWING"}, "BUY") is None
    assert _validate_entry_decision(
        {**base, "holding_period": "SWING"}, "BUY", allowed_horizons=("INTRADAY",),
    ) is not None
    assert _validate_entry_decision({**base, "holding_period": "DELIVERY"}, "BUY") is not None


def test_tick_alignment_moves_levels_toward_the_entry_only():
    from agents.graph import _tick_aligned_levels

    tick = Decimal("0.05")
    assert _tick_aligned_levels("BUY", 100.03, 98.52, 103.77, tick) == (100.05, 98.55, 103.75)
    assert _tick_aligned_levels("SELL", 100.0, 101.52, 96.28, tick) == (100.0, 101.5, 96.3)
    assert _tick_aligned_levels("BUY", 100.0, 98.5, 103.75, tick) == (100.0, 98.5, 103.75)  # unchanged


def _entry_state(monkeypatch, tmp_path, strategy="CompositeTrend", horizon="INTRADAY", persist_strategy="CompositeTrend", technical_extra=None):
    from datetime import datetime as real_datetime

    import numpy as np
    import pandas as pd

    from agents import graph
    from agents.portfolio_manager_agent import PortfolioManagerAgent
    from broker.portfolio_tracker import portfolio_tracker
    from data import indicators
    from operations.candidate_reviews import persist_candidate_reviews
    from operations.control_state import ControlStateStore
    from risk import dead_mans_switch as dms
    from risk import hmm_regime_classifier as hmm
    from risk import daily_portfolio_risk as daily_risk
    from operations import control_state as controls
    from tests.sentiment_fixtures import reviewed_debate, reviewed_decision, sentiment_state
    from tests.test_hmm_regime import _sector_data

    class MidSession(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return graph.IST.localize(real_datetime(2026, 11, 12, 11, 0))

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.set_paused(False)
    monkeypatch.setattr(controls, "control_state", store)
    monkeypatch.setattr(dms, "dead_mans_switch", dms.DeadMansSwitch(store=store))
    monkeypatch.setattr(hmm, "entry_regime_block_reason", lambda _r: None)
    monkeypatch.setattr(graph, "datetime", MidSession)
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(portfolio_tracker, "daily_entry_count", lambda: 0)
    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "update_unrealised_pnl", lambda: True)
    monkeypatch.setattr(graph, "get_token", lambda _s: "token")
    monkeypatch.setattr(indicators, "average_daily_turnover_inr", lambda *_a, **_k: 1_000_000_000.0)
    monkeypatch.setattr(daily_risk, "daily_returns", lambda *_a: pd.Series(np.linspace(-0.01, 0.01, 20)))
    calls = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: calls.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": len(calls),
    })
    symbol = "OK"
    evidence = sentiment_state([symbol])
    evidence["technical_signals"] = {symbol: {
        **evidence["technical_signals"][symbol],
        "signal": "BUY", "strategy": persist_strategy, "stop_loss": 97, "target": 109,
        "indicators": {"close": 100, "daily_atr_14": 3.0}, **(technical_extra or {}),
    }}
    risk = {symbol: {
        "approved": True, "deterministic_action": "BUY", "stop_loss": 97, "target": 109,
        "holding_period": horizon, "vix_factor": 1.0, **_sector_data(),
    }}
    state = {
        "execution_enabled": True, "portfolio_snapshot_valid": True, **evidence,
        "watchlist": [symbol], "screener_results": {"qualified": [symbol], "rejected": []},
        "current_positions": {}, "risk_assessment": risk,
        "debate_results": {symbol: reviewed_debate(evidence, symbol, "BUY")},
        "hmm_regime": {"degraded": False, "action_directive": {
            "allow_new_positions": True, "max_position_size_pct": 0.1,
            "allowed_strategies": ["trend_following"], "hedge_required": False,
        }},
        "available_capital": 25_000, "portfolio_value": 25_000,
    }
    decision = {"symbol": symbol, "action": "BUY", "quantity": 10, "price": 100, "stop_loss": 97,
                "target": 109, "holding_period": horizon}
    state["trading_decisions"] = [reviewed_decision(evidence, decision)]
    unique = {d["candidate_id"]: d for d in state["trading_decisions"]}
    context = PortfolioManagerAgent._portfolio_context(state, risk)
    persist_candidate_reviews(state, list(unique.values()), context)
    state["technical_signals"][symbol]["strategy"] = strategy  # swapped after the review was stored
    return graph, state, calls


def test_composite_trend_entry_carries_horizon_product_and_policy_metadata(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    result = graph._order_router_node(state)
    assert len(calls) == 1, result["errors"]
    call = calls[0]
    assert (call["price"], call["stop_loss"], call["target"]) == (100, 97.0, 109.0)
    assert call["product_type"] == "INTRADAY" and call["horizon"] == "INTRADAY"
    assert call["strategy_version"] == "composite-trend-v1"
    assert call["exit_policy"] == "daily_atr_baseline_v1"
    assert call["atr_timeframe"] == "DAILY_14_COMPLETED_SESSIONS"
    assert call["book"] == "core"


def test_unknown_strategy_blocks_entries(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path, strategy="Mystery")
    result = graph._order_router_node(state)
    assert calls == []
    assert result["errors"] and all("[OrderRouter]" in error for error in result["errors"])


def test_horizon_outside_the_contract_blocks_the_entry(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path, horizon="SWING")
    result = graph._order_router_node(state)
    assert calls == []
    assert any("Unsupported horizon SWING" in error for error in result["errors"]), result["errors"]


def test_missing_atr_blocks_the_candidate():
    from agents.graph import _build_trade_candidate
    from strategies.contracts import COMPOSITE_TREND

    kwargs = dict(
        symbol="OK", action="BUY", technical={"indicators": {"close": 100}}, contract=COMPOSITE_TREND,
        quantity=10, tick=Decimal("0.05"), lot=1, now=NOW,
        decision={"price": 100, "stop_loss": 97, "target": 109, "holding_period": "INTRADAY"},
    )
    with pytest.raises(CandidateInvalid, match="atr"):
        _build_trade_candidate(atr_value=None, **kwargs)
    candidate = _build_trade_candidate(atr_value=3.0, **kwargs)
    assert candidate.exit_policy == "daily_atr_baseline_v1" and candidate.horizon == "INTRADAY"


def test_unknown_strategy_never_blocks_an_exit(monkeypatch):
    from agents import graph
    from risk import stop_loss_manager as slm

    placed = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: placed.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": 1, "quantity": kw["quantity"],
    })
    monkeypatch.setattr(graph, "get_token", lambda _s: "token")
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(slm.stop_loss_manager, "pending_exits", lambda: {})
    state = {
        "execution_enabled": True,
        "current_positions": {"OLD": {"quantity": 3, "entry_trade_ids": [1], "holding_period": "DELIVERY"}},
        "technical_signals": {"OLD": {"strategy": "RetiredStrategy"}},
        "trading_decisions": [{"symbol": "OLD", "action": "SELL", "quantity": 3, "price": 100,
                               "holding_period": "DELIVERY"}],
    }
    graph._order_router_node(state)
    assert [(c["symbol"], c["product_type"], c["is_exit"]) for c in placed] == [("OLD", "DELIVERY", True)]


# ── persistence and legacy rows ───────────────────────────────────────────────

@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 't06.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "max_daily_entries", 100)
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 100)
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 0)
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_a: {
        "price": 100.0, "source": "ws_tick", "age_seconds": 1.0,
    })
    yield engine
    engine.dispose()


def test_trade_row_persists_horizon_policy_and_version(paper_db):
    result = order_manager.place_order(
        "TEST", "token", "BUY", 5, price=100, product_type="DELIVERY", stop_loss=95, target=110,
        strategy="Sleeve", horizon="SWING", strategy_version="v7", exit_policy="daily_atr_baseline_v1",
        atr_timeframe="DAILY_14_COMPLETED_SESSIONS", book="sleeve_a", expected_paper_book={},
    )
    assert result["holding_period"] == "SWING" and result["horizon"] == "SWING"
    with db.get_db() as session:
        trade = session.query(Trade).one()
        assert (trade.holding_period, trade.horizon, trade.strategy_version) == ("SWING", "SWING", "v7")
        assert (trade.exit_policy, trade.atr_timeframe, trade.book) == (
            "daily_atr_baseline_v1", "DAILY_14_COMPLETED_SESSIONS", "sleeve_a",
        )
        position = summarize_positions(replay_orders([trade]))["TEST"]
    assert position["holding_period"] == "SWING" and position["exit_policy"] == "daily_atr_baseline_v1"


def test_horizon_must_match_the_product(paper_db):
    with pytest.raises(ValueError, match="cannot be carried"):
        order_manager.place_order(
            "TEST", "token", "BUY", 5, price=100, product_type="INTRADAY", stop_loss=95,
            horizon="SWING", expected_paper_book={},
        )


def test_legacy_delivery_row_still_exits_with_the_delivery_product(paper_db):
    with db.get_db() as session:
        session.add(Trade(
            timestamp=datetime.now(), symbol="OLD", action="BUY", quantity=4, price=100.0,
            holding_period="DELIVERY", stop_loss=95, status="EXECUTED", is_paper=True, order_id="L1",
        ))
        lots = replay_orders(session.query(Trade).all()).open_lots["OLD"]
        ids = [lot.trade_id for lot in lots]
    result = order_manager.place_order(
        "OLD", "token", "SELL", 4, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    assert result is not None and result["status"] == "EXECUTED"
    # An exit with the wrong product for a delivery lot is refused.
    with db.get_db() as session:
        session.add(Trade(
            timestamp=datetime.now(), symbol="OLD2", action="BUY", quantity=4, price=100.0,
            holding_period="SWING", stop_loss=95, status="EXECUTED", is_paper=True, order_id="L2",
        ))
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).filter(Trade.symbol == "OLD2").all()).open_lots["OLD2"]]
    assert order_manager.place_order(
        "OLD2", "token", "SELL", 4, price=100, product_type="INTRADAY", is_exit=True,
        expected_entry_trade_ids=ids,
    ) is None


def test_init_db_adds_new_columns_to_an_existing_database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.sqlite'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY, timestamp DATETIME NOT NULL, symbol VARCHAR(50) NOT NULL,"
            " exchange VARCHAR(10), action VARCHAR(10) NOT NULL, quantity INTEGER NOT NULL, order_type VARCHAR(20),"
            " price FLOAT NOT NULL, holding_period VARCHAR(20), status VARCHAR(20), is_paper BOOLEAN)"
        ))
        connection.execute(text(
            "CREATE TABLE closed_trades (id INTEGER PRIMARY KEY, entry_order_id INTEGER NOT NULL,"
            " exit_order_id INTEGER NOT NULL, symbol VARCHAR(50) NOT NULL, action VARCHAR(10) NOT NULL,"
            " quantity INTEGER NOT NULL, entry_price FLOAT NOT NULL, exit_price FLOAT NOT NULL,"
            " entry_timestamp DATETIME NOT NULL, exit_timestamp DATETIME NOT NULL, gross_pnl FLOAT NOT NULL,"
            " strategy VARCHAR(50) NOT NULL, holding_period VARCHAR(20) NOT NULL, is_paper BOOLEAN NOT NULL)"
        ))
    monkeypatch.setattr(db, "engine", engine)
    db.init_db()
    trade_columns = {c["name"] for c in inspect(engine).get_columns("trades")}
    closed_columns = {c["name"] for c in inspect(engine).get_columns("closed_trades")}
    assert {"horizon", "strategy_version", "exit_policy", "atr_timeframe", "book", "atr", "sector"} <= trade_columns
    assert {"venue", "currency", "exit_policy", "book"} <= closed_columns
    db.init_db()  # idempotent
    engine.dispose()


# ── square-off is horizon-aware ───────────────────────────────────────────────

def test_square_off_and_time_exit_leave_swing_positions_alone(monkeypatch, tmp_path):
    import asyncio

    import pytz

    from broker.portfolio_tracker import PortfolioTracker
    from operations.control_state import ControlStateStore
    from risk.dead_mans_switch import DeadMansSwitch
    from risk.stop_loss_manager import StopLossManager
    from scheduler import market_scheduler as ms

    store = ControlStateStore(tmp_path / "sq.sqlite")
    tracker = PortfolioTracker(store)
    monkeypatch.setattr(ms, "portfolio_tracker", tracker)
    monkeypatch.setattr(ms, "stop_loss_manager", StopLossManager(store))
    monkeypatch.setattr(ms, "dead_mans_switch", DeadMansSwitch(store=store))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(tracker, "sync_from_broker", lambda: None)
    monkeypatch.setattr(tracker, "update_unrealised_pnl", lambda: None)
    tracker._positions = {
        "IN": {"symbol": "IN", "quantity": 1, "avg_price": 100.0, "ltp": 100.0, "holding_period": "INTRADAY"},
        "SW": {"symbol": "SW", "quantity": 1, "avg_price": 100.0, "ltp": 100.0, "holding_period": "SWING"},
        "BT": {"symbol": "BT", "quantity": 1, "avg_price": 100.0, "ltp": 100.0, "holding_period": "BTST"},
        "PO": {"symbol": "PO", "quantity": 1, "avg_price": 100.0, "ltp": 100.0, "holding_period": "POSITIONAL"},
        "LG": {"symbol": "LG", "quantity": 1, "avg_price": 100.0, "ltp": 100.0, "holding_period": "DELIVERY"},
    }
    exited = []

    def close(symbol, *_args):
        exited.append(symbol)
        tracker._positions.pop(symbol, None)
        return {"status": "EXECUTED"}

    scheduler = ms.MarketScheduler()
    monkeypatch.setattr(scheduler, "_submit_protective_exit", lambda symbol, action, pos: close(symbol))
    asyncio.run(scheduler._intraday_squareoff())
    assert exited == ["IN"]
    assert set(tracker._positions) == {"SW", "BT", "PO", "LG"}


def test_protective_time_exit_is_intraday_only():
    from risk.stop_loss_manager import StopLossManager
    import risk.stop_loss_manager as module
    import inspect as pyinspect

    source = pyinspect.getsource(StopLossManager._time_exit)
    assert '!= "INTRADAY"' in source  # anything not INTRADAY returns before the 15:15 rule
