"""T09: paper settlement, daily P&L continuity, BTST delivery, corporate-action blocks, attribution."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from broker import cash_ledger, order_manager
from broker.paper_ledger import replay_orders
from broker.portfolio_tracker import PortfolioTracker
from config.settings import settings
from database import db
from database.models import (
    Base, CashLedgerEntry, ClosedTrade, CorporateAction, DailyPnL, PaperFillAudit, Trade,
)
from operations import corporate_actions
from operations.control_state import ControlStateStore

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 't09.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "max_daily_entries", 100)
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 100)
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 0)
    price = {"value": 100.0}
    monkeypatch.setattr(order_manager, "_market_reference", lambda *_a: {
        "price": price["value"], "source": "ws_tick", "age_seconds": 1.0,
    })
    engine.price = price
    yield engine
    engine.dispose()


def _clock(monkeypatch, when: datetime):
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return IST.localize(when) if tz is not None else when

    monkeypatch.setattr(order_manager, "datetime", Fixed)
    import broker.portfolio_tracker as tracker_module

    monkeypatch.setattr(tracker_module, "datetime", Fixed)


def _add_trade(**kwargs) -> int:
    defaults = dict(
        timestamp=datetime(2026, 11, 12, 10, 0), symbol="TEST", action="BUY", quantity=10, price=100.0,
        holding_period="INTRADAY", status="EXECUTED", is_paper=True, order_id="X", stop_loss=95, target=120,
    )
    defaults.update(kwargs)
    with db.get_db() as session:
        trade = Trade(**defaults)
        session.add(trade)
        session.flush()
        return trade.id


# ── settlement ───────────────────────────────────────────────────────────────

def test_intraday_proceeds_net_the_same_day_and_delivery_proceeds_settle_next_session(paper_db):
    intraday = _add_trade(action="SELL", holding_period="INTRADAY", order_id="a")
    swing = _add_trade(action="SELL", holding_period="SWING", symbol="SW", order_id="b",
                       timestamp=datetime(2026, 11, 13, 14, 0))  # a Friday
    with db.get_db() as session:
        cash_ledger.backfill_missing(session, PortfolioTracker._trade_cost)
        rows = {row.trade_id: row for row in session.query(CashLedgerEntry).filter(CashLedgerEntry.kind == "TRADE")}
        assert rows[intraday].settled is True and rows[intraday].settles_on is None
        assert rows[swing].settled is False and rows[swing].settles_on == "2026-11-16"  # Monday
        assert cash_ledger.unsettled_cash(session) == 1000.0
        assert cash_ledger.settle_due(session, date(2026, 11, 14)) == 0  # weekend: still waiting
        assert cash_ledger.settle_due(session, date(2026, 11, 16)) == 1
        assert cash_ledger.unsettled_cash(session) == 0.0


def test_backfill_is_idempotent_and_records_fees(paper_db):
    trade = _add_trade()
    with db.get_db() as session:
        assert cash_ledger.backfill_missing(session, PortfolioTracker._trade_cost) == 1
        assert cash_ledger.backfill_missing(session, PortfolioTracker._trade_cost) == 0
        kinds = sorted(row.kind for row in session.query(CashLedgerEntry).filter(CashLedgerEntry.trade_id == trade))
        assert kinds == ["FEE", "TRADE"]
        fee = session.query(CashLedgerEntry).filter(CashLedgerEntry.kind == "FEE").one()
        assert fee.amount < 0


def test_settlement_skips_exchange_holidays(paper_db):
    assert cash_ledger.settlement_date(date(2026, 11, 23)) == "2026-11-25"  # 24 Nov 2026 is a holiday
    assert cash_ledger.settlement_date(date(2031, 6, 6)) == "2031-06-09"  # unknown calendar: next weekday


def test_sizing_uses_settled_cash_only_after_a_delivery_sale(paper_db, monkeypatch):
    _clock(monkeypatch, datetime(2026, 11, 13, 14, 30))
    paper_db.price["value"] = 100.0
    assert order_manager.place_order(
        "TEST", "token", "BUY", 50, price=100, product_type="DELIVERY", horizon="SWING",
        stop_loss=95, target=120, expected_paper_book={},
    )
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    assert order_manager.place_order(
        "TEST", "token", "SELL", 50, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    tracker = PortfolioTracker()
    tracker.sync_from_broker()
    snapshot = tracker.snapshot()
    assert snapshot["unsettled_cash"] == 5000.0
    assert snapshot["settled_cash"] == pytest.approx(tracker._available_capital - 5000.0)
    assert tracker.available_capital == pytest.approx(snapshot["settled_cash"], abs=0.01)
    assert tracker.total_value == pytest.approx(tracker._available_capital + tracker._cost_basis())
    # Next session: the proceeds settle and become available again.
    _clock(monkeypatch, datetime(2026, 11, 16, 10, 0))
    tracker.sync_from_broker()
    assert tracker.unsettled_cash == 0.0
    assert tracker.available_capital == pytest.approx(tracker._available_capital)


def test_intraday_round_trip_leaves_no_unsettled_cash(paper_db, monkeypatch):
    _clock(monkeypatch, datetime(2026, 11, 13, 11, 0))
    assert order_manager.place_order(
        "TEST", "token", "BUY", 10, price=100, product_type="INTRADAY", stop_loss=95, expected_paper_book={},
    )
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    assert order_manager.place_order(
        "TEST", "token", "SELL", 10, price=100, product_type="INTRADAY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    tracker = PortfolioTracker()
    tracker.sync_from_broker()
    assert tracker.unsettled_cash == 0.0 and tracker.snapshot()["unsettled_cash"] == 0.0


# ── daily P&L continuity ─────────────────────────────────────────────────────

def _carried_swing(paper_db, monkeypatch, tmp_path, *, record_marks: bool):
    from backtester.costs import calculate_indian_transaction_cost

    _clock(monkeypatch, datetime(2026, 11, 13, 10, 0))
    trade_id = _add_trade(
        symbol="SWG", quantity=100, price=100.0, holding_period="SWING",
        timestamp=datetime(2026, 11, 12, 10, 0), order_id="swg",
    )
    entry_fee = calculate_indian_transaction_cost(100.0, 100, is_buy=True, is_intraday=False)
    equity = settings.initial_capital + (110.0 - 100.0) * 100 - entry_fee  # marked at 110 yesterday
    with db.get_db() as session:
        session.add(DailyPnL(date="2026-11-12", portfolio_value=equity))
    store = ControlStateStore(tmp_path / "marks.sqlite3")
    if record_marks:
        store.write("closing_marks", {"date": "2026-11-12", "marks": {
            "SWG": {"ltp": 110.0, "quantity": 100, "entry_trade_ids": [trade_id]},
        }})
    return PortfolioTracker(store), equity


def test_a_carried_gain_is_not_counted_as_todays_gain(paper_db, monkeypatch, tmp_path):
    tracker, equity = _carried_swing(paper_db, monkeypatch, tmp_path, record_marks=True)
    tracker.sync_from_broker()
    position = tracker.positions["SWG"]
    assert position["ltp"] == 110.0 and position["mark_source"] == "previous_close"
    assert position["mark_at"] is None  # not a fresh mark: fresh-mark entry gates still apply
    assert tracker.daily_pnl == pytest.approx(0.0, abs=1e-6)
    tracker.mark_price("SWG", 110.0)  # today's mark equals yesterday's
    assert tracker.daily_pnl == pytest.approx(0.0, abs=1e-6)
    assert tracker.snapshot()["day_start_equity"] == pytest.approx(equity)
    assert tracker.snapshot()["daily_loss_baseline_valid"] is True
    tracker.mark_price("SWG", 111.0)
    assert tracker.daily_pnl == pytest.approx(100.0, abs=1e-6)


def test_missing_closing_marks_make_the_baseline_unavailable_not_a_guess(paper_db, monkeypatch, tmp_path):
    tracker, _equity = _carried_swing(paper_db, monkeypatch, tmp_path, record_marks=False)
    tracker.sync_from_broker()
    assert tracker.snapshot()["daily_loss_baseline_valid"] is False
    assert tracker.has_hit_daily_loss_limit() is True  # entries blocked while it lasts, exits unaffected


def test_closing_marks_are_saved_with_the_daily_snapshot(paper_db, monkeypatch, tmp_path):
    store = ControlStateStore(tmp_path / "save.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._positions = {
        "FRESH": {"quantity": 5, "ltp": 101.0, "mark_at": "2026-11-13T15:29:00+05:30", "entry_trade_ids": [3]},
        "STALE": {"quantity": 5, "ltp": 90.0, "mark_at": None, "entry_trade_ids": [4]},
    }
    tracker.save_daily_snapshot()
    record = store.read("closing_marks")
    assert set(record["marks"]) == {"FRESH"} and record["marks"]["FRESH"]["ltp"] == 101.0


# ── BTST delivery ────────────────────────────────────────────────────────────

def _btst_position(paper_db, monkeypatch):
    _clock(monkeypatch, datetime(2026, 11, 12, 15, 10))  # Thursday
    assert order_manager.place_order(
        "TEST", "token", "BUY", 10, price=100, product_type="DELIVERY", horizon="BTST",
        stop_loss=95, target=110, expected_paper_book={},
    )
    with db.get_db() as session:
        return [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]


def test_a_btst_exit_before_delivery_is_refused_then_allowed_after_it(paper_db, monkeypatch):
    ids = _btst_position(paper_db, monkeypatch)

    def sell():
        return order_manager.place_order(
            "TEST", "token", "SELL", 10, price=100, product_type="DELIVERY", is_exit=True,
            expected_entry_trade_ids=ids,
        )

    _clock(monkeypatch, datetime(2026, 11, 12, 15, 20))  # same day: not delivered
    assert sell() is None
    _clock(monkeypatch, datetime(2026, 11, 13, 9, 20))  # next open, before the 15-minute delivery model
    assert sell() is None
    with db.get_db() as session:
        assert [a.outcome for a in session.query(PaperFillAudit).order_by(PaperFillAudit.id)][-2:] == [
            "SHORT_DELIVERY_RISK", "SHORT_DELIVERY_RISK",
        ]
        assert session.query(Trade).count() == 1  # nothing was sold
    _clock(monkeypatch, datetime(2026, 11, 13, 9, 30))
    assert sell() is not None


def test_non_btst_delivery_exits_are_never_blocked_by_the_delivery_model(paper_db, monkeypatch):
    _clock(monkeypatch, datetime(2026, 11, 12, 14, 0))
    assert order_manager.place_order(
        "TEST", "token", "BUY", 10, price=100, product_type="DELIVERY", horizon="SWING",
        stop_loss=95, target=110, expected_paper_book={},
    )
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    _clock(monkeypatch, datetime(2026, 11, 12, 14, 30))
    assert order_manager.place_order(
        "TEST", "token", "SELL", 10, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )


def test_a_short_delivery_exit_is_retried_by_the_next_protective_pass(paper_db, monkeypatch, tmp_path):
    """The scheduler keeps the position protected and does not mark it exited."""
    from risk.stop_loss_manager import StopLossManager

    ids = _btst_position(paper_db, monkeypatch)
    manager = StopLossManager()
    manager.register_trade({
        "symbol": "TEST", "action": "BUY", "price": 100, "stop_loss": 95, "target": 110, "atr": 2,
        "holding_period": "BTST", "entry_trade_ids": ids,
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
    })
    _clock(monkeypatch, datetime(2026, 11, 13, 9, 17))
    result = order_manager.place_order(
        "TEST", "token", "SELL", 10, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    assert result is None and "TEST" in manager.snapshot()  # still tracked
    _clock(monkeypatch, datetime(2026, 11, 13, 9, 45))
    assert order_manager.place_order(
        "TEST", "token", "SELL", 10, price=100, product_type="DELIVERY", is_exit=True,
        expected_entry_trade_ids=ids,
    )


def test_paper_sell_entries_remain_intraday_only():
    import inspect

    from agents import graph

    source = inspect.getsource(graph._order_router_node)
    assert 'holding_period == "INTRADAY"' in source and "Naked short sale blocked" in source


# ── corporate actions ────────────────────────────────────────────────────────

def test_unprocessed_action_blocks_entries_but_not_exits(paper_db, monkeypatch, caplog):
    assert corporate_actions.import_actions([
        {"symbol": "OK", "action_type": "split", "ex_date": "2026-11-12", "source": "NSE announcement (test)"},
        {"symbol": "OK", "action_type": "SPLIT", "ex_date": "2026-11-12", "source": "duplicate"},
    ]) == 1
    assert corporate_actions.blocking_action("OK", date(2026, 11, 11)) is None  # ex-date not reached
    assert corporate_actions.blocking_action("OK", date(2026, 11, 12))["action_type"] == "SPLIT"
    assert corporate_actions.blocking_action("OTHER", date(2026, 11, 12)) is None
    assert corporate_actions.mark_processed("OK", "SPLIT", "2026-11-12") is True
    assert corporate_actions.blocking_action("OK", date(2026, 11, 12)) is None


def test_corporate_action_rows_need_a_source_and_a_known_type(paper_db):
    for bad in (
        {"symbol": "X", "action_type": "SPLIT", "ex_date": "2026-11-12"},
        {"symbol": "X", "action_type": "MAGIC", "ex_date": "2026-11-12", "source": "s"},
        {"symbol": "", "action_type": "SPLIT", "ex_date": "2026-11-12", "source": "s"},
    ):
        with pytest.raises(ValueError):
            corporate_actions.import_actions([bad])
    with pytest.raises(ValueError):
        corporate_actions.import_actions([{"symbol": "X", "action_type": "SPLIT", "ex_date": "12-11-2026", "source": "s"}])


def test_router_blocks_an_entry_with_an_unprocessed_action_but_routes_the_exit(paper_db, monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    corporate_actions.import_actions([
        {"symbol": "OK", "action_type": "BONUS", "ex_date": "2026-11-12", "source": "NSE announcement (test)"},
    ])
    result = graph._order_router_node(state)
    assert calls == []
    assert any("unprocessed corporate action BONUS" in error for error in result["errors"]), result["errors"]
    corporate_actions.mark_processed("OK", "BONUS", "2026-11-12")
    assert graph._order_router_node(state)["executed_orders"]

    corporate_actions.import_actions([
        {"symbol": "HELD", "action_type": "SPLIT", "ex_date": "2026-11-01", "source": "NSE announcement (test)"},
    ])
    from risk import stop_loss_manager as slm

    monkeypatch.setattr(slm.stop_loss_manager, "pending_exits", lambda: {})
    exit_state = {
        "execution_enabled": True,
        "current_positions": {"HELD": {"quantity": 2, "entry_trade_ids": [1], "holding_period": "INTRADAY"}},
        "trading_decisions": [{"symbol": "HELD", "action": "SELL", "quantity": 2, "price": 100,
                               "holding_period": "INTRADAY"}],
    }
    calls.clear()
    graph._order_router_node(exit_state)
    assert [(c["symbol"], c["is_exit"]) for c in calls] == [("HELD", True)]


def test_an_unreadable_corporate_action_table_blocks_entries(paper_db, monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(corporate_actions, "blocking_action", broken)
    result = graph._order_router_node(state)
    assert calls == [] and any("corporate action check unavailable" in e for e in result["errors"])


# ── attribution ──────────────────────────────────────────────────────────────

def test_closed_trades_carry_venue_currency_policy_and_book(paper_db, monkeypatch):
    _clock(monkeypatch, datetime(2026, 11, 13, 11, 0))
    assert order_manager.place_order(
        "TEST", "token", "BUY", 10, price=100, product_type="INTRADAY", horizon="INTRADAY",
        stop_loss=95, strategy="CompositeTrend", exit_policy="daily_atr_baseline_v1",
        atr_timeframe="DAILY_14_COMPLETED_SESSIONS", book="core", expected_paper_book={},
    )
    with db.get_db() as session:
        ids = [lot.trade_id for lot in replay_orders(session.query(Trade).all()).open_lots["TEST"]]
    paper_db.price["value"] = 101.0
    assert order_manager.place_order(
        "TEST", "token", "SELL", 10, price=101, product_type="INTRADAY", is_exit=True,
        expected_entry_trade_ids=ids,
    )
    with db.get_db() as session:
        closed = session.query(ClosedTrade).one()
        assert (closed.venue, closed.currency, closed.exit_policy, closed.book) == (
            "NSE", "INR", "daily_atr_baseline_v1", "core",
        )
        assert closed.strategy == "CompositeTrend" and closed.holding_period == "INTRADAY"
