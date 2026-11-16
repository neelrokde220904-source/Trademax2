"""Paper fills, FIFO ledger, reconciliation and resume gates, without external calls."""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from broker import order_manager
from broker.paper_ledger import position_book_identity, replay_orders, summarize_positions
from broker.portfolio_tracker import PortfolioTracker, portfolio_tracker
from config.settings import settings
from database import db
from database.models import Base, ClosedTrade, PaperFillAudit, Trade
from operations.control_state import ControlStateStore
from operations.paper_reconciliation import IST, reconcile_paper
from risk.dead_mans_switch import DeadMansSwitch
from risk.stop_loss_manager import StopLossManager
from scheduler import market_scheduler as scheduler_module


@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'paper.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "slippage_pct", 0.0)
    monkeypatch.setattr(settings, "max_daily_entries", 100)
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 100)
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 0)
    yield engine
    engine.dispose()


def _references(monkeypatch, *prices):
    queue = list(prices)
    def reference(*_args):
        price = queue.pop(0)
        return None if price is None else {"price": price, "source": "ws_tick", "age_seconds": 1.0}
    monkeypatch.setattr(order_manager, "_market_reference", reference)


def _order(action, quantity, price, **kwargs):
    if not kwargs.get("is_exit"):
        kwargs.setdefault("expected_paper_book", _paper_book())
    return order_manager.place_order(
        "TEST", "token", action, quantity, price=price, product_type="INTRADAY",
        strategy="test", **kwargs,
    )


def _paper_book():
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED", Trade.is_paper == True).all()
        return position_book_identity(summarize_positions(replay_orders(trades)))


def _entry_ids():
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED", Trade.is_paper == True).all()
        return [lot.trade_id for lot in replay_orders(trades).open_lots.get("TEST", [])]


def _morning():
    return datetime.now(IST).replace(hour=10, minute=0, second=0, microsecond=0)


def _process_order(database_path, barrier, results, exit_ids=None, symbol="TEST", daily_cap=100, symbol_cap=1):
    engine = create_engine(f"sqlite:///{database_path}", connect_args={"check_same_thread": False})
    db.engine = engine
    db.SessionLocal = sessionmaker(bind=engine)
    settings.paper_trading = True
    settings.slippage_pct = 0.0
    settings.max_daily_entries = daily_cap
    settings.max_daily_entries_per_symbol = symbol_cap
    settings.symbol_reentry_cooldown_minutes = 0
    order_manager._market_reference = (
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0}
    )
    count = portfolio_tracker.daily_entry_count

    def delayed_count(*args, **kwargs):
        result = count(*args, **kwargs)
        time.sleep(0.15)
        return result

    portfolio_tracker.daily_entry_count = delayed_count
    barrier.wait(timeout=10)
    try:
        if exit_ids is None:
            order = order_manager.place_order(
                symbol, "token", "BUY", 1, price=100, product_type="INTRADAY", stop_loss=95,
                expected_paper_book={},
            )
        else:
            order = _order("SELL", 1, 100.0, is_exit=True, expected_entry_trade_ids=exit_ids)
        results.put(("ok", order is not None))
    except Exception as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "symbols,daily_cap,symbol_cap",
    [(["TEST", "TEST"], 100, 1), (["TEST", "OTHER"], 1, 100)],
)
def test_processes_cannot_race_paper_entry_caps(paper_db, symbols, daily_cap, symbol_cap):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    database_path = str(paper_db.url.database)
    processes = [
        context.Process(
            target=_process_order,
            args=(database_path, barrier, results, None, symbol, daily_cap, symbol_cap),
        )
        for symbol in symbols
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
    assert sorted(outcomes) == [("ok", False), ("ok", True)]
    with db.get_db() as session:
        assert session.query(Trade).count() == 1
        assert sorted(row.outcome for row in session.query(PaperFillAudit).all()) == [
            "FILLED", "REJECTED_ENTRY_LIMIT",
        ]


def test_processes_cannot_both_enter_using_the_same_risk_book(paper_db):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_order,
            args=(str(paper_db.url.database), barrier, results, None, symbol, 100, 100),
        )
        for symbol in ("FIRST", "SECOND")
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
    assert sorted(outcomes) == [("ok", False), ("ok", True)]
    with db.get_db() as session:
        assert session.query(Trade).count() == 1
        assert sorted(row.outcome for row in session.query(PaperFillAudit).all()) == [
            "FILLED", "REJECTED_STALE_BOOK",
        ]


def test_paper_entry_requires_matching_position_snapshot(paper_db, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert order_manager.place_order("FIRST", "token", "BUY", 1, price=100, stop_loss=95) is None
    first = order_manager.place_order(
        "FIRST", "token", "BUY", 1, price=100, stop_loss=95, expected_paper_book={},
    )
    assert first is not None
    projected_book = {
        "FIRST": {"quantity": 1, "holding_period": "INTRADAY", "entry_trade_ids": [first["trade_id"]]},
    }
    assert projected_book == _paper_book()
    assert order_manager.place_order(
        "SECOND", "token", "BUY", 1, price=100, stop_loss=95, expected_paper_book={},
    ) is None
    assert order_manager.place_order(
        "SECOND", "token", "BUY", 1, price=100, stop_loss=95, expected_paper_book=projected_book,
    ) is not None
    with db.get_db() as session:
        assert sorted(row.outcome for row in session.query(PaperFillAudit).all()) == [
            "FILLED", "FILLED", "REJECTED_STALE_BOOK", "REJECTED_STALE_BOOK",
        ]


def test_paper_entry_rejects_reopened_lot_with_same_symbol_and_quantity(paper_db, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 1, 100, stop_loss=95)
    old_book = _paper_book()
    assert _order("SELL", 1, 100, is_exit=True, expected_entry_trade_ids=_entry_ids())
    assert _order("BUY", 1, 100, stop_loss=95)
    assert _paper_book()["TEST"]["quantity"] == old_book["TEST"]["quantity"]
    assert _order("BUY", 1, 100, stop_loss=95, expected_paper_book=old_book) is None


def test_processes_cannot_both_close_the_same_paper_lot(paper_db, monkeypatch):
    _references(monkeypatch, 100)
    assert _order("BUY", 1, 100, stop_loss=95)
    ids = _entry_ids()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_order, args=(str(paper_db.url.database), barrier, results, ids),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
    assert sorted(outcomes) == [("ok", False), ("ok", True)]
    with db.get_db() as session:
        assert session.query(Trade).count() == 2
        assert session.query(ClosedTrade).count() == 1
        assert sorted(row.outcome for row in session.query(PaperFillAudit).all()) == [
            "FILLED", "FILLED", "REJECTED_POSITION_CHANGED",
        ]


def test_paper_fill_and_audit_roll_back_together_on_ledger_failure(paper_db, monkeypatch):
    _references(monkeypatch, 100)

    def fail_ledger(*_args):
        raise RuntimeError("ledger failure")

    monkeypatch.setattr(order_manager, "_record_closed_lots", fail_ledger)
    with pytest.raises(RuntimeError, match="ledger failure"):
        _order("BUY", 1, 100, stop_loss=95)
    with db.get_db() as session:
        assert session.query(Trade).count() == 0
        assert session.query(PaperFillAudit).count() == 0


def test_daily_entry_count_tracks_closed_and_open_entries(paper_db):
    now = _morning()
    yesterday = now - timedelta(days=1)
    with db.get_db() as session:
        closed_entry = Trade(
            timestamp=now, symbol="CLOSED", action="BUY", quantity=4, price=100,
            status="EXECUTED", is_paper=True,
        )
        closed_exit = Trade(
            timestamp=now + timedelta(minutes=1), symbol="CLOSED", action="SELL",
            quantity=4, price=101, status="EXECUTED", is_paper=True,
        )
        open_today = Trade(
            timestamp=now, symbol="OPEN", action="BUY", quantity=2, price=200,
            status="EXECUTED", is_paper=True,
        )
        open_old = Trade(
            timestamp=yesterday, symbol="OLD", action="BUY", quantity=1, price=300,
            status="EXECUTED", is_paper=True,
        )
        session.add_all([closed_entry, closed_exit, open_today, open_old])
        session.flush()
        for quantity in (1, 3):
            session.add(ClosedTrade(
                entry_order_id=closed_entry.id, exit_order_id=closed_exit.id,
                symbol="CLOSED", action="BUY", quantity=quantity,
                entry_price=100, exit_price=101, entry_timestamp=now,
                exit_timestamp=now + timedelta(minutes=1), gross_pnl=quantity,
                strategy="test", holding_period="INTRADAY", is_paper=True,
            ))

    tracker = PortfolioTracker()
    assert tracker.daily_entry_count() == 2
    assert tracker.daily_entry_count("CLOSED") == 1
    assert tracker.daily_entry_count("OPEN") == 1
    assert tracker.daily_entry_count("OLD") == 0


def test_daily_entry_count_can_limit_one_symbol(paper_db):
    now = _morning()
    with db.get_db() as session:
        session.add_all([
            Trade(timestamp=now, symbol="FIRST", action="BUY", quantity=1, price=100,
                  status="EXECUTED", is_paper=True),
            Trade(timestamp=now, symbol="SECOND", action="BUY", quantity=1, price=100,
                  status="EXECUTED", is_paper=True),
        ])
    tracker = PortfolioTracker()
    assert tracker.daily_entry_count("FIRST") == 1
    assert tracker.daily_entry_count("SECOND") == 1
    assert tracker.daily_entry_count("OTHER") == 0


def test_legacy_closed_orders_still_count_and_enforce_caps(paper_db, monkeypatch):
    now = datetime.now(IST)
    with db.get_db() as session:
        session.add_all([
            Trade(timestamp=now, symbol="TEST", action="BUY",
                  quantity=2, price=100, status="EXECUTED", is_paper=True),
            Trade(timestamp=now, symbol="TEST", action="SELL",
                  quantity=2, price=101, status="EXECUTED", is_paper=True),
        ])
    tracker = PortfolioTracker()
    assert tracker.daily_entry_count(as_of=now) == 1
    assert tracker.daily_entry_count("TEST", as_of=now) == 1
    assert tracker.last_symbol_exit_time("TEST").replace(tzinfo=None) == (
        now
    ).replace(tzinfo=None)
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 1)
    monkeypatch.setattr(order_manager, "_market_reference",
                        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0})
    assert _order("BUY", 1, 100, stop_loss=95) is None
    with db.get_db() as session:
        assert session.query(Trade).count() == 2
        assert session.query(PaperFillAudit).one().outcome == "REJECTED_ENTRY_LIMIT"


def test_legacy_closed_orders_trigger_cooldown_without_closed_trade_row(paper_db, monkeypatch):
    now = datetime.now(IST)
    with db.get_db() as session:
        session.add_all([
            Trade(timestamp=now, symbol="TEST", action="SELL",
                  quantity=1, price=100, status="EXECUTED", is_paper=True),
            Trade(timestamp=now, symbol="TEST", action="BUY",
                  quantity=1, price=99, status="EXECUTED", is_paper=True),
        ])
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 30)
    assert "cooldown" in order_manager._paper_entry_limit_reason("TEST", now)
    assert order_manager._paper_entry_limit_reason("TEST", now + timedelta(minutes=30)) is None


def test_replayed_partial_closes_and_reversal_count_distinct_entries(paper_db):
    now = _morning()
    with db.get_db() as session:
        session.add_all([
            Trade(timestamp=now, symbol="TEST", action="BUY", quantity=4, price=100,
                  status="EXECUTED", is_paper=True),
            Trade(timestamp=now + timedelta(minutes=1), symbol="TEST", action="SELL",
                  quantity=1, price=101, status="EXECUTED", is_paper=True),
            Trade(timestamp=now + timedelta(minutes=2), symbol="TEST", action="SELL",
                  quantity=5, price=102, status="EXECUTED", is_paper=True),
            Trade(timestamp=now + timedelta(days=1), symbol="FUTURE", action="BUY",
                  quantity=1, price=100, status="EXECUTED", is_paper=True),
        ])
    tracker = PortfolioTracker()
    assert tracker.daily_entry_count(as_of=now) == 2
    assert tracker.daily_entry_count("TEST", as_of=now) == 2
    assert tracker.daily_entry_count("FUTURE", as_of=now) == 0
    assert tracker.last_symbol_exit_time("TEST") == (now + timedelta(minutes=2)).replace(tzinfo=None)


def test_concurrent_entries_cannot_exceed_symbol_daily_limit(paper_db, monkeypatch):
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 1)
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    barrier = Barrier(2)

    def entry():
        barrier.wait(timeout=5)
        return _order("BUY", 1, 100.0, stop_loss=95)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: entry(), range(2)))
    assert sum(result is not None for result in results) == 1
    with db.get_db() as session:
        assert session.query(Trade).count() == 1
        assert sorted(audit.outcome for audit in session.query(PaperFillAudit)) == [
            "FILLED", "REJECTED_ENTRY_LIMIT",
        ]


def test_concurrent_different_symbols_obey_global_daily_limit(paper_db, monkeypatch):
    monkeypatch.setattr(settings, "max_daily_entries", 1)
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    barrier = Barrier(2)

    def entry(symbol):
        barrier.wait(timeout=5)
        return order_manager.place_order(
            symbol, "token", "BUY", 1, price=100, product_type="INTRADAY", stop_loss=95,
            expected_paper_book={},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(entry, ("FIRST", "SECOND")))
    assert sum(result is not None for result in results) == 1
    with db.get_db() as session:
        assert session.query(Trade).count() == 1
        assert sorted(audit.outcome for audit in session.query(PaperFillAudit)) == [
            "FILLED", "REJECTED_ENTRY_LIMIT",
        ]


def test_symbol_reentry_cooldown_blocks_but_does_not_block_exit(paper_db, monkeypatch):
    monkeypatch.setattr(settings, "max_daily_entries_per_symbol", 2)
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 30)
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 1, 100.0, stop_loss=95) is not None
    assert _order("SELL", 1, 100.0, is_exit=True, expected_entry_trade_ids=_entry_ids()) is not None
    assert _order("BUY", 1, 100.0, stop_loss=95) is None
    with db.get_db() as session:
        assert session.query(Trade).count() == 2
        assert session.query(PaperFillAudit).order_by(PaperFillAudit.id.desc()).first().outcome == "REJECTED_ENTRY_LIMIT"
        old_exit = datetime.now(IST) - timedelta(minutes=31)
        session.query(ClosedTrade).one().exit_timestamp = old_exit
        session.query(Trade).filter(Trade.action == "BUY").one().timestamp = old_exit - timedelta(minutes=1)
        session.query(Trade).filter(Trade.action == "SELL").one().timestamp = old_exit
    assert _order("BUY", 1, 100.0, stop_loss=95) is not None


def test_symbol_cooldown_applies_across_midnight(paper_db, monkeypatch):
    monkeypatch.setattr(settings, "symbol_reentry_cooldown_minutes", 30)
    midnight = IST.localize(datetime(2026, 11, 16, 0, 5))
    with db.get_db() as session:
        session.add(ClosedTrade(
            entry_order_id=1, exit_order_id=2, symbol="TEST", action="BUY", quantity=1,
            entry_price=100, exit_price=101,
            entry_timestamp=datetime(2026, 11, 15, 23, 45),
            exit_timestamp=datetime(2026, 11, 15, 23, 55),
            gross_pnl=1, strategy="test", holding_period="INTRADAY", is_paper=True,
        ))
    assert "cooldown" in order_manager._paper_entry_limit_reason("TEST", midnight)
    assert "cooldown" in order_manager._paper_entry_limit_reason("TEST", midnight + timedelta(minutes=19))
    assert order_manager._paper_entry_limit_reason("TEST", midnight + timedelta(minutes=20)) is None


def test_retired_tatamotors_symbol_maps_to_tmpv(monkeypatch):
    from config import instruments
    from config.settings import Settings
    from config.symbol_changes import canonical_nse_symbol, canonical_watchlist

    assert canonical_nse_symbol(" tatamotors ") == "TMPV"
    assert canonical_nse_symbol("TATAMOTORS-EQ") == "TMPV-EQ"
    assert canonical_watchlist(["TATAMOTORS", "INFY", "TMPV"]) == ["TMPV", "INFY"]
    assert Settings.canonicalise_default_watchlist("TATAMOTORS,INFY") == ["TMPV", "INFY"]
    assert "TATAMOTORS" not in settings.default_watchlist
    entry = {"token": "3456", "symbol": "TMPV-EQ", "exchange": "NSE"}
    monkeypatch.setattr(instruments, "_CACHE", {"NSE:TMPV-EQ": entry})
    assert instruments.get_instrument("TATAMOTORS") == entry


@pytest.mark.parametrize(("action", "order_type", "limit", "reference", "slippage", "expected"), [
    ("BUY", "LIMIT", 100.0, 101.0, 0.0, ("NOT_MARKETABLE", None)),
    ("BUY", "LIMIT", 100.0, 99.9, 0.005, ("FILLED", 100.0)),
    ("BUY", "LIMIT", 100.0, 99.0, 0.001, ("FILLED", 99.1)),
    ("SELL", "LIMIT", 100.0, 99.0, 0.0, ("NOT_MARKETABLE", None)),
    ("SELL", "LIMIT", 100.0, 100.2, 0.005, ("FILLED", 100.0)),
    ("BUY", "MARKET", 0.0, 100.0, 0.001, ("FILLED", 100.1)),
    ("SELL", "LIMIT", 0.0, 100.0, 0.0, ("INVALID_PRICE", None)),
])
def test_limit_fills_need_marketable_reference_and_never_exceed_limit(
    action, order_type, limit, reference, slippage, expected,
):
    outcome, price, filled = order_manager.simulate_paper_fill(
        action, order_type, limit, reference, slippage, quantity=10,
    )
    assert (outcome, price) == expected
    assert filled == (10 if outcome == "FILLED" else 0)


def test_entry_without_live_reference_is_audited_but_not_filled(paper_db, monkeypatch):
    _references(monkeypatch, None, 100.0, None)
    assert _order("BUY", 5, 100.0, stop_loss=95.0) is None
    assert _order("BUY", 5, 100.0, stop_loss=95.0)["fill_price"] == 100.0
    exit_order = _order("SELL", 5, 102.0, allow_requested_price_reference=True, is_exit=True, expected_entry_trade_ids=_entry_ids())
    assert exit_order["reference_source"] == "requested_price"
    with db.get_db() as session:
        outcomes = [audit.outcome for audit in session.query(PaperFillAudit).order_by(PaperFillAudit.id)]
        assert outcomes == ["REJECTED_NO_REFERENCE", "FILLED", "FILLED"]
        assert session.query(Trade).count() == 2
        assert [row.gross_pnl for row in session.query(ClosedTrade)] == [10.0]
    report = reconcile_paper(now=_morning())
    assert report["status"] == "warn"
    assert [issue["code"] for issue in report["issues"]] == ["requested_price_reference"]


def test_exit_without_live_reference_cannot_fabricate_a_fill(paper_db, monkeypatch):
    _references(monkeypatch, 100.0, None)
    assert _order("BUY", 5, 100.0, stop_loss=95.0) is not None
    assert _order("SELL", 5, 102.0, allow_requested_price_reference=False, is_exit=True) is None
    with db.get_db() as session:
        assert [audit.outcome for audit in session.query(PaperFillAudit).order_by(PaperFillAudit.id)] == [
            "FILLED", "REJECTED_NO_REFERENCE",
        ]
        assert session.query(Trade).count() == 1
        assert session.query(ClosedTrade).count() == 0


def test_concurrent_paper_exits_cannot_reverse_a_position(paper_db, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0, stop_loss=95.0) is not None
    entry_ids = _entry_ids()
    barrier = Barrier(2)

    def exit_position():
        barrier.wait(timeout=5)
        return _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: exit_position(), range(2)))
    assert sum(result is not None for result in results) == 1
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED").all()
        assert len(trades) == 2
        assert summarize_positions(replay_orders(trades)) == {}
        assert sorted(audit.outcome for audit in session.query(PaperFillAudit)) == [
            "FILLED", "FILLED", "REJECTED_POSITION_CHANGED",
        ]


@pytest.mark.parametrize(("entry", "exit_side"), [("BUY", "SELL"), ("SELL", "BUY")])
def test_paper_exit_rechecks_side_quantity_and_remaining_lots(paper_db, monkeypatch, entry, exit_side):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order(entry, 5, 100.0) is not None
    entry_ids = _entry_ids()
    assert _order(entry, 1, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids) is None
    assert _order(exit_side, 6, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids) is None
    assert _order(exit_side, 2, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids) is not None
    assert _order(exit_side, 3, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids) is not None
    assert _order(exit_side, 1, 100.0, is_exit=True, expected_entry_trade_ids=entry_ids) is None
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED").all()
        assert summarize_positions(replay_orders(trades)) == {}
        assert session.query(ClosedTrade).count() == 2


def test_paper_exit_of_mixed_horizon_lots_must_flatten_the_whole_symbol(paper_db, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0) is not None
    # A second entry is refused (one identity per symbol), so seed an older mixed ledger directly.
    with db.get_db() as session:
        session.add(Trade(
            timestamp=_morning(), symbol="TEST", action="BUY", quantity=2, price=100.0,
            holding_period="DELIVERY", status="EXECUTED", is_paper=True, order_id="SEED",
        ))
    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=_entry_ids()) is None
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED").all()
        position = summarize_positions(replay_orders(trades))["TEST"]
        assert position["quantity"] == 7 and position["mixed_lots"] is True
        assert position["holding_period"] == "INTRADAY"
    assert _order("SELL", 7, 100.0, is_exit=True, expected_entry_trade_ids=_entry_ids()) is not None
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED").all()
        assert summarize_positions(replay_orders(trades)) == {}


def test_paper_exit_cannot_close_a_reopened_lot_with_old_identity(paper_db, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0) is not None
    old_ids = _entry_ids()
    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=old_ids) is not None
    assert _order("BUY", 5, 100.0) is not None
    new_ids = _entry_ids()
    assert new_ids != old_ids
    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=old_ids) is None
    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=None) is None
    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=new_ids) is not None


def test_paper_intent_cannot_adopt_a_reopened_lot(paper_db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0, stop_loss=95, target=110) is not None
    store = ControlStateStore(tmp_path / "intent-controls.sqlite")
    tracker = PortfolioTracker(store)
    tracker.sync_from_broker()
    stops = StopLossManager(store)
    assert stops.restore_from_positions(tracker.positions) == []
    old_ids = _entry_ids()
    stops.begin_exit("TEST")
    marker = stops.pending_exits()["TEST"]
    switch = DeadMansSwitch(store=store)
    switch._notifier = SimpleNamespace(send_message=AsyncMock(return_value=True))
    monkeypatch.setattr(scheduler_module, "portfolio_tracker", tracker)
    monkeypatch.setattr(scheduler_module, "stop_loss_manager", stops)
    monkeypatch.setattr(scheduler_module, "dead_mans_switch", switch)
    scheduler = scheduler_module.MarketScheduler(store)

    assert _order("SELL", 5, 100.0, is_exit=True, expected_entry_trade_ids=old_ids) is not None
    assert _order("BUY", 5, 100.0, stop_loss=90, target=120) is not None
    assert _entry_ids() != old_ids

    asyncio.run(scheduler._reconcile_pending_exits())

    assert stops.pending_exits() == {"TEST": marker}
    assert stops.snapshot()["TEST"]["entry_trade_ids"] == old_ids
    assert switch.is_halted
    assert scheduler._exit_outcomes["TEST"]["outcome"] == "IDENTITY_MISMATCH"
    assert tracker.positions["TEST"]["stop_loss"] == 90
    with db.get_db() as session:
        assert session.query(Trade).count() == 3


@pytest.mark.parametrize("committed_quantity", [0, 2, 5])
def test_paper_exit_restart_recovers_committed_ledger_without_duplicate_fills(
    paper_db, tmp_path, monkeypatch, committed_quantity,
):
    from config import instruments

    monkeypatch.setattr(instruments, "get_token", lambda *_args: "token")
    monkeypatch.setattr(instruments, "get_instrument", lambda *_args: {"tick_size": 5})
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0, stop_loss=95, target=110) is not None
    store = ControlStateStore(tmp_path / "intent-controls.sqlite")
    tracker = PortfolioTracker(store)
    tracker.sync_from_broker()
    stops = StopLossManager(store)
    assert stops.restore_from_positions(tracker.positions) == []
    stops._tracking["TEST"].update(stop_loss=97, highest_since_entry=104)
    stops.begin_exit("TEST")
    if committed_quantity:
        assert _order(
            "SELL", committed_quantity, 100.0, is_exit=True,
            expected_entry_trade_ids=_entry_ids(),
        ) is not None

    # Recreate runtime objects at the interruption boundary, before the intent was settled.
    tracker = PortfolioTracker(store)
    tracker.sync_from_broker()
    stops = StopLossManager(store)
    assert stops.restore_from_positions(tracker.positions) == []
    switch = DeadMansSwitch(store=store)
    switch._notifier = SimpleNamespace(send_message=AsyncMock(return_value=True))
    switch.force_halt("Interrupted protective exit")
    monkeypatch.setattr(scheduler_module, "portfolio_tracker", tracker)
    monkeypatch.setattr(scheduler_module, "stop_loss_manager", stops)
    monkeypatch.setattr(scheduler_module, "dead_mans_switch", switch)
    monkeypatch.setattr(
        scheduler_module.angel_client, "get_order_book",
        lambda: pytest.fail("Paper recovery must use the ledger, not the broker"),
    )
    scheduler = scheduler_module.MarketScheduler(store)
    asyncio.run(scheduler._reconcile_pending_exits())
    assert not stops.pending_exits()
    if committed_quantity < 5:
        assert tracker.positions["TEST"]["quantity"] == 5 - committed_quantity
        assert stops.snapshot()["TEST"]["stop_loss"] == 97
        result = scheduler._submit_protective_exit("TEST", "SELL", tracker.positions["TEST"])
        assert scheduler._settle_exit_result("TEST", result) == "CLOSED"
    asyncio.run(scheduler._reconcile_pending_exits())
    assert not tracker.positions and not stops.snapshot()
    assert switch.is_halted
    assert store.read("protective_stops")["positions"] == {}
    with db.get_db() as session:
        exits = session.query(Trade).filter(Trade.action == "SELL").all()
        assert sum(trade.quantity for trade in exits) == 5
        assert len(exits) == (2 if committed_quantity == 2 else 1)
        assert sum(row.quantity for row in session.query(ClosedTrade).all()) == 5
    report = reconcile_paper(tracker_positions={}, stop_snapshot={}, now=_morning())
    assert report["status"] == "pass", report["issues"]


def test_strategy_and_protective_paper_exits_cannot_both_fill(paper_db, monkeypatch):
    from agents import graph
    from config import instruments

    monkeypatch.setattr(settings, "trademax_broker_mode", "angel_only")
    monkeypatch.setattr(graph, "get_token", lambda *_args: "token")
    monkeypatch.setattr(instruments, "get_token", lambda *_args: "token")
    monkeypatch.setattr(instruments, "get_instrument", lambda *_args: {"tick_size": 5})
    monkeypatch.setattr(
        order_manager, "_market_reference",
        lambda *_args: {"price": 100.0, "source": "ws_tick", "age_seconds": 1.0},
    )
    assert _order("BUY", 5, 100.0, stop_loss=95) is not None
    position = {
        "quantity": 5, "avg_price": 100, "ltp": 100, "holding_period": "INTRADAY",
        "entry_trade_ids": _entry_ids(),
    }
    state = {
        "execution_enabled": True, "current_positions": {"TEST": position},
        "trading_decisions": [{
            "symbol": "TEST", "action": "SELL", "quantity": 5, "price": 100,
            "holding_period": "INTRADAY",
        }],
    }
    barrier = Barrier(2)

    def strategy_exit():
        barrier.wait(timeout=5)
        return graph._order_router_node(state)["executed_orders"]

    def protective_exit():
        barrier.wait(timeout=5)
        result = scheduler_module.MarketScheduler._place_exit_order("TEST", "SELL", position)
        return [result] if result else []

    with ThreadPoolExecutor(max_workers=2) as pool:
        strategy = pool.submit(strategy_exit)
        protective = pool.submit(protective_exit)
        assert len(strategy.result()) + len(protective.result()) == 1
    with db.get_db() as session:
        trades = session.query(Trade).filter(Trade.status == "EXECUTED").all()
        assert len(trades) == 2
        assert summarize_positions(replay_orders(trades)) == {}


def test_fifo_replay_has_no_phantom_closes_and_flips_cleanly():
    stamp = datetime(2026, 1, 5, 10)
    def trade(trade_id, action, quantity, price):
        return SimpleNamespace(
            id=trade_id, symbol="TEST", action=action, quantity=quantity, price=price,
            timestamp=stamp + timedelta(minutes=trade_id), holding_period="INTRADAY",
            stop_loss=None, target=None,
        )
    replay = replay_orders([trade(3, "BUY", 5, 102.0), trade(1, "BUY", 10, 100.0), trade(2, "SELL", 10, 105.0)])
    assert [(c.entry_id, c.exit_id, c.quantity, c.gross_pnl) for c in replay.closed] == [(1, 2, 10, 50.0)]
    assert summarize_positions(replay)["TEST"]["quantity"] == 5

    flipped = replay_orders([trade(1, "BUY", 10, 100.0), trade(2, "SELL", 15, 101.0)])
    position = summarize_positions(flipped)["TEST"]
    assert [c.quantity for c in flipped.closed] == [10]
    assert (position["quantity"], position["avg_price"]) == (-5, 101.0)


def test_sector_metadata_survives_paper_ledger_replay(paper_db):
    entry = Trade(
        id=1, timestamp=_morning(), symbol="TEST", action="BUY", quantity=2,
        price=100, holding_period="DELIVERY", stop_loss=95, target=110,
        sector="financial services", sector_source="yfinance.Ticker.info",
        sector_observed_at=datetime.now().astimezone().isoformat(),
    )
    position = summarize_positions(replay_orders([entry]))["TEST"]
    assert position["sector"] == "financial services"
    assert position["sector_source"] == "yfinance.Ticker.info"

    legacy_entry = Trade(
        id=2, timestamp=_morning(), symbol="OLD", action="BUY", quantity=1,
        price=100, holding_period="DELIVERY", stop_loss=95, target=110,
    )
    assert summarize_positions(replay_orders([legacy_entry]))["OLD"]["sector"] is None


def test_partial_exit_syncs_tracker_and_reconciles(paper_db, monkeypatch):
    _references(monkeypatch, 100.0, 105.0)
    _order("BUY", 10, 100.0, stop_loss=95.0, target=110.0)
    _order("SELL", 4, 105.0, is_exit=True, expected_entry_trade_ids=_entry_ids())
    tracker = PortfolioTracker()
    tracker.sync_from_broker()
    assert tracker.positions["TEST"]["quantity"] == 6
    assert tracker.positions["TEST"]["avg_price"] == 100.0
    from backtester.costs import calculate_indian_transaction_cost
    costs = (
        calculate_indian_transaction_cost(100.0, 10, is_buy=True, is_intraday=True)
        + calculate_indian_transaction_cost(105.0, 4, is_buy=False, is_intraday=True)
    )
    assert tracker.available_capital == pytest.approx(settings.initial_capital + 20.0 - 600.0 - costs)
    assert tracker.total_value == pytest.approx(settings.initial_capital + 20.0 - costs)
    assert tracker.daily_transaction_costs == pytest.approx(costs)
    assert tracker.daily_pnl == pytest.approx(20.0 - costs)

    stops = StopLossManager()
    assert stops.restore_from_positions(tracker.positions) == []
    report = reconcile_paper(tracker_positions=tracker.positions, stop_snapshot=stops.snapshot(), now=_morning())
    assert report["status"] == "pass", report["issues"]
    assert report["summary"]["realised_pnl"] == 20.0
    assert report["summary"]["open_positions"] == 1

    def failures(**kwargs):
        defaults = {"tracker_positions": tracker.positions, "stop_snapshot": stops.snapshot(), "now": _morning()}
        result = reconcile_paper(**{**defaults, **kwargs})
        assert result["status"] == "fail"
        return {issue["code"] for issue in result["issues"] if issue["severity"] == "fail"}

    assert failures(tracker_positions={}) == {"tracker_position_mismatch"}
    assert failures(stop_snapshot={}) == {"missing_stop"}
    assert failures(now=_morning().replace(hour=15, minute=25)) == {"intraday_open_after_squareoff"}
    assert failures(now=_morning() + timedelta(days=1)) == {"intraday_carried_overnight"}

    with db.get_db() as session:
        session.query(Trade).filter(Trade.action == "SELL").update({"price": 104.0})
    assert {"trade_price_mismatch", "closed_lots_mismatch"} <= failures()


def test_restore_flags_missing_or_wrong_side_stops_without_triggering_exits():
    stops = StopLossManager()
    invalid = stops.restore_from_positions({
        "LONG": {"quantity": 5, "avg_price": 100.0, "stop_loss": 95.0, "holding_period": "DELIVERY"},
        "SHORT": {"quantity": -5, "avg_price": 100.0, "stop_loss": 95.0, "holding_period": "DELIVERY"},
        "BARE": {"quantity": -3, "avg_price": 100.0, "stop_loss": None, "holding_period": "DELIVERY"},
    })
    assert sorted(invalid) == ["BARE", "SHORT"]
    snapshot = stops.snapshot()
    assert snapshot["LONG"]["action"] == "BUY" and snapshot["BARE"]["action"] == "SELL"
    assert stops._check_exit("BARE", 100.0, stops._tracking["BARE"]) is None


def test_trailing_stop_and_atr_survive_restart(monkeypatch):
    from risk import stop_loss_manager as stop_module

    class MidSession(datetime):
        @classmethod
        def now(cls, tz=None):
            moment = IST.localize(datetime(2026, 11, 16, 11, 0))
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(stop_module, "datetime", MidSession)

    class MemoryStore:
        def __init__(self):
            self.values = {}

        def read(self, key):
            return self.values.get(key)

        def write(self, key, value):
            self.values[key] = value

    store = MemoryStore()
    stops = StopLossManager(store)
    stops.register_trade({
        "exit_policy": "daily_atr_baseline_v1", "atr_timeframe": "DAILY_14_COMPLETED_SESSIONS",
        "symbol": "TEST", "action": "BUY", "price": 100.0,
        "stop_loss": 95.0, "target": 120.0, "atr": 2.0,
    })
    monkeypatch.setattr(portfolio_tracker, "_positions", {
        "TEST": {
            "symbol": "TEST", "quantity": 2, "avg_price": 100.0,
            "ltp": 110.0, "pnl": 20.0,
            "mark_at": IST.localize(datetime(2026, 11, 16, 11, 0)).isoformat(),
        }
    })
    assert stops.check_all_positions() == []
    assert store.values["protective_stops"]["positions"]["TEST"]["stop_loss"] == 106.0

    restored = StopLossManager(store)
    assert restored.restore_from_positions({
        "TEST": {
            "quantity": 2, "avg_price": 100.0, "stop_loss": 95.0,
            "target": 120.0, "atr": 2.0, "holding_period": "INTRADAY",
        }
    }) == []
    assert restored.snapshot()["TEST"]["stop_loss"] == 106.0
    assert restored.snapshot()["TEST"]["highest_since_entry"] == 110.0


@pytest.fixture
def isolated_scheduler(tmp_path, monkeypatch):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    switch = DeadMansSwitch(store=store)
    monkeypatch.setattr(scheduler_module, "dead_mans_switch", switch)
    monkeypatch.setattr(settings, "paper_trading", True)
    return scheduler_module.MarketScheduler(store=store), store


FAILED_RECONCILIATION = {"status": "fail", "issues": [{"severity": "fail", "code": "missing_stop"}]}


def test_resume_is_refused_while_paper_reconciliation_fails(isolated_scheduler, monkeypatch):
    scheduler, store = isolated_scheduler
    monkeypatch.setattr(scheduler, "reconcile_paper", lambda: FAILED_RECONCILIATION)
    with pytest.raises(ValueError, match="missing_stop"):
        scheduler.resume_trading()
    assert scheduler.is_paused and store.paused()


def test_failed_pre_scan_reconciliation_pauses_without_scanning(isolated_scheduler, monkeypatch):
    scheduler, store = isolated_scheduler
    from config import universe

    monkeypatch.setattr(universe, "get_active_watchlist", lambda: ["TEST"])
    scheduler._trading_paused = False
    store.set_paused(False)
    alerts = []
    async def notify_error(message, **_kwargs):
        alerts.append(message)
    async def pipeline(*_args, **_kwargs):
        pytest.fail("scan ran despite failed reconciliation")
    monkeypatch.setattr(scheduler, "reconcile_paper", lambda: FAILED_RECONCILIATION)
    monkeypatch.setattr(scheduler_module.telegram_notifier, "notify_error", notify_error)
    monkeypatch.setattr(scheduler_module, "run_trading_pipeline", pipeline)
    asyncio.run(scheduler._run_scan())
    assert scheduler.is_paused and store.paused()
    assert alerts and "missing_stop" in alerts[0]
