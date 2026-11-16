import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytz
import pytest

from config.settings import settings
from scheduler import market_scheduler


IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture(autouse=True)
def isolated_exit_controls(tmp_path, monkeypatch):
    from broker.portfolio_tracker import PortfolioTracker
    from operations.control_state import ControlStateStore
    from risk.dead_mans_switch import DeadMansSwitch
    from risk.stop_loss_manager import StopLossManager

    store = ControlStateStore(tmp_path / "exit-controls.sqlite")
    monkeypatch.setattr(market_scheduler, "portfolio_tracker", PortfolioTracker(store))
    monkeypatch.setattr(market_scheduler, "stop_loss_manager", StopLossManager(store))
    monkeypatch.setattr(market_scheduler, "dead_mans_switch", DeadMansSwitch(store=store))


def test_squareoff_retries_failed_position_and_does_not_disable_stops(monkeypatch):
    scheduler = market_scheduler.MarketScheduler()
    symbol = "TEST"
    market_scheduler.portfolio_tracker._positions = {
        symbol: {
            "symbol": symbol, "quantity": 2, "avg_price": 100.0,
            "ltp": 100.0, "holding_period": "INTRADAY",
        }
    }
    attempts = []
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(market_scheduler.portfolio_tracker, "sync_from_broker", lambda: None)
    monkeypatch.setattr(market_scheduler.portfolio_tracker, "update_unrealised_pnl", lambda: None)
    monkeypatch.setattr(
        scheduler, "_place_exit_order",
        lambda *_args: attempts.append(symbol) or (None if len(attempts) == 1 else {"status": "EXECUTED"}),
    )
    monkeypatch.setattr(market_scheduler.telegram_notifier, "notify_error", AsyncMock())

    asyncio.run(scheduler._intraday_squareoff())
    assert scheduler._squareoff_date is None

    asyncio.run(scheduler._intraday_squareoff())
    assert attempts == [symbol, symbol]

    stop_checks = []
    monkeypatch.setattr(market_scheduler.stop_loss_manager, "check_all_positions", lambda: stop_checks.append(True) or [])
    asyncio.run(scheduler._check_stops())
    assert stop_checks == [True]


def test_squareoff_marks_complete_only_after_intraday_positions_are_flat(monkeypatch):
    scheduler = market_scheduler.MarketScheduler()
    symbol = "TEST"
    market_scheduler.portfolio_tracker._positions = {
        symbol: {
            "symbol": symbol, "quantity": 2, "avg_price": 100.0,
            "ltp": 100.0, "holding_period": "INTRADAY",
        }
    }
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(market_scheduler.portfolio_tracker, "sync_from_broker", lambda: None)
    monkeypatch.setattr(market_scheduler.portfolio_tracker, "update_unrealised_pnl", lambda: None)

    def close_position(*_args):
        market_scheduler.portfolio_tracker._positions.clear()
        return {"status": "EXECUTED"}

    monkeypatch.setattr(scheduler, "_place_exit_order", close_position)
    asyncio.run(scheduler._intraday_squareoff())
    assert scheduler._squareoff_date == datetime.now(IST).date()
