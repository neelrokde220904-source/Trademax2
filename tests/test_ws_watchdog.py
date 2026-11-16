import asyncio
from datetime import datetime

import pytest

from scheduler import market_scheduler as ms


class _Clock:
    def __init__(self, value):
        self.value = value

    def now(self, tz=None):
        return ms.IST.localize(self.value)


@pytest.mark.parametrize("when,running,holiday,expected", [
    (datetime(2026, 11, 13, 12, 10), False, False, 1),
    (datetime(2026, 11, 13, 12, 10), True, False, 0),
    (datetime(2026, 11, 13, 15, 40), False, False, 0),
    (datetime(2026, 11, 13, 8, 58), False, False, 0),
    (datetime(2026, 11, 13, 12, 10), False, True, 0),
])
def test_ws_watchdog_connects_only_when_needed(monkeypatch, when, running, holiday, expected):
    calls = []
    scheduler = ms.MarketScheduler()

    async def fake_connect():
        calls.append(1)

    monkeypatch.setattr(ms, "datetime", _Clock(when))
    monkeypatch.setattr(type(ms.ws_feed), "is_running", property(lambda _self: running))
    monkeypatch.setattr(ms.settings, "nse_holidays", ["2026-11-13"] if holiday else [])
    monkeypatch.setattr(scheduler, "_connect_ws", fake_connect)

    asyncio.run(scheduler._ensure_ws())
    assert len(calls) == expected


def test_ws_watchdog_job_is_scheduled():
    scheduler = ms.MarketScheduler()
    scheduler.setup()
    assert scheduler.scheduler.get_job("ws_watchdog") is not None
