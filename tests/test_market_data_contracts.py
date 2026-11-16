import asyncio
import json
import os
import time

import pandas as pd
import pytest

from data.instrument_master import find_nfo_options
from data.options_data import calculate_max_pain, calculate_pcr, find_max_oi_strikes
from risk.fast_execution_loop import FiveMinuteBarAggregator


def _tick(when, price, volume):
    return {"token": "1", "timestamp": when, "ltp": price, "volume": volume}


def test_five_minute_bar_keeps_first_tick_of_next_bucket():
    bars = FiveMinuteBarAggregator({"1": "TEST"})
    assert bars.update(_tick("2026-11-13T09:15:00+05:30", 100, 1000)) is None
    assert bars.update(_tick("2026-11-13T09:19:00+05:30", 101, 1010)) is None
    first = bars.update(_tick("2026-11-13T09:20:00+05:30", 105, 1015))
    assert first[1].iloc[-1].to_dict() == {
        "open": 100, "high": 101, "low": 100, "close": 101, "volume": 10,
    }
    bars.update(_tick("2026-11-13T09:21:00+05:30", 103, 1020))
    second = bars.update(_tick("2026-11-13T09:25:00+05:30", 102, 1025))
    assert second[1].iloc[-1].to_dict() == {
        "open": 105, "high": 105, "low": 103, "close": 103, "volume": 10,
    }


def test_old_ticks_cannot_corrupt_volume_and_new_session_resets_baseline():
    bars = FiveMinuteBarAggregator({"1": "TEST"})
    bars.update(_tick("2026-11-13T09:15:00+05:30", 100, 1000))
    bars.update(_tick("2026-11-13T09:19:00+05:30", 101, 1020))
    bars.update(_tick("2026-11-13T09:18:00+05:30", 500, 9999))
    bars.update(_tick("2026-11-16T09:15:00+05:30", 110, 30))
    bars.update(_tick("2026-11-16T09:19:00+05:30", 111, 40))
    completed = bars.update(_tick("2026-11-16T09:20:00+05:30", 112, 50))
    assert completed[1].iloc[-1]["open"] == 110
    assert completed[1].iloc[-1]["volume"] == 10
    assert completed[1].iloc[-2]["high"] == 101
    assert completed[1].iloc[-2]["volume"] == 20


def test_invalid_tick_does_not_change_five_minute_bar():
    bars = FiveMinuteBarAggregator({"1": "TEST"})
    bars.update(_tick("2026-11-13T09:15:00+05:30", 100, 100))
    assert bars.update(_tick("2026-11-13T09:16:00+05:30", float("inf"), 110)) is None
    assert bars.update(_tick("2026-11-13T09:17:00+05:30", 102, float("nan"))) is None
    result = bars.update(_tick("2026-11-13T09:20:00+05:30", 101, 105))
    assert result[1].iloc[-1]["close"] == 100


def _contract(symbol, *, side=None, expiry="29DEC2026"):
    return {
        "exch_seg": "NFO", "name": "NIFTY", "symbol": symbol, "token": "123",
        "expiry": expiry, "strike": "2300000.000000", "lotsize": "65",
        "instrumenttype": "OPTIDX", "optiontype": side,
    }


def test_cached_angel_options_derive_side_from_symbol():
    instruments = [
        _contract("NIFTY29DEC2623000CE"),
        _contract("NIFTY29DEC2623000PE"),
        _contract("NIFTY29DEC2623000CE", side="PE"),
    ]
    calls = find_nfo_options("NIFTY", option_type="CE", instruments=instruments)
    puts = find_nfo_options("NIFTY", option_type="PE", instruments=instruments)
    assert [(x["option_type"], x["strike"]) for x in calls] == [("CE", 23000)]
    assert [(x["option_type"], x["strike"]) for x in puts] == [("PE", 23000)]


@pytest.mark.parametrize("oi,expiry", [
    ((0, 0), ("29DEC2026", "29DEC2026")),
    ((100, None), ("29DEC2026", "29DEC2026")),
    ((100, -1), ("29DEC2026", "29DEC2026")),
    ((100, 130), ("29DEC2026", "05JAN2027")),
])
def test_options_metrics_abstain_on_missing_or_mixed_oi(oi, expiry):
    frame = pd.DataFrame([
        {"strike": 23000, "option_type": "CE", "oi": oi[0], "expiry": expiry[0]},
        {"strike": 23000, "option_type": "PE", "oi": oi[1], "expiry": expiry[1]},
    ])
    assert calculate_pcr(frame) is None
    assert calculate_max_pain(frame, 23000) is None
    assert find_max_oi_strikes(frame) == {"max_ce_oi_strike": None, "max_pe_oi_strike": None}


def test_options_context_does_not_issue_thousands_of_broker_quotes(monkeypatch):
    from data import options_data
    from broker.angel_client import angel_client

    sample = [_contract("NIFTY29DEC2623000CE"), _contract("NIFTY29DEC2623000PE")]
    monkeypatch.setattr(options_data, "load_instrument_master", lambda: sample)
    monkeypatch.setattr(angel_client, "get_ltp", lambda *_args: pytest.fail("Unexpected quote call"))
    chain = options_data.fetch_options_chain("NIFTY")
    assert len(chain) == 2
    assert chain["oi"].isna().all()
    assert chain["ltp"].isna().all()


def _master(size=1000):
    return [
        {"symbol": f"TEST{i}-EQ", "exch_seg": "NSE", "token": str(i + 1)}
        for i in range(size)
    ]


def test_failed_instrument_refresh_preserves_valid_cache(tmp_path, monkeypatch):
    from config.settings import settings
    from data import instrument_master

    path = tmp_path / "master.json"
    previous = _master()
    path.write_text(json.dumps(previous))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))

    class Response:
        def raise_for_status(self): pass
        def json(self): return []

    class Client:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def get(self, _url): return Response()

    monkeypatch.setattr(instrument_master.httpx, "AsyncClient", Client)
    assert asyncio.run(instrument_master.download_instrument_master()) == previous
    assert json.loads(path.read_text()) == previous


@pytest.mark.parametrize("fail_replace", [False, True])
def test_instrument_refresh_replaces_atomically_or_preserves_old_cache(tmp_path, monkeypatch, fail_replace):
    from config import instruments
    from config.settings import settings
    from data import instrument_master

    path = tmp_path / "master.json"
    previous, updated = _master(), _master(1001)
    path.write_text(json.dumps(previous))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))
    invalidations = []
    monkeypatch.setattr(instruments, "invalidate_cache", lambda: invalidations.append(True))

    class Response:
        def raise_for_status(self): pass
        def json(self): return updated

    class Client:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def get(self, _url): return Response()

    monkeypatch.setattr(instrument_master.httpx, "AsyncClient", Client)
    if fail_replace:
        def fail(*_args):
            raise OSError("replacement failed")
        monkeypatch.setattr(instrument_master.os, "replace", fail)

    result = asyncio.run(instrument_master.download_instrument_master())
    assert result == (previous if fail_replace else updated)
    assert json.loads(path.read_text()) == result
    assert invalidations == ([] if fail_replace else [True])
    assert list(tmp_path.iterdir()) == [path]


def test_failed_instrument_refresh_rejects_stale_fallback(tmp_path, monkeypatch):
    from config.settings import settings
    from data import instrument_master

    path = tmp_path / "master.json"
    path.write_text(json.dumps(_master()))
    timestamp = time.time() - 10 * 24 * 3600
    os.utime(path, (timestamp, timestamp))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))

    class Client:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def get(self, _url): raise OSError("network unavailable")

    monkeypatch.setattr(instrument_master.httpx, "AsyncClient", Client)
    with pytest.raises(instrument_master.InstrumentMasterUnavailableError, match="stale|expired|too old"):
        asyncio.run(instrument_master.download_instrument_master())


def test_instrument_lookup_rejects_master_replaced_during_load(tmp_path, monkeypatch):
    from config import instruments
    from config.settings import settings
    from data import instrument_master

    path = tmp_path / "master.json"
    replacement = tmp_path / "replacement.json"
    old, new = _master(), _master(1001)
    path.write_text(json.dumps(old))
    replacement.write_text(json.dumps(new))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))
    monkeypatch.setattr(instruments, "_CACHE", {})
    monkeypatch.setattr(instruments, "_CACHE_IDENTITY", None)

    def replace_during_load():
        os.replace(replacement, path)
        return old

    monkeypatch.setattr(instrument_master, "load_instrument_master", replace_during_load)
    with pytest.raises(instrument_master.InstrumentMasterUnavailableError, match="changed"):
        instruments.load_instrument_cache()
    assert instruments._CACHE == {}


def test_instrument_lookup_detects_replacement_with_same_size_and_mtime(tmp_path, monkeypatch):
    from config import instruments
    from config.settings import settings

    path = tmp_path / "master.json"
    replacement = tmp_path / "replacement.json"
    old, new = _master(), _master()
    new[0]["token"] = "9"
    path.write_text(json.dumps(old))
    replacement.write_text(json.dumps(new))
    original = path.stat()
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert replacement.stat().st_size == original.st_size
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))
    monkeypatch.setattr(instruments, "_CACHE", {})
    monkeypatch.setattr(instruments, "_CACHE_IDENTITY", None)
    assert instruments.get_token("TEST0") == "1"
    os.replace(replacement, path)
    assert instruments.get_token("TEST0") == "9"


def test_valid_single_expiry_oi_still_produces_metrics():
    chain = pd.DataFrame([
        {"strike": 23000, "option_type": "CE", "oi": 100, "expiry": "29DEC2026"},
        {"strike": 23000, "option_type": "PE", "oi": 130, "expiry": "29DEC2026"},
        {"strike": 23100, "option_type": "CE", "oi": 200, "expiry": "29DEC2026"},
        {"strike": 23100, "option_type": "PE", "oi": 70, "expiry": "29DEC2026"},
    ])
    assert calculate_pcr(chain) == pytest.approx(200 / 300, abs=0.001)
    assert calculate_max_pain(chain, 23000) == 23000
    assert find_max_oi_strikes(chain) == {"max_ce_oi_strike": 23100, "max_pe_oi_strike": 23000}


def test_in_memory_instrument_lookup_cannot_outlive_stale_cache(tmp_path, monkeypatch):
    from config import instruments
    from config.settings import settings
    from data.instrument_master import InstrumentMasterUnavailableError

    path = tmp_path / "master.json"
    path.write_text(json.dumps(_master()))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))
    monkeypatch.setattr(instruments, "_CACHE", {})
    monkeypatch.setattr(instruments, "_CACHE_IDENTITY", None)
    assert instruments.get_token("TEST1") == "2"
    timestamp = time.time() - 10 * 24 * 3600
    os.utime(path, (timestamp, timestamp))
    with pytest.raises(InstrumentMasterUnavailableError, match="stale"):
        instruments.get_token("TEST1")


def test_invalid_cache_never_leaves_partially_populated_in_memory_map(tmp_path, monkeypatch):
    from config import instruments
    from config.settings import settings
    from data.instrument_master import InstrumentMasterUnavailableError

    path = tmp_path / "master.json"
    entries = _master()
    entries[-1]["lotsize"] = "not-a-number"
    path.write_text(json.dumps(entries))
    monkeypatch.setattr(settings, "instrument_cache_path", str(path))
    monkeypatch.setattr(instruments, "_CACHE", {})
    monkeypatch.setattr(instruments, "_CACHE_IDENTITY", None)
    with pytest.raises(InstrumentMasterUnavailableError, match="precision"):
        instruments.get_token("TEST1")
    assert not instruments._CACHE


def test_options_agent_explicitly_reports_missing_oi_without_spot_fetch(monkeypatch):
    from agents import options_agent
    from broker.angel_client import angel_client

    chain = pd.DataFrame([
        {"strike": 23000, "option_type": "CE", "oi": None, "expiry": "29DEC2026"},
        {"strike": 23000, "option_type": "PE", "oi": None, "expiry": "29DEC2026"},
    ])
    monkeypatch.setattr(options_agent, "fetch_options_chain", lambda _underlying: chain)
    monkeypatch.setattr(angel_client, "get_ltp", lambda *_args: pytest.fail("Unexpected spot call"))
    result = options_agent.OptionsAgent().run({"live_quotes": {}})
    assert not result["errors"]
    for signal in result["options_signals"].values():
        assert signal["data_status"] == "unavailable"
        assert signal["signal"] == "HOLD"
        assert signal["confidence"] == 0
        assert signal["pcr"] is None and signal["max_pain"] is None
