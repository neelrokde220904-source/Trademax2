"""Historical candle requests share a bounded cross-process request schedule."""

import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


def _process_candle_calls(path, barrier, starts):
    from broker.candle_rate_limiter import CandleRateLimiter

    limiter = CandleRateLimiter(interval_seconds=0.1, coordination_path=Path(path))
    barrier.wait(timeout=10)
    for _ in range(2):
        limiter.call(lambda: starts.put(time.monotonic()))


def _crash_during_candle_call(path):
    from broker.candle_rate_limiter import CandleRateLimiter

    limiter = CandleRateLimiter(interval_seconds=0.1, coordination_path=Path(path))
    limiter.call(lambda: os._exit(0))


def test_cooperating_processes_space_actual_request_starts(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    starts = context.Queue()
    path = str(tmp_path / "candle.state")
    processes = [
        context.Process(target=_process_candle_calls, args=(path, barrier, starts))
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    try:
        observed = sorted(starts.get(timeout=20) for _ in range(6))
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        starts.close()
    assert min(b - a for a, b in zip(observed, observed[1:])) >= 0.095


def test_process_death_leaves_inflight_state_and_delays_next_request(tmp_path):
    from broker.candle_rate_limiter import CandleRateLimiter

    path = tmp_path / "candle.state"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_during_candle_call, args=(str(path),))
    process.start()
    try:
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    assert path.read_text(encoding="ascii") == "inflight"
    started = time.monotonic()
    observed = CandleRateLimiter(interval_seconds=0.1, coordination_path=path).call(time.monotonic)
    assert observed - started >= 0.095


@pytest.mark.parametrize("initial", ["", "inflight", "complete:1000"])
def test_coordinated_limiter_waits_after_unknown_or_previous_boot_state(tmp_path, initial):
    from broker.candle_rate_limiter import CandleRateLimiter

    path = tmp_path / "candle.state"
    path.write_text(initial, encoding="ascii")
    now = [0.0]
    delays = []
    def sleep(delay):
        delays.append(delay)
        now[0] += delay
    limiter = CandleRateLimiter(clock=lambda: now[0], sleep=sleep, coordination_path=path)

    assert limiter.call(lambda: now[0]) == 1.0
    assert limiter.call(lambda: now[0]) == 2.0
    assert delays == [1.0, 1.0]
    assert path.read_text(encoding="ascii") == "complete:2.0"


@pytest.mark.parametrize("initial", ["broken", "complete:nan", "complete:-1"])
def test_coordinated_limiter_rejects_invalid_state_without_request(tmp_path, initial):
    from broker.candle_rate_limiter import CandleRateLimiter

    path = tmp_path / "candle.state"
    path.write_text(initial, encoding="ascii")
    limiter = CandleRateLimiter(coordination_path=path)
    with pytest.raises(ValueError, match="rate-limit state"):
        limiter.call(lambda: pytest.fail("invalid state must not call the provider"))


def test_failed_coordinated_request_still_consumes_a_slot(tmp_path):
    from broker.candle_rate_limiter import CandleRateLimiter

    now = [0.0]
    def sleep(delay):
        now[0] += delay
    limiter = CandleRateLimiter(clock=lambda: now[0], sleep=sleep, coordination_path=tmp_path / "candle.state")
    def failed():
        raise RuntimeError("request failed")
    with pytest.raises(RuntimeError, match="request failed"):
        limiter.call(failed)
    assert limiter.call(lambda: now[0]) == 2.0


def test_coordination_waits_after_completion_even_if_durable_write_is_slow(tmp_path, monkeypatch):
    from broker.candle_rate_limiter import CandleRateLimiter

    now = [0.0]
    def sleep(delay):
        now[0] += delay
    limiter = CandleRateLimiter(clock=lambda: now[0], sleep=sleep, coordination_path=tmp_path / "candle.state")
    write_state = limiter._write_state
    def slow_write(state, value):
        write_state(state, value)
        if value == "inflight":
            now[0] += 1.5
    monkeypatch.setattr(limiter, "_write_state", slow_write)

    assert limiter.call(lambda: now[0]) == 2.5
    assert limiter.call(lambda: now[0]) == 5.0


def test_coordinated_limiter_does_not_call_provider_if_state_write_fails(tmp_path, monkeypatch):
    from broker.candle_rate_limiter import CandleRateLimiter

    limiter = CandleRateLimiter(sleep=lambda _: None, coordination_path=tmp_path / "candle.state")
    def fail_fsync(_fd):
        raise OSError("state disk unavailable")
    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="state disk unavailable"):
        limiter.call(lambda: pytest.fail("state must be durable before a request"))


def test_independent_coordination_paths_do_not_share_wait_state(tmp_path):
    from broker.candle_rate_limiter import CandleRateLimiter

    now = [0.0]
    delays = []
    def sleep(delay):
        delays.append(delay)
        now[0] += delay
    first = CandleRateLimiter(clock=lambda: now[0], sleep=sleep, coordination_path=tmp_path / "first")
    second = CandleRateLimiter(clock=lambda: now[0], sleep=sleep, coordination_path=tmp_path / "second")
    first.call(lambda: None)
    second.call(lambda: None)
    assert delays == [1.0, 1.0]
    assert (tmp_path / "first").read_text(encoding="ascii") == "complete:1.0"
    assert (tmp_path / "second").read_text(encoding="ascii") == "complete:2.0"


def test_candle_limiter_spaces_concurrent_calls_and_counts_failed_requests():
    from broker.candle_rate_limiter import CandleRateLimiter

    now = [0.0]
    starts = []

    def sleep(delay):
        assert delay > 0
        now[0] += delay

    limiter = CandleRateLimiter(interval_seconds=1.0, clock=lambda: now[0], sleep=sleep)

    def request(index):
        def action():
            starts.append(now[0])
            if index == 2:
                raise RuntimeError("request failed")
            return index
        try:
            return limiter.call(action)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(request, range(8)))

    assert starts == pytest.approx(list(range(8)))


def test_candle_limiter_stays_below_published_sliding_window_limits():
    from broker.candle_rate_limiter import CandleRateLimiter

    now = [0.0]
    def sleep(delay):
        now[0] += delay
    limiter = CandleRateLimiter(clock=lambda: now[0], sleep=sleep)
    starts = [limiter.call(lambda: now[0]) for _ in range(5001)]
    assert min(b - a for a, b in zip(starts, starts[1:])) >= 1.0
    for window, limit in ((1.0, 2), (60.0, 80), (3600.0, 5000)):
        left = 0
        for right, start in enumerate(starts):
            while start - starts[left] >= window:
                left += 1
            assert right - left + 1 <= limit


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_candle_limiter_rejects_invalid_interval(interval):
    from broker.candle_rate_limiter import CandleRateLimiter

    with pytest.raises(ValueError):
        CandleRateLimiter(interval_seconds=interval)


def test_normal_and_daily_history_requests_share_the_limiter(monkeypatch):
    from broker import angel_client as angel_module
    from broker.candle_rate_limiter import candle_rate_limiter
    from data import market_data

    calls = []

    def limited(operation, *args, **kwargs):
        calls.append("limited")
        return operation(*args, **kwargs)

    api = SimpleNamespace(getCandleData=lambda _params: {"status": True, "data": []})
    client = angel_module.AngelClient()
    monkeypatch.setattr(client, "ensure_session", lambda: None)
    monkeypatch.setattr(client, "_api", api)
    monkeypatch.setattr(market_data, "angel_client", SimpleNamespace(api=api))
    monkeypatch.setattr(candle_rate_limiter, "call", limited)

    assert client.get_candle_data("NSE", "123", "FIFTEEN_MINUTE", "start", "end") == []
    assert market_data._daily_candle_chunk("123", date(2026, 1, 1), date(2026, 1, 2), ()) == []
    assert calls == ["limited", "limited"]


def test_daily_history_retries_explicit_rate_limit_response(monkeypatch):
    from broker.candle_rate_limiter import candle_rate_limiter
    from data import market_data

    responses = iter([
        {"status": False, "message": "Access denied because of exceeding access rate"},
        {"status": True, "data": [["2026-01-01", 1, 1, 1, 1, 0]]},
    ])
    api = SimpleNamespace(getCandleData=lambda _params: next(responses))
    monkeypatch.setattr(market_data, "angel_client", SimpleNamespace(api=api))
    monkeypatch.setattr(market_data.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(candle_rate_limiter, "call", lambda action, *args: action(*args))

    assert len(market_data._daily_candle_chunk("123", date(2026, 1, 1), date(2026, 1, 2), (1.0,))) == 1
