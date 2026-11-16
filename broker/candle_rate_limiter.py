"""Schedule Angel One historical-candle requests across cooperating processes."""

from __future__ import annotations

import fcntl
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TextIO, TypeVar

from config.settings import PROJECT_ROOT

P = ParamSpec("P")
T = TypeVar("T")


class CandleRateLimiter:
    def __init__(
        self, interval_seconds: float = 1.0, *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        coordination_path: Path | None = None,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Candle request interval must be positive")
        self._interval = interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None
        self._lock = threading.Lock()
        self._coordination_path = coordination_path

    def call(self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        with self._lock:
            if self._coordination_path is not None:
                return self._coordinated_call(operation, *args, **kwargs)
            self._wait_after(self._last_start)
            self._last_start = self._clock()
            return operation(*args, **kwargs)

    def _wait_after(self, last_start: float | None) -> None:
        if last_start is None:
            return
        now = self._clock()
        if last_start > now:
            self._sleep(self._interval)
            return
        delay = last_start + self._interval - now
        while delay > 0:
            self._sleep(delay)
            delay = last_start + self._interval - self._clock()

    def _coordinated_call(self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        path = self._coordination_path
        if path is None:
            raise RuntimeError("Candle coordination path is unavailable")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "r+", encoding="ascii") as state:
            fcntl.flock(state.fileno(), fcntl.LOCK_EX)
            try:
                text = state.read().strip()
                if text and text != "inflight":
                    if not text.startswith("complete:"):
                        raise ValueError("Invalid candle rate-limit state")
                    try:
                        last_start = float(text.removeprefix("complete:"))
                    except ValueError as exc:
                        raise ValueError("Invalid candle rate-limit state") from exc
                    if not math.isfinite(last_start) or last_start < 0:
                        raise ValueError("Invalid candle rate-limit state")
                    self._wait_after(last_start)
                else:
                    self._sleep(self._interval)
                self._write_state(state, "inflight")
                try:
                    return operation(*args, **kwargs)
                finally:
                    self._write_state(state, f"complete:{self._clock()!r}")
            finally:
                fcntl.flock(state.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_state(state: TextIO, value: str) -> None:
        state.seek(0)
        state.write(value)
        state.truncate()
        state.flush()
        os.fsync(state.fileno())


candle_rate_limiter = CandleRateLimiter(
    coordination_path=PROJECT_ROOT / "database" / "candle_rate_limit.state",
)
