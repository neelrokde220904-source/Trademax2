"""Sliding-window order-submission limiter shared by entries and exits.

Entries that exceed the window are refused. Exits are never dropped: they wait
at most ``EXIT_MAX_WAIT_SECONDS`` and then proceed with a warning, because a
protective exit must not depend on a throttle.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from loguru import logger

from config.settings import settings

EXIT_MAX_WAIT_SECONDS = 1.0


class OrderRateLimiter:
    def __init__(
        self,
        *,
        per_second: int | None = None,
        per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._per_second = per_second
        self._per_minute = per_minute
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _limits(self) -> tuple[int, int]:
        return (
            self._per_second or settings.order_rate_per_second,
            self._per_minute or settings.order_rate_per_minute,
        )

    def _wait_needed(self, now: float) -> float:
        """Seconds until a slot is free (0 when one is available now)."""
        per_second, per_minute = self._limits()
        while self._stamps and now - self._stamps[0] >= 60.0:
            self._stamps.popleft()
        wait = 0.0
        recent = [stamp for stamp in self._stamps if now - stamp < 1.0]
        if len(recent) >= per_second:
            wait = max(wait, 1.0 - (now - recent[-per_second]))
        if len(self._stamps) >= per_minute:
            wait = max(wait, 60.0 - (now - self._stamps[-per_minute]))
        return max(0.0, wait)

    def acquire(self, *, is_exit: bool) -> bool:
        """Reserve a submission slot; return False only when an entry is throttled."""
        deadline = self._clock() + (EXIT_MAX_WAIT_SECONDS if is_exit else 0.0)
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._stamps.append(now)
                    return True
                if not is_exit:
                    return False
                if now + wait > deadline:
                    logger.warning(
                        "Order rate limit still saturated; sending protective exit without waiting."
                    )
                    self._stamps.append(now)
                    return True
            self._sleep(min(wait, max(0.0, deadline - self._clock())))


order_rate_limiter = OrderRateLimiter()
