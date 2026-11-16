"""Prevent concurrent processes from running trading scans or the scheduler."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config.settings import PROJECT_ROOT


DEFAULT_LOCK_PATH = PROJECT_ROOT / "database" / "trading_service.lock"


@contextmanager
def trading_process_lock(lock_path: Path | None = None) -> Iterator[None]:
    """Hold an OS-released exclusive lock for the lifetime of a trading process."""
    lock_path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another trading scheduler or scan already holds the process lock."
            ) from exc
        locked = True
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
