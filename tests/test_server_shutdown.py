"""Shutdown behavior with an open dashboard stream, without trading."""

import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def test_scheduler_server_has_bounded_graceful_shutdown(monkeypatch):
    import main
    from broker.websocket_feed import ws_feed
    from config.settings import settings
    from scheduler import market_scheduler as scheduler_module

    class Scheduler:
        def restore_paper_state(self):
            return []

        def reconcile_paper(self):
            return {"positions": [], "status": "pass"}

        def setup(self):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    captured = {}

    def config(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    class Server:
        def __init__(self, _config):
            pass

        async def serve(self):
            pass

    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(scheduler_module, "market_scheduler", Scheduler())
    monkeypatch.setattr(ws_feed, "disconnect", lambda: None)
    monkeypatch.setattr(main.uvicorn, "Config", config)
    monkeypatch.setattr(main.uvicorn, "Server", Server)

    asyncio.run(main.run_scheduler_mode())

    assert 0 < captured["timeout_graceful_shutdown"] <= 5


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM shutdown behavior is Unix-specific")
def test_dashboard_shutdown_with_open_stream(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = os.environ.copy()
    env.update(
        DATABASE_URL=f"sqlite:///{tmp_path / 'dashboard.sqlite3'}",
        DASHBOARD_PORT=str(port),
        PAPER_TRADING="true",
        ANGEL_API_KEY="",
        ANGEL_CLIENT_ID="",
        ANGEL_MPIN="",
        ANGEL_TOTP_SECRET="",
        DEEPSEEK_API_KEY="",
        ANTHROPIC_API_KEY="",
    )
    process = subprocess.Popen(
        [sys.executable, str(root / "main.py"), "--dashboard"],
        cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    stream = None
    try:
        deadline = time.monotonic() + 25
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2):
                    break
            except (OSError, urllib.error.URLError):
                if process.poll() is not None:
                    pytest.fail(f"Dashboard exited before becoming ready: {process.returncode}")
                if time.monotonic() >= deadline:
                    pytest.fail("Dashboard did not become ready")
                time.sleep(0.25)
        stream = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stream", timeout=5)
        process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            pytest.fail("Dashboard did not stop while a stream was open")
    finally:
        if stream is not None:
            stream.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
