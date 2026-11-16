"""Shared test isolation: never touch the live process lock or live .env sizing."""

from __future__ import annotations

import os
import tempfile

import pytest

# Redirect the module-level control-state singleton before anything imports it, so no test can
# pause, halt or otherwise change the real operational state of a running service.
os.environ["TRADING_CONTROL_STATE_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="trader-test-controls-"), "operational_state.sqlite3",
)

from config.settings import Settings, settings as _settings

# Tests assume the code defaults, not whatever the operator's live .env says.
# Pinned at import time because module-level singletons (e.g. the portfolio
# tracker) read these values when the test modules are imported.
for _name in ("initial_capital", "max_daily_entries"):
    setattr(_settings, _name, Settings.model_fields[_name].default)

# No test may reach a real external service, even when run next to a live .env.
for _name in (
    "telegram_bot_token", "telegram_chat_id", "angel_api_key", "angel_client_id", "angel_mpin",
    "angel_totp_secret", "anthropic_api_key", "deepseek_api_key", "openai_api_key",
):
    setattr(_settings, _name, "")


@pytest.fixture(autouse=True)
def _isolated_process_lock(tmp_path, monkeypatch):
    import operations.process_lock as process_lock

    monkeypatch.setattr(
        process_lock, "DEFAULT_LOCK_PATH", tmp_path / "trading_service.lock"
    )


@pytest.fixture(autouse=True)
def _fresh_order_rate_limiter():
    from operations.order_rate_limiter import order_rate_limiter

    order_rate_limiter._stamps.clear()


@pytest.fixture(autouse=True)
def _scratch_database(tmp_path_factory, monkeypatch):
    """Tests never write to the real trading database; tests needing their own engine patch it again."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import db
    from database.models import Base

    engine = create_engine(
        f"sqlite:///{tmp_path_factory.mktemp('scratch') / 'scratch.sqlite'}", connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    yield
    engine.dispose()
