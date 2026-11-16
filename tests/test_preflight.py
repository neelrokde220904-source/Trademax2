"""Regression tests for configuration and non-mutating operational checks."""

from operations.preflight import run_preflight


def test_preflight_accepts_default_paper_configuration():
    """The checked-in paper-trading defaults must be runnable without credentials."""
    result = run_preflight()

    assert result.passed
    assert result.errors == []


def test_global_watchlist_accepts_comma_separated_environment_value(monkeypatch):
    """Documented .env syntax must not prevent settings from loading."""
    monkeypatch.setenv("GLOBAL_WATCHLIST", "spy, qqq, AAPL")

    from config.settings import Settings

    assert Settings(_env_file=None).global_watchlist == ["SPY", "QQQ", "AAPL"]


def test_preflight_does_not_create_missing_database(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from database import db

    path = tmp_path / "not-created.db"
    engine = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(db, "engine", engine)
    assert run_preflight().passed
    assert not path.exists()
    engine.dispose()
