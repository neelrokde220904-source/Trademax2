"""Database session management using SQLAlchemy."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from database.models import Base

engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Additive column migrations for tables that already exist in a live database.
# Every new column on an existing model must be listed here or the live DB breaks.
SQLITE_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "trades": {
        "atr": "FLOAT",
        "sector": "VARCHAR(100)",
        "sector_source": "VARCHAR(80)",
        "sector_observed_at": "VARCHAR(40)",
        "horizon": "VARCHAR(20)",
        "strategy_version": "VARCHAR(40)",
        "exit_policy": "VARCHAR(60)",
        "atr_timeframe": "VARCHAR(40)",
        "book": "VARCHAR(40)",
    },
    "closed_trades": {
        "venue": "VARCHAR(20)",
        "currency": "VARCHAR(8)",
        "exit_policy": "VARCHAR(60)",
        "book": "VARCHAR(40)",
    },
}


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        inspector = inspect(engine)
        with engine.begin() as connection:
            for table, columns in SQLITE_ADDED_COLUMNS.items():
                existing = {column["name"] for column in inspector.get_columns(table)}
                for name, column_type in columns.items():
                    if name not in existing:
                        connection.exec_driver_sql(
                            f"ALTER TABLE {table} ADD COLUMN {name} {column_type}"
                        )
    logger.info("Database initialised at {}", settings.database_url)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Yield a database session, with automatic commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Return a raw session (caller manages lifecycle)."""
    return SessionLocal()
