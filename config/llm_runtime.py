"""Persistent model selection and isolated execution profiles."""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from importlib.util import find_spec
from uuid import uuid4

from loguru import logger
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from config.settings import settings
from database.db import get_db
from database.models import LLMModeSetting, LLMRun


@dataclass(frozen=True)
class Profile:
    mode: str
    revision: int
    fast_model: str
    decision_model: str
    decision_max_tokens: int
    timeout: float


@dataclass(frozen=True)
class RunContext:
    profile: Profile
    run_id: str
    forced_profile: bool = False


_context: ContextVar[RunContext | None] = ContextVar("llm_run", default=None)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def profile_for(mode: str, revision: int = 0) -> Profile:
    if mode not in ("claude", "economy", "rules"):
        raise ValueError("Invalid LLM mode")
    if mode == "rules":
        return Profile(mode, revision, "rule-based", "rule-based", 0, settings.llm_timeout_seconds)
    return Profile(
        mode, revision,
        settings.llm_fast_model if mode == "claude" else settings.economy_fast_model,
        settings.llm_decision_model if mode == "claude" else settings.economy_decision_model,
        settings.economy_decision_max_tokens, settings.llm_timeout_seconds,
    )


def configured(profile: Profile) -> bool:
    if profile.mode == "rules":
        return False
    key = settings.anthropic_api_key if profile.mode == "claude" else settings.deepseek_api_key
    module = "anthropic" if profile.mode == "claude" else "httpx"
    return bool(key.strip() and not key.startswith("your_") and find_spec(module))


def selected_profile() -> Profile:
    with get_db() as database:
        row = database.get(LLMModeSetting, 1)
        if row is not None:
            return profile_for(row.mode, row.revision)
    try:
        with get_db() as database:
            database.add(LLMModeSetting(id=1, mode="claude", revision=0))
    except IntegrityError:
        pass
    with get_db() as database:
        row = database.get(LLMModeSetting, 1)
        return profile_for(row.mode, row.revision)


def select_mode(mode: str, expected_revision: int) -> Profile:
    target = profile_for(mode)
    current = selected_profile()
    if current.revision != expected_revision:
        raise ValueError("Selection changed elsewhere; refresh and retry.")
    if current.mode == mode:
        return current
    if not configured(target):
        raise ValueError("Selected provider is not configured. Set its API key on the server and restart.")
    with get_db() as database:
        result = database.execute(
            update(LLMModeSetting)
            .where(LLMModeSetting.id == 1, LLMModeSetting.revision == expected_revision)
            .values(mode=mode, revision=expected_revision + 1)
        )
        if result.rowcount != 1:
            raise ValueError("Selection changed elsewhere; refresh and retry.")
    return profile_for(mode, expected_revision + 1)


def current_context() -> RunContext | None:
    return _context.get()


def current_profile() -> Profile:
    context = current_context()
    return context.profile if context else selected_profile()


def mark_degraded() -> None:
    context = current_context()
    if context:
        with get_db() as database:
            database.execute(update(LLMRun).where(LLMRun.id == context.run_id).values(degraded=True))


def _heartbeat(run_id: str, stopped: threading.Event) -> None:
    while not stopped.wait(15):
        try:
            with get_db() as database:
                database.execute(update(LLMRun).where(LLMRun.id == run_id).values(heartbeat_at=utcnow()))
        except Exception:
            logger.warning("LLM run heartbeat could not be recorded")


@contextmanager
def llm_run(kind: str, *, profile: Profile | None = None):
    existing = current_context()
    if existing:
        if profile is not None and profile != existing.profile:
            raise ValueError("Cannot change an active LLM run profile")
        yield existing
        return
    forced_profile = profile is not None
    profile = profile or selected_profile()
    context = RunContext(profile, str(uuid4()), forced_profile)
    with get_db() as database:
        database.add(LLMRun(
            id=context.run_id, kind=kind, mode=profile.mode, revision=profile.revision,
            fast_model=profile.fast_model, decision_model=profile.decision_model,
            started_at=utcnow(), heartbeat_at=utcnow(), status="running",
        ))
    token = _context.set(context)
    stopped = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(context.run_id, stopped), daemon=True)
    heartbeat.start()
    outcome = "completed"
    try:
        yield context
    except BaseException:
        outcome = "failed"
        raise
    finally:
        stopped.set()
        heartbeat.join(timeout=1)
        _context.reset(token)
        try:
            with get_db() as database:
                database.execute(update(LLMRun).where(LLMRun.id == context.run_id).values(
                    status=outcome, finished_at=utcnow(), heartbeat_at=utcnow(),
                ))
        except Exception:
            logger.error("LLM run completion could not be recorded")


def llm_operation(kind: str):
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with llm_run(kind):
                    return await function(*args, **kwargs)
            return asynchronous
        @wraps(function)
        def synchronous(*args, **kwargs):
            with llm_run(kind):
                return function(*args, **kwargs)
        return synchronous
    return decorate


def mode_status() -> dict:
    selected = selected_profile()
    profiles = {}
    for mode in ("claude", "economy"):
        profile = profile_for(mode)
        profiles[mode] = {"fast_model": profile.fast_model, "decision_model": profile.decision_model,
                          "configured": configured(profile)}
    with get_db() as database:
        runs = database.query(LLMRun).filter(LLMRun.status == "running").all()
        active = []
        for run in runs:
            if run.heartbeat_at < utcnow() - timedelta(minutes=2):
                run.status = "interrupted"
                continue
            active.append({"id": run.id, "kind": run.kind, "mode": run.mode,
                           "revision": run.revision, "degraded": run.degraded})
        latest = database.query(LLMRun).filter(
            LLMRun.kind != "research_comparison",
        ).order_by(LLMRun.started_at.desc()).first()
        degraded = bool(latest and latest.degraded)
    return {"selected_mode": selected.mode, "revision": selected.revision,
            "profiles": profiles, "active_runs": active, "degraded": degraded,
            "pending": any(run["kind"] != "research_comparison" and run["revision"] != selected.revision for run in active),
            "observed_at": utcnow().isoformat() + "Z"}