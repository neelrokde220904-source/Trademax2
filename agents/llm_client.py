"""Provider-neutral completions with per-run routing and durable usage estimates."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from config.llm_runtime import (
    configured, current_context, current_profile, llm_operation, mark_degraded, utcnow,
)
from config.settings import settings
from config.llm_factory import create_llm, get_deep_llm, get_quick_llm, resolve_configuration
from database.db import get_db
from database.models import LLMUsage


@dataclass(frozen=True)
class Completion:
    content: str


def _dms(event: str, error: str = "") -> None:
    """Report provider health to the Dead Man's Switch; it must never break a completion."""
    try:
        from risk.dead_mans_switch import dead_mans_switch

        if event == "success":
            dead_mans_switch.record_success("llm_provider")
        else:
            dead_mans_switch.record_failure("llm_provider", error)
    except Exception:
        logger.debug("DMS could not record LLM provider health")


def estimate_cost(mode: str, model: str, usage: dict, timestamp: datetime):
    version = "2026-09-28"
    if mode == "claude":
        rates = {"claude-opus-4-6": ("5", "25"), "claude-haiku-4-5-20251001": ("1", "5")}
        if model not in rates or "input_tokens" not in usage or "output_tokens" not in usage:
            return None, version
        input_rate, output_rate = map(Decimal, rates[model])
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        long_write = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0) or 0
        cost = (Decimal(usage["input_tokens"]) * input_rate
                + Decimal(usage["output_tokens"]) * output_rate
                + Decimal(usage.get("cache_read_input_tokens", 0) or 0) * input_rate * Decimal("0.1")
                + Decimal(cache_write - long_write) * input_rate * Decimal("1.25")
                + Decimal(long_write) * input_rate * 2)
    else:
        rates = {"deepseek-flash": ("0.30", "1.20", "0.006"),
                 "deepseek-v4-pro": ("1.32", "3.96", "0.044")}
        if model not in rates or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            return None, version
        input_rate, output_rate, cache_rate = map(Decimal, rates[model])
        moment = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
        peak = moment.weekday() < 5 and (1 <= moment.hour < 4 or 6 <= moment.hour < 10)
        multiplier = Decimal("1") if peak else Decimal("0.5")
        version += ":peak-upper-bound" if peak else ":off-peak"
        cached = usage.get("prompt_cache_hit_tokens", 0) or 0
        cost = (Decimal(usage["prompt_tokens"] - cached) * input_rate
                + Decimal(cached) * cache_rate
                + Decimal(usage["completion_tokens"]) * output_rate) * multiplier
    return cost / Decimal(1_000_000), version


def _claude_request(profile, model, prompt, max_tokens, temperature):
    from anthropic import Anthropic

    options = {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        options["temperature"] = temperature
    with Anthropic(api_key=settings.anthropic_api_key, timeout=profile.timeout, max_retries=0) as client:
        response = client.messages.create(**options)
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.model_dump(), response.stop_reason, response.id


def _deepseek_request(profile, role, model, prompt, max_tokens, temperature):
    thinking = role == "decision"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": profile.decision_max_tokens if thinking else max_tokens,
            "thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking:
        body["reasoning_effort"] = "high"
    elif temperature is not None:
        body["temperature"] = temperature
    with httpx.Client(timeout=profile.timeout, follow_redirects=False, trust_env=False) as client:
        response = client.post("https://api.deepseek.com/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {settings.deepseek_api_key}"})
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]
    return choice["message"].get("content") or "", payload.get("usage", {}), choice.get("finish_reason"), payload.get("id")


def _validate_text(text: str, intent: str) -> None:
    if not text.strip():
        raise ValueError("Empty LLM answer")
    if intent == "text":
        return
    opening, closing, expected = ("[", "]", list) if intent == "array" else ("{", "}", dict)
    start, end = text.find(opening), text.rfind(closing) + 1
    if start < 0 or not isinstance(json.loads(text[start:end]), expected):
        raise ValueError("Invalid LLM JSON")


def _use_universal_factory(role: str) -> bool:
    context = current_context()
    if context and context.forced_profile:
        return False
    tier = "deep" if role == "decision" else "quick"
    explicit = (
        settings.trademax_llm_provider.strip().lower() not in ("", "auto")
        or bool(getattr(settings, f"trademax_{tier}_think_provider").strip())
    )
    if explicit:
        return True
    profile = current_profile()
    if configured(profile):
        return False
    try:
        resolve_configuration(tier)
        return True
    except (RuntimeError, ValueError):
        return False


def _response_text(response: object) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content if isinstance(item, (str, dict))
        )
    return str(content)


def _complete_with_factory(
    role: str, prompt: str, max_tokens: int, temperature: float | None,
    task: str, intent: str, context: RunContext,
) -> str:
    tier = "deep" if role == "decision" else "quick"
    configuration = resolve_configuration(tier)
    model = create_llm(tier, temperature)
    token_option = "max_output_tokens" if configuration.provider == "google" else "max_tokens"
    if configuration.provider == "ollama" and not (
        configuration.base_url and configuration.base_url.rstrip("/").endswith("/v1")
    ):
        token_option = "num_predict"
    model = model.bind(**{token_option: max_tokens})
    attempt_id = str(uuid4())
    started = utcnow()
    with get_db() as database:
        database.add(LLMUsage(
            id=attempt_id, run_id=context.run_id, mode=configuration.provider,
            model=configuration.model, task=task, created_at=started, status="pending",
        ))
    usage: dict = {}
    request_id = None
    outcome = "error"
    clock = time.monotonic()
    try:
        response = model.invoke(prompt)
        text = _response_text(response)
        metadata = getattr(response, "response_metadata", {}) or {}
        raw_usage = getattr(response, "usage_metadata", None) or metadata.get("token_usage") or {}
        usage = {
            "input_tokens": raw_usage.get("input_tokens", raw_usage.get("prompt_tokens")),
            "output_tokens": raw_usage.get("output_tokens", raw_usage.get("completion_tokens")),
        }
        request_id = metadata.get("id") or metadata.get("request_id")
        finish = metadata.get("finish_reason") or metadata.get("stop_reason")
        if finish in {"length", "max_tokens", "MAX_TOKENS"}:
            raise ValueError("Incomplete LLM answer")
        _validate_text(text, intent)
        outcome = "success"
        _dms("success")
        return text
    except Exception as exc:
        _dms("failure", type(exc).__name__)
        mark_degraded()
        logger.warning("{} {} LLM call failed ({})", configuration.provider, task, type(exc).__name__)
        raise RuntimeError(
            f"{configuration.provider} LLM request failed; no provider fallback used"
        ) from None
    finally:
        cost, version = estimate_cost(configuration.provider, configuration.model, usage, started)
        with get_db() as database:
            record = database.get(LLMUsage, attempt_id)
            record.status = outcome
            record.request_id = request_id
            record.input_tokens = usage.get("input_tokens")
            record.output_tokens = usage.get("output_tokens")
            record.latency_ms = int((time.monotonic() - clock) * 1000)
            record.estimated_usd = cost
            record.pricing_version = version


@llm_operation("completion")
def complete(role: str, prompt: str, max_tokens: int = 1500, temperature=None,
             task: str = "analysis", intent: str = "object") -> str:
    if role not in ("fast", "decision"):
        raise ValueError("Invalid LLM role")
    context = current_context()
    if _use_universal_factory(role):
        return _complete_with_factory(
            role, prompt, max_tokens, temperature, task, intent, context
        )
    profile = context.profile
    if not configured(profile):
        mark_degraded()
        raise RuntimeError("Selected LLM provider is not configured")
    model = profile.fast_model if role == "fast" else profile.decision_model
    attempt_id = str(uuid4())
    started = utcnow()
    with get_db() as database:
        database.add(LLMUsage(id=attempt_id, run_id=context.run_id, mode=profile.mode,
                              model=model, task=task, created_at=started, status="pending"))
    usage, request_id, outcome = {}, None, "error"
    clock = time.monotonic()
    try:
        if profile.mode == "claude":
            text, usage, finish, request_id = _claude_request(profile, model, prompt, max_tokens, temperature)
        else:
            text, usage, finish, request_id = _deepseek_request(profile, role, model, prompt, max_tokens, temperature)
        if finish not in ("end_turn", "stop", "stop_sequence"):
            raise ValueError("Incomplete LLM answer")
        _validate_text(text, intent)
        outcome = "success"
        _dms("success")
        return text
    except Exception as exc:
        _dms("failure", type(exc).__name__)
        mark_degraded()
        logger.warning("{} {} LLM call failed ({})", profile.mode, task, type(exc).__name__)
        raise RuntimeError(f"{profile.mode} LLM request failed; no provider fallback used") from None
    finally:
        cost, version = estimate_cost(profile.mode, model, usage, started)
        with get_db() as database:
            record = database.get(LLMUsage, attempt_id)
            record.status = outcome
            record.request_id = request_id
            record.input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            record.output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            record.cache_read_tokens = usage.get("cache_read_input_tokens", usage.get("prompt_cache_hit_tokens", 0))
            record.cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
            record.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            record.latency_ms = int((time.monotonic() - clock) * 1000)
            record.estimated_usd = cost
            record.pricing_version = version


@dataclass(frozen=True)
class LLMClient:
    role: str
    task: str
    max_tokens: int = 1024
    temperature: float | None = None

    def __bool__(self):
        if current_profile().mode == "rules":
            return False
        if _use_universal_factory(self.role):
            try:
                resolve_configuration("deep" if self.role == "decision" else "quick")
                return True
            except (RuntimeError, ValueError):
                mark_degraded()
                return False
        available = configured(current_profile())
        if not available:
            mark_degraded()
        return available

    def invoke(self, prompt: str) -> Completion:
        return Completion(complete(self.role, prompt, self.max_tokens, self.temperature, self.task))


def usage_summary(now: datetime | None = None) -> dict:
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Asia/Kolkata"))
    start = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    totals = {mode: {"requests": 0, "estimated_usd": Decimal(0), "unknown_requests": 0,
                     "input_tokens": 0, "output_tokens": 0} for mode in ("claude", "economy")}
    with get_db() as database:
        rows = database.query(LLMUsage).filter(
            LLMUsage.created_at >= start.astimezone(timezone.utc).replace(tzinfo=None),
            LLMUsage.created_at < end.astimezone(timezone.utc).replace(tzinfo=None),
        ).all()
        for row in rows:
            total = totals.setdefault(row.mode, {
                "requests": 0, "estimated_usd": Decimal(0), "unknown_requests": 0,
                "input_tokens": 0, "output_tokens": 0,
            })
            total["requests"] += 1
            total["unknown_requests"] += int(row.estimated_usd is None)
            total["estimated_usd"] += row.estimated_usd or Decimal(0)
            total["input_tokens"] += row.input_tokens or 0
            total["output_tokens"] += row.output_tokens or 0
    for total in totals.values():
        total["estimated_inr"] = float(total["estimated_usd"] * Decimal(str(settings.llm_usd_inr_rate)))
        total["estimated_usd"] = float(total["estimated_usd"])
    return {"month": start.strftime("%Y-%m"), "modes": totals,
            "estimated_usd": sum(total["estimated_usd"] for total in totals.values()),
            "estimated_inr": sum(total["estimated_inr"] for total in totals.values()),
            "incomplete": any(total["unknown_requests"] for total in totals.values()),
            "usd_inr_rate": settings.llm_usd_inr_rate, "hard_limit": None,
            "updated_at": utcnow().isoformat() + "Z"}

def llm_spend_today_inr(now: datetime | None = None) -> dict:
    """Today's (IST) LLM spend in ₹; a successful call with unknown pricing makes it unreliable."""
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Asia/Kolkata"))
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    spend_usd, unknown = Decimal(0), 0
    with get_db() as database:
        rows = database.query(LLMUsage).filter(
            LLMUsage.created_at >= start.astimezone(timezone.utc).replace(tzinfo=None),
            LLMUsage.created_at < end.astimezone(timezone.utc).replace(tzinfo=None),
        ).all()
        for row in rows:
            if row.estimated_usd is None:
                unknown += int(row.status == "success")
            else:
                spend_usd += row.estimated_usd
    return {
        "date": start.strftime("%Y-%m-%d"),
        "spend_inr": float(spend_usd * Decimal(str(settings.llm_usd_inr_rate))),
        "unknown_pricing_calls": unknown,
        "budget_inr": settings.llm_daily_budget_inr,
    }


def llm_budget_exceeded(now: datetime | None = None) -> tuple[bool, dict]:
    """True when the daily budget is spent or when spend cannot be established (fail closed)."""
    summary = llm_spend_today_inr(now)
    exceeded = summary["unknown_pricing_calls"] > 0 or summary["spend_inr"] >= settings.llm_daily_budget_inr
    return exceeded, summary
