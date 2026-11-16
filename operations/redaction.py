"""Mask credentials before they reach a log line or a log file.

``redact`` is applied by a loguru patcher (every message the application logs) and can be used
to scrub existing files. It masks, without ever printing the matched value:
- Angel client codes that follow the word "client",
- JWTs and ``Bearer`` tokens,
- values after ``api_key=``, ``apikey``, ``totp``, ``mpin``, ``password``, ``secret``, ``X-PrivateKey``,
- ``Authorization`` header values,
- any configured secret (API keys, client ID, MPIN, TOTP secret, bot token) wherever it appears.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

MASK = "[REDACTED]"

_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_AUTH_HEADER = re.compile(r"""(?i)(authorization['"]?\s*[:=]\s*['"]?)([^'",}\n]+)""")
_PRIVATE_KEY_HEADER = re.compile(r"""(?i)(x-?privatekey['"]?\s*[:=]\s*['"]?)([^'",}\s]+)""")
_SECRET_ASSIGNMENT = re.compile(
    r"""(?i)\b(api[_-]?key|apikey|totp(?:[_-]?secret)?|mpin|password|passwd|secret|access[_-]?token|"""
    r"""refresh[_-]?token|feed[_-]?token|jwt[_-]?token)(['"]?\s*[=:]\s*['"]?)([^\s,'"&;)}\]]+)"""
)
_CLIENT_CODE = re.compile(r"(?i)\b(client(?:[\s_-]*(?:id|code))?\W{0,3})([A-Z]\d{5,})\b")

_SECRET_SETTINGS = (
    "angel_api_key", "angel_client_id", "angel_mpin", "angel_totp_secret", "telegram_bot_token",
    "anthropic_api_key", "deepseek_api_key", "openai_api_key", "alpaca_api_key", "alpaca_secret_key",
    "zerodha_api_key", "zerodha_access_token",
)


def _known_secrets() -> list[str]:
    from config.settings import settings

    values = []
    for name in _SECRET_SETTINGS:
        value = getattr(settings, name, "")
        if isinstance(value, str) and len(value) >= 6 and not value.startswith("your_"):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact(text: str, *, include_configured: bool = True) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = _JWT.sub(MASK, text)
    text = _BEARER.sub("Bearer " + MASK, text)
    text = _AUTH_HEADER.sub(lambda m: m.group(1) + MASK, text)
    text = _PRIVATE_KEY_HEADER.sub(lambda m: m.group(1) + MASK, text)
    text = _SECRET_ASSIGNMENT.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    text = _CLIENT_CODE.sub(lambda m: m.group(1) + MASK, text)
    if include_configured:
        for secret in _known_secrets():
            text = text.replace(secret, MASK)
    return text


def loguru_patcher(record: dict[str, Any]) -> None:
    """Mask the message of every log record before any sink formats it."""
    record["message"] = redact(record["message"])


def silence_sdk_logging() -> None:
    """The SmartAPI SDK logs full request headers (API key, JWT) to ``logs/<date>/app.log``.

    Remove its file handler and raise its levels so nothing it formats is written to disk.
    Our own code logs failures without headers.
    """
    for name in ("SmartApi", "SmartApi.smartConnect", "SmartApi.smartWebSocketV2", "websocket"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        import logzero

        logzero.logfile(None)  # drops the date-wise file handler the SDK installs
        logzero.logger.setLevel(logging.CRITICAL)
    except Exception:
        pass


def scrub_file(path: Path) -> int:
    """Redact a log file in place; returns the number of lines changed (never the values)."""
    changed = 0
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    out = []
    for line in lines:
        cleaned = redact(line)
        changed += cleaned != line
        out.append(cleaned)
    if changed:
        path.write_text("".join(out))
    return changed
