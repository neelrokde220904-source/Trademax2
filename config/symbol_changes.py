"""Exchange-confirmed NSE symbol changes.

Only add entries backed by an exchange circular for the same listed security.
Demergers that create a new company (for example TMCV) are not aliases.
"""

from __future__ import annotations

# NSE circular CML70875 (2025-10-16): TATAMOTORS -> TMPV, effective 2025-10-24.
NSE_SYMBOL_CHANGES: dict[str, str] = {
    "TATAMOTORS": "TMPV",
}


def canonical_nse_symbol(symbol: str) -> str:
    """Return the current NSE symbol, preserving any '-EQ' suffix."""
    cleaned = symbol.strip().upper()
    base, suffix = (cleaned[:-3], "-EQ") if cleaned.endswith("-EQ") else (cleaned, "")
    return NSE_SYMBOL_CHANGES.get(base, base) + suffix


def canonical_watchlist(symbols: list[str]) -> list[str]:
    """Canonicalise renamed symbols and drop duplicates, keeping order."""
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        current = canonical_nse_symbol(symbol)
        if current and current not in seen:
            seen.add(current)
            result.append(current)
    return result
