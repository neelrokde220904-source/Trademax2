"""ETF benchmark rotation: hold the strongest ETF while it is trending, otherwise stay in cash.

Daily completed bars only. An ETF qualifies when its 63-session return is positive and it
closes above its 200-session average; the single best qualifier is the candidate. Being in
cash is the default, so a weak market produces no candidate.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


class ETFRotation:
    name = "ETFRotation"
    momentum_sessions = 63
    trend_sessions = 200

    def rank(self, bars_by_symbol: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
        """Qualifying ETFs by momentum, best first (empty means stay in cash)."""
        ranked: list[tuple[str, float]] = []
        for symbol, bars in bars_by_symbol.items():
            if not isinstance(bars, pd.DataFrame) or "close" not in bars:
                continue
            close = pd.to_numeric(bars["close"], errors="coerce").dropna()
            if len(close) < self.trend_sessions + 1:
                continue
            latest = float(close.iloc[-1])
            momentum = latest / float(close.iloc[-1 - self.momentum_sessions]) - 1.0
            trend = float(close.iloc[-self.trend_sessions:].mean())
            if math.isfinite(momentum) and momentum > 0 and latest > trend:
                ranked.append((symbol, momentum))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def scan(
        self, symbol: str, bars: pd.DataFrame, peers: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any] | None:
        universe = {**(peers or {}), symbol: bars}
        ranked = self.rank(universe)
        if not ranked or ranked[0][0] != symbol:
            return None
        close = float(pd.to_numeric(bars["close"], errors="coerce").dropna().iloc[-1])
        return {
            "symbol": symbol, "action": "BUY", "price": close, "holding_period": "POSITIONAL",
            "strategy": self.name,
            "reasoning": (
                f"Top-ranked ETF: {self.momentum_sessions}-session momentum {ranked[0][1]:.1%} "
                f"and above its {self.trend_sessions}-session average."
            ),
        }


etf_rotation = ETFRotation()
