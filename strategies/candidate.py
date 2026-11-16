"""Typed trade-candidate contract: asset identity, horizon and broker product kept apart.

``Horizon`` is how long the strategy intends to hold; ``Product`` is the Indian
broker product that carries it. A candidate that cannot prove it is well formed
raises ``CandidateInvalid`` and the router refuses the entry (fail closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

Venue = Literal["NSE", "BSE", "US", "CRYPTO_SIM"]
Segment = Literal["CASH", "ETF"]
Currency = Literal["INR", "USD", "USDT"]
Horizon = Literal["INTRADAY", "SWING", "BTST", "POSITIONAL"]
Product = Literal["INTRADAY", "DELIVERY"]
Side = Literal["BUY", "SELL"]

HORIZONS: tuple[str, ...] = ("INTRADAY", "SWING", "BTST", "POSITIONAL")
# Rows written before horizons were modelled stored the broker product as the holding period.
LEGACY_HORIZON_ALIASES = {"DELIVERY": "SWING"}


class CandidateInvalid(ValueError):
    """The candidate failed contract validation."""


def horizon_to_product(horizon: str) -> Product:
    """INTRADAY maps to the intraday product; every longer horizon is carried as DELIVERY."""
    value = str(horizon).upper()
    if value == "INTRADAY":
        return "INTRADAY"
    if value in HORIZONS or value in LEGACY_HORIZON_ALIASES:
        return "DELIVERY"
    raise CandidateInvalid(f"Unknown horizon: {horizon!r}")


def normalize_horizon(value: Any, *, legacy_rows: bool = False) -> Horizon:
    """Return a canonical horizon; the legacy DELIVERY alias is accepted only when reading old rows."""
    text = str(value or "").upper()
    if text in HORIZONS:
        return text  # type: ignore[return-value]
    if legacy_rows and text in LEGACY_HORIZON_ALIASES:
        return LEGACY_HORIZON_ALIASES[text]  # type: ignore[return-value]
    raise CandidateInvalid(f"Unknown horizon: {value!r}")


def is_intraday_horizon(value: Any) -> bool:
    """Square-off applies to INTRADAY only; unknown or empty values are treated as intraday (legacy default)."""
    text = str(value or "INTRADAY").upper()
    return text == "INTRADAY"


@dataclass(frozen=True)
class AssetId:
    venue: Venue
    segment: Segment
    currency: Currency
    symbol: str

    def __post_init__(self) -> None:
        if self.venue not in ("NSE", "BSE", "US", "CRYPTO_SIM"):
            raise CandidateInvalid(f"Unknown venue: {self.venue!r}")
        if self.segment not in ("CASH", "ETF"):
            raise CandidateInvalid(f"Unknown segment: {self.segment!r}")
        if self.currency not in ("INR", "USD", "USDT"):
            raise CandidateInvalid(f"Unknown currency: {self.currency!r}")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise CandidateInvalid("Asset symbol is required")


@dataclass(frozen=True)
class TradeCandidate:
    asset: AssetId
    candidate_id: str
    strategy: str
    strategy_version: str
    side: Side
    horizon: Horizon
    product: Product
    signal_time: datetime
    expires_at: datetime
    tick_size: Decimal
    lot_size: int
    entry_price: Decimal
    stop_loss: Decimal
    target: Decimal
    exit_policy: str
    atr_value: Decimal
    atr_timeframe: str
    thesis: str
    review_contract_version: str | None = None

    def validate(self, quantity: int | None = None, now: datetime | None = None) -> "TradeCandidate":
        """Raise ``CandidateInvalid`` unless every contract rule holds; return self when valid."""
        now = now or datetime.now(timezone.utc)
        if self.side not in ("BUY", "SELL"):
            raise CandidateInvalid("Side must be BUY or SELL")
        if self.horizon not in HORIZONS:
            raise CandidateInvalid(f"Unknown horizon: {self.horizon!r}")
        if self.product != horizon_to_product(self.horizon):
            raise CandidateInvalid(
                f"Product {self.product} does not match horizon {self.horizon}"
            )
        for name in ("tick_size", "entry_price", "stop_loss", "target", "atr_value"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise CandidateInvalid(f"{name} must be a positive finite Decimal")
        if not isinstance(self.lot_size, int) or isinstance(self.lot_size, bool) or self.lot_size <= 0:
            raise CandidateInvalid("lot_size must be a positive integer")
        if not self.atr_timeframe:
            raise CandidateInvalid("atr_timeframe is required (an ambiguous ATR timeframe is refused)")
        if self.side == "BUY" and not self.stop_loss < self.entry_price < self.target:
            raise CandidateInvalid("BUY requires stop_loss < entry_price < target")
        if self.side == "SELL" and not self.target < self.entry_price < self.stop_loss:
            raise CandidateInvalid("SELL requires target < entry_price < stop_loss")
        for name in ("entry_price", "stop_loss", "target"):
            if getattr(self, name) % self.tick_size != 0:
                raise CandidateInvalid(f"{name} is not a multiple of tick size {self.tick_size}")
        if quantity is not None:
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
                raise CandidateInvalid("quantity must be a positive integer")
            if quantity % self.lot_size != 0:
                raise CandidateInvalid(f"quantity {quantity} is not a multiple of lot size {self.lot_size}")
        for name in ("signal_time", "expires_at"):
            moment = getattr(self, name)
            if not isinstance(moment, datetime) or moment.tzinfo is None:
                raise CandidateInvalid(f"{name} must be timezone-aware")
        if self.expires_at <= now:
            raise CandidateInvalid("Candidate has expired")
        if self.expires_at <= self.signal_time:
            raise CandidateInvalid("expires_at must be after signal_time")
        from risk.exit_policy import get_atr_exit_policy

        try:
            get_atr_exit_policy(self.exit_policy)
        except ValueError as exc:
            raise CandidateInvalid(str(exc)) from exc
        return self


def tick_aligned_levels(
    action: str, price: Any, stop: Any, target: Any, tick: Any,
) -> tuple[float, float, float]:
    """Round the entry to the nearest tick and the stop/target one way: toward the entry.

    Levels move by less than a tick and only ever toward the entry, so planned risk is
    never understated and the stop/target never cross the entry.
    """
    from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

    entry = round_to_tick(price, tick, ROUND_HALF_UP)
    buy = action == "BUY"
    stop_rounded = round_to_tick(stop, tick, ROUND_CEILING if buy else ROUND_FLOOR)
    target_rounded = round_to_tick(target, tick, ROUND_FLOOR if buy else ROUND_CEILING)
    return float(entry), float(stop_rounded), float(target_rounded)


def to_decimal(value: Any, name: str = "value") -> Decimal:
    """Convert via ``str`` so binary float noise never enters tick arithmetic."""
    if isinstance(value, bool):
        raise CandidateInvalid(f"{name} must be a number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CandidateInvalid(f"{name} must be a number") from exc
    if not result.is_finite():
        raise CandidateInvalid(f"{name} must be finite")
    return result


def round_to_tick(value: Any, tick: Any, rounding: str) -> Decimal:
    """Round ``value`` to a multiple of ``tick`` using a ``decimal`` rounding mode name."""
    price, step = to_decimal(value, "price"), to_decimal(tick, "tick_size")
    if step <= 0:
        raise CandidateInvalid("tick_size must be positive")
    return (price / step).to_integral_value(rounding=rounding) * step
