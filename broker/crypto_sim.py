"""Local crypto spot simulator (disabled unless ``CRYPTO_SIM_ENABLED``).

Nothing here talks to an exchange account. It models a long-only, unlevered spot book in USDT with
fractional quantities, tick and step filters, a minimum notional, maker/taker fees and a 24/7 UTC
session reported in IST. USDT→INR uses a *configured* rate that is labelled as such. Market data
comes from a public REST endpoint only when it answers without keys; otherwise recorded fixtures.

Not modelled and not possible: withdrawals, leverage, futures, margin, staking or lending.
The crypto book has its own daily-loss halt that never touches the Indian book, while the
portfolio-wide position and exposure caps still apply to every entry.

Indian tax output (``vda_tax_estimate``) follows the VDA rules: 30% on each gain with no set-off
of losses, and 1% TDS on the sale consideration shown as withholding. It is an estimate for
planning, not tax advice.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from loguru import logger

from config.settings import settings

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc
VDA_TAX_RATE = Decimal("0.30")
VDA_TDS_RATE = Decimal("0.01")
DEFAULT_MAKER_FEE = Decimal("0.001")
DEFAULT_TAKER_FEE = Decimal("0.001")
PUBLIC_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
PUBLIC_KLINES_URL = "https://api.binance.com/api/v3/klines"

FX_LABEL = "configured static USDT/INR rate (CRYPTO_USDT_INR_RATE), not a market quote"


class CryptoSimDisabled(RuntimeError):
    """CRYPTO_SIM_ENABLED is false."""


class OrderRejected(ValueError):
    """The simulated order fails a filter, a cap or a halt (never raised for a no-fill)."""


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal  # USDT

    @classmethod
    def defaults(cls) -> dict[str, "SymbolFilters"]:
        return {
            "BTCUSDT": cls(Decimal("0.01"), Decimal("0.00001"), Decimal("5")),
            "ETHUSDT": cls(Decimal("0.01"), Decimal("0.0001"), Decimal("5")),
            "SOLUSDT": cls(Decimal("0.01"), Decimal("0.001"), Decimal("5")),
        }


@dataclass
class CryptoTrade:
    time_utc: str
    symbol: str
    side: str
    quantity: str
    price: str
    fee_usdt: str
    liquidity: str  # MAKER | TAKER


def ist_label(moment: datetime) -> str:
    """UTC instants are reported in IST for the operator."""
    return moment.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def session_is_open(_moment: datetime | None = None) -> bool:
    """Crypto trades 24/7 in UTC; there is no holiday calendar."""
    return True


# ── market data ──────────────────────────────────────────────────────────────

FIXTURE_PRICES = {"BTCUSDT": Decimal("65000.00"), "ETHUSDT": Decimal("3200.00"), "SOLUSDT": Decimal("150.00")}


def fetch_public_price(symbol: str, *, timeout: float = 4.0, getter: Callable[..., Any] | None = None) -> Decimal | None:
    """Last price from the keyless public endpoint, or None when it is not reachable."""
    try:
        if getter is None:
            import httpx

            getter = httpx.get
        response = getter(PUBLIC_PRICE_URL, params={"symbol": symbol}, timeout=timeout)
        if response.status_code != 200:
            logger.warning("[CryptoSim] Public ticker returned HTTP {} for {}", response.status_code, symbol)
            return None
        price = Decimal(str(response.json()["price"]))
        return price if price > 0 and price.is_finite() else None
    except Exception as exc:
        logger.warning("[CryptoSim] Public ticker unavailable for {} ({})", symbol, type(exc).__name__)
        return None


def fetch_public_closes(
    symbol: str, *, interval: str = "1h", limit: int = 60,
    timeout: float = 6.0, getter: Callable[..., Any] | None = None,
) -> list[Decimal] | None:
    """Fetch completed Binance candles; return None for an unavailable or malformed feed."""
    try:
        if getter is None:
            import httpx

            getter = httpx.get
        response = getter(
            PUBLIC_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit + 1},
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.warning("[CryptoSim] Public candles returned HTTP {} for {}", response.status_code, symbol)
            return None
        rows = response.json()
        if not isinstance(rows, list) or len(rows) < limit:
            return None
        closes = [Decimal(str(row[4])) for row in rows[:-1]]
        if len(closes) < limit or any(value <= 0 or not value.is_finite() for value in closes):
            return None
        return closes[-limit:]
    except Exception as exc:
        logger.warning("[CryptoSim] Public candles unavailable for {} ({})", symbol, type(exc).__name__)
        return None


class PriceSource:
    """Public REST when reachable (keyless), recorded fixtures otherwise; the source is always labelled."""

    def __init__(self, *, use_public: bool = True, getter: Callable[..., Any] | None = None) -> None:
        self.use_public, self._getter = use_public, getter

    def price(self, symbol: str) -> tuple[Decimal, str]:
        if self.use_public:
            live = fetch_public_price(symbol, getter=self._getter)
            if live is not None:
                return live, "public_rest"
        if symbol in FIXTURE_PRICES:
            return FIXTURE_PRICES[symbol], "fixture"
        raise OrderRejected(f"No price available for {symbol}")


# ── the book ─────────────────────────────────────────────────────────────────

@dataclass
class CryptoSimState:
    cash_usdt: str
    positions: dict[str, str] = field(default_factory=dict)  # symbol -> quantity
    trades: list[CryptoTrade] = field(default_factory=list)
    day: str = ""
    day_start_equity_usdt: str = "0"
    average_entry_prices: dict[str, str] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""


class CryptoSim:
    def __init__(
        self, starting_usdt: Decimal | float = 1000, *, prices: PriceSource | None = None,
        filters: dict[str, SymbolFilters] | None = None, slippage_bps: Decimal = Decimal("5"),
        maker_fee: Decimal = DEFAULT_MAKER_FEE, taker_fee: Decimal = DEFAULT_TAKER_FEE,
    ) -> None:
        if not settings.crypto_sim_enabled:
            raise CryptoSimDisabled("The crypto simulator is disabled (CRYPTO_SIM_ENABLED=false)")
        start = Decimal(str(starting_usdt))
        if start <= 0:
            raise ValueError("starting_usdt must be positive")
        self.prices = prices or PriceSource(use_public=False)
        self.filters = filters or SymbolFilters.defaults()
        self.slippage = slippage_bps / Decimal(10000)
        self.maker_fee, self.taker_fee = maker_fee, taker_fee
        self.state = CryptoSimState(cash_usdt=str(start), day_start_equity_usdt=str(start))

    # -- valuation ---------------------------------------------------------------
    @property
    def cash(self) -> Decimal:
        return Decimal(self.state.cash_usdt)

    def quantity(self, symbol: str) -> Decimal:
        return Decimal(self.state.positions.get(symbol, "0"))

    def equity_usdt(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        total = self.cash
        for symbol, qty in self.state.positions.items():
            price = (marks or {}).get(symbol) or self.prices.price(symbol)[0]
            total += Decimal(qty) * price
        return total

    def gross_exposure_usdt(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        return sum(
            (Decimal(qty) * ((marks or {}).get(symbol) or self.prices.price(symbol)[0])
             for symbol, qty in self.state.positions.items()), Decimal(0),
        )

    def to_inr(self, usdt: Decimal) -> Decimal:
        """INR value at the configured static rate (see ``FX_LABEL``)."""
        return usdt * Decimal(str(settings.crypto_usdt_inr_rate))

    # -- halt --------------------------------------------------------------------
    def _roll_day(self, now: datetime, equity: Decimal) -> None:
        day = now.astimezone(UTC).strftime("%Y-%m-%d")
        if self.state.day != day:
            self.state.day, self.state.day_start_equity_usdt = day, str(equity)
            self.state.halted, self.state.halt_reason = False, ""

    def _update_halt(self, equity: Decimal) -> None:
        start = Decimal(self.state.day_start_equity_usdt)
        if start > 0 and (start - equity) / start >= Decimal(str(settings.crypto_halt_daily_loss_pct)):
            self.state.halted = True
            self.state.halt_reason = f"crypto daily loss reached {settings.crypto_halt_daily_loss_pct:.0%}"

    # -- orders ------------------------------------------------------------------
    def place_order(
        self, symbol: str, side: str, quantity: Decimal | float | str, order_type: str = "MARKET",
        limit_price: Decimal | float | str | None = None, *, now: datetime | None = None,
        portfolio_equity_inr: Decimal | float | None = None, portfolio_gross_inr: Decimal | float = 0,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        """Simulate one order. Returns a fill dict; a non-marketable LIMIT returns status NOT_MARKETABLE."""
        now = now or datetime.now(UTC)
        side, order_type = side.upper(), order_type.upper()
        if leverage != 1.0:
            raise OrderRejected("Leverage, margin and futures are not available")
        if side not in {"BUY", "SELL"} or order_type not in {"MARKET", "LIMIT"}:
            raise OrderRejected("side must be BUY/SELL and order_type MARKET/LIMIT")
        flt = self.filters.get(symbol)
        if flt is None:
            raise OrderRejected(f"{symbol} has no trading filters; it is not tradable")
        reference, source = self.prices.price(symbol)
        marks = {symbol: reference}
        equity = self.equity_usdt(marks)
        self._roll_day(now, equity)
        qty = Decimal(str(quantity)).quantize(flt.step_size, rounding=ROUND_DOWN)
        if qty <= 0:
            raise OrderRejected("quantity is below the step size")
        if side == "BUY" and self.state.halted:
            raise OrderRejected(f"Crypto entries are halted: {self.state.halt_reason}")

        limit = None
        if order_type == "LIMIT":
            if limit_price is None:
                raise OrderRejected("a LIMIT order needs a price")
            limit = Decimal(str(limit_price))
            if limit % flt.tick_size != 0:
                raise OrderRejected(f"price is not a multiple of the tick size {flt.tick_size}")
            marketable = reference <= limit if side == "BUY" else reference >= limit
            if not marketable:
                return {"status": "NOT_MARKETABLE", "symbol": symbol, "reference": reference, "source": source}

        slipped = reference * (1 + self.slippage if side == "BUY" else 1 - self.slippage)
        fill = (min(slipped, limit) if side == "BUY" else max(slipped, limit)) if limit is not None else slipped
        fill = fill.quantize(flt.tick_size, rounding=ROUND_DOWN)
        notional = qty * fill
        if notional < flt.min_notional:
            raise OrderRejected(f"notional {notional:.2f} USDT is below the minimum {flt.min_notional}")
        liquidity = "TAKER"  # every simulated order is IOC against the reference price
        fee = notional * (self.taker_fee if liquidity == "TAKER" else self.maker_fee)

        if side == "BUY":
            if notional + fee > self.cash:
                raise OrderRejected("insufficient USDT")
            self._check_portfolio_caps(notional, portfolio_equity_inr, portfolio_gross_inr, equity, marks)
            old_quantity = self.quantity(symbol)
            old_entry = Decimal(self.state.average_entry_prices.get(symbol, str(fill)))
            self.state.average_entry_prices[symbol] = str(
                ((old_quantity * old_entry) + (qty * fill)) / (old_quantity + qty)
            )
            self.state.cash_usdt = str(self.cash - notional - fee)
            self.state.positions[symbol] = str(self.quantity(symbol) + qty)
        else:
            if qty > self.quantity(symbol):
                raise OrderRejected("selling more than held: spot only, no shorting")
            self.state.cash_usdt = str(self.cash + notional - fee)
            remaining = self.quantity(symbol) - qty
            if remaining > 0:
                self.state.positions[symbol] = str(remaining)
            else:
                self.state.positions.pop(symbol, None)
                self.state.average_entry_prices.pop(symbol, None)
        self.state.trades.append(CryptoTrade(
            time_utc=now.astimezone(UTC).isoformat(), symbol=symbol, side=side, quantity=str(qty),
            price=str(fill), fee_usdt=str(fee), liquidity=liquidity,
        ))
        self._update_halt(self.equity_usdt(marks))
        return {
            "status": "FILLED", "symbol": symbol, "side": side, "quantity": qty, "price": fill, "fee_usdt": fee,
            "source": source, "time_ist": ist_label(now), "fx": FX_LABEL,
        }

    def _check_portfolio_caps(
        self, notional: Decimal, portfolio_equity_inr: Any, portfolio_gross_inr: Any,
        book_equity: Decimal, marks: dict[str, Decimal],
    ) -> None:
        """The portfolio-wide limits (10% per position, 50% gross) apply to crypto entries too."""
        equity_inr = Decimal(str(portfolio_equity_inr)) if portfolio_equity_inr else self.to_inr(book_equity)
        if equity_inr <= 0:
            raise OrderRejected("portfolio equity is unavailable")
        entry_inr = self.to_inr(notional)
        if entry_inr > equity_inr * Decimal(str(settings.max_capital_per_trade_pct)):
            raise OrderRejected("entry exceeds the per-position cap of portfolio equity")
        gross_inr = Decimal(str(portfolio_gross_inr)) + self.to_inr(self.gross_exposure_usdt(marks))
        if gross_inr + entry_inr > equity_inr * Decimal(str(settings.max_total_exposure_pct)):
            raise OrderRejected("entry would exceed the gross exposure cap of portfolio equity")

    # -- persistence -------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self.state), indent=2))
        temporary.replace(path)

    def load(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        raw["trades"] = [CryptoTrade(**trade) for trade in raw.get("trades", [])]
        if "average_entry_prices" not in raw:
            lots: dict[str, deque[list[Decimal]]] = {}
            for trade in raw["trades"]:
                quantity, price = Decimal(trade.quantity), Decimal(trade.price)
                queue = lots.setdefault(trade.symbol, deque())
                if trade.side == "BUY":
                    queue.append([quantity, price])
                    continue
                remaining = quantity
                while remaining > 0 and queue:
                    lot = queue[0]
                    used = min(remaining, lot[0])
                    lot[0] -= used
                    remaining -= used
                    if lot[0] == 0:
                        queue.popleft()
            raw["average_entry_prices"] = {
                symbol: str(sum((qty * price for qty, price in lots.get(symbol, [])), Decimal(0))
                            / Decimal(quantity))
                for symbol, quantity in raw.get("positions", {}).items()
                if Decimal(quantity) > 0 and sum((qty for qty, _ in lots.get(symbol, [])), Decimal(0)) > 0
            }
        self.state = CryptoSimState(**raw)


# ── tax ──────────────────────────────────────────────────────────────────────

def realised_gains(trades: list[CryptoTrade]) -> list[dict[str, Decimal]]:
    """FIFO gains per sale in USDT (cost = purchase price x quantity; fees are not deducted for tax)."""
    lots: dict[str, deque[list[Decimal]]] = {}
    gains: list[dict[str, Decimal]] = []
    for trade in trades:
        quantity, price = Decimal(trade.quantity), Decimal(trade.price)
        queue = lots.setdefault(trade.symbol, deque())
        if trade.side == "BUY":
            queue.append([quantity, price])
            continue
        remaining, cost, proceeds = quantity, Decimal(0), quantity * price
        while remaining > 0 and queue:
            lot = queue[0]
            take = min(remaining, lot[0])
            cost += take * lot[1]
            lot[0] -= take
            remaining -= take
            if lot[0] == 0:
                queue.popleft()
        gains.append({"proceeds": proceeds, "cost": cost, "gain": proceeds - cost})
    return gains


def vda_tax_estimate(trades: list[CryptoTrade], usdt_inr: Decimal | float | None = None) -> dict[str, Any]:
    """30% of each positive gain, no set-off of losses, plus the 1% TDS shown as withholding."""
    rate = Decimal(str(usdt_inr if usdt_inr is not None else settings.crypto_usdt_inr_rate))
    gains = realised_gains(trades)
    taxable_gain = sum((g["gain"] for g in gains if g["gain"] > 0), Decimal(0))
    losses_ignored = sum((-g["gain"] for g in gains if g["gain"] < 0), Decimal(0))
    sale_value = sum((g["proceeds"] for g in gains), Decimal(0))
    tax = taxable_gain * VDA_TAX_RATE
    tds = sale_value * VDA_TDS_RATE
    return {
        "currency": "INR", "fx": FX_LABEL, "usdt_inr": rate,
        "taxable_gain_inr": taxable_gain * rate, "tax_30pct_inr": tax * rate,
        "losses_not_set_off_inr": losses_ignored * rate,
        "tds_1pct_withholding_inr": tds * rate,
        "net_tax_after_tds_inr": max(Decimal(0), tax - tds) * rate,
        "notes": [
            "Gains are taxed per transaction at 30%; losses cannot be set off against gains or other income.",
            "1% TDS on the sale consideration is shown as withholding and is creditable against the tax.",
            "An estimate for planning only; it is not tax advice.",
        ],
    }
