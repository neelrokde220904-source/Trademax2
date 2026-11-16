"""Scheduled, paper-only US equity and local crypto-simulation strategies."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from broker.alpaca_api import alpaca_client
from broker.alpaca_paper_adapter import AlpacaPaperAdapter, DATA_FEED_LABEL
from broker.crypto_sim import (
    CryptoSim, CryptoSimDisabled, OrderRejected, PriceSource, fetch_public_closes,
    fetch_public_price,
)
from broker.portfolio_tracker import portfolio_tracker
from config.settings import PROJECT_ROOT, settings
from data.us_calendar import NEW_YORK, session_bounds
from operations.control_state import control_state
from risk.dead_mans_switch import dead_mans_switch

UTC = timezone.utc
FAST_WINDOW = 20
SLOW_WINDOW = 50
US_STOP_PCT = 0.02
US_TARGET_PCT = 0.05
CRYPTO_STOP_PCT = 0.02
CRYPTO_TARGET_PCT = 0.05
CRYPTO_ENTRY_EQUITY_PCT = 0.05
MAX_QUOTE_AGE_SECONDS = 90
MAX_SPREAD_PCT = 0.002


def moving_average_signal(closes: list[float], price: float) -> tuple[bool, float | None]:
    """Return a long-only 20/50 trend signal and its fast moving average."""
    if len(closes) < SLOW_WINDOW or price <= 0 or any(value <= 0 for value in closes):
        return False, None
    fast = sum(closes[-FAST_WINDOW:]) / FAST_WINDOW
    slow = sum(closes[-SLOW_WINDOW:]) / SLOW_WINDOW
    return fast > slow and price > fast, fast


class _SnapshotPrices(PriceSource):
    """A cycle-scoped crypto price source that never falls back to fixtures."""

    def __init__(self, prices: dict[str, Decimal]) -> None:
        super().__init__(use_public=False)
        self._prices = prices

    def price(self, symbol: str) -> tuple[Decimal, str]:
        if symbol not in self._prices:
            raise OrderRejected(f"No fresh public price for {symbol}")
        return self._prices[symbol], "public_rest"


class ForeignMarketScheduler:
    """Run independent US market-hours and 24/7 crypto simulation cycles.

    Entries share the durable India pause, circuit breaker and drawdown gate. Existing
    foreign positions are monitored and exited before any entry gate is checked.
    """

    STATE_KEY = "foreign_markets"
    CRYPTO_STATE_PATH = PROJECT_ROOT / "database" / "crypto_sim_state.json"

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=UTC, job_defaults={"misfire_grace_time": 120, "coalesce": True})
        self._running = False
        self._cycle_lock = asyncio.Lock()
        self._crypto_sim: CryptoSim | None = None
        self._state = self._load_state()
        self._last_us: dict[str, Any] = {"status": "not_started"}
        self._last_crypto: dict[str, Any] = {"status": "not_started"}

    @classmethod
    def _load_state(cls) -> dict[str, Any]:
        saved = control_state.read(cls.STATE_KEY)
        if saved is None:
            return {"us": {"date": "", "entries": [], "managed": {}, "day_start_equity_usd": None}}
        if not isinstance(saved.get("us"), dict):
            raise ValueError("Invalid persisted foreign-market state; startup must be reviewed")
        saved.setdefault("us", {})
        saved["us"].setdefault("date", "")
        saved["us"].setdefault("entries", [])
        saved["us"].setdefault("managed", {})
        saved["us"].setdefault("day_start_equity_usd", None)
        return saved

    def _persist(self) -> None:
        control_state.write(self.STATE_KEY, self._state)

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        us_job = self.scheduler.get_job("foreign_us_scan") if self._running else None
        crypto_job = self.scheduler.get_job("foreign_crypto_scan") if self._running else None
        return {
            "running": self._running,
            "strategy": "long-only 20/50 moving-average trend; 2% stop, 5% target",
            "us": {
                "enabled": settings.foreign_us_paper_enabled,
                "paper_only": True,
                "watchlist": settings.global_watchlist,
                "feed": DATA_FEED_LABEL,
                "last_run": self._last_us,
                "next_run": us_job.next_run_time.isoformat() if us_job and us_job.next_run_time else None,
            },
            "crypto": {
                "enabled": settings.crypto_sim_enabled,
                "simulated_only": True,
                "watchlist": settings.crypto_sim_watchlist,
                "last_run": self._last_crypto,
                "next_run": crypto_job.next_run_time.isoformat() if crypto_job and crypto_job.next_run_time else None,
                "state_file": str(self.CRYPTO_STATE_PATH),
            },
        }

    def setup(self) -> None:
        if settings.foreign_us_paper_enabled:
            self.scheduler.add_job(
                self._run_us_cycle, IntervalTrigger(minutes=15, timezone=UTC),
                id="foreign_us_scan", name="US paper trend scan (15m)", replace_existing=True,
                max_instances=1,
            )
            self.scheduler.add_job(
                self._run_us_cycle, DateTrigger(run_date=datetime.now(UTC)),
                id="foreign_us_scan_startup", name="US paper startup check", replace_existing=True,
                max_instances=1,
            )
        if settings.crypto_sim_enabled:
            self.scheduler.add_job(
                self._run_crypto_cycle, IntervalTrigger(minutes=15, timezone=UTC),
                id="foreign_crypto_scan", name="Crypto spot simulation scan (15m)", replace_existing=True,
                max_instances=1,
            )
            self.scheduler.add_job(
                self._run_crypto_cycle, DateTrigger(run_date=datetime.now(UTC)),
                id="foreign_crypto_scan_startup", name="Crypto simulation startup check", replace_existing=True,
                max_instances=1,
            )

    def start(self) -> None:
        if self._running or not (settings.foreign_us_paper_enabled or settings.crypto_sim_enabled):
            return
        self.setup()
        self.scheduler.start()
        self._running = True
        logger.info("[ForeignMarkets] Paper/simulation schedules started.")

    def stop(self) -> None:
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[ForeignMarkets] Schedules stopped.")

    def _entry_gate(self) -> tuple[bool, str]:
        if not settings.paper_trading:
            return False, "PAPER_TRADING must remain true"
        from scheduler.market_scheduler import market_scheduler

        if market_scheduler.is_paused:
            return False, "global entries are paused"
        allowed, reason = dead_mans_switch.check_allowed()
        if not allowed:
            return False, f"dead-man's switch: {reason}"
        try:
            if portfolio_tracker.has_hit_portfolio_drawdown_limit():
                return False, "portfolio drawdown halt"
            if portfolio_tracker.has_hit_daily_loss_limit():
                return False, "portfolio daily-loss halt"
        except Exception as exc:
            return False, f"portfolio risk unavailable ({type(exc).__name__})"
        return True, "entry gates clear"

    def _india_gross_inr(self) -> float:
        gross = 0.0
        for symbol, position in portfolio_tracker.positions.items():
            quantity = abs(float(position.get("quantity") or 0))
            mark = float(position.get("ltp") or position.get("avg_price") or 0)
            if quantity and mark <= 0:
                raise ValueError(f"Cannot value open Indian position {symbol}; foreign entry blocked")
            gross += quantity * mark
        return gross

    def _total_equity_inr(self) -> float:
        equity = float(portfolio_tracker.total_value or settings.initial_capital)
        if equity <= 0:
            raise ValueError("Portfolio equity is unavailable")
        return min(equity, float(settings.initial_capital))

    def _us_positions(self) -> list[dict[str, Any]]:
        return alpaca_client.get_positions_strict()

    def _us_gross_inr(
        self, positions: list[dict[str, Any]], open_orders: list[dict[str, Any]] | None = None,
    ) -> float:
        rate = float(settings.foreign_usd_inr_rate)
        gross = sum(abs(float(position.get("market_value") or 0)) * rate for position in positions)
        if open_orders is None and settings.foreign_us_paper_enabled:
            open_orders = alpaca_client.get_open_orders()
        for order in open_orders or []:
            if order.get("side") == "buy":
                remaining = max(0.0, float(order.get("quantity") or 0) - float(order.get("filled_quantity") or 0))
                gross += remaining * float(order.get("limit_price") or 0) * rate
        return gross

    def _crypto_gross_inr(self, prices: dict[str, Decimal] | None = None) -> float:
        if self._crypto_sim is None and settings.crypto_sim_enabled:
            self._get_crypto_sim()
        if not self._crypto_sim or not self._crypto_sim.state.positions:
            return 0.0
        marks = prices if prices is not None else {
            symbol: price
            for symbol in self._crypto_sim.state.positions
            if (price := fetch_public_price(symbol)) is not None
        }
        if any(symbol not in marks for symbol in self._crypto_sim.state.positions):
            raise ValueError("Cannot value every open simulated crypto position; entries blocked")
        gross = self._crypto_sim.gross_exposure_usdt(marks)
        return float(self._crypto_sim.to_inr(gross))

    @staticmethod
    def _quote_is_usable(quote: dict[str, Any], now: datetime) -> tuple[bool, str]:
        try:
            bid, ask = float(quote["bid"]), float(quote["ask"])
            timestamp = datetime.fromisoformat(str(quote["timestamp"]).replace("Z", "+00:00"))
            age = (now - timestamp.astimezone(UTC)).total_seconds()
            spread = (ask - bid) / ((ask + bid) / 2)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False, "quote missing or malformed"
        if bid <= 0 or ask < bid:
            return False, "quote prices invalid"
        if age < -5 or age > MAX_QUOTE_AGE_SECONDS:
            return False, f"quote stale ({age:.0f}s)"
        if spread > MAX_SPREAD_PCT:
            return False, f"spread too wide ({spread:.2%})"
        return True, "fresh quote"

    async def _run_us_cycle(self) -> None:
        async with self._cycle_lock:
            await asyncio.to_thread(self._run_us_cycle_sync)

    def _run_us_cycle_sync(self) -> None:
        now = datetime.now(UTC)
        result: dict[str, Any] = {"at": now.isoformat(), "status": "skipped"}
        try:
            if not settings.foreign_us_paper_enabled:
                result["reason"] = "FOREIGN_US_PAPER_ENABLED is false"
                return
            local = now.astimezone(NEW_YORK)
            bounds = session_bounds(local.date())
            if bounds is None or not bounds[0] <= local < bounds[1]:
                result["reason"] = "US regular session closed or calendar unknown"
                return
            adapter = AlpacaPaperAdapter(alpaca_client)
            account = alpaca_client.get_account()
            if "error" in account or account.get("status") not in {"ACTIVE", "AccountStatus.ACTIVE", None}:
                raise RuntimeError("Alpaca paper account is unavailable or inactive")
            positions = self._us_positions()
            open_orders = alpaca_client.get_open_orders()
            open_symbols = {order["symbol"] for order in open_orders}
            positions_by_symbol = {str(item["tradingsymbol"]).upper(): item for item in positions}
            state = self._state["us"]
            day = local.date().isoformat()
            if state["date"] != day:
                state.update(date=day, entries=[], day_start_equity_usd=float(account["equity"]))
            elif state["day_start_equity_usd"] is None:
                state["day_start_equity_usd"] = float(account["equity"])
            self._persist()

            managed = state["managed"]
            force_flat = local >= bounds[1] - timedelta(minutes=15)
            checks: dict[str, dict[str, Any]] = {}
            for symbol, meta in list(managed.items()):
                position = positions_by_symbol.get(symbol)
                if position is None:
                    if symbol not in open_symbols:
                        managed.pop(symbol, None)
                    continue
                quote = alpaca_client.get_quote(symbol)
                usable, reason = self._quote_is_usable(quote, now)
                if not usable:
                    checks[symbol] = {"status": "exit_check_delayed", "reason": reason}
                    continue
                current = (float(quote["bid"]) + float(quote["ask"])) / 2
                entry = float(meta.get("entry_price") or position.get("average_price") or 0)
                bars = alpaca_client.get_bars(
                    symbol, start=now - timedelta(days=10), end=now - timedelta(minutes=15),
                    minutes=15, limit=500,
                )
                closes = [float(bar["close"]) for bar in bars if float(bar.get("close") or 0) > 0]
                _bullish, fast = moving_average_signal(closes, current)
                should_exit = (
                    force_flat or meta.get("entry_date") != day
                    or (entry > 0 and (current <= entry * (1 - US_STOP_PCT) or current >= entry * (1 + US_TARGET_PCT)))
                    or (fast is not None and current < fast)
                )
                if should_exit and symbol not in open_symbols:
                    response = adapter.place_order({
                        "tradingsymbol": symbol, "transactiontype": "SELL",
                        "quantity": float(position["quantity"]), "price": float(quote["bid"]), "ordertype": "LIMIT",
                    })
                    if response.get("status") not in {"REJECTED", "ERROR", "NOT_CONFIGURED", "FEMA_BLOCKED"}:
                        open_symbols.add(symbol)
                        checks[symbol] = {"status": "exit_submitted", "order_id": response.get("orderid")}
                else:
                    checks[symbol] = {"status": "held", "price": current}
            for symbol in list(managed):
                if symbol not in positions_by_symbol and symbol not in open_symbols:
                    managed.pop(symbol, None)

            # Run exits before checking pause, global halts or entry permissions.
            gate, gate_reason = self._entry_gate()
            if force_flat:
                result.update(status="completed", reason="US close: flattening managed intraday positions")
                result["symbols"] = checks
                self._last_us = result
                self._persist()
                return
            if not gate:
                result["reason"] = gate_reason
                result["symbols"] = checks
                return
            if local < bounds[0] + timedelta(minutes=30) or local >= bounds[1] - timedelta(minutes=30):
                result["reason"] = "outside US strategy entry window"
                return
            equity_inr = self._total_equity_inr()
            india_gross = self._india_gross_inr()
            crypto_gross = self._crypto_gross_inr()
            us_gross = self._us_gross_inr(positions, open_orders)
            gross_limit = equity_inr * settings.max_total_exposure_pct
            daily_loss_limit_usd = (
                equity_inr * settings.max_daily_loss_pct / settings.foreign_usd_inr_rate
            )
            start_equity = float(state["day_start_equity_usd"])
            if start_equity - float(account["equity"]) >= daily_loss_limit_usd:
                result["reason"] = "US paper daily loss cap reached"
                return
            if len(state["entries"]) >= settings.max_daily_entries:
                result["reason"] = "US paper daily entry limit reached"
                return

            entries: dict[str, Any] = {}
            for symbol in settings.global_watchlist:
                symbol = symbol.upper()
                if (
                    symbol in positions_by_symbol or symbol in open_symbols
                    or symbol in state["entries"] or symbol in managed
                ):
                    continue
                quote = alpaca_client.get_quote(symbol)
                usable, reason = self._quote_is_usable(quote, now)
                if not usable:
                    entries[symbol] = {"status": "skipped", "reason": reason}
                    continue
                price = (float(quote["bid"]) + float(quote["ask"])) / 2
                bars = alpaca_client.get_bars(
                    symbol, start=now - timedelta(days=10), end=now - timedelta(minutes=15),
                    minutes=15, limit=500,
                )
                closes = [float(bar["close"]) for bar in bars if float(bar.get("close") or 0) > 0]
                bullish, _fast = moving_average_signal(closes, price)
                if not bullish:
                    entries[symbol] = {"status": "no_signal"}
                    continue
                per_position_limit = equity_inr * settings.max_capital_per_trade_pct
                risk_limit = equity_inr * settings.risk_per_trade_pct / US_STOP_PCT
                gross_available = max(0.0, gross_limit - india_gross - us_gross - crypto_gross)
                order_inr = min(per_position_limit, risk_limit, gross_available)
                available_usd = min(
                    order_inr / settings.foreign_usd_inr_rate,
                    float(account.get("cash") or 0),
                    float(account.get("buying_power") or 0),
                )
                quantity = int(available_usd // float(quote["ask"]))
                if quantity < 1:
                    entries[symbol] = {"status": "risk_cap_or_cash_too_small"}
                    continue
                # Persist the intent before submission so a restart cannot duplicate an order.
                managed[symbol] = {
                    "entry_date": day, "entry_price": float(quote["ask"]), "quantity": quantity,
                    "intent_at": now.isoformat(),
                }
                self._persist()
                response = adapter.place_order({
                    "tradingsymbol": symbol, "transactiontype": "BUY", "quantity": quantity,
                    "price": float(quote["ask"]), "ordertype": "LIMIT",
                })
                if response.get("status") in {"REJECTED", "ERROR", "NOT_CONFIGURED", "FEMA_BLOCKED"}:
                    managed.pop(symbol, None)
                    entries[symbol] = {"status": "rejected"}
                else:
                    state["entries"].append(symbol)
                    open_symbols.add(symbol)
                    us_gross += quantity * float(quote["ask"]) * settings.foreign_usd_inr_rate
                    entries[symbol] = {"status": "submitted", "quantity": quantity, "order_id": response.get("orderid")}
                self._persist()
                if len(state["entries"]) >= settings.max_daily_entries or us_gross + india_gross + crypto_gross >= gross_limit:
                    break
            result.update(status="completed", reason=gate_reason, symbols={**checks, **entries})
        except Exception as exc:
            logger.error("[ForeignMarkets] US cycle failed closed ({})", type(exc).__name__)
            result.update(status="error", reason=type(exc).__name__)
        finally:
            self._last_us = result
            try:
                self._persist()
            except Exception:
                logger.exception("[ForeignMarkets] Could not persist US scheduler state.")

    def _get_crypto_sim(self) -> CryptoSim:
        if self._crypto_sim is None:
            self._crypto_sim = CryptoSim(
                settings.crypto_sim_starting_usdt, prices=PriceSource(use_public=False),
            )
            if self.CRYPTO_STATE_PATH.exists():
                self._crypto_sim.load(self.CRYPTO_STATE_PATH)
            else:
                self._crypto_sim.save(self.CRYPTO_STATE_PATH)
        return self._crypto_sim

    async def _run_crypto_cycle(self) -> None:
        async with self._cycle_lock:
            await asyncio.to_thread(self._run_crypto_cycle_sync)

    def _run_crypto_cycle_sync(self) -> None:
        now = datetime.now(UTC)
        result: dict[str, Any] = {"at": now.isoformat(), "status": "skipped"}
        try:
            if not settings.crypto_sim_enabled:
                result["reason"] = "CRYPTO_SIM_ENABLED is false"
                return
            sim = self._get_crypto_sim()
            symbols = settings.crypto_sim_watchlist
            if not sim.state.positions:
                gate, reason = self._entry_gate()
                if not gate:
                    result.update(
                        reason=reason,
                        symbols={symbol: {"status": "entries_paused", "reason": reason} for symbol in symbols},
                    )
                    return
            prices: dict[str, Decimal] = {}
            signals: dict[str, tuple[bool, Decimal, list[Decimal] | None, str]] = {}
            for symbol in symbols:
                closes = fetch_public_closes(symbol, interval="1h", limit=SLOW_WINDOW)
                price = fetch_public_price(symbol)
                if closes is None or price is None:
                    signals[symbol] = (False, Decimal(0), closes, "fresh public market data unavailable")
                    continue
                prices[symbol] = price
                bullish, _fast = moving_average_signal([float(value) for value in closes], float(price))
                signals[symbol] = (bullish, price, closes, "trend up" if bullish else "no trend signal")
            sim.prices = _SnapshotPrices(prices)
            marks = {symbol: price for symbol, price in prices.items() if symbol in sim.state.positions}
            if any(symbol not in marks for symbol in sim.state.positions):
                result.update(status="blocked", reason="fresh public prices unavailable for an open crypto position")
                return
            equity = sim.equity_usdt(marks)
            utc_day = now.strftime("%Y-%m-%d")
            if sim.state.day != utc_day:
                sim.state.day = utc_day
                sim.state.day_start_equity_usdt = str(equity)
                sim.state.halted = False
                sim.state.halt_reason = ""
            total_equity_inr = self._total_equity_inr()
            day_start = Decimal(sim.state.day_start_equity_usdt)
            overall_daily_loss_usdt = Decimal(str(
                total_equity_inr * settings.max_daily_loss_pct / settings.crypto_usdt_inr_rate
            ))
            crypto_daily_loss_usdt = min(
                day_start * Decimal(str(settings.crypto_halt_daily_loss_pct)),
                overall_daily_loss_usdt,
            )
            if day_start - equity >= crypto_daily_loss_usdt:
                sim.state.halted = True
                sim.state.halt_reason = "crypto or portfolio daily loss cap reached"

            details: dict[str, Any] = {}
            # Always manage simulated positions before the global entry pause/halt checks.
            for symbol in list(sim.state.positions):
                bullish, price, _closes, reason = signals.get(
                    symbol, (False, Decimal(0), None, "fresh public market data unavailable"),
                )
                if price <= 0:
                    details[symbol] = {"status": "exit_check_delayed", "reason": reason}
                    continue
                entry = Decimal(sim.state.average_entry_prices.get(symbol, "0"))
                stop = entry > 0 and price <= entry * Decimal(str(1 - CRYPTO_STOP_PCT))
                target = entry > 0 and price >= entry * Decimal(str(1 + CRYPTO_TARGET_PCT))
                if stop or target or not bullish:
                    fill = sim.place_order(
                        symbol, "SELL", sim.quantity(symbol), "MARKET", now=now,
                        portfolio_equity_inr=total_equity_inr,
                    )
                    details[symbol] = {"status": "simulated_exit", "reason": "stop" if stop else "target" if target else "trend exit",
                                       "price": str(fill["price"])}
                else:
                    details[symbol] = {"status": "held", "price": str(price)}
            gate, gate_reason = self._entry_gate()
            india_gross = self._india_gross_inr()
            us_gross = self._us_gross_inr(self._us_positions()) if settings.foreign_us_paper_enabled else 0.0
            crypto_gross = self._crypto_gross_inr(prices)
            gross_limit = total_equity_inr * settings.max_total_exposure_pct
            if not gate:
                result.update(reason=gate_reason, symbols=details)
            elif sim.state.halted:
                result.update(reason=sim.state.halt_reason, symbols=details)
            else:
                for symbol in symbols:
                    bullish, price, _closes, reason = signals[symbol]
                    if not bullish or price <= 0 or symbol in sim.state.positions:
                        details.setdefault(symbol, {"status": "no_entry", "reason": reason})
                        continue
                    position_limit = total_equity_inr * min(settings.max_capital_per_trade_pct, CRYPTO_ENTRY_EQUITY_PCT)
                    risk_limit = total_equity_inr * settings.risk_per_trade_pct / CRYPTO_STOP_PCT
                    gross_available = max(0.0, gross_limit - india_gross - us_gross - crypto_gross)
                    budget_inr = min(position_limit, risk_limit, gross_available)
                    budget_usdt = Decimal(str(budget_inr / settings.crypto_usdt_inr_rate))
                    budget_usdt = min(budget_usdt, sim.cash * Decimal("0.97"))
                    quantity = (budget_usdt / price).quantize(
                        sim.filters[symbol].step_size, rounding=ROUND_DOWN,
                    )
                    if quantity <= 0:
                        details[symbol] = {"status": "risk_cap_or_cash_too_small"}
                        continue
                    fill = sim.place_order(
                        symbol, "BUY", quantity, "MARKET", now=now,
                        portfolio_equity_inr=total_equity_inr,
                        portfolio_gross_inr=india_gross + us_gross,
                    )
                    details[symbol] = {
                        "status": "simulated_entry", "quantity": str(fill["quantity"]),
                        "price": str(fill["price"]), "source": fill["source"],
                    }
                    crypto_gross += float(fill["quantity"] * fill["price"] * Decimal(str(settings.crypto_usdt_inr_rate)))
                    sim.save(self.CRYPTO_STATE_PATH)
                    if crypto_gross + india_gross + us_gross >= gross_limit:
                        break
                result.update(status="completed", reason=gate_reason, symbols=details)
            sim.save(self.CRYPTO_STATE_PATH)
        except (CryptoSimDisabled, OrderRejected, ValueError, RuntimeError) as exc:
            logger.warning("[ForeignMarkets] Crypto cycle blocked ({})", type(exc).__name__)
            result.update(status="blocked", reason=str(exc)[:240])
        except Exception as exc:
            logger.error("[ForeignMarkets] Crypto cycle failed closed ({})", type(exc).__name__)
            result.update(status="error", reason=type(exc).__name__)
        finally:
            self._last_crypto = result
            try:
                self._persist()
            except Exception:
                logger.exception("[ForeignMarkets] Could not persist crypto scheduler state.")


foreign_market_scheduler = ForeignMarketScheduler()
