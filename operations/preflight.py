"""Fail-closed operational readiness checks.

Run with ``python3 main.py --preflight`` before a scheduled session.  The checker
does not authenticate, submit orders, or modify state, so it is safe in all modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import settings


@dataclass(frozen=True)
class PreflightResult:
    """A machine- and human-readable preflight result."""

    passed: bool
    errors: list[str]
    warnings: list[str]


def run_preflight() -> PreflightResult:
    """Validate safe configuration and local prerequisites without calling brokers."""
    errors: list[str] = []
    warnings: list[str] = []

    if settings.initial_capital <= 0:
        errors.append("INITIAL_CAPITAL must be greater than zero.")
    if not 0 < settings.max_capital_per_trade_pct <= 0.10:
        errors.append("MAX_CAPITAL_PER_TRADE_PCT must be in (0, 0.10].")
    if settings.max_positions < 1:
        errors.append("MAX_POSITIONS must be at least 1.")
    if not 0 < settings.max_daily_loss_pct <= 0.02:
        errors.append("MAX_DAILY_LOSS_PCT must be in (0, 0.02].")
    if not 0 < settings.max_portfolio_drawdown_pct <= 0.10:
        errors.append("MAX_PORTFOLIO_DRAWDOWN_PCT must be in (0, 0.10].")
    if settings.max_nifty_vix >= settings.vix_halt_threshold:
        errors.append("MAX_NIFTY_VIX must be lower than VIX_HALT_THRESHOLD.")

    if not 120 <= settings.scan_entry_deadline_seconds <= 1800:
        errors.append("SCAN_ENTRY_DEADLINE_SECONDS must be in [120, 1800].")
    if not 0 < settings.llm_daily_budget_inr <= 5000:
        errors.append("LLM_DAILY_BUDGET_INR must be in (0, 5000].")
    if not (1 <= settings.order_rate_per_second <= 10 and 1 <= settings.order_rate_per_minute <= 200):
        errors.append("ORDER_RATE_PER_SECOND must be in [1, 10] and ORDER_RATE_PER_MINUTE in [1, 200].")

    if not (1 <= settings.swing_max_holding_days <= 60 and 1 <= settings.positional_max_holding_days <= 250):
        errors.append("SWING_MAX_HOLDING_DAYS must be in [1, 60] and POSITIONAL_MAX_HOLDING_DAYS in [1, 250].")
    if not 0 <= settings.btst_delivery_minutes_after_open <= 120:
        errors.append("BTST_DELIVERY_MINUTES_AFTER_OPEN must be in [0, 120].")

    if not 0 < settings.crypto_usdt_inr_rate <= 1000:
        errors.append("CRYPTO_USDT_INR_RATE must be in (0, 1000].")
    if not 0 < settings.crypto_halt_daily_loss_pct <= 0.20:
        errors.append("CRYPTO_HALT_DAILY_LOSS_PCT must be in (0, 0.20].")
    if not 0 < settings.crypto_sim_starting_usdt <= 100_000:
        errors.append("CRYPTO_SIM_STARTING_USDT must be in (0, 100000].")
    if not 0 < settings.foreign_usd_inr_rate <= 1000:
        errors.append("FOREIGN_USD_INR_RATE must be in (0, 1000].")
    if settings.foreign_us_paper_enabled:
        if settings.alpaca_paper is not True:
            errors.append("FOREIGN_US_PAPER_ENABLED requires ALPACA_PAPER=true; live US trading is not supported.")
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            warnings.append("FOREIGN_US_PAPER_ENABLED is true but Alpaca paper keys are missing; US orders will be refused.")
    if settings.crypto_sim_enabled:
        warnings.append("CRYPTO_SIM_ENABLED is true: a local simulation only, with a static USDT/INR rate and no real exchange.")

    enabled_sleeves = [
        name for name in type(settings).model_fields
        if name.startswith("strategy_") and name.endswith("_enabled") and getattr(settings, name)
    ]
    if enabled_sleeves:
        warnings.append(
            "Strategy sleeves enabled: " + ", ".join(sorted(enabled_sleeves))
            + ". Each needs T10 out-of-sample evidence before being relied on."
        )

    if settings.signal_gate_enabled:
        from models.signal_gate.gate import SignalGate

        try:
            SignalGate()._load()
        except Exception as exc:
            errors.append(f"SIGNAL_GATE_ENABLED is true but the gate model is unusable ({exc}); entries would all be blocked.")

    from datetime import date

    from data.nse_calendar import calendar_year_known

    today = date.today()
    if not calendar_year_known(today.year):
        errors.append(f"No NSE holiday calendar for {today.year}; entries are blocked until it is added.")
    elif today.month == 12 and not calendar_year_known(today.year + 1):
        warnings.append(
            f"No NSE holiday calendar for {today.year + 1} yet; add the official NSE circular "
            "to data/calendars/nse_equity_holidays.json before 1 January."
        )

    from compliance.broker_preflight import check_broker_readiness
    broker_readiness = check_broker_readiness()
    errors.extend(broker_readiness.errors)
    warnings.extend(broker_readiness.warnings)

    cache_path = Path(settings.instrument_cache_path)
    if not cache_path.exists():
        warnings.append("Instrument cache is absent; the 08:00 pre-market job must download it.")
    else:
        from data.instrument_master import InstrumentMasterUnavailableError, load_instrument_master

        try:
            load_instrument_master()
        except InstrumentMasterUnavailableError as exc:
            warnings.append(f"Instrument cache is unusable: {exc}")
    if settings.trading_universe_mode == "nifty50":
        from config.universe import UniverseUnavailableError, load_nifty50_membership

        try:
            load_nifty50_membership()
        except UniverseUnavailableError as exc:
            warnings.append(f"Nifty 50 universe is not ready: {exc}")
    if not settings.paper_trading:
        required = {}
        if settings.trademax_broker_mode in {"angel_only", "dual_active_failover"}:
            required.update({
                "ANGEL_API_KEY": settings.angel_api_key,
                "ANGEL_CLIENT_ID": settings.angel_client_id,
                "ANGEL_MPIN": settings.angel_mpin,
                "ANGEL_TOTP_SECRET": settings.angel_totp_secret,
            })
        if settings.trademax_broker_mode in {"zerodha_only", "dual_active_failover"}:
            required.update({
                "ZERODHA_API_KEY": settings.zerodha_api_key,
                "ZERODHA_ACCESS_TOKEN": settings.zerodha_access_token,
            })
        missing = [name for name, value in required.items() if not value or value.startswith("your_")]
        if missing:
            errors.append(f"Live trading requires configured broker credentials: {', '.join(missing)}.")
    elif not settings.angel_api_key or settings.angel_api_key.startswith("your_"):
        warnings.append("No Angel One credentials: paper fills will use requested prices when LTP is unavailable.")

    from sqlalchemy import inspect
    from config.llm_runtime import configured, profile_for
    from database.db import engine, get_db
    from database.models import LLMModeSetting

    try:
        mode = "claude"
        database_path = engine.url.database
        database_exists = (engine.dialect.name != "sqlite" or database_path == ":memory:"
                   or bool(database_path and Path(database_path).exists()))
        if database_exists and inspect(engine).has_table(LLMModeSetting.__tablename__):
            with get_db() as database:
                selection = database.get(LLMModeSetting, 1)
                if selection is not None:
                    mode = selection.mode
        if not configured(profile_for(mode)):
            warnings.append(f"Selected LLM mode '{mode}' is not configured; existing non-LLM fallbacks will be used.")
    except Exception:
        errors.append("Persisted LLM selection could not be validated.")

    return PreflightResult(not errors, errors, warnings)
