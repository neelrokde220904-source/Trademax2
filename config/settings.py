"""Pydantic BaseSettings — centralised application configuration loaded from .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration for the India AI Trader system."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # List settings are deliberately parsed by explicit validators so the
        # documented comma-separated .env syntax is accepted.
        enable_decoding=False,
    )

    # --- Angel One SmartAPI ---
    angel_api_key: str = Field(default="", repr=False, description="Angel One API key")
    angel_client_id: str = Field(default="", repr=False, description="Angel One client ID")
    angel_mpin: str = Field(default="", repr=False, description="Angel One MPIN")
    angel_totp_secret: str = Field(default="", repr=False, description="TOTP secret for Angel One 2FA")

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", repr=False, description="Anthropic API key")
    openai_api_key: str = Field(default="", repr=False)
    google_api_key: str = Field(default="", repr=False)
    xai_api_key: str = Field(default="", repr=False)
    dashscope_api_key: str = Field(default="", repr=False)
    dashscope_cn_api_key: str = Field(default="", repr=False)
    zhipu_api_key: str = Field(default="", repr=False)
    zhipu_cn_api_key: str = Field(default="", repr=False)
    minimax_api_key: str = Field(default="", repr=False)
    minimax_cn_api_key: str = Field(default="", repr=False)
    mistral_api_key: str = Field(default="", repr=False)
    moonshot_api_key: str = Field(default="", repr=False)
    groq_api_key: str = Field(default="", repr=False)
    nvidia_api_key: str = Field(default="", repr=False)
    openrouter_api_key: str = Field(default="", repr=False)
    azure_openai_api_key: str = Field(default="", repr=False)
    azure_openai_endpoint: str = Field(default="")
    azure_openai_api_version: str = Field(default="2024-10-21")
    aws_default_region: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    openai_compatible_api_key: str = Field(default="", repr=False)
    trademax_llm_backend_url: str = Field(default="")

    trademax_llm_provider: str = Field(default="auto")
    trademax_quick_think_provider: str = Field(default="")
    trademax_quick_think_llm: str = Field(default="")
    trademax_deep_think_provider: str = Field(default="")
    trademax_deep_think_llm: str = Field(default="")
    trademax_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    trademax_llm_retry_budget: int = Field(default=0, ge=0, le=5)

    # --- Zerodha Kite Connect ---
    zerodha_api_key: str = Field(default="", repr=False)
    zerodha_api_secret: str = Field(default="", repr=False)
    zerodha_access_token: str = Field(default="", repr=False)
    trademax_broker_mode: Literal[
        "zerodha_only", "angel_only", "dual_active_failover"
    ] = "angel_only"
    trademax_primary_broker: Literal["zerodha", "angel"] = "angel"
    trademax_static_outbound_ip: str = Field(default="")
    broker_static_ip_allowlist_confirmed: bool = Field(default=False)

    # --- News ---
    news_api_key: str = Field(default="", repr=False, description="NewsAPI.org API key")

    # --- Telegram ---
    telegram_bot_token: str = Field(default="", repr=False, description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID for notifications")

    # --- Trading Config ---
    paper_trading: bool = Field(default=True, description="Paper trading mode (default ON)")
    live_trading_acknowledgement: str = Field(
        default="",
        description="Required literal acknowledgement before live orders can be enabled.",
    )
    initial_capital: float = Field(default=25_000.0, description="Starting capital in ₹")
    max_capital_per_trade_pct: float = Field(default=0.10, description="Max 10% per trade")
    risk_per_trade_pct: float = Field(default=0.005, gt=0.0, le=0.02, description="Maximum planned loss per entry as a share of equity")
    max_total_exposure_pct: float = Field(default=0.50, gt=0.0, le=1.0, description="Maximum gross open exposure as a share of equity")
    max_portfolio_var_pct: float = Field(
        default=0.02, gt=0.0, le=0.02,
        description="Maximum projected one-day 95% historical portfolio VaR as a share of equity",
    )
    max_sector_exposure_pct: float = Field(
        default=0.25, gt=0.0, le=0.50,
        description="Maximum gross exposure to one sector as a share of equity",
    )
    sector_metadata_max_age_hours: float = Field(
        default=168.0, gt=0.0, le=168.0,
        description="Maximum age of a candidate sector classification used for entry risk",
    )
    max_daily_entries: int = Field(default=3, ge=1, description="Maximum new entries per trading day")
    max_daily_entries_per_symbol: int = Field(
        default=1, ge=1, description="Maximum new entries in one symbol per trading day"
    )
    symbol_reentry_cooldown_minutes: int = Field(
        default=30, ge=0, le=1440,
        description="Minimum time after a symbol exit before another entry",
    )
    min_net_reward_risk: float = Field(default=2.0, gt=0.0, description="Minimum reward/risk after estimated execution friction")
    max_cost_to_gross_reward_pct: float = Field(default=0.25, gt=0.0, le=1.0, description="Maximum estimated fees and slippage as a share of target profit")
    max_positions: int = Field(default=8, description="Max simultaneous open positions")
    max_daily_loss_pct: float = Field(default=0.02, gt=0.0, le=0.02, description="Block new entries at a 2% daily equity loss")
    max_open_stop_risk_pct: float = Field(
        default=0.02, gt=0.0, le=0.02,
        description="Maximum estimated aggregate open-stop risk as a share of current equity",
    )
    max_portfolio_drawdown_pct: float = Field(
        default=0.10, gt=0.0, le=0.10,
        description="Persistently pause new entries after a 10% peak-to-trough equity drawdown",
    )
    trading_universe_mode: Literal["nifty50", "custom"] = Field(
        default="nifty50", description="Use a dated Nifty 50 list or the configured custom watchlist"
    )
    scan_entry_deadline_seconds: int = Field(
        default=900, ge=120, le=1800,
        description="Entries are blocked once a scan has run longer than this; exits are never affected",
    )
    paper_partial_fill_enabled: bool = Field(
        default=False, description="Model partial paper fills from top-of-book depth when depth is known",
    )
    btst_delivery_minutes_after_open: int = Field(
        default=15, ge=0, le=120,
        description="Paper model: BTST shares are delivered this many minutes after the next session opens",
    )
    foreign_us_paper_enabled: bool = Field(
        default=False,
        description="Allow the Alpaca PAPER adapter for US orders (never a live endpoint; off by default)",
    )
    crypto_sim_enabled: bool = Field(
        default=False, description="Enable the local crypto spot simulator (no real exchange; off by default)",
    )
    crypto_usdt_inr_rate: float = Field(
        default=88.0, gt=0, le=1000,
        description="Static USDT→INR rate for reporting only; labelled as configured, not a market quote",
    )
    crypto_halt_daily_loss_pct: float = Field(
        default=0.05, gt=0, le=0.20, description="Daily loss at which the crypto simulator stops entering",
    )
    crypto_sim_starting_usdt: float = Field(
        default=500.0, gt=0, le=100_000,
        description="Starting paper balance for the independent crypto spot simulator",
    )
    crypto_sim_watchlist: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        description="Spot pairs supported by the local crypto simulator",
    )
    foreign_usd_inr_rate: float = Field(
        default=90.0, gt=0, le=1000,
        description="Static USD/INR conversion used only for foreign portfolio risk caps",
    )
    signal_gate_enabled: bool = Field(
        default=False,
        description="Consult the calibrated signal gate and half-Kelly reducer for entries (off by default)",
    )
    # --- Strategy sleeves: every sleeve ships disabled and needs its own explicit opt-in ---
    strategy_orb_intraday_enabled: bool = Field(default=False)
    strategy_supertrend_adx_enabled: bool = Field(default=False)
    strategy_mean_reversion_enabled: bool = Field(default=False)
    strategy_momentum_breakout_enabled: bool = Field(default=False)
    strategy_vcp_nr7_swing_enabled: bool = Field(default=False)
    strategy_btst_strength_enabled: bool = Field(default=False)
    strategy_etf_rotation_enabled: bool = Field(default=False)
    etf_rotation_universe: list[str] = Field(
        default=["NIFTYBEES", "BANKBEES", "GOLDBEES"],
        description="ETF symbols the rotation sleeve ranks (they must also be in the watchlist for data)",
    )
    swing_max_holding_days: int = Field(
        default=10, ge=1, le=60, description="Calendar days after which a SWING position is exited",
    )
    positional_max_holding_days: int = Field(
        default=60, ge=1, le=250, description="Calendar days after which a POSITIONAL position is exited",
    )
    order_rate_per_second: int = Field(default=5, ge=1, le=10, description="Maximum order submissions per second")
    order_rate_per_minute: int = Field(default=100, ge=1, le=200, description="Maximum order submissions per minute")
    protective_exit_max_reprice_steps: int = Field(
        default=3, ge=1, le=3,
        description="Maximum collar multiple (steps of protective_exit_collar_pct) for an unfilled protective exit",
    )

    # --- Risk Hard Limits ---
    min_adx_for_trade: float = Field(default=20.0)
    no_trade_window_open_minutes: int = Field(default=15)
    no_trade_window_close_minutes: int = Field(default=15)
    max_nifty_vix: float = Field(default=22.0)
    vix_halt_threshold: float = Field(default=28.0)
    stop_loss_atr_multiplier: float = Field(default=1.5)
    trailing_stop_atr_multiplier: float = Field(default=2.0)
    protective_exit_collar_pct: float = Field(
        default=0.003, gt=0.0, le=0.02,
        description="Maximum adverse price move from a fresh reference for protective limit exits",
    )
    technical_stop_daily_atr_multiplier: float = Field(default=1.0, gt=0.0, description="Entry stop distance in daily ATR(14) units")
    technical_target_daily_atr_multiplier: float = Field(default=2.5, gt=0.0, description="Entry target distance in daily ATR(14) units")
    screener_min_relative_volume: float = Field(default=1.0, ge=0.0, description="Session-to-date volume vs prior sessions at the same time")
    screener_min_average_daily_turnover_inr: float = Field(
        default=100_000_000.0, ge=0.0, description="Minimum mean turnover across complete prior sessions"
    )
    screener_min_liquidity_sessions: int = Field(default=10, ge=1)
    screener_max_spread_pct: float = Field(default=0.003, gt=0.0, le=0.05)
    screener_max_quote_age_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    max_order_participation_pct: float = Field(default=0.01, gt=0.0, le=0.10)
    instrument_cache_max_age_hours: float = Field(
        default=168.0, gt=0.0, le=168.0,
        description="Maximum age of a validated cached broker instrument master",
    )
    screener_breakout_min_relative_volume: float = Field(
        default=1.5, ge=0.0, description="Minimum same-time-of-day relative volume for a confirmed breakout"
    )
    screener_min_atr_pct: float = Field(default=0.002, ge=0.0, description="Minimum 15-minute ATR as a share of price")
    screener_vwap_chop_bars: int = Field(default=8, ge=2)
    screener_vwap_chop_band_pct: float = Field(default=0.005, gt=0.0, le=0.02)
    screener_breakout_lookback_bars: int = Field(default=20, ge=2)

    # --- System ---
    database_url: str = Field(default="sqlite:///./india_trader.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    dashboard_port: int = Field(default=8000)

    # --- LLM Models ---
    llm_decision_model: str = Field(default="claude-opus-4-6", description="Primary decision model")
    llm_fast_model: str = Field(default="claude-haiku-4-5-20251001", description="Fast signal model")
    deepseek_api_key: str = Field(default="", repr=False)
    economy_fast_model: str = Field(default="deepseek-flash", min_length=1)
    economy_decision_model: str = Field(default="deepseek-v4-pro", min_length=1)
    economy_decision_max_tokens: int = Field(default=8192, ge=1024, le=32768)
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=180)
    llm_usd_inr_rate: float = Field(default=90.0, gt=0)
    llm_daily_budget_inr: float = Field(
        default=50.0, gt=0, le=5000, description="Daily LLM spend (₹) above which paid scans are skipped",
    )

    # --- Market Constants ---
    risk_free_rate: float = Field(default=0.065, description="India 10Y bond yield ~6.5%")
    brokerage_per_order: float = Field(default=20.0, description="Maximum Angel One equity brokerage per order")
    brokerage_rate_pct: float = Field(default=0.001, description="Angel One equity brokerage rate (0.1%)")
    brokerage_min_per_order: float = Field(default=5.0, description="Minimum Angel One equity brokerage per order")
    stt_delivery_pct: float = Field(default=0.001, description="STT 0.1% on delivery")
    stt_intraday_pct: float = Field(default=0.00025, description="STT 0.025% on intraday")
    exchange_charges_pct: float = Field(default=0.0000297, description="NSE equity transaction charge: ₹2.97 per lakh")
    sebi_turnover_fee_pct: float = Field(default=0.000001)
    gst_on_brokerage_pct: float = Field(default=0.18)
    stamp_duty_pct: float = Field(default=0.00015, description="Equity delivery stamp duty: 0.015% on buy")
    stamp_duty_intraday_pct: float = Field(default=0.00003, description="Equity intraday stamp duty: 0.003% on buy")
    slippage_pct: float = Field(default=0.0005, description="0.05% conservative slippage")

    # --- Instrument Master ---
    instrument_master_url: str = Field(
        default="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    )
    instrument_cache_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "cache" / "instrument_master.json")
    )

    # --- WebSocket ---
    ws_max_tokens: int = Field(default=1000, description="Max tokens per WS session")
    ws_max_connections: int = Field(default=3, description="Max concurrent WS connections")

    # --- Watchlist Defaults ---
    default_watchlist: list[str] = Field(
        default=[
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
            "SUNPHARMA", "TMPV", "WIPRO", "HCLTECH", "ADANIENT",
        ],
    )

    # --- Index Tokens (Angel One) ---
    nifty50_token: str = Field(default="99926000")
    india_vix_token: str = Field(default="99926017", description="Angel One India VIX index token")
    banknifty_token: str = Field(default="99926009")
    nifty_midcap_token: str = Field(default="99926037")

    # --- NSE equity trading holidays ---
    # The verified yearly lists live in data/calendars/nse_equity_holidays.json and are read
    # through data.nse_calendar. This field is an extra override merged with that file
    # (comma-separated ISO dates in .env); it cannot remove a holiday.
    nse_holidays: list[str] = Field(default=[])

    # ────────────── v2.0 additions ──────────────────────

    # --- Interactive Brokers ---
    ibkr_host: str = Field(default="127.0.0.1", description="TWS/Gateway host")
    ibkr_port: int = Field(default=7497, description="7497=TWS paper, 7496=TWS live")
    ibkr_client_id: int = Field(default=1)
    ibkr_account: str = Field(default="", description="IBKR account ID")

    # --- Alpaca ---
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="Use Alpaca paper trading")

    # --- Setu Banking ---
    setu_api_key: str = Field(default="", description="Setu UPI API key")
    setu_base_url: str = Field(default="https://sandbox.setu.co")

    # --- Kafka ---
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_group_id: str = Field(default="india-ai-trader")
    kafka_events_topic: str = Field(default="market-events")
    kafka_enabled: bool = Field(default=False, description="Enable Kafka consumer")

    # --- FEMA / LRS ---
    lrs_annual_limit_usd: float = Field(default=250_000.0, description="RBI LRS annual limit")
    tcs_threshold_inr: float = Field(default=700_000.0, description="TCS 20% threshold ₹7L")

    # --- HMM Regime ---
    hmm_model_path: str = Field(default="", description="Path to the fitted HMM model JSON (default models/hmm_regime_classifier.json)")
    hmm_n_states: int = Field(default=4)
    hmm_retrain_days: int = Field(default=90, description="Retrain HMM every N days")

    # --- Vector DB / ChromaDB ---
    chromadb_path: str = Field(default="", description="ChromaDB persistent path (default: vector_db/trade_memory)")

    # --- Alpha Miner ---
    alpha_miner_enabled: bool = Field(default=True)
    alpha_miner_max_hypotheses: int = Field(default=3, description="Hypotheses per overnight run")
    alpha_miner_max_generations: int = Field(default=5, description="Max mutation rounds")

    # --- Global Watchlist (US/EU instruments) ---
    global_watchlist: list[str] = Field(
        default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        description="US/global symbols for IBKR/Alpaca",
    )

    # --- Bull/Bear Debate ---
    debate_conviction_threshold: int = Field(default=70, description="Min conviction to act (0-100)")

    # --- Order Routing ---
    large_order_threshold_inr: int = Field(default=500_000, description="₹5L triggers VWAP/TWAP")
    large_order_threshold_usd: int = Field(default=10_000, description="$10K triggers VWAP/TWAP")

    @field_validator("global_watchlist", mode="before")
    @classmethod
    def parse_global_watchlist(cls, value: Any) -> Any:
        """Accept JSON arrays and the comma-separated .env form documented for users."""
        if isinstance(value, str):
            return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
        return value

    @field_validator("etf_rotation_universe", mode="before")
    @classmethod
    def parse_etf_universe(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
        return value

    @field_validator("crypto_sim_watchlist", mode="before")
    @classmethod
    def parse_crypto_watchlist(cls, value: Any) -> Any:
        if isinstance(value, str):
            symbols = [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
            if not set(symbols) <= {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
                raise ValueError("CRYPTO_SIM_WATCHLIST supports BTCUSDT, ETHUSDT and SOLUSDT")
            return symbols
        if isinstance(value, list) and not set(value) <= {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            raise ValueError("CRYPTO_SIM_WATCHLIST supports BTCUSDT, ETHUSDT and SOLUSDT")
        return value

    @field_validator("nse_holidays", mode="before")
    @classmethod
    def parse_nse_holidays(cls, value: Any) -> Any:
        """Accept comma-separated ISO dates from .env; reject anything that is not a date."""
        from datetime import date

        if isinstance(value, str):
            value = [day.strip() for day in value.split(",") if day.strip()]
        if isinstance(value, list):
            for day in value:
                date.fromisoformat(str(day))
        return value

    @field_validator("default_watchlist", mode="before")
    @classmethod
    def canonicalise_default_watchlist(cls, value: Any) -> Any:
        """Map exchange-confirmed renames so orders never use a retired NSE symbol."""
        from config.symbol_changes import canonical_watchlist

        if isinstance(value, str):
            value = [symbol for symbol in value.split(",") if symbol.strip()]
        return canonical_watchlist(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_explicit_live_trading_acknowledgement(self) -> "Settings":
        """Fail closed: changing PAPER_TRADING alone can never enable live orders."""
        if not self.paper_trading and self.live_trading_acknowledgement != "I_ACCEPT_REAL_MONEY_RISK":
            raise ValueError(
                "Live trading requires LIVE_TRADING_ACKNOWLEDGEMENT=I_ACCEPT_REAL_MONEY_RISK. "
                "Keep PAPER_TRADING=true until paper results and broker controls are reviewed."
            )
        return self


# Singleton
settings = Settings()
