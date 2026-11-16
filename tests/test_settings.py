"""Tests for config/settings.py."""

from config.settings import Settings, settings


def test_settings_defaults():
    """Settings should load with sensible defaults."""
    defaults = Settings(_env_file=None)
    assert settings.paper_trading is True
    assert defaults.initial_capital == 25_000.0
    assert defaults.max_positions == 8
    assert defaults.max_daily_loss_pct == 0.02
    assert defaults.max_capital_per_trade_pct == 0.10
    assert defaults.risk_per_trade_pct == 0.005
    assert defaults.max_total_exposure_pct == 0.50
    assert defaults.max_portfolio_var_pct == 0.02
    assert defaults.max_sector_exposure_pct == 0.25
    assert defaults.sector_metadata_max_age_hours == 168.0
    assert defaults.max_daily_entries == 3
    assert defaults.max_daily_entries_per_symbol == 1
    assert defaults.symbol_reentry_cooldown_minutes == 30
    assert defaults.min_net_reward_risk == 2.0
    assert defaults.max_cost_to_gross_reward_pct == 0.25


def test_settings_market_constants():
    assert settings.brokerage_per_order == 20.0
    assert settings.brokerage_min_per_order == 5.0
    assert settings.brokerage_rate_pct == 0.001
    assert settings.risk_free_rate == 0.065
    assert settings.slippage_pct == 0.0005


def test_llm_profiles_preserve_claude_defaults():
    config = Settings(_env_file=None)
    assert config.llm_fast_model == "claude-haiku-4-5-20251001"
    assert config.llm_decision_model == "claude-opus-4-6"
    assert config.economy_fast_model == "deepseek-flash"
    assert config.economy_decision_model == "deepseek-v4-pro"
    assert config.paper_trading is True


def test_settings_watchlist():
    assert len(settings.default_watchlist) == 20
    assert "RELIANCE" in settings.default_watchlist
    assert "TCS" in settings.default_watchlist


def test_nse_holidays_load_from_the_calendar_file():
    from data.nse_calendar import nse_holiday_set

    holidays = nse_holiday_set()
    assert len(holidays) > 10
    assert "2026-01-26" in holidays  # Republic Day
