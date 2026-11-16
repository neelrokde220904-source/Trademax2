"""Tests for risk engine and position sizer."""

import pytest

from risk.risk_engine import RiskEngine
from risk.position_sizer import PositionSizer


def test_risk_engine_fno_ban():
    """Should block trades for F&O banned symbols."""
    engine = RiskEngine()
    engine.update_fno_ban(["INFY", "TCS"])

    allowed, reason = engine._check_fno_ban({"symbol": "INFY"})
    assert allowed is False
    assert "F&O ban" in reason

    allowed, _ = engine._check_fno_ban({"symbol": "RELIANCE"})
    assert allowed is True


def test_risk_engine_vix_halt():
    """Should block trades when VIX is above halt threshold."""
    engine = RiskEngine()
    engine.update_vix(30.0)

    allowed, reason = engine._check_vix({})
    assert allowed is False
    assert "VIX" in reason

    engine.update_vix(15.0)
    allowed, _ = engine._check_vix({})
    assert allowed is True


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_missing_or_invalid_vix_blocks_entries(value):
    engine = RiskEngine()
    engine.update_vix(value)
    assert engine._check_vix({})[0] is False


def test_position_sizer_kelly():
    """Kelly calculation should return a sensible fraction."""
    sizer = PositionSizer()

    # 55% win rate, 1.5:1 ratio
    kelly = sizer._kelly(0.55, 1.5)
    assert 0 < kelly < 0.25

    # 30% win rate should give 0 or very small
    kelly_low = sizer._kelly(0.30, 1.0)
    assert kelly_low == 0.0


def test_position_sizer_var():
    """VaR calculation should return a positive number."""
    sizer = PositionSizer()
    import numpy as np

    np.random.seed(42)
    returns = list(np.random.randn(60) * 0.01)
    var = sizer._calculate_var(returns, 100_000)
    assert var > 0


def test_position_sizer_uses_stop_risk_cash_and_total_exposure():
    sizer = PositionSizer()
    signal = {
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 98, "target": 118, "holding_period": "INTRADAY",
    }

    quantity, reason = sizer.size(
        signal, portfolio_value=25_000, available_capital=25_000,
        current_exposure=0,
    )
    assert reason is None
    assert quantity == 24
    assert sizer.validate_quantity(
        signal, quantity, portfolio_value=25_000,
        available_capital=25_000, current_exposure=0,
    ) is None

    exposure_limited, reason = sizer.size(
        signal, portfolio_value=25_000, available_capital=25_000,
        current_exposure=12_000,
    )
    assert reason is None
    assert exposure_limited == 4

    symbol_limited, reason = sizer.size(
        signal, portfolio_value=25_000, available_capital=25_000,
        current_exposure=2_000, current_symbol_exposure=2_000,
    )
    assert reason is None
    assert symbol_limited == 4


def test_position_sizer_rejects_insufficient_net_reward_to_risk():
    sizer = PositionSizer()
    quantity, reason = sizer.size(
        {
            "symbol": "TEST", "action": "BUY", "price": 100,
            "stop_loss": 98, "target": 104, "holding_period": "INTRADAY",
        },
        portfolio_value=25_000, available_capital=25_000,
        current_exposure=0,
    )
    assert quantity == 0
    assert reason == "Net reward-to-risk is below the configured minimum"


def test_position_sizer_respects_tighter_regime_notional_cap():
    sizer = PositionSizer()
    signal = {
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 98, "target": 118, "holding_period": "INTRADAY",
    }
    kwargs = dict(
        portfolio_value=25_000, available_capital=25_000,
        current_exposure=0, max_position_pct=0.04,
    )
    quantity, reason = sizer.size(signal, **kwargs)
    assert reason is None
    assert quantity == 9
    assert sizer.validate_quantity(signal, quantity, **kwargs) is None
    assert "exceeds" in sizer.validate_quantity(signal, quantity + 1, **kwargs)


def test_risk_manager_requires_deterministic_technical_trigger(monkeypatch):
    from agents.base_agent import create_initial_state
    from agents.risk_manager_agent import RiskManagerAgent
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    state = create_initial_state(["TEST"])
    state["india_vix"] = 15
    state["portfolio_snapshot_valid"] = True
    state["hmm_regime"] = {
        "degraded": False,
        "action_directive": {"allow_new_positions": True, "max_position_size_pct": 0.1},
    }
    state["technical_signals"] = {"TEST": {"signal": "HOLD"}}
    state["sentiment_signals"] = {"TEST": {"signal": "BUY", "confidence": 95}}

    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert assessment["rejection_reason"] == "No deterministic technical entry signal"


def test_risk_manager_uses_risk_budget_instead_of_fixed_kelly(monkeypatch):
    from agents.base_agent import create_initial_state
    from agents.risk_manager_agent import RiskManagerAgent
    from broker.portfolio_tracker import portfolio_tracker
    from tests.test_daily_portfolio_risk import AS_OF, history_for

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    state = create_initial_state(["TEST"])
    state["india_vix"] = 15
    state["portfolio_value"] = 25_000
    state["available_capital"] = 25_000
    state["research_mode"] = True
    state["research_as_of"] = AS_OF
    state["market_data"] = {"TEST": history_for()}
    state["technical_signals"] = {
        "TEST": {
            "signal": "BUY", "confidence": 90, "stop_loss": 98, "target": 118,
            "indicators": {"close": 100, "atr_14": 2},
        },
    }

    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is True
    assert assessment["max_quantity"] == 24
    assert assessment["deterministic_action"] == "BUY"
