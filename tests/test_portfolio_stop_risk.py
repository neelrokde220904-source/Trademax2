"""Portfolio stop risk is measured from current marks and fails closed."""

import pytest

from backtester.costs import calculate_indian_transaction_cost


def _position(quantity=10, mark=100.0, stop=95.0, ids=None):
    return {
        "quantity": quantity, "ltp": mark, "avg_price": mark, "stop_loss": stop,
        "holding_period": "INTRADAY", "entry_trade_ids": [1] if ids is None else ids,
    }


def _protection(position):
    return {
        "action": "BUY" if position["quantity"] > 0 else "SELL",
        "stop_loss": position["stop_loss"],
        "holding_period": position["holding_period"],
        "entry_trade_ids": position["entry_trade_ids"],
        "exit_pending_order_id": None,
    }


def test_open_stop_risk_uses_fresh_mark_exit_slippage_and_exit_cost(monkeypatch):
    from config.settings import settings
    from risk.portfolio_stop_risk import open_stop_risk

    monkeypatch.setattr(settings, "slippage_pct", 0.001)
    positions = {
        "LONG": _position(),
        "SHORT": _position(quantity=-4, mark=200, stop=210, ids=[2]),
    }
    long_exit = 95 * 0.999
    short_exit = 210 * 1.001
    expected = (
        (100 - long_exit) * 10
        + calculate_indian_transaction_cost(long_exit, 10, is_buy=False, is_intraday=True)
        + (short_exit - 200) * 4
        + calculate_indian_transaction_cost(short_exit, 4, is_buy=True, is_intraday=True)
    )
    protection = {symbol: _protection(pos) for symbol, pos in positions.items()}
    assert open_stop_risk(positions, protection) == pytest.approx(expected)
    assert open_stop_risk({}, {}) == 0


@pytest.mark.parametrize("position", [
    _position(stop=0),
    _position(stop=float("nan")),
    _position(mark=float("nan")),
    _position(quantity=True),
    _position(quantity=1.5),
    _position(ids=[]),
    _position(ids=[1, 2]),
    _position(mark=95, stop=95),
    _position(quantity=-1, mark=105, stop=100),
    {**_position(), "holding_period": "UNKNOWN"},
])
def test_open_stop_risk_rejects_missing_ambiguous_or_breached_protection(position):
    from risk.portfolio_stop_risk import StopRiskUnavailable, open_stop_risk

    with pytest.raises(StopRiskUnavailable):
        open_stop_risk({"TEST": position}, {"TEST": _protection(position)})


@pytest.mark.parametrize("change", [
    {"entry_trade_ids": [9]}, {"action": "SELL"}, {"stop_loss": 94},
    {"stop_loss": 100}, {"exit_pending_order_id": "PENDING"},
    {"holding_period": "DELIVERY"},
])
def test_open_stop_risk_rejects_mismatched_or_pending_live_protection(change):
    from risk.portfolio_stop_risk import StopRiskUnavailable, open_stop_risk

    position = _position()
    with pytest.raises(StopRiskUnavailable):
        open_stop_risk({"TEST": position}, {"TEST": {**_protection(position), **change}})


def test_open_stop_risk_uses_ratchet_without_widening():
    from risk.portfolio_stop_risk import StopRiskUnavailable, open_stop_risk

    position = _position()
    original = _protection(position)
    ratchet = {**original, "stop_loss": 98}
    assert 0 < open_stop_risk({"TEST": position}, {"TEST": ratchet}) < open_stop_risk(
        {"TEST": position}, {"TEST": original}
    )
    with pytest.raises(StopRiskUnavailable):
        open_stop_risk({"TEST": position}, {})


def test_open_stop_risk_rejects_stop_crossed_since_original_entry():
    from risk.portfolio_stop_risk import StopRiskUnavailable, open_stop_risk

    position = {**_position(), "ltp": 94, "avg_price": 100}
    with pytest.raises(StopRiskUnavailable, match="already breached"):
        open_stop_risk({"TEST": position}, {"TEST": _protection(position)})


def test_position_sizer_combines_entry_and_portfolio_stop_budgets():
    from risk.position_sizer import PositionSizer
    from risk.portfolio_stop_risk import entry_stop_risk

    signal = {
        "symbol": "TEST", "action": "BUY", "price": 100,
        "stop_loss": 98, "target": 118, "holding_period": "INTRADAY",
    }
    sizer = PositionSizer()
    kwargs = {
        "portfolio_value": 25_000, "available_capital": 25_000,
        "current_exposure": 0, "remaining_stop_risk": 35.0,
    }
    quantity, reason = sizer.size(signal, **kwargs)
    assert reason is None and 0 < quantity < 24
    assert entry_stop_risk(signal, quantity) <= 35
    assert entry_stop_risk(signal, quantity + 1) > 35
    assert sizer.validate_quantity(signal, quantity, **kwargs) is None
    assert "exceeds" in sizer.validate_quantity(signal, quantity + 1, **kwargs)


@pytest.mark.parametrize("remaining", [-1, float("nan"), float("inf"), True, "100"])
def test_position_sizer_rejects_invalid_portfolio_stop_budget(remaining):
    from risk.position_sizer import PositionSizer

    signal = {"action": "BUY", "price": 100, "stop_loss": 98, "target": 118}
    quantity, reason = PositionSizer().size(
        signal, portfolio_value=25_000, available_capital=25_000,
        current_exposure=0, remaining_stop_risk=remaining,
    )
    assert quantity == 0
    assert "risk" in reason.lower()
