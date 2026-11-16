"""Daily return provenance and directional portfolio risk."""

from datetime import date

import numpy as np
import pandas as pd
import pytest


AS_OF = "2026-10-19T10:00:00+05:30"


def history_for(returns=None):
    days = pd.bdate_range("2026-09-17", "2026-10-16")
    days = days[days != pd.Timestamp("2026-10-02")][-21:]
    changes = returns if returns is not None else [(-0.002 if i % 2 else 0.003) for i in range(20)]
    prices = [100.0]
    for change in changes:
        prices.append(prices[-1] * (1 + change))
    rows = []
    for day, close in zip(days, prices, strict=True):
        for stamp in pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=25, freq="15min"):
            rows.append((stamp, close, close, close, close, 1000))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def test_daily_returns_use_session_closes_not_intraday_bars():
    from risk.daily_portfolio_risk import daily_returns, historical_var

    changes = [0.001] * 19 + [-0.08]
    result = daily_returns(history_for(changes), AS_OF)
    assert len(result) == 20
    assert result.iloc[-1] == pytest.approx(-0.08)
    assert historical_var(result, "BUY") == pytest.approx(0.08 * 0.05 - 0.001 * 0.95)
    assert historical_var(result, "SELL") == pytest.approx(0.001)


@pytest.mark.parametrize("damage", ["missing_bar", "missing_session", "invalid_price", "conflicting_duplicate"])
def test_daily_returns_reject_incomplete_or_corrupt_history(damage):
    from risk.daily_portfolio_risk import RiskHistoryUnavailable, daily_returns

    history = history_for()
    if damage == "missing_bar":
        history = history.drop(index=100)
    elif damage == "missing_session":
        day = history.iloc[100]["datetime"].date()
        history = history[history["datetime"].dt.date != day]
    elif damage == "invalid_price":
        history.loc[100, "close"] = float("nan")
    else:
        extra = history.iloc[[100]].copy()
        extra["close"] += 1
        history = pd.concat([history, extra], ignore_index=True)
    with pytest.raises(RiskHistoryUnavailable):
        daily_returns(history, AS_OF)


def test_daily_returns_fail_closed_for_unverified_calendar_and_missing_special_session():
    from risk.daily_portfolio_risk import RiskHistoryUnavailable, daily_returns

    with pytest.raises(RiskHistoryUnavailable, match="calendar"):
        daily_returns(history_for(), "2027-01-20T10:00:00+05:30")
    with pytest.raises(RiskHistoryUnavailable, match="missing or invalid"):
        daily_returns(history_for(), "2026-11-16T10:00:00+05:30")


def test_muhurat_session_is_in_daily_returns_and_atr():
    from data.indicators import daily_atr
    from risk.daily_portfolio_risk import RiskHistoryUnavailable, daily_returns

    days = pd.bdate_range("2026-10-01", "2026-11-13")
    days = [day for day in days if day.date() not in {
        date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 10),
    }]
    days = sorted([*days, pd.Timestamp("2026-11-08")])[-21:]
    rows = []
    for index, day in enumerate(days):
        is_muhurat = day.date() == date(2026, 11, 8)
        start = pd.Timedelta(hours=14, minutes=20) if is_muhurat else pd.Timedelta(hours=9, minutes=15)
        for stamp in pd.date_range(day + start, periods=5 if is_muhurat else 25, freq="15min"):
            price = 100.0 + index * 0.1
            rows.append((stamp, price, price + 1, price - 1, price, 1000))
    bars = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    as_of = "2026-11-16T10:00:00+05:30"
    returns = daily_returns(bars, as_of)
    assert date(2026, 11, 8) in returns.index
    assert len(returns) == 20
    assert daily_atr(bars, as_of) is not None
    missing_auction = bars[bars["datetime"] != pd.Timestamp("2026-11-08 15:20")]
    with pytest.raises(RiskHistoryUnavailable, match="2026-11-08"):
        daily_returns(missing_auction, as_of)
    assert daily_atr(missing_auction, as_of) is None


def test_2026_budget_sunday_is_an_expected_regular_session():
    from risk.daily_portfolio_risk import _session_dates

    dates = _session_dates("2026-02-02T10:00:00+05:30")
    assert dates[-1] == date(2026, 2, 1)
    assert date(2026, 1, 15) not in dates
    assert date(2026, 1, 26) not in dates


def test_directional_correlation_aligns_by_session_and_requires_overlap():
    from risk.daily_portfolio_risk import RiskHistoryUnavailable, adverse_correlation

    index = pd.Index([date(2026, 9, 1) + pd.Timedelta(days=i) for i in range(20)])
    left = pd.Series(np.linspace(-0.01, 0.01, 20), index=index)
    right = left.copy()
    assert adverse_correlation(left, "BUY", right, 1) == pytest.approx(1.0)
    assert adverse_correlation(left, "BUY", right, -1) == pytest.approx(-1.0)
    assert adverse_correlation(left, "SELL", right, -1) == pytest.approx(1.0)
    with pytest.raises(RiskHistoryUnavailable, match="aligned"):
        adverse_correlation(left, "BUY", right.iloc[1:], 1)
    with pytest.raises(RiskHistoryUnavailable, match="variance"):
        adverse_correlation(left, "BUY", pd.Series(0.0, index=index), 1)


def _risk_state():
    from agents.base_agent import create_initial_state

    state = create_initial_state(["TEST"])
    state.update({
        "research_mode": True, "research_as_of": AS_OF,
        "india_vix": 15, "portfolio_value": 25_000, "available_capital": 25_000,
        "technical_signals": {"TEST": {
            "signal": "BUY", "stop_loss": 98, "target": 118,
            "indicators": {"close": 100},
        }},
        "market_data": {"TEST": history_for()},
    })
    return state


def test_risk_manager_rejects_missing_candidate_and_held_history():
    from agents.risk_manager_agent import RiskManagerAgent

    state = _risk_state()
    state["market_data"] = {}
    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "history unavailable" in assessment["rejection_reason"].lower()
    state = _risk_state()
    state["current_positions"] = {"HELD": {"quantity": 1, "ltp": 100}}
    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "history unavailable" in assessment["rejection_reason"].lower()


def test_risk_manager_assesses_candidates_in_stable_symbol_order():
    from agents.risk_manager_agent import RiskManagerAgent

    state = _risk_state()
    state["watchlist"] = ["ZZZ", "AAA"]
    state["technical_signals"] = {
        symbol: {
            "signal": "BUY", "stop_loss": 98, "target": 118,
            "indicators": {"close": 100},
        }
        for symbol in ("ZZZ", "AAA")
    }
    state["market_data"] = {"ZZZ": history_for(), "AAA": history_for()}
    assessment = RiskManagerAgent().run(state)["risk_assessment"]
    assert list(assessment) == ["blocked", "AAA", "ZZZ"]


def test_risk_manager_applies_hmm_position_cap(monkeypatch):
    from agents import risk_manager_agent as module
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "has_hit_portfolio_drawdown_limit", lambda: False)
    monkeypatch.setattr(module, "daily_returns", lambda *_args: pd.Series(np.linspace(-0.01, 0.01, 20)))
    state = _risk_state()
    state.update({
        "research_mode": False, "portfolio_snapshot_valid": True,
        "hmm_regime": {"degraded": False, "action_directive": {
            "allow_new_positions": True, "max_position_size_pct": 0.04,
            "allowed_strategies": ["trend_following"], "hedge_required": False,
        }},
    })
    state["technical_signals"]["TEST"]["strategy"] = "CompositeTrend"
    assessment = module.RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "sector classification unavailable" in assessment["rejection_reason"].lower()

    state["research_fundamental_inputs"] = {"TEST": {
            "sector": "Financial Services",
            "sector_source": "yfinance.Ticker.info",
            "sector_observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }}
    state["technical_signals"]["TEST"]["strategy"] = "CompositeTrend"
    assessment = module.RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is True
    assert assessment["max_quantity"] == 9
    state["hmm_regime"]["action_directive"]["allowed_strategies"] = ["mean_reversion"]
    assessment = module.RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "does not permit" in assessment["rejection_reason"]
    state["hmm_regime"]["action_directive"]["allowed_strategies"] = ["trend_following"]
    state["hmm_regime"]["action_directive"]["hedge_required"] = True
    assessment = module.RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "requires a hedge" in assessment["rejection_reason"]


def test_risk_manager_blocks_ambiguous_protection_before_approval(monkeypatch):
    from agents import risk_manager_agent as module
    from broker.portfolio_tracker import portfolio_tracker
    from risk.stop_loss_manager import stop_loss_manager

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    monkeypatch.setattr(portfolio_tracker, "has_hit_portfolio_drawdown_limit", lambda: False)
    monkeypatch.setattr(stop_loss_manager, "snapshot", lambda: {})
    state = _risk_state()
    state.update({
        "research_mode": False, "portfolio_snapshot_valid": True,
        "current_positions": {"HELD": {
            "quantity": 1, "ltp": 100, "avg_price": 100, "stop_loss": 95,
            "entry_trade_ids": [1], "holding_period": "INTRADAY",
        }},
        "hmm_regime": {"degraded": False, "action_directive": {
            "allow_new_positions": True, "max_position_size_pct": 0.1,
            "allowed_strategies": ["trend_following"], "hedge_required": False,
        }},
    })
    assessment = module.RiskManagerAgent().run(state)["risk_assessment"]
    assert assessment["blocked"] is True
    assert "active stop is missing" in assessment["reason"]


def test_risk_manager_blocks_only_adverse_directional_correlation():
    from agents.risk_manager_agent import RiskManagerAgent

    state = _risk_state()
    state["market_data"]["HELD"] = history_for()
    state["current_positions"] = {"HELD": {"quantity": 1, "ltp": 100}}
    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is False
    assert "Adversely correlated" in assessment["rejection_reason"]
    state["current_positions"]["HELD"]["quantity"] = -1
    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment["approved"] is True
