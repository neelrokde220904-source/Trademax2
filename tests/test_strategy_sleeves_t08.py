"""T08: strategy adapters, sleeve flags, candidate node, technical overlay, router gates."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from agents import strategy_candidates as node_module
from config.settings import Settings, settings
from strategies import adapters
from strategies.candidate import CandidateInvalid
from strategies.contracts import SLEEVE_CONTRACTS, STRATEGY_CONTRACTS, require_strategy_contract

IST = ZoneInfo("Asia/Kolkata")
FLAGS = [c.enabled_setting for c in SLEEVE_CONTRACTS]


def ist(year, month, day, hour=11, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def enable(monkeypatch, *flags):
    for flag in flags:
        monkeypatch.setattr(settings, flag, True)


def intraday_bars(sessions=12, minutes=15, start="2026-10-26", drift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    price = 100.0
    for day in pd.bdate_range(start, periods=sessions):
        index = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=375 // minutes,
                              freq=f"{minutes}min", tz=IST)
        steps = rng.normal(drift, 0.15, len(index))
        close = price + np.cumsum(steps)
        price = float(close[-1])
        frames.append(pd.DataFrame({
            "open": close - steps, "high": close + 0.3, "low": close - 0.3, "close": close,
            "volume": rng.integers(900, 1100, len(index)).astype(float),
        }, index=index))
    return pd.concat(frames)


def daily_bars(rows=260, drift=0.001, seed=5, start="2025-06-02"):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, rows))
    index = pd.bdate_range(start, periods=rows)
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
        "volume": np.full(rows, 1e6),
    }, index=index)


# ── shipped disabled ──────────────────────────────────────────────────────────

def test_every_sleeve_flag_defaults_to_false_and_nothing_is_enabled():
    for flag in FLAGS:
        assert Settings.model_fields[flag].default is False, flag
        assert getattr(settings, flag) is False
    assert len(FLAGS) == 7 and len(set(FLAGS)) == 7
    assert adapters.enabled_adapters() == []
    assert {c.name for c in SLEEVE_CONTRACTS} <= set(STRATEGY_CONTRACTS)
    assert STRATEGY_CONTRACTS["CompositeTrend"].enabled_setting is None  # core stays always on


def test_a_disabled_sleeve_produces_nothing_and_cannot_enter():
    for adapter in adapters.ADAPTERS.values():
        with pytest.raises(ValueError, match="disabled"):
            adapter.generate("TCS", {}, ist(2026, 11, 13))
    with pytest.raises(ValueError):
        require_strategy_contract("ORBIntraday")


def test_the_candidate_node_is_a_no_op_with_every_flag_off():
    state = {"market_data": {"X": intraday_bars()}, "screener_results": {"qualified": ["X"]}}
    assert node_module.strategy_candidates_node(state) == {}


def test_enabling_one_flag_enables_only_that_sleeve(monkeypatch):
    enable(monkeypatch, "strategy_btst_strength_enabled")
    assert [a.key for a in adapters.enabled_adapters()] == ["btst_strength"]
    assert require_strategy_contract("BTSTStrength").horizons == ("BTST",)
    with pytest.raises(ValueError):
        require_strategy_contract("ORBIntraday")


def test_sleeve_contracts_are_ai_required_with_their_own_review():
    for contract in SLEEVE_CONTRACTS:
        assert contract.review is not None and contract.review.strategy == contract.name
        assert contract.review.required == STRATEGY_CONTRACTS["CompositeTrend"].review.required
        assert contract.book == contract.name and contract.asset_venues == ("NSE",)
    assert STRATEGY_CONTRACTS["ETFRotation"].segment == "ETF"
    assert STRATEGY_CONTRACTS["MeanReversion"].hmm_tag == "mean_reversion"


# ── completed-bar normalisation (no look-ahead) ───────────────────────────────

def test_only_completed_intraday_bars_are_used():
    bars = intraday_bars(sessions=3)
    as_of = ist(2026, 10, 28, 10, 7)  # bar starting 10:00 is still in progress
    completed = adapters.completed_intraday(bars, 15, as_of)
    assert completed.index[-1] == pd.Timestamp("2026-10-28 09:45", tz=IST)
    assert (completed.index + pd.Timedelta(minutes=15) <= as_of).all()
    assert adapters.completed_intraday(bars, 15, ist(2026, 10, 26, 9, 0)) is None


def test_only_finished_daily_sessions_are_used():
    bars = daily_bars(rows=30, start="2026-10-01")
    today = pd.Timestamp(bars.index[-1]).to_pydatetime()
    during = adapters.completed_daily(bars, datetime(today.year, today.month, today.day, 12, 0, tzinfo=IST))
    after = adapters.completed_daily(bars, datetime(today.year, today.month, today.day, 15, 30, tzinfo=IST))
    assert len(during) == len(bars) - 1 and len(after) == len(bars)


def test_daily_bars_derive_only_from_finished_sessions():
    bars = intraday_bars(sessions=5)
    as_of = ist(2026, 10, 30, 12, 0)  # the last session is still running
    daily = adapters.daily_from_intraday(bars, as_of)
    assert daily.index.max() < pd.Timestamp("2026-10-30")
    full = adapters.daily_from_intraday(bars, ist(2026, 10, 30, 15, 45))
    assert full.index.max() == pd.Timestamp("2026-10-30")


def test_strategies_never_receive_future_or_in_progress_bars(monkeypatch):
    enable(monkeypatch, "strategy_supertrend_adx_enabled")
    seen = {}

    class Spy:
        def _analyse(self, symbol, bars):
            seen["last_end"] = bars.index[-1] + pd.Timedelta(minutes=15)
            return None

    import strategies.supertrend_adx as module

    monkeypatch.setattr(module, "SupertrendADX", lambda: Spy())
    bars = intraday_bars(sessions=6)
    poisoned = bars.copy()
    poisoned.iloc[-40:, poisoned.columns.get_indexer(["close"])] += 1000
    as_of = ist(2026, 10, 30, 11, 3)
    assert adapters.ADAPTERS["supertrend_adx"].generate("TCS", {"15m": poisoned}, as_of) is None
    assert seen["last_end"] <= as_of


# ── candidate construction ───────────────────────────────────────────────────

def _stub(monkeypatch, adapter_key, signal):
    adapter = adapters.ADAPTERS[adapter_key]
    monkeypatch.setattr(adapter, "signal", lambda *_a, **_k: dict(signal))
    return adapter


def test_intraday_sleeve_candidate_uses_the_15m_policy_and_validates(monkeypatch):
    enable(monkeypatch, "strategy_supertrend_adx_enabled")
    as_of = ist(2026, 11, 3, 11, 0)
    bars = intraday_bars(sessions=12, start="2026-10-19")
    adapter = _stub(monkeypatch, "supertrend_adx", {
        "action": "BUY", "price": 100.03, "holding_period": "INTRADAY", "reasoning": "flip",
    })
    candidate = adapter.generate("TCS", {"15m": bars}, as_of)
    assert candidate.strategy == "SupertrendADX" and candidate.horizon == "INTRADAY"
    assert candidate.product == "INTRADAY" and candidate.exit_policy == "intraday_atr_2x_4x_v1"
    assert candidate.atr_timeframe == "INTRADAY_15M_14"
    for level in (candidate.entry_price, candidate.stop_loss, candidate.target):
        assert level % Decimal("0.05") == 0
    assert candidate.stop_loss < candidate.entry_price < candidate.target
    assert candidate.review_contract_version == "review-v2"
    # 2 ATR stop, 4 ATR target (moved at most one tick toward the entry)
    risk, reward = candidate.entry_price - candidate.stop_loss, candidate.target - candidate.entry_price
    assert abs(float(reward / risk) - 2.0) < 0.2


def test_swing_sleeves_use_daily_atr22_and_the_delivery_product(monkeypatch):
    enable(monkeypatch, "strategy_momentum_breakout_enabled", "strategy_vcp_nr7_swing_enabled",
           "strategy_mean_reversion_enabled")
    as_of = ist(2026, 11, 13, 16, 0)
    bars = {"1d": daily_bars(rows=260)}
    for key, name in (("momentum_breakout", "MomentumBreakout"), ("vcp_nr7_swing", "VCPNR7Swing"),
                      ("mean_reversion", "MeanReversion")):
        adapter = _stub(monkeypatch, key, {"action": "BUY", "price": 120.0, "holding_period": "SWING"})
        candidate = adapter.generate("INFY", bars, as_of)
        assert candidate.strategy == name and candidate.product == "DELIVERY"
        assert candidate.exit_policy == "daily_atr22_2x_4x_v1"
        assert candidate.atr_timeframe == "DAILY_22_COMPLETED_SESSIONS"


def test_mean_reversion_is_long_only(monkeypatch):
    enable(monkeypatch, "strategy_mean_reversion_enabled")
    import strategies.mean_reversion as module

    class Short:
        def _analyse(self, symbol, bars):
            return {"action": "SELL", "price": 100.0, "holding_period": "SWING"}

    monkeypatch.setattr(module, "MeanReversion", lambda: Short())
    assert adapters.ADAPTERS["mean_reversion"].generate(
        "INFY", {"1d": daily_bars()}, ist(2026, 11, 13, 16, 0),
    ) is None


def test_orb_uses_a_structure_stop_and_skips_when_the_structure_is_wrong(monkeypatch):
    enable(monkeypatch, "strategy_orb_intraday_enabled")
    as_of = ist(2026, 11, 3, 11, 0)
    bars = intraday_bars(sessions=12, start="2026-10-19")
    from risk.exit_policy import atr_before

    atr = atr_before(adapters.completed_intraday(bars, 15, as_of), 14, as_of, bar_minutes=15)
    signal = {"action": "BUY", "price": 100.0, "holding_period": "INTRADAY",
              "opening_range_high": 100.0 - 0.0, "opening_range_low": 100.0 - 2 * atr * 1.5}
    adapter = _stub(monkeypatch, "orb_intraday", signal)
    candidate = adapter.generate("TCS", {"15m": bars, "5m": None}, as_of)
    midpoint = 100.0 - atr * 1.5
    assert candidate is not None and float(candidate.stop_loss) <= midpoint  # beyond the structure
    assert candidate.exit_policy == "intraday_atr_2x_4x_v1"
    too_tight = {**signal, "opening_range_low": 99.9}
    _stub(monkeypatch, "orb_intraday", too_tight)
    assert adapter.generate("TCS", {"15m": bars}, as_of) is None  # structure inside 0.5 ATR: skipped


def test_not_enough_history_skips_instead_of_guessing(monkeypatch):
    enable(monkeypatch, "strategy_momentum_breakout_enabled")
    adapter = _stub(monkeypatch, "momentum_breakout", {"action": "BUY", "price": 120.0, "holding_period": "SWING"})
    assert adapter.generate("INFY", {"1d": daily_bars(rows=15)}, ist(2026, 11, 13, 16, 0)) is None
    assert adapter.generate("INFY", {"1d": None}, ist(2026, 11, 13, 16, 0)) is None
    with pytest.raises(CandidateInvalid):
        adapter.generate("INFY", {"1d": daily_bars()}, datetime(2026, 11, 13, 16, 0))  # naive as_of


# ── the two new sleeves on real fixtures ─────────────────────────────────────

def test_etf_rotation_holds_only_the_strongest_trending_etf(monkeypatch):
    enable(monkeypatch, "strategy_etf_rotation_enabled")
    as_of = ist(2026, 11, 13, 16, 0)
    rising = daily_bars(rows=300, drift=0.002, seed=1, start="2025-09-01")
    slower = daily_bars(rows=300, drift=0.0008, seed=2, start="2025-09-01")
    falling = daily_bars(rows=300, drift=-0.002, seed=3, start="2025-09-01")
    adapter = adapters.ADAPTERS["etf_rotation"]
    bars = lambda own, peers: {"1d": own, "1d_peers": peers}
    winner = adapter.generate("NIFTYBEES", bars(rising, {"BANKBEES": slower, "GOLDBEES": falling}), as_of)
    assert winner is not None and winner.asset.segment == "ETF" and winner.horizon == "POSITIONAL"
    assert winner.exit_policy == "chandelier_atr22_k3_v1"
    assert adapter.generate("BANKBEES", bars(slower, {"NIFTYBEES": rising, "GOLDBEES": falling}), as_of) is None
    assert adapter.generate("GOLDBEES", bars(falling, {"NIFTYBEES": rising}), as_of) is None  # cash default
    assert adapter.generate("X", bars(falling, {}), as_of) is None


def _late_session_bars(strong=True):
    bars = intraday_bars(sessions=30, start="2026-09-21", seed=11)
    today = bars.index[-1].date()
    session = bars[bars.index.date == today].copy()
    if strong:
        ramp = np.linspace(100, 103, len(session))
        session["close"], session["open"] = ramp, ramp - 0.05
        session["high"], session["low"] = ramp + 0.1, ramp - 0.1
        session.iloc[-1, session.columns.get_loc("close")] = session["high"].max()
        session["volume"] = 3000.0
    bars = bars[bars.index.date != today]
    return pd.concat([bars, session])


def test_btst_needs_late_session_strength_volume_and_the_entry_window(monkeypatch):
    enable(monkeypatch, "strategy_btst_strength_enabled")
    bars = _late_session_bars(strong=True)
    today = bars.index[-1].date()
    daily = adapters.daily_from_intraday(bars, ist(today.year, today.month, today.day, 15, 45))
    adapter = adapters.ADAPTERS["btst_strength"]
    as_of = ist(today.year, today.month, today.day, 15, 10)
    candidate = adapter.generate("TCS", {"15m": bars, "1d": daily.iloc[:-1]}, as_of)
    assert candidate is not None and candidate.horizon == "BTST" and candidate.product == "DELIVERY"
    assert candidate.exit_policy == "daily_atr_baseline_v1"
    # Before 15:00 or after 15:20 there is no BTST entry.
    assert adapter.generate("TCS", {"15m": bars, "1d": daily.iloc[:-1]}, ist(today.year, today.month, today.day, 14, 30)) is None
    assert adapter.generate("TCS", {"15m": bars, "1d": daily.iloc[:-1]}, ist(today.year, today.month, today.day, 15, 40)) is None
    weak = _late_session_bars(strong=False)
    assert adapter.generate("TCS", {"15m": weak, "1d": daily.iloc[:-1]}, as_of) is None


# ── node + selector ──────────────────────────────────────────────────────────

def _node_state(monkeypatch, vix=15.0, regime="LOW_VOL_TREND"):
    bars = intraday_bars(sessions=12, start="2026-10-19", seed=9)
    monkeypatch.setattr(node_module, "_market_indicators", lambda _s, _q: (30.0, 30.0))
    from broker.portfolio_tracker import portfolio_tracker

    monkeypatch.setattr(type(portfolio_tracker), "daily_loss_pct", property(lambda _self: 0.0))
    return {
        "market_data": {"TCS": bars, "INFY": bars}, "india_vix": vix,
        "screener_results": {"qualified": ["TCS", "INFY"]}, "hmm_regime": {"regime": regime},
    }


def test_node_collects_candidates_only_from_active_enabled_sleeves(monkeypatch):
    enable(monkeypatch, "strategy_supertrend_adx_enabled", "strategy_mean_reversion_enabled")
    state = _node_state(monkeypatch)
    adapter = _stub(monkeypatch, "supertrend_adx", {"action": "BUY", "price": 100.0, "holding_period": "INTRADAY"})
    _stub(monkeypatch, "mean_reversion", {"action": "BUY", "price": 100.0, "holding_period": "SWING"})
    result = node_module.strategy_candidates_node(state)
    # Trending regime: supertrend is active, mean reversion is not.
    assert set(result["strategy_candidates"]) == {"TCS", "INFY"}
    assert {c["strategy"] for c in result["strategy_candidates"].values()} == {"SupertrendADX"}
    one = result["strategy_candidates"]["TCS"]
    assert one["exit_policy"] == "intraday_atr_2x_4x_v1" and one["atr_timeframe"] == "INTRADAY_15M_14"
    assert isinstance(one["stop_loss"], float) and isinstance(one["signal_time"], str)  # checkpoint-safe


def test_node_halts_in_high_volatility_and_when_vix_is_unknown(monkeypatch):
    enable(monkeypatch, "strategy_supertrend_adx_enabled")
    _stub(monkeypatch, "supertrend_adx", {"action": "BUY", "price": 100.0, "holding_period": "INTRADAY"})
    assert node_module.strategy_candidates_node(_node_state(monkeypatch, vix=35.0))["strategy_candidates"] == {}
    assert node_module.strategy_candidates_node(_node_state(monkeypatch, vix=0.0))["strategy_candidates"] == {}
    assert node_module.strategy_candidates_node(
        _node_state(monkeypatch, regime="SYSTEMIC_PANIC"),
    )["strategy_candidates"] == {}


def test_selector_includes_the_new_sleeves():
    from models.signal_gate.strategy_selector import select_strategies

    trending = select_strategies(adx=30, choppiness=30, india_vix=15, daily_drawdown=0.0)
    assert {"btst_strength", "etf_rotation"} <= set(trending.active_strategies)
    ranging = select_strategies(adx=15, choppiness=70, india_vix=15, daily_drawdown=0.0)
    assert "etf_rotation" in ranging.active_strategies and "mean_reversion" in ranging.active_strategies


# ── technical overlay ────────────────────────────────────────────────────────

def _technical(monkeypatch, signal):
    from agents import technical_agent

    frame = pd.DataFrame({
        "datetime": pd.date_range("2026-06-01T09:15:00+05:30", periods=40, freq="15min"),
        "close": [100.0] * 40, "high": [101.0] * 40, "low": [99.0] * 40, "open": [100.0] * 40,
        "volume": [1000] * 40,
    })
    monkeypatch.setattr(technical_agent, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _df: {
        "signal": signal, "confidence": 70, "reasoning": "core",
        "indicators": {"close": 100.0, "rsi_14": 50.0},
    })
    monkeypatch.setattr(technical_agent, "daily_atr", lambda _df, _a=None: 2.0)
    return technical_agent, frame


CANDIDATE = {
    "symbol": "KEEP", "strategy": "ORBIntraday", "side": "BUY", "horizon": "INTRADAY",
    "stop_loss": 98.0, "target": 104.0, "exit_policy": "intraday_atr_2x_4x_v1",
    "atr_timeframe": "INTRADAY_15M_14", "candidate_id": "c-1", "thesis": "breakout", "atr_value": 1.2,
}


def test_technical_output_is_identical_when_no_sleeve_supplies_a_candidate(monkeypatch):
    agent, frame = _technical(monkeypatch, "BUY")
    base = {"watchlist": ["KEEP"], "market_data": {"KEEP": frame},
            "market_data_provenance": {"KEEP": {"provider": "angel_one", "symbol": "KEEP"}}}
    without_key = agent.TechnicalAgent().run(dict(base))
    empty = agent.TechnicalAgent().run({**base, "strategy_candidates": {}})
    assert repr(without_key) == repr(empty)
    assert without_key["technical_signals"]["KEEP"]["strategy"] == "CompositeTrend"


def test_sleeve_candidate_overlays_only_symbols_the_core_strategy_holds(monkeypatch):
    agent, frame = _technical(monkeypatch, "HOLD")
    state = {"watchlist": ["KEEP"], "market_data": {"KEEP": frame},
             "market_data_provenance": {"KEEP": {"provider": "angel_one", "symbol": "KEEP"}},
             "strategy_candidates": {"KEEP": CANDIDATE}}
    signal = agent.TechnicalAgent().run(state)["technical_signals"]["KEEP"]
    assert signal["strategy"] == "ORBIntraday" and signal["signal"] == "BUY"
    assert (signal["stop_loss"], signal["target"], signal["holding_period"]) == (98.0, 104.0, "INTRADAY")
    assert signal["exit_policy"] == "intraday_atr_2x_4x_v1" and signal["indicators"]["policy_atr"] == 1.2
    assert signal["history_provenance"]["provider"] == "angel_one"  # review evidence is preserved
    agent2, frame2 = _technical(monkeypatch, "BUY")
    state["market_data"] = {"KEEP": frame2}
    core = agent2.TechnicalAgent().run(state)["technical_signals"]["KEEP"]
    assert core["strategy"] == "CompositeTrend"  # the core signal is never replaced


# ── router: an enabled sleeve still faces every gate ─────────────────────────

def _sleeve_router_state(monkeypatch, tmp_path, strategy="ORBIntraday", horizon="INTRADAY"):
    """A reviewed, stored entry for a sleeve whose own flag is on."""
    from tests.test_candidate_contract import _entry_state

    enable(monkeypatch, next(c.enabled_setting for c in SLEEVE_CONTRACTS if c.name == strategy))
    return _entry_state(
        monkeypatch, tmp_path, strategy=strategy, horizon=horizon, persist_strategy=strategy,
        technical_extra={"exit_policy": STRATEGY_CONTRACTS[strategy].default_exit_policy},
    )


def test_sleeve_entries_route_with_their_own_book_policy_and_horizon(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _sleeve_router_state(monkeypatch, tmp_path)
    result = graph._order_router_node(state)
    assert result["review_contract_results"]["OK"]["strategy"] == "ORBIntraday"
    assert len(calls) == 1, result["errors"]
    call = calls[0]
    assert call["book"] == "ORBIntraday" and call["horizon"] == "INTRADAY"
    assert call["exit_policy"] == "intraday_atr_2x_4x_v1" and call["atr_timeframe"] == "INTRADAY_15M_14"
    assert call["strategy_version"] == "orbintraday-v1"


def test_a_swing_sleeve_routes_through_the_delivery_product(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _sleeve_router_state(
        monkeypatch, tmp_path, strategy="MomentumBreakout", horizon="SWING",
    )
    result = graph._order_router_node(state)
    assert len(calls) == 1, result["errors"]
    assert calls[0]["product_type"] == "DELIVERY" and calls[0]["horizon"] == "SWING"
    assert calls[0]["book"] == "MomentumBreakout" and calls[0]["exit_policy"] == "daily_atr22_2x_4x_v1"


def test_a_disabled_sleeve_candidate_is_blocked(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path, strategy="ORBIntraday")
    result = graph._order_router_node(state)
    assert calls == [] and result["errors"]


def test_unapproved_paused_and_wrong_horizon_sleeve_entries_are_blocked(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification

    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _sleeve_router_state(monkeypatch, tmp_path)
    state["risk_assessment"]["OK"]["approved"] = False
    result = graph._order_router_node(state)
    assert calls == [] and any("risk-approved" in e for e in result["errors"])

    graph, state, calls = _sleeve_router_state(monkeypatch, tmp_path)
    from operations import control_state as controls

    controls.control_state.set_paused(True)
    result = graph._order_router_node(state)
    assert calls == [] and any("operational safety controls" in e for e in result["errors"])

    graph, state, calls = _sleeve_router_state(monkeypatch, tmp_path)
    state["risk_assessment"]["OK"]["holding_period"] = "SWING"  # ORB may only hold INTRADAY
    result = graph._order_router_node(state)
    assert calls == [] and any("Unsupported horizon SWING" in e for e in result["errors"])


def test_sleeve_exits_never_depend_on_the_sleeve_flag(monkeypatch):
    from agents import graph
    from risk import stop_loss_manager as slm

    placed = []
    monkeypatch.setattr(graph, "place_order", lambda **kw: placed.append(kw) or {
        "symbol": kw["symbol"], "status": "EXECUTED", "trade_id": 1, "quantity": kw["quantity"],
    })
    monkeypatch.setattr(graph, "get_token", lambda _s: "token")
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(slm.stop_loss_manager, "pending_exits", lambda: {})
    state = {
        "execution_enabled": True,  # every sleeve flag is off
        "current_positions": {"OLD": {"quantity": 3, "entry_trade_ids": [1], "holding_period": "SWING"}},
        "trading_decisions": [{"symbol": "OLD", "action": "SELL", "quantity": 3, "price": 100,
                               "holding_period": "SWING"}],
    }
    graph._order_router_node(state)
    assert [(c["symbol"], c["product_type"], c["is_exit"]) for c in placed] == [("OLD", "DELIVERY", True)]


def test_preflight_warns_when_a_sleeve_is_enabled(monkeypatch):
    from operations import preflight

    assert not any("Strategy sleeves enabled" in w for w in preflight.run_preflight().warnings)
    enable(monkeypatch, "strategy_etf_rotation_enabled")
    warnings = preflight.run_preflight().warnings
    assert any("strategy_etf_rotation_enabled" in w for w in warnings)
