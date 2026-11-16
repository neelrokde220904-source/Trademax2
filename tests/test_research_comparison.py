"""Comparison runs must be isolated, point-in-time and broker-free."""

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from tests.test_llm_runtime import llm_database
from tests.test_research_tracking import _state


def _forbid(*_args, **_kwargs):
    raise AssertionError("Research replay attempted an external or broker call")


def test_full_replay_uses_only_captured_inputs(llm_database, monkeypatch):
    from agents import graph, fundamental_agent, india_specific_agent, macro_agent, options_agent
    from agents.base_agent import create_initial_state
    from config.llm_runtime import llm_run, profile_for
    from data import news_fetcher

    snapshot = _state()
    snapshot["research_as_of"] = "2026-06-05T05:00:00+00:00"
    monkeypatch.setattr(graph, "market_data_agent", _forbid)
    monkeypatch.setattr(graph, "place_order", _forbid)
    monkeypatch.setattr(news_fetcher, "fetch_news_snapshot", _forbid)
    monkeypatch.setattr(fundamental_agent.FundamentalAgent, "_fetch_fundamentals", _forbid)
    monkeypatch.setattr(options_agent, "fetch_options_chain", _forbid)
    for name in ("_fetch_global_cues", "_fetch_usdinr", "_fetch_crude_oil", "_fetch_fii_dii"):
        monkeypatch.setattr(macro_agent.MacroAgent, name, _forbid)
    for name in ("_fetch_fo_ban_list", "_fetch_bulk_deals", "_fetch_block_deals",
                 "_check_promoter_pledging", "_check_circuit_limits", "_is_expiry_day"):
        monkeypatch.setattr(india_specific_agent.IndiaSpecificAgent, name, _forbid)

    state = create_initial_state(["TEST"])
    state.update(snapshot)
    state.update(research_mode=True, execution_enabled=False, research_daily_loss_hit=False,
                 current_positions={}, available_capital=100000.0, portfolio_value=100000.0)
    with llm_run("research_test", profile=profile_for("rules")):
        result = asyncio.run(graph.build_trading_graph(research=True).ainvoke(state))
    assert result["executed_orders"] == []
    assert result["research_options_inputs"] == snapshot["research_options_inputs"]
    assert result["research_macro_inputs"] == snapshot["research_macro_inputs"]


def _snapshot(at, bars):
    state = _state()
    state["market_data"] = {"TEST": pd.DataFrame(bars)}
    state["hmm_regime"] = {
        "regime": "LOW_VOL_TREND", "degraded": False,
        "action_directive": {
            "allow_new_positions": True, "max_position_size_pct": 0.1,
            "allowed_strategies": ["trend_following"], "hedge_required": False,
        },
    }
    return state, datetime.fromisoformat(at).astimezone(timezone.utc)


def test_separate_paper_portfolios_use_next_observed_bar(llm_database, monkeypatch):
    from research import comparison, tracking

    monkeypatch.setattr(comparison.settings, "paper_trading", True)
    monkeypatch.setattr(comparison, "configured", lambda _profile: True)
    monkeypatch.setattr(comparison, "_run_research_pipeline", lambda state, profile: {
        **state,
        "technical_signals": {"TEST": {
            "signal": "BUY", "strategy": "CompositeTrend", "indicators": {"close": 100},
            "stop_loss": 95, "target": 125, "holding_period": "INTRADAY",
        }},
        "risk_assessment": {"blocked": False, "TEST": {
            "approved": True, "deterministic_action": "BUY", "stop_loss": 95,
            "target": 125, "vix_factor": 1.0, "holding_period": "INTRADAY",
        }},
        "india_signals": {"fo_ban_list": []},
        "trading_decisions": [{
            "symbol": "TEST", "action": "BUY", "quantity": 1,
            "order_type": "MARKET", "price": 100, "stop_loss": 95,
            "target": 125, "holding_period": "INTRADAY",
        }] if profile.mode != "economy" else [],
        "errors": [],
    })
    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    later = initial + [{
        "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
        "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000,
    }, {
        "datetime": pd.Timestamp("2026-06-05T11:00:00+05:30"),
        "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 1000,
    }]
    first, first_time = _snapshot("2026-06-05T10:31:00+05:30", initial)
    second, second_time = _snapshot("2026-06-05T11:16:00+05:30", later)
    tracking.record_scan(first, "one", observed_at=first_time)
    tracking.record_scan(second, "two", observed_at=second_time)

    result = comparison.run_comparison(["one", "two"])
    assert result["status"] == "completed"
    profiles = {row["mode"]: row for row in result["variants"]}
    assert len(profiles["rules"]["open_positions"]) == 1
    assert profiles["economy"]["open_positions"] == []
    assert profiles["rules"]["open_positions"][0]["entry_time"].startswith("2026-06-05T10:45")
    assert profiles["rules"]["open_positions"][0]["entry_price"] > 101
    assert profiles["rules"]["final_equity"] != profiles["economy"]["final_equity"]


def test_comparison_rejects_missing_inputs_and_unconfigured_provider(llm_database, monkeypatch):
    from research import comparison, tracking

    monkeypatch.setattr(comparison.settings, "paper_trading", True)
    first, at = _snapshot("2026-06-05T10:31:00+05:30", [])
    first["research_fundamental_inputs"] = {}
    tracking.record_scan(first, "incomplete", observed_at=at)
    with pytest.raises(ValueError, match="incomplete"):
        comparison.run_comparison(["incomplete", "incomplete"])


def test_comparison_rejects_an_unconfigured_model_before_replay(llm_database, monkeypatch):
    from research import comparison, tracking

    monkeypatch.setattr(comparison.settings, "paper_trading", True)
    monkeypatch.setattr(comparison, "configured", lambda profile: profile.mode != "economy")
    monkeypatch.setattr(comparison, "_run_research_pipeline", _forbid)
    for identifier, when in (("one", "2026-06-05T10:31:00+05:30"),
                             ("two", "2026-06-05T11:31:00+05:30")):
        state = _state()
        tracking.record_scan(state, identifier, observed_at=datetime.fromisoformat(when))
    with pytest.raises(ValueError, match="economy"):
        comparison.run_comparison(["one", "two"])


def test_provider_failure_does_not_fall_back_to_another_model(llm_database, monkeypatch):
    from agents import graph, llm_client
    from agents.base_agent import create_initial_state
    from config.llm_runtime import profile_for
    from research.comparison import _run_research_pipeline

    monkeypatch.setattr(llm_client.settings, "trademax_llm_provider", "openai")
    monkeypatch.setattr(llm_client, "_claude_request", lambda *_args: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llm_client, "resolve_configuration", _forbid)
    monkeypatch.setattr(graph, "place_order", _forbid)
    snapshot = _state()
    snapshot["news_snapshot"]["articles"] = [{"title": "Market headline"}]
    snapshot["research_as_of"] = "2026-06-05T05:00:00+00:00"
    state = create_initial_state(["TEST"])
    state.update(snapshot)
    state.update(research_mode=True, execution_enabled=False, research_daily_loss_hit=False,
                 current_positions={}, available_capital=100000.0, portfolio_value=100000.0)
    with pytest.raises(RuntimeError, match="degraded"):
        _run_research_pipeline(state, profile_for("claude"))


def test_full_replay_routes_each_model_to_its_own_provider(llm_database, monkeypatch):
    from agents import graph, llm_client
    from agents.base_agent import create_initial_state
    from config.llm_runtime import profile_for
    from research.comparison import _run_research_pipeline

    calls = []
    answer = '{"identified_risks": [], "overall_risk_level": "LOW"}'
    monkeypatch.setattr(llm_client.settings, "trademax_llm_provider", "openai")
    monkeypatch.setattr(llm_client, "resolve_configuration", _forbid)
    monkeypatch.setattr(llm_client, "_claude_request", lambda profile, *_args: (
        calls.append(profile.mode) or (answer, {"input_tokens": 1, "output_tokens": 1}, "end_turn", "c")
    ))
    monkeypatch.setattr(llm_client, "_deepseek_request", lambda profile, *_args: (
        calls.append(profile.mode) or (answer, {"prompt_tokens": 1, "completion_tokens": 1}, "stop", "d")
    ))
    monkeypatch.setattr(graph, "place_order", _forbid)
    snapshot = _state()
    snapshot["news_snapshot"]["articles"] = [{"title": "Market headline"}]
    snapshot["research_as_of"] = "2026-06-05T05:00:00+00:00"
    for mode in ("claude", "economy"):
        state = create_initial_state(["TEST"])
        state.update(snapshot)
        state.update(research_mode=True, execution_enabled=False, research_daily_loss_hit=False,
                     current_positions={}, available_capital=100000.0, portfolio_value=100000.0)
        result = _run_research_pipeline(state, profile_for(mode))
        assert result["executed_orders"] == []
    assert calls == ["claude", "economy"]


def test_full_comparison_runs_real_graph_and_persists_results(llm_database, monkeypatch):
    from agents import graph, llm_client, fundamental_agent, india_specific_agent, macro_agent, options_agent
    from config import llm_runtime
    from data import news_fetcher
    from research import comparison, tracking

    calls = []
    answer = '{"identified_risks": [], "overall_risk_level": "LOW"}'
    monkeypatch.setattr(comparison.settings, "paper_trading", True)
    monkeypatch.setattr(llm_client, "resolve_configuration", _forbid)
    monkeypatch.setattr(llm_client, "_claude_request", lambda profile, *_args: (
        calls.append(profile.mode) or (answer, {"input_tokens": 1, "output_tokens": 1}, "end_turn", "c")
    ))
    monkeypatch.setattr(llm_client, "_deepseek_request", lambda profile, *_args: (
        calls.append(profile.mode) or (answer, {"prompt_tokens": 1, "completion_tokens": 1}, "stop", "d")
    ))
    monkeypatch.setattr(graph, "market_data_agent", _forbid)
    monkeypatch.setattr(graph, "place_order", _forbid)
    monkeypatch.setattr(news_fetcher, "fetch_news_snapshot", _forbid)
    monkeypatch.setattr(fundamental_agent.FundamentalAgent, "_fetch_fundamentals", _forbid)
    monkeypatch.setattr(options_agent, "fetch_options_chain", _forbid)
    for name in ("_fetch_global_cues", "_fetch_usdinr", "_fetch_crude_oil", "_fetch_fii_dii"):
        monkeypatch.setattr(macro_agent.MacroAgent, name, _forbid)
    for name in ("_fetch_fo_ban_list", "_fetch_bulk_deals", "_fetch_block_deals",
                 "_check_promoter_pledging", "_check_circuit_limits", "_is_expiry_day"):
        monkeypatch.setattr(india_specific_agent.IndiaSpecificAgent, name, _forbid)

    for identifier, when in (("one", "2026-06-05T10:31:00+05:30"),
                             ("two", "2026-06-05T11:31:00+05:30")):
        state = _state()
        state["news_snapshot"]["articles"] = [{"title": "Market headline"}]
        assert tracking.record_scan(state, identifier, observed_at=datetime.fromisoformat(when))["snapshot_status"] == "complete"
    selected = llm_runtime.select_mode("economy", 0)
    result = comparison.run_comparison(["one", "two"])

    assert result["status"] == "completed", result.get("error")
    assert [row["mode"] for row in result["variants"]] == ["claude", "economy", "rules"]
    assert all(not row["open_positions"] for row in result["variants"])
    assert calls == ["claude", "economy", "claude", "economy"]
    assert comparison.list_comparisons()[0]["id"] == result["id"]
    assert llm_runtime.selected_profile() == selected


def test_real_graph_model_decisions_reach_separate_paper_books(llm_database, monkeypatch):
    from agents import graph, llm_client, technical_agent
    from config import llm_runtime
    from research import comparison, tracking

    calls = []

    def answer(profile, model, prompt):
        calls.append((profile.mode, model))
        if "Portfolio Manager of an AI hedge fund" in prompt:
            decisions = [] if profile.mode == "economy" else [{
                "symbol": "TEST", "action": "BUY", "quantity": 1,
                "order_type": "MARKET", "confidence": 80,
            }]
            return json.dumps({"decisions": decisions})
        return '{"signal":"BUY","confidence":80,"reasoning":"Synthetic evidence","key_risks":[],"action":"BUY","conviction_score":80,"identified_risks":[]}'

    def claude(profile, model, prompt, *_args):
        return answer(profile, model, prompt), {"input_tokens": 1, "output_tokens": 1}, "end_turn", "c"

    def economy(profile, _role, model, prompt, *_args):
        return answer(profile, model, prompt), {"prompt_tokens": 1, "completion_tokens": 1}, "stop", "d"

    monkeypatch.setattr(comparison.settings, "paper_trading", True)
    monkeypatch.setattr(comparison.settings, "screener_min_liquidity_sessions", 1)
    monkeypatch.setattr(comparison.settings, "screener_min_average_daily_turnover_inr", 0.0)
    # Relax only this test fixture so synthetic ATR targets exercise the decision path.
    monkeypatch.setattr(comparison.settings, "min_net_reward_risk", 1.0)
    monkeypatch.setattr(llm_client, "_claude_request", claude)
    monkeypatch.setattr(llm_client, "_deepseek_request", economy)
    monkeypatch.setattr(graph, "market_data_agent", _forbid)
    monkeypatch.setattr(graph, "place_order", _forbid)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _frame: {
        "signal": "BUY", "confidence": 80, "reasoning": "Synthetic test signal",
        "indicators": {"close": 100.0, "atr_14": 3.0, "rsi_14": 50.0},
    })
    # Pin the ATR to keep the test's original 95.5/106 decision levels.
    monkeypatch.setattr(technical_agent, "daily_atr", lambda _frame, _as_of=None: 3.0)
    monkeypatch.setattr(comparison.settings, "technical_stop_daily_atr_multiplier", 1.5)
    monkeypatch.setattr(comparison.settings, "technical_target_daily_atr_multiplier", 2.0)
    stamps = list(pd.date_range("2026-06-04T09:15:00+05:30", periods=25, freq="15min").tz_convert("Asia/Kolkata"))
    stamps += list(pd.date_range("2026-06-05T09:15:00+05:30", periods=5, freq="15min").tz_convert("Asia/Kolkata"))
    days = pd.bdate_range("2026-05-06", "2026-06-03", tz="Asia/Kolkata")
    days = days[days.date != datetime(2026, 5, 28).date()]
    history = [{
        "datetime": stamp, "open": 99 + day_index * 0.02,
        "high": 100 + day_index * 0.02, "low": 98 + day_index * 0.02,
        "close": 99 + day_index * 0.02, "volume": 1000,
    } for day_index, day in enumerate(days) for stamp in pd.date_range(
        day + pd.Timedelta(hours=9, minutes=15), periods=25, freq="15min",
    )]
    bars = history + [{
        "datetime": stamp, "open": 97.0 + index * 0.1, "high": 98.0 + index * 0.1,
        "low": 96.0 + index * 0.1, "close": 97.0 + index * 0.1,
        "volume": 5000 if index == len(stamps) - 1 else 1000,
    } for index, stamp in enumerate(stamps)]
    later = bars + [{
        "datetime": pd.Timestamp(f"2026-06-05T{time}:00+05:30").tz_convert("Asia/Kolkata"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 5000 if time == "10:45" else 1000,
    } for time in ("10:30", "10:45")]

    for identifier, when, captured in (
        ("one", "2026-06-05T10:31:00+05:30", bars),
        ("two", "2026-06-05T11:01:00+05:30", later),
    ):
        state, at = _snapshot(when, captured)
        state["live_quotes"] = {"TEST": {
            "timestamp": when,
            "exchange_timestamp": when,
            "ltp": 100.0,
            "best_bid": [{"price": 99.95}],
            "best_ask": [{"price": 100.05}],
        }}
        state["news_snapshot"]["articles"] = [{"title": "Market headline"}]
        assert tracking.record_scan(state, identifier, observed_at=at)["snapshot_status"] == "complete"
    result = comparison.run_comparison(["one", "two"])

    assert result["status"] == "completed", result.get("error")
    variants = {row["mode"]: row for row in result["variants"]}
    assert len(variants["claude"]["open_positions"]) == 1
    assert not variants["economy"]["open_positions"]
    assert len(variants["rules"]["open_positions"]) == 1
    assert ("claude", llm_runtime.profile_for("claude").decision_model) in calls
    assert ("economy", llm_runtime.profile_for("economy").decision_model) in calls


def _approved_buy(state):
    return {
        **state,
        "technical_signals": {"TEST": {
            "signal": "BUY", "strategy": "CompositeTrend",
            "indicators": {"close": 100}, "stop_loss": 95,
            "target": 125, "holding_period": "INTRADAY",
        }},
        "risk_assessment": {"blocked": False, "TEST": {
            "approved": True, "deterministic_action": "BUY", "stop_loss": 95,
            "target": 125, "vix_factor": 1.0, "holding_period": "INTRADAY",
        }},
        "india_signals": {"fo_ban_list": []},
        "trading_decisions": [{
            "symbol": "TEST", "action": "BUY", "quantity": 1, "order_type": "MARKET",
        }],
    }


def test_replay_session_grid_handles_verified_budget_sunday():
    from research.comparison import _session_bars

    start = datetime.fromisoformat("2026-02-01T09:00:00+05:30")
    end = datetime.fromisoformat("2026-02-01T15:30:00+05:30")
    bars = _session_bars(start, end)
    assert len(bars) == 25
    assert bars[0].strftime("%H:%M") == "09:15"
    assert bars[-1].strftime("%H:%M") == "15:15"


def test_replay_refuses_unimplemented_muhurat_and_unverified_calendar_years():
    from research.comparison import _session_bars

    with pytest.raises(ValueError, match="Muhurat"):
        _session_bars(
            datetime.fromisoformat("2026-11-08T14:00:00+05:30"),
            datetime.fromisoformat("2026-11-08T16:00:00+05:30"),
        )
    with pytest.raises(ValueError, match="calendar"):
        _session_bars(
            datetime.fromisoformat("2027-01-04T09:00:00+05:30"),
            datetime.fromisoformat("2027-01-04T10:00:00+05:30"),
        )


def test_replay_does_not_process_unexpected_weekend_bars():
    from research.comparison import _PaperBook

    def bar(stamp):
        return {
            "datetime": pd.Timestamp(f"2026-06-07T{stamp}:00+05:30"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
        }

    first, first_time = _snapshot("2026-06-07T10:31:00+05:30", [bar("10:15")])
    second, second_time = _snapshot("2026-06-07T11:00:00+05:30", [bar("10:15"), bar("10:45")])
    first["research_as_of"], second["research_as_of"] = first_time.isoformat(), second_time.isoformat()
    book = _PaperBook()
    book.advance(first)
    with pytest.raises(ValueError, match="Unexpected NSE"):
        book.advance(second)


@pytest.mark.parametrize("directive,reason", [
    ({"allowed_strategies": ["mean_reversion"]}, "strategy"),
    ({"hedge_required": True}, "hedge"),
    ({"hedge_required": None}, "hedge"),
    ({"max_position_size_pct": None}, "position"),
])
def test_research_entries_abstain_when_hmm_directives_fail(directive, reason):
    from research.comparison import _PaperBook

    snapshot, at = _snapshot("2026-06-05T10:31:00+05:30", [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }])
    snapshot["research_as_of"] = at.isoformat()
    snapshot["hmm_regime"]["action_directive"].update(directive)
    book = _PaperBook()
    book.advance(snapshot)
    book.submit(_approved_buy(book.state(snapshot)))
    assert not book.pending
    assert reason in book.audit[-1]["reason"].lower()


def test_research_fill_respects_captured_hmm_notional_cap(monkeypatch):
    from config.settings import settings
    from research.comparison import _PaperBook

    monkeypatch.setattr(settings, "initial_capital", 25_000)
    first_bar = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    fill_bar = {
        "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }
    first, first_time = _snapshot("2026-06-05T10:31:00+05:30", first_bar)
    second, second_time = _snapshot("2026-06-05T11:00:00+05:30", first_bar + [fill_bar])
    first["research_as_of"], second["research_as_of"] = first_time.isoformat(), second_time.isoformat()
    first["hmm_regime"]["action_directive"]["max_position_size_pct"] = 0.04
    book = _PaperBook()
    book.advance(first)
    book.submit(_approved_buy(book.state(first)))
    book.advance(second)
    assert book.engine._positions["TEST"].quantity == 9
    book.loss_halted = True
    exit_state = book.state(second)
    exit_state.update(
        risk_assessment={"blocked": True}, hmm_regime=None,
        trading_decisions=[{
            "symbol": "TEST", "action": "SELL", "quantity": 9, "order_type": "MARKET",
        }],
    )
    book.submit(exit_state)
    assert any(order["kind"] == "exit" for order in book.pending.values())


def test_missing_next_bar_is_not_replaced_by_a_later_fill():
    from research.comparison import _PaperBook

    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    later = initial + [{
        "datetime": pd.Timestamp("2026-06-05T11:00:00+05:30"),
        "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000,
    }]
    first, first_time = _snapshot("2026-06-05T10:31:00+05:30", initial)
    second, second_time = _snapshot("2026-06-05T11:16:00+05:30", later)
    first["research_as_of"], second["research_as_of"] = first_time.isoformat(), second_time.isoformat()
    book = _PaperBook()
    book.advance(first)
    book.submit(_approved_buy(book.state(first)))
    book.advance(second)
    assert book.report("rules")["open_positions"] == []
    assert book.audit[0]["status"] == "missing_next_bar"
    assert "10:45" in book.audit[0]["bar"]


def test_missing_intermediate_bar_cannot_hide_a_stop_or_target():
    from research.comparison import _PaperBook

    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    fill = {
        "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
        "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000,
    }
    after_gap = {
        "datetime": pd.Timestamp("2026-06-05T11:15:00+05:30"),
        "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 1000,
    }
    first, at1 = _snapshot("2026-06-05T10:31:00+05:30", initial)
    second, at2 = _snapshot("2026-06-05T11:00:00+05:30", initial + [fill])
    third, at3 = _snapshot("2026-06-05T11:31:00+05:30", initial + [fill, after_gap])
    for snapshot, at in ((first, at1), (second, at2), (third, at3)):
        snapshot["research_as_of"] = at.isoformat()
    book = _PaperBook()
    book.advance(first)
    book.submit(_approved_buy(book.state(first)))
    book.advance(second)
    assert "TEST" in book.engine._positions
    with pytest.raises(ValueError, match="Missing.*bar.*TEST"):
        book.advance(third)


def test_late_bar_is_not_replayed_before_an_existing_position_entry():
    from research.comparison import _PaperBook

    def bar(stamp, low=99.0):
        return {
            "datetime": pd.Timestamp(f"2026-06-05T{stamp}:00+05:30"),
            "open": 100.0, "high": 101.0, "low": low, "close": 100.0, "volume": 1000,
        }

    first, at1 = _snapshot("2026-06-05T10:01:00+05:30", [bar("09:45")])
    second, at2 = _snapshot("2026-06-05T11:01:00+05:30", [bar("09:45"), bar("10:45")])
    third, at3 = _snapshot("2026-06-05T11:31:00+05:30", [bar("09:45"), bar("10:45"), bar("11:15")])
    fourth, at4 = _snapshot("2026-06-05T12:01:00+05:30", [
        bar("09:45"), bar("10:30", low=90.0), bar("10:45"), bar("11:15"), bar("11:30"), bar("11:45"),
    ])
    for snapshot, at in ((first, at1), (second, at2), (third, at3), (fourth, at4)):
        snapshot["research_as_of"] = at.isoformat()
    book = _PaperBook()
    book.advance(first)
    book.advance(second)
    book.submit(_approved_buy(book.state(second)))
    book.advance(third)
    assert "TEST" in book.engine._positions
    book.advance(fourth)
    assert not book.engine._closed_trades
    assert "TEST" in book.engine._positions
    assert any(item["status"] == "late_bar_ignored" for item in book.audit)


def test_simultaneous_entries_are_processed_in_stable_symbol_order(monkeypatch):
    from research.comparison import _PaperBook

    symbols = ["ZZZ", "AAA", "MMM"]
    first, at1 = _snapshot("2026-06-05T10:31:00+05:30", [])
    second, at2 = _snapshot("2026-06-05T11:00:00+05:30", [])
    for snapshot, at, stamp in ((first, at1, "10:15"), (second, at2, "10:45")):
        snapshot["research_as_of"] = at.isoformat()
        snapshot["market_data"] = {
            symbol: pd.DataFrame([{
                "datetime": pd.Timestamp(f"2026-06-05T{stamp}:00+05:30"),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
            }]) for symbol in symbols
        }
    book = _PaperBook()
    book.advance(first)
    stamp = pd.Timestamp("2026-06-05T10:45:00+05:30")
    for symbol in symbols:
        book.pending[(symbol, stamp)] = {
            "kind": "entry", "action": "BUY", "signal": {"symbol": symbol}, "limit_price": None,
        }
    seen = []
    monkeypatch.setattr(book.engine, "_process_signal", lambda signal, *_args: seen.append(signal["symbol"]))
    book.advance(second)
    assert seen == sorted(symbols)


def test_intraday_squareoff_requires_the_actual_1515_bar():
    from research.comparison import _PaperBook

    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    fill = {
        "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
        "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000,
    }
    squareoff = {
        "datetime": pd.Timestamp("2026-06-05T15:15:00+05:30"),
        "open": 103.0, "high": 104.0, "low": 102.0, "close": 103.0, "volume": 1000,
    }
    first, at1 = _snapshot("2026-06-05T10:31:00+05:30", initial)
    intervening = [{
        "datetime": stamp, "open": 102.0, "high": 104.0,
        "low": 101.0, "close": 103.0, "volume": 1000,
    } for stamp in pd.date_range(
        "2026-06-05T11:00:00+05:30", "2026-06-05T15:00:00+05:30", freq="15min",
    )]
    second, at2 = _snapshot("2026-06-05T11:00:00+05:30", initial + [fill])
    end, at3 = _snapshot("2026-06-05T15:31:00+05:30", initial + [fill] + intervening + [squareoff])
    for snapshot, at in ((first, at1), (second, at2), (end, at3)):
        snapshot["research_as_of"] = at.isoformat()
    book = _PaperBook()
    book.advance(first)
    book.submit(_approved_buy(book.state(first)))
    book.advance(second)
    assert len(book.engine._positions) == 1
    missing, _ = _snapshot("2026-06-05T15:31:00+05:30", initial + [fill] + intervening)
    missing["research_as_of"] = at3.isoformat()
    with pytest.raises(ValueError, match="square-off"):
        deepcopy(book).advance(missing)
    book.advance(end)
    report = book.report("rules")
    assert report["open_positions"] == []
    assert report["closed_trades"][0]["exit_reason"] == "INTRADAY_SQUARE_OFF"
    assert report["closed_trades"][0]["costs"] > 0


def test_entry_is_blocked_if_next_bar_reaches_intraday_cutoff():
    from research.comparison import _PaperBook

    bars = [{
        "datetime": pd.Timestamp("2026-06-05T14:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    snapshot, at = _snapshot("2026-06-05T14:44:00+05:30", bars)
    snapshot["research_as_of"] = at.isoformat()
    book = _PaperBook()
    book.advance(snapshot)
    book.submit(_approved_buy(book.state(snapshot)))
    assert not book.pending
    assert book.audit[0]["status"] == "outside_entry_window"


@pytest.mark.parametrize("limit,expected", [(100.0, "limit_not_marketable"), (101.0, "filled")])
def test_limit_orders_are_not_filled_beyond_the_limit(limit, expected):
    from research.comparison import _PaperBook

    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    fill = {
        "datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
        "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000,
    }
    first, at1 = _snapshot("2026-06-05T10:31:00+05:30", initial)
    second, at2 = _snapshot("2026-06-05T11:00:00+05:30", initial + [fill])
    first["research_as_of"], second["research_as_of"] = at1.isoformat(), at2.isoformat()
    book = _PaperBook()
    book.advance(first)
    state = _approved_buy(book.state(first))
    state["technical_signals"]["TEST"]["indicators"]["close"] = limit
    state["trading_decisions"][0]["order_type"] = "LIMIT"
    book.submit(state)
    book.advance(second)
    assert book.audit[0]["status"] == expected
    if expected == "filled":
        assert book.engine._positions["TEST"].entry_price <= limit
    else:
        assert not book.engine._positions


def test_daily_loss_halt_blocks_pending_entries_before_the_next_open():
    from research.comparison import _PaperBook

    initial = [{
        "datetime": pd.Timestamp("2026-06-05T10:15:00+05:30"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000,
    }]
    following = initial + [
        {"datetime": pd.Timestamp("2026-06-05T10:30:00+05:30"),
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        {"datetime": pd.Timestamp("2026-06-05T10:45:00+05:30"),
         "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000},
    ]
    first, at1 = _snapshot("2026-06-05T10:31:00+05:30", initial)
    second, at2 = _snapshot("2026-06-05T11:16:00+05:30", following)
    first["research_as_of"], second["research_as_of"] = at1.isoformat(), at2.isoformat()
    book = _PaperBook()
    book.advance(first)
    book.submit(_approved_buy(book.state(first)))
    book.day_start_equity = book.engine.initial_capital / 0.96
    book.advance(second)
    assert book.loss_halted
    assert not book.engine._positions
    assert book.audit[0]["status"] == "daily_loss_blocked"
