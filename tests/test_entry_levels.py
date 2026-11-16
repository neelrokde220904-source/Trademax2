"""Daily-ATR entry levels and the pre-LLM screener's completed-bar volume check."""

import pandas as pd
import pytest

from data.indicators import bar_relative_volume, best_quote_spread_pct, daily_atr, session_relative_volume


def _sessions(days: int, bars_per_day: int = 25, today_bars: int = 0, today_volume: int = 1000):
    rows = []
    start = pd.Timestamp("2026-06-01")
    for day in range(days + (1 if today_bars else 0)):
        date = start + pd.Timedelta(days=day)
        is_today = day == days
        for index in range(today_bars if is_today else bars_per_day):
            stamp = pd.Timestamp(f"{date.date()} 09:15:00+05:30") + pd.Timedelta(minutes=15 * index)
            rows.append({
                "datetime": stamp, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": today_volume if is_today else 1000,
            })
    return pd.DataFrame(rows), start + pd.Timedelta(days=days)


def _quote(today, *, bid=99.95, ask=100.05, at="10:00:00"):
    return {
        "timestamp": f"{today.date()}T{at}+05:30",
        "exchange_timestamp": f"{today.date()}T{at}+05:30",
        "best_bid": [{"price": bid}],
        "best_ask": [{"price": ask}],
    }


@pytest.mark.parametrize("exchange_at", [None, "09:59:29", "10:00:03"])
def test_spread_rejects_missing_stale_or_future_exchange_time(exchange_at):
    quote = _quote(pd.Timestamp("2026-06-22"))
    if exchange_at is not None:
        quote["exchange_timestamp"] = f"2026-06-22T{exchange_at}+05:30"
    else:
        del quote["exchange_timestamp"]
    assert best_quote_spread_pct(quote, "2026-06-22T10:00:00+05:30") is None


def test_spread_accepts_current_exchange_and_receipt_times():
    quote = _quote(pd.Timestamp("2026-06-22"))
    quote["exchange_timestamp"] = "2026-06-22T09:59:45+05:30"
    assert best_quote_spread_pct(quote, "2026-06-22T10:00:00+05:30") == pytest.approx(0.001)
    assert best_quote_spread_pct(quote, "NaT") is None


def test_daily_atr_uses_completed_prior_sessions_only():
    frame, today = _sessions(20, today_bars=3)
    # A huge bar today must not leak into the daily ATR.
    frame.loc[frame.index[-1], ["high", "low"]] = [150.0, 50.0]
    value = daily_atr(frame, f"{today.date()}T10:00:00+05:30")
    assert value == pytest.approx(2.0)


def test_daily_atr_requires_enough_sessions():
    frame, today = _sessions(10)
    assert daily_atr(frame, f"{today.date()}T10:00:00+05:30") is None


def test_daily_atr_rejects_an_incomplete_prior_session_in_lookback():
    frame, today = _sessions(20, today_bars=3)
    last_prior_day = today - pd.Timedelta(days=1)
    missing = (frame["datetime"].dt.date == last_prior_day.date()) & (
        frame["datetime"].dt.hour == 12
    ) & (frame["datetime"].dt.minute == 15)
    frame = frame.loc[~missing]
    assert daily_atr(frame, f"{today.date()}T10:00:00+05:30") is None


def test_daily_atr_rejects_invalid_or_missing_ohlc_in_recent_session():
    frame, today = _sessions(20, today_bars=3)
    last_prior_day = today - pd.Timedelta(days=1)
    index = frame.index[frame["datetime"].dt.date == last_prior_day.date()][0]
    frame.loc[index, "high"] = float("nan")
    assert daily_atr(frame, f"{today.date()}T10:00:00+05:30") is None


def test_relative_volume_ignores_in_progress_bar_and_compares_same_time_of_day():
    frame, today = _sessions(5, today_bars=3, today_volume=1500)
    # The 09:45 bar is still open at 09:50, so only two bars count: 3000 vs 2000.
    assert session_relative_volume(frame, f"{today.date()}T09:50:00+05:30") == pytest.approx(1.5)
    assert session_relative_volume(frame, f"{today.date()}T09:20:00+05:30") is None


def test_breakout_bar_volume_uses_matching_closed_bar_and_requires_five_samples():
    frame, today = _sessions(5, today_bars=3, today_volume=1000)
    today_rows = frame.index[frame["datetime"].dt.date == today.date()]
    frame.loc[today_rows[1], "volume"] = 1500
    frame.loc[today_rows[2], "volume"] = 10_000
    as_of = f"{today.date()}T09:50:00+05:30"
    assert bar_relative_volume(frame, as_of) == pytest.approx(1.5)
    assert session_relative_volume(frame, as_of) == pytest.approx(1.25)
    first_day = today - pd.Timedelta(days=5)
    missing = (frame["datetime"].dt.date == first_day.date()) & (
        frame["datetime"].dt.hour == 9
    ) & (frame["datetime"].dt.minute == 15)
    assert bar_relative_volume(frame.loc[~missing], as_of) is None


def test_relative_volume_rejects_gaps_in_current_session():
    frame, today = _sessions(5, today_bars=3, today_volume=1500)
    missing = (frame["datetime"].dt.date == today.date()) & (
        frame["datetime"].dt.hour == 9
    ) & (frame["datetime"].dt.minute == 30)
    assert session_relative_volume(frame.loc[~missing], f"{today.date()}T10:00:00+05:30") is None


def test_relative_volume_excludes_prior_sessions_missing_matching_time_slots():
    frame, today = _sessions(2, today_bars=3, today_volume=1500)
    incomplete_day = today - pd.Timedelta(days=1)
    missing = (frame["datetime"].dt.date == incomplete_day.date()) & (
        frame["datetime"].dt.hour == 9
    ) & (frame["datetime"].dt.minute == 15)
    later = (frame["datetime"].dt.date == incomplete_day.date()) & (
        frame["datetime"].dt.hour == 10
    ) & (frame["datetime"].dt.minute == 0)
    frame.loc[later, "volume"] = 9000
    assert session_relative_volume(frame.loc[~missing], f"{today.date()}T10:00:00+05:30") == pytest.approx(1.5)


@pytest.mark.parametrize("invalid", [float("nan"), -1])
def test_relative_volume_rejects_invalid_current_volume(invalid):
    frame, today = _sessions(5, today_bars=3, today_volume=1500)
    frame.loc[frame.index[-1], "volume"] = invalid
    assert session_relative_volume(frame, f"{today.date()}T10:00:00+05:30") is None


def test_screener_passes_normal_volume_and_large_cap_atr(monkeypatch):
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3)
    frame["volume"] = 50_000
    frame["high"] = frame["close"] * 1.003
    frame["low"] = frame["close"] * 0.997
    frame["close"] = frame["close"] + frame.index % 7 * 0.2  # enough trend to pass ADX
    screener = QuantitativeScreener(min_adx=0)
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }
    assert screener.screen(state)["watchlist"] == ["X"]

    state["live_quotes"]["X"]["exchange_timestamp"] = f"{today.date()}T09:59:29+05:30"
    assert screener.screen(state)["screener_results"]["reasons"] == {"X": "spread_unavailable"}
    state["live_quotes"]["X"] = _quote(today)
    frame.loc[frame["datetime"].dt.date == today.date(), "volume"] = 100
    assert screener.screen(state)["screener_results"]["reasons"] == {"X": "low_volume"}


@pytest.mark.parametrize("missing", ["current_session", "current_bar", "baseline"])
def test_screener_blocks_unavailable_same_time_volume(missing):
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3)
    frame["volume"] = 50_000
    today_mask = frame["datetime"].dt.date == today.date()
    if missing == "current_session":
        frame = frame.loc[~today_mask]
    elif missing == "current_bar":
        frame = frame.loc[~(today_mask & (frame["datetime"].dt.minute == 30))]
    else:
        morning = frame["datetime"].dt.hour == 9
        frame.loc[~today_mask & morning, "volume"] = 0
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }

    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["screener_results"]["reasons"] == {"X": "volume_unavailable"}


def test_screener_trend_indicators_exclude_in_progress_candle(monkeypatch):
    from agents import quantitative_screener

    frame, today = _sessions(10, today_bars=4)
    frame["volume"] = 50_000
    in_progress = (frame["datetime"].dt.date == today.date()) & (
        frame["datetime"].dt.hour == 10
    ) & (frame["datetime"].dt.minute == 0)
    assert in_progress.sum() == 1
    frame.loc[in_progress, ["high", "low", "close"]] = [1000, 900, 950]
    observed = {}

    def adx(high, low, close, length):
        observed["adx"] = close.iloc[-1]
        return pd.DataFrame({"ADX_14": [50.0]})

    def atr(high, low, close, length):
        observed["atr"] = close.iloc[-1]
        return pd.Series([1.0])

    monkeypatch.setattr(quantitative_screener.ta, "adx", adx)
    monkeypatch.setattr(quantitative_screener.ta, "atr", atr)
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }

    assert quantitative_screener.QuantitativeScreener().screen(state)["watchlist"] == ["X"]
    assert observed == {"adx": 100.0, "atr": 100.0}


def test_screener_requires_1_5_relative_volume_for_breakout():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3, today_volume=50_000)
    frame.loc[frame["datetime"].dt.date < today.date(), "volume"] = 50_000
    frame.loc[frame.index[-1], ["close", "high"]] = [103.0, 104.0]
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }

    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["screener_results"]["reasons"] == {"X": "breakout_relative_volume"}


def test_screener_accepts_breakout_at_or_above_relative_volume_threshold():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3, today_volume=75_000)
    frame.loc[frame["datetime"].dt.date < today.date(), "volume"] = 50_000
    frame.loc[frame.index[-1], ["close", "high"]] = [103.0, 104.0]
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }

    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["watchlist"] == ["X"]


def test_breakout_requires_volume_on_the_breakout_bar_not_just_earlier_bars():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3, today_volume=50_000)
    frame.loc[frame["datetime"].dt.date < today.date(), "volume"] = 50_000
    today_rows = frame.index[frame["datetime"].dt.date == today.date()]
    frame.loc[today_rows[0], "volume"] = 150_000
    frame.loc[today_rows[-1], ["close", "high", "volume"]] = [103.0, 104.0, 25_000]
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }

    assert session_relative_volume(frame, state["research_as_of"]) == pytest.approx(1.5)
    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["screener_results"]["reasons"] == {"X": "breakout_relative_volume"}


def test_screener_blocks_extended_vwap_chop():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=8)
    frame["volume"] = 50_000
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today, at="11:15:00")},
        "research_as_of": f"{today.date()}T11:15:00+05:30",
    }
    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["screener_results"]["reasons"] == {"X": "vwap_chop"}


def test_screener_rejects_missing_or_wide_market_depth():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3)
    frame["volume"] = 50_000
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }
    screener = QuantitativeScreener(min_adx=0)
    assert screener.screen(state)["screener_results"]["reasons"] == {"X": "spread_unavailable"}
    state["live_quotes"] = {"X": _quote(today, bid=99, ask=101)}
    assert screener.screen(state)["screener_results"]["reasons"] == {"X": "wide_spread"}


def test_screener_rejects_low_average_daily_turnover():
    from agents.quantitative_screener import QuantitativeScreener

    frame, today = _sessions(10, today_bars=3)
    state = {
        "watchlist": ["X"], "market_data": {"X": frame}, "hmm_regime": {},
        "live_quotes": {"X": _quote(today)},
        "research_as_of": f"{today.date()}T10:00:00+05:30",
    }
    result = QuantitativeScreener(min_adx=0).screen(state)
    assert result["screener_results"]["reasons"] == {"X": "low_daily_turnover"}


def test_screener_rejects_all_candidates_when_market_data_is_missing():
    from agents.quantitative_screener import QuantitativeScreener

    result = QuantitativeScreener().screen({"watchlist": ["X"], "market_data": {}})
    assert result["watchlist"] == []
    assert result["screener_results"]["reasons"] == {"X": "no_data"}


def test_technical_agent_only_analyzes_the_screened_watchlist(monkeypatch):
    from agents import technical_agent

    frame = pd.DataFrame({
        "datetime": pd.date_range("2026-06-01T09:15:00+05:30", periods=30, freq="15min"),
        "close": [100.0] * 30, "high": [101.0] * 30, "low": [99.0] * 30,
        "open": [100.0] * 30, "volume": [1000] * 30,
    })
    monkeypatch.setattr(technical_agent, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _df: {
        "signal": "BUY", "confidence": 70, "reasoning": "test",
        "indicators": {"close": 100.0, "rsi_14": 50.0},
    })
    monkeypatch.setattr(technical_agent, "daily_atr", lambda _df, _as_of=None: 2.0)

    result = technical_agent.TechnicalAgent().run({
        "watchlist": ["KEEP"],
        "market_data": {"KEEP": frame, "DROP": frame},
        "market_data_provenance": {"KEEP": {"provider": "angel_one", "symbol": "KEEP"}},
    })

    assert list(result["technical_signals"]) == ["KEEP"]
    assert result["technical_signals"]["KEEP"]["history_provenance"]["provider"] == "angel_one"


def test_technical_agent_sets_daily_atr_levels(monkeypatch):
    from agents import technical_agent

    frame, _today = _sessions(2)
    monkeypatch.setattr(technical_agent, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _df: {
        "signal": "SELL", "confidence": 70, "reasoning": "test",
        "indicators": {"close": 100.0, "atr_14": 0.3, "rsi_14": 50.0},
    })
    monkeypatch.setattr(technical_agent, "daily_atr", lambda _df, _as_of=None: 2.0)
    monkeypatch.setattr(technical_agent.settings, "technical_stop_daily_atr_multiplier", 1.0)
    monkeypatch.setattr(technical_agent.settings, "technical_target_daily_atr_multiplier", 2.5)

    signal = technical_agent.TechnicalAgent().run({"market_data": {"X": frame}})["technical_signals"]["X"]
    assert (signal["stop_loss"], signal["target"]) == (102.0, 95.0)
    assert signal["indicators"]["daily_atr_14"] == 2.0


def test_technical_agent_reports_invalid_symbol_provenance(monkeypatch):
    from agents import technical_agent

    frame, _today = _sessions(2)
    monkeypatch.setattr(technical_agent, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _df: {
        "signal": "BUY", "confidence": 70, "reasoning": "test",
        "indicators": {"close": 100.0, "rsi_14": 50.0},
    })
    result = technical_agent.TechnicalAgent().run({
        "market_data": {"X": frame}, "market_data_provenance": {"X": "not a source record"},
    })
    assert "X" not in result["technical_signals"]
    assert any("Invalid historical data provenance" in error for error in result["errors"])


def test_technical_agent_blocks_long_signal_outside_rsi_momentum_zone(monkeypatch):
    from agents import technical_agent

    frame, _today = _sessions(2)
    monkeypatch.setattr(technical_agent, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(technical_agent, "get_latest_signals", lambda _df: {
        "signal": "BUY", "confidence": 70, "reasoning": "test",
        "indicators": {"close": 100.0, "rsi_14": 35.0},
    })
    monkeypatch.setattr(technical_agent, "daily_atr", lambda _df, _as_of=None: 2.0)

    signal = technical_agent.TechnicalAgent().run({"market_data": {"X": frame}})["technical_signals"]["X"]
    assert signal["signal"] == "HOLD"
    assert "RSI 40–70" in signal["reasoning"]
    assert "stop_loss" not in signal


def test_risk_manager_rejects_signal_without_daily_atr_levels(monkeypatch):
    from agents.base_agent import create_initial_state
    from agents.risk_manager_agent import RiskManagerAgent
    from broker.portfolio_tracker import portfolio_tracker
    from tests.test_daily_portfolio_risk import AS_OF, history_for

    monkeypatch.setattr(portfolio_tracker, "has_hit_daily_loss_limit", lambda: False)
    state = create_initial_state(["TEST"])
    state["india_vix"] = 15
    state["research_mode"] = True
    state["research_as_of"] = AS_OF
    state["market_data"] = {"TEST": history_for()}
    state["technical_signals"] = {
        "TEST": {"signal": "BUY", "confidence": 90, "indicators": {"close": 100, "atr_14": 2}},
    }

    assessment = RiskManagerAgent().run(state)["risk_assessment"]["TEST"]
    assert assessment == {"approved": False, "rejection_reason": "Daily-ATR stop/target unavailable"}


def test_daily_atr_level_defaults():
    from config.settings import Settings

    fields = Settings.model_fields
    assert fields["technical_stop_daily_atr_multiplier"].default == 1.0
    assert fields["technical_target_daily_atr_multiplier"].default == 2.5
