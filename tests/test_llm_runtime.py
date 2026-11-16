"""Offline model routing, persistence and billing regression tests."""

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import llm_runtime as runtime
from config.settings import settings
from database import db
from database.models import Base, LLMRun
from agents import llm_client
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import httpx


@pytest.fixture
def llm_database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'llm.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(settings, "anthropic_api_key", "test-claude")
    monkeypatch.setattr(settings, "deepseek_api_key", "test-economy")
    yield engine
    engine.dispose()


def test_persist_switch_and_revision(llm_database):
    assert runtime.selected_profile().mode == "claude"
    assert runtime.select_mode("economy", 0).revision == 1
    assert runtime.selected_profile().decision_model == settings.economy_decision_model
    with pytest.raises(ValueError):
        runtime.select_mode("claude", 0)
    assert runtime.select_mode("claude", 1).revision == 2
    assert runtime.select_mode("claude", 2).revision == 2


def test_missing_provider_does_not_change_mode(llm_database, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "your_key")
    with pytest.raises(ValueError, match="not configured"):
        runtime.select_mode("economy", 0)
    assert runtime.selected_profile().mode == "claude"


def test_run_keeps_profile_across_switch_and_threads(llm_database):
    async def exercise():
        with runtime.llm_run("scan") as original:
            runtime.select_mode("economy", 0)
            assert runtime.mode_status()["pending"] is True
            assert await asyncio.to_thread(runtime.current_profile) == original.profile
            with runtime.llm_run("nested") as nested:
                assert nested == original
        with runtime.llm_run("next") as next_run:
            assert next_run.profile.mode == "economy"
            runtime.select_mode("claude", 1)
            assert runtime.current_profile().mode == "economy"
    asyncio.run(exercise())
    assert runtime.current_context() is None
    with db.get_db() as database:
        assert [row.status for row in database.query(LLMRun).all()] == ["completed", "completed"]


def test_adapter_routes_both_roles_and_records_cost(llm_database, monkeypatch):
    calls = []
    def claude(profile, model, prompt, max_tokens, temperature):
        calls.append((profile.mode, model, max_tokens, temperature))
        return '{"signal":"HOLD"}', {"input_tokens": 100, "output_tokens": 20}, "end_turn", "request-1"
    def deepseek(profile, role, model, prompt, max_tokens, temperature):
        calls.append((profile.mode, model, max_tokens, temperature))
        return '{"signal":"HOLD"}', {"prompt_tokens": 100, "completion_tokens": 20}, "stop", "request-2"
    monkeypatch.setattr(llm_client, "_claude_request", claude)
    monkeypatch.setattr(llm_client, "_deepseek_request", deepseek)
    for mode in ("claude", "economy", "claude"):
        runtime.select_mode(mode, runtime.selected_profile().revision)
        for role in ("fast", "decision"):
            llm_client.complete(role, "test", max_tokens=123, temperature=0.2)
    assert [call[1] for call in calls] == [settings.llm_fast_model, settings.llm_decision_model,
        settings.economy_fast_model, settings.economy_decision_model, settings.llm_fast_model, settings.llm_decision_model]
    summary = llm_client.usage_summary()
    assert summary["modes"]["claude"]["requests"] == 4
    assert summary["modes"]["economy"]["requests"] == 2
    assert summary["estimated_usd"] > 0
    assert summary["hard_limit"] is None


def test_failed_provider_is_not_free_and_never_falls_back(llm_database, monkeypatch):
    runtime.select_mode("economy", 0)
    def failed(*args):
        raise OSError("private provider error")
    monkeypatch.setattr(llm_client, "_deepseek_request", failed)
    monkeypatch.setattr(llm_client, "_claude_request", lambda *args: pytest.fail("Claude fallback"))
    with pytest.raises(RuntimeError, match="no provider fallback"):
        llm_client.complete("fast", "test")
    assert llm_client.usage_summary()["incomplete"]
    assert runtime.mode_status()["degraded"]


def test_deepseek_read_timeout_is_single_attempt_and_degraded(llm_database, monkeypatch):
    runtime.select_mode("economy", 0)
    attempts = []
    def respond(request):
        attempts.append(request)
        raise httpx.ReadTimeout("mock timeout", request=request)
    client_class = httpx.Client
    monkeypatch.setattr(llm_client.httpx, "Client", lambda **kwargs:
        client_class(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(llm_client, "_claude_request", lambda *args: pytest.fail("Claude fallback"))
    with pytest.raises(RuntimeError, match="no provider fallback"):
        llm_client.complete("decision", "test")
    assert len(attempts) == 1
    assert runtime.mode_status()["degraded"]
    assert llm_client.usage_summary()["incomplete"]


def test_reasoning_and_caches_not_double_counted():
    peak = datetime(2026, 9, 28, 7)
    cost, _ = llm_client.estimate_cost("economy", "deepseek-flash",
        {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 500, "completion_tokens": 100,
         "completion_tokens_details": {"reasoning_tokens": 80}}, peak)
    assert cost == Decimal("0.000273")
    cost, _ = llm_client.estimate_cost("claude", "claude-opus-4-6",
        {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 100,
         "cache_creation_input_tokens": 100, "cache_creation": {"ephemeral_1h_input_tokens": 40}}, peak)
    assert cost == Decimal("0.001825")


@pytest.mark.parametrize("role", ["fast", "decision"])
def test_deepseek_wire_request(role, monkeypatch):
    profile = runtime.profile_for("economy")
    model = profile.fast_model if role == "fast" else profile.decision_model
    def respond(request):
        import json
        body = json.loads(request.content)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert body["model"] == model
        assert body["thinking"]["type"] == ("enabled" if role == "decision" else "disabled")
        assert body["max_tokens"] == (settings.economy_decision_max_tokens if role == "decision" else 1000)
        assert ("temperature" in body) == (role == "fast")
        return httpx.Response(200, json={"id": "mock", "choices": [{"finish_reason": "stop",
            "message": {"content": "{}", "reasoning_content": "private reasoning"}}], "usage": {}})
    client_class = httpx.Client
    monkeypatch.setattr(llm_client.httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs))
    assert llm_client._deepseek_request(profile, role, model, "test", 1000, 0.1)[0] == "{}"


def test_claude_request_preserves_generation_settings(monkeypatch):
    import anthropic
    captured = {}
    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")],
                usage=SimpleNamespace(model_dump=lambda: {}), stop_reason="end_turn", id="mock")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    llm_client._claude_request(runtime.profile_for("claude"), settings.llm_decision_model, "prompt", 4096, 0.2)
    assert captured["model"] == settings.llm_decision_model
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.2
    assert captured["max_retries"] == 0


@pytest.mark.parametrize("text,intent,finish,valid", [
    ('{"a":1}', "object", "stop", True), ('[{}]', "array", "stop", True),
    ('def factor(data): return data', "text", "stop", True),
    ('not json', "object", "stop", False), ('{}', "object", "length", False),
    ('', "text", "stop", False),
])
def test_output_validation_accounts_for_invalid_answers(llm_database, monkeypatch, text, intent, finish, valid):
    monkeypatch.setattr(llm_client, "_claude_request", lambda *args:
        (text, {"input_tokens": 10, "output_tokens": 10}, finish, "mock"))
    if valid:
        assert llm_client.complete("fast", "test", intent=intent) == text
    else:
        with pytest.raises(RuntimeError):
            llm_client.complete("fast", "test", intent=intent)
    assert llm_client.usage_summary()["estimated_usd"] > 0


def test_langgraph_inherits_profile_and_following_scan_switches(llm_database, monkeypatch, tmp_path):
    from langgraph.graph import StateGraph, END
    from agents import graph as trading_graph
    calls = []
    def record(profile, model, *args):
        calls.append((profile.mode, model))
        return "{}", {"input_tokens": 1, "output_tokens": 1}, "stop", "mock"
    monkeypatch.setattr(llm_client, "_claude_request", record)
    monkeypatch.setattr(llm_client, "_deepseek_request", lambda profile, role, model, *args: record(profile, model))
    def first(state):
        llm_client.complete("fast", "test")
        if runtime.selected_profile().mode == "claude":
            runtime.select_mode("economy", runtime.selected_profile().revision)
        return state
    def last(state):
        llm_client.complete("decision", "test")
        return state
    graph = StateGraph(dict)
    graph.add_node("first", first)
    graph.add_node("last", last)
    graph.set_entry_point("first")
    graph.add_edge("first", "last")
    graph.add_edge("last", END)
    monkeypatch.setattr(trading_graph, "build_trading_graph", graph.compile)
    monkeypatch.setattr(trading_graph, "__file__", str(tmp_path / "agents" / "graph.py"))
    asyncio.run(trading_graph.run_trading_pipeline(["TEST"]))
    asyncio.run(trading_graph.run_trading_pipeline(["TEST"]))
    assert calls == [("claude", settings.llm_fast_model), ("claude", settings.llm_decision_model),
                     ("economy", settings.economy_fast_model), ("economy", settings.economy_decision_model)]


def test_real_checkpoint_round_trips_market_dataframe(llm_database, monkeypatch, tmp_path):
    import pandas as pd
    import sqlite3
    from langgraph.graph import StateGraph, END
    from agents import graph as trading_graph

    frame = pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=2, tz="Asia/Kolkata"),
        "close": [100.25, 101.5],
        "volume": [20, 30],
    })
    seen = []

    def market(state):
        return {**state, "market_data": {"TEST": frame}}

    def consumer(state):
        pd.testing.assert_frame_equal(state["market_data"]["TEST"], frame)
        seen.append(state["execution_enabled"])
        return state

    graph = StateGraph(dict)
    graph.add_node("market", market)
    graph.add_node("consumer", consumer)
    graph.set_entry_point("market")
    graph.add_edge("market", "consumer")
    graph.add_edge("consumer", END)
    monkeypatch.setattr(trading_graph, "build_trading_graph", graph.compile)
    monkeypatch.setattr(trading_graph, "__file__", str(tmp_path / "agents" / "graph.py"))

    for _ in range(2):
        result = asyncio.run(trading_graph.run_trading_pipeline(["TEST"], execute_orders=False))
        pd.testing.assert_frame_equal(result["market_data"]["TEST"], frame)
    assert seen == [False, False]
    with sqlite3.connect(tmp_path / "database" / "checkpoints" / "trading.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
        thread_ids = [row[0] for row in connection.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    assert len(thread_ids) == 2

    async def load_checkpoint():
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from agents.checkpoint_serializer import TradingCheckpointSerializer
        async with aiosqlite.connect(tmp_path / "database" / "checkpoints" / "trading.sqlite") as connection:
            saver = AsyncSqliteSaver(connection, serde=TradingCheckpointSerializer())
            compiled = graph.compile(checkpointer=saver)
            snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_ids[0]}})
            pd.testing.assert_frame_equal(snapshot.values["market_data"]["TEST"], frame)
    asyncio.run(load_checkpoint())


def test_checkpoint_dataframe_round_trip_and_reserved_marker():
    import pandas as pd
    from agents.checkpoint_serializer import TradingCheckpointSerializer

    frame = pd.DataFrame({"close": [1.25, 2.5]}, index=pd.date_range(
        "2026-01-01", periods=2, tz="Asia/Kolkata",
    ))
    serializer = TradingCheckpointSerializer()
    result = serializer.loads_typed(serializer.dumps_typed({"bars": [frame]}))
    pd.testing.assert_frame_equal(result["bars"][0], frame, check_freq=False)
    with pytest.raises(ValueError, match="Reserved"):
        serializer.dumps_typed({"__india_trader_dataframe_v2__": "not a frame"})


def test_checkpoint_handles_angel_timestamps_and_exact_float_values():
    import pandas as pd
    from agents.checkpoint_serializer import TradingCheckpointSerializer

    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01T09:15:00+05:30", "2026-01-01T09:30:00+05:30"]),
        "close": [123.12345678901234, float("inf")],
        "volume": [20, 30],
    })
    serializer = TradingCheckpointSerializer()
    result = serializer.loads_typed(serializer.dumps_typed(frame))
    pd.testing.assert_frame_equal(result, frame, check_exact=True)


def test_checkpoint_rejects_unsupported_objects():
    import pandas as pd
    from agents.checkpoint_serializer import TradingCheckpointSerializer

    with pytest.raises(TypeError, match="Unsupported object"):
        TradingCheckpointSerializer().dumps_typed(pd.DataFrame({"value": [object()]}))


def test_full_graph_analysis_only_with_market_checkpoints(llm_database, monkeypatch, tmp_path):
    import pandas as pd
    from agents import graph as trading_graph

    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01T09:15:00+05:30"]),
        "open": [100.0], "high": [102.0], "low": [99.0],
        "close": [101.0], "volume": [1000],
    })
    monkeypatch.setattr(trading_graph, "__file__", str(tmp_path / "agents" / "graph.py"))
    monkeypatch.setattr(trading_graph, "market_data_agent", lambda state: {"market_data": {"TEST": frame}})
    for node in (
        "_kafka_events_node", "_l2_order_flow_node", "_hmm_regime_node",
        "quantitative_screener_node", "_shared_news_node", "technical_agent", "fundamental_agent",
        "sentiment_agent", "options_agent", "macro_agent", "india_specific_agent",
        "_geopolitical_nlp_node", "_bull_bear_debate_node", "risk_manager_agent",
        "_marl_optimizer_node", "_delta_hedger_node", "_compliance_check_node",
        "_episodic_reflection_node",
    ):
        monkeypatch.setattr(trading_graph, node, lambda state: {})
    monkeypatch.setattr(trading_graph, "portfolio_manager_agent", lambda state: {
        "trading_decisions": [{"symbol": "TEST", "action": "BUY", "quantity": 1, "price": 101.0}]
    })
    monkeypatch.setattr(trading_graph, "place_order", lambda *args, **kwargs: pytest.fail("order placed"))

    result = asyncio.run(trading_graph.run_trading_pipeline(["TEST"], execute_orders=False))
    pd.testing.assert_frame_equal(result["market_data"]["TEST"], frame)
    assert result["executed_orders"] == []
    assert any("Execution disabled" in entry for entry in result["agent_logs"])


def test_no_agent_bypasses_shared_provider():
    root = Path(__file__).resolve().parents[1] / "agents"
    for path in root.glob("*.py"):
        if path.name == "llm_client.py":
            continue
        content = path.read_text()
        assert "from anthropic import" not in content, path.name
        assert "ChatAnthropic(" not in content, path.name
        assert '"claude-haiku-4-5-20251001"' not in content, path.name


def test_usage_month_boundary_unknown_model_and_no_cap(llm_database):
    from database.models import LLMUsage
    from datetime import timezone
    with db.get_db() as database:
        for record_id, timestamp, amount in [
            ("previous", datetime(2026, 9, 30, 18, 29), Decimal("5")),
            ("current", datetime(2026, 9, 30, 18, 30), Decimal("100")),
        ]:
            database.add(LLMUsage(id=record_id, run_id="test", mode="economy", model="test", task="test",
                created_at=timestamp, status="success", estimated_usd=amount))
    summary = llm_client.usage_summary(datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert summary["modes"]["economy"]["requests"] == 1
    assert summary["estimated_inr"] == 100 * settings.llm_usd_inr_rate
    assert summary["hard_limit"] is None
    assert llm_client.estimate_cost("claude", "unknown-model", {"input_tokens": 1, "output_tokens": 1}, datetime.now())[0] is None


def test_run_cancellation_and_stale_heartbeat(llm_database):
    async def cancel():
        with runtime.llm_run("cancelled"):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel())
    assert runtime.current_context() is None
    with db.get_db() as database:
        assert database.query(LLMRun).one().status == "failed"
        database.add(LLMRun(id="abandoned", kind="scan", mode="claude", revision=0,
            fast_model="fast", decision_model="decision", status="running",
            started_at=datetime(2000, 1, 1), heartbeat_at=datetime(2000, 1, 1)))
    assert runtime.mode_status()["active_runs"] == []


def test_preflight_reads_selected_provider_without_mutating(llm_database, monkeypatch):
    from operations.preflight import run_preflight
    runtime.select_mode("economy", 0)
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    result = run_preflight()
    assert result.passed
    assert any("economy" in warning for warning in result.warnings)
    assert runtime.selected_profile().revision == 1