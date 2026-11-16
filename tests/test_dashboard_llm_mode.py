"""Local-only dashboard control contracts, without provider or broker calls."""

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from config.llm_runtime import llm_run
from dashboard.backend.app import app
from tests.test_llm_runtime import llm_database

HEADERS = {"Origin": "http://localhost:3000", "X-Dashboard-Control": "1"}


@pytest.fixture
def client(llm_database):
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as browser:
        yield browser


def test_mode_endpoints_and_paper_mode_unchanged(client):
    paper = settings.paper_trading
    initial = client.get("/api/llm-mode").json()
    assert initial["selected_mode"] == "claude"
    with llm_run("scan"):
        switched = client.put("/api/llm-mode", json={"mode": "economy", "expected_revision": 0}, headers=HEADERS)
        assert switched.status_code == 200
        assert switched.json()["pending"]
        assert switched.json()["active_runs"][0]["mode"] == "claude"
    assert client.get("/api/llm-mode").json()["selected_mode"] == "economy"
    assert settings.paper_trading == paper
    assert client.put("/api/llm-mode", json={"mode": "claude", "expected_revision": 0}, headers=HEADERS).status_code == 409
    assert client.put("/api/llm-mode", json={"mode": "claude", "expected_revision": 1}, headers=HEADERS).status_code == 200
    usage = client.get("/api/llm-usage")
    assert usage.status_code == 200
    assert usage.json()["hard_limit"] is None
    assert "test-claude" not in str(initial)
    assert "test-economy" not in str(initial)


@pytest.mark.parametrize("headers", [{}, {"Origin": "https://evil.example", "X-Dashboard-Control": "1"},
                                     {"Origin": "null", "X-Dashboard-Control": "1"},
                                     {"Origin": "http://localhost:3000"}])
def test_untrusted_mutation_rejected(client, headers):
    result = client.put("/api/llm-mode", json={"mode": "economy", "expected_revision": 0}, headers=headers)
    assert result.status_code == 403
    assert client.get("/api/llm-mode").json()["revision"] == 0


def test_invalid_payload_and_unconfigured_provider(client, monkeypatch):
    assert client.put("/api/llm-mode", json={"mode": "live", "expected_revision": 0}, headers=HEADERS).status_code == 422
    assert client.put("/api/llm-mode", json={"mode": "economy", "expected_revision": 0, "api_key": "no"}, headers=HEADERS).status_code == 422
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    assert client.put("/api/llm-mode", json={"mode": "economy", "expected_revision": 0}, headers=HEADERS).status_code == 409
    assert client.get("/api/llm-mode").json()["selected_mode"] == "claude"


def test_dns_rebinding_host_rejected(client):
    assert client.get("/api/llm-mode", headers={"Host": "evil.example"}).status_code == 400