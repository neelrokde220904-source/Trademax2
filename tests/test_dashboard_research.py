"""Research controls remain local-only and require cost acknowledgement."""

from tests.test_dashboard_llm_mode import HEADERS, client
from tests.test_llm_runtime import llm_database


def test_research_controls_require_trusted_origin_and_cost_acknowledgement(client, monkeypatch):
    from research import comparison

    body = {"snapshot_ids": ["one", "two"], "acknowledge_model_cost": True}
    assert client.post("/api/research/compare", json=body).status_code == 403
    assert client.post("/api/research/settle", json={}).status_code == 403
    assert client.post("/api/research/compare", headers=HEADERS, json={
        **body, "acknowledge_model_cost": False,
    }).status_code == 400
    monkeypatch.setattr(comparison, "validate_comparison", lambda _ids: (_ for _ in ()).throw(ValueError("snapshot incomplete")))
    response = client.post("/api/research/compare", headers=HEADERS, json=body)
    assert response.status_code == 409
    assert "incomplete" in response.json()["detail"]
    summary = client.get("/api/research")
    assert summary.status_code == 200
    assert summary.json()["forecast_summary"]["forecasts"] == []


def test_dashboard_settlement_returns_data_errors(client, monkeypatch):
    from research import tracking

    monkeypatch.setattr(tracking.settings, "paper_trading", True)
    monkeypatch.setattr(tracking, "settle_forecasts", lambda: {
        "settled": 0, "pending": 1, "errors": ["NSE data unavailable"],
    })
    response = client.post("/api/research/settle", headers=HEADERS, json={})
    assert response.status_code == 200
    assert response.json()["errors"] == ["NSE data unavailable"]
