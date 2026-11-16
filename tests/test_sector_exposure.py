from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from risk.sector_exposure import (
    SECTOR_SOURCE,
    SectorMetadataUnavailable,
    sector_exposure_by_name,
    sector_limit_block_reason,
    validate_candidate_sector,
)


def _classification(**overrides):
    value = {
        "sector": "Financial Services",
        "sector_source": SECTOR_SOURCE,
        "sector_observed_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**value, **overrides}


def test_candidate_sector_contract_normalizes_name_and_validates_provenance():
    result = validate_candidate_sector(" test ", _classification())
    assert result["symbol"] == "TEST"
    assert result["sector"] == "financial services"
    assert result["sector_source"] == SECTOR_SOURCE


def test_fundamental_fetch_records_sector_source_and_observation_time(monkeypatch):
    from agents import fundamental_agent

    monkeypatch.setattr(
        fundamental_agent.yf, "Ticker",
        lambda _symbol: SimpleNamespace(info={"sector": "Industrials"}),
    )
    result = fundamental_agent.FundamentalAgent._fetch_fundamentals(object(), "TEST")

    assert result["sector"] == "Industrials"
    assert result["sector_source"] == SECTOR_SOURCE
    assert datetime.fromisoformat(result["sector_observed_at"]).tzinfo is not None


@pytest.mark.parametrize("data", [
    None,
    _classification(sector="Unknown"),
    _classification(sector_source="unverified"),
    _classification(sector_observed_at="2026-11-14T10:00:00"),
    _classification(sector_observed_at=(datetime.now(timezone.utc) - timedelta(days=8)).isoformat()),
])
def test_candidate_sector_contract_fails_closed(data):
    with pytest.raises(SectorMetadataUnavailable):
        validate_candidate_sector("TEST", data)


def test_sector_exposure_uses_gross_signed_notional_and_fails_closed(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "max_sector_exposure_pct", 0.25)
    positions = {
        "LONG": {
            "quantity": 2, "ltp": 100, "sector": "financial services",
            "sector_source": SECTOR_SOURCE,
            "sector_observed_at": datetime.now(timezone.utc).isoformat(),
        },
        "SHORT": {
            "quantity": -1, "ltp": 100, "sector": "Financial Services",
            "sector_source": SECTOR_SOURCE,
            "sector_observed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    assert sector_exposure_by_name(positions, 1_000) == {"financial services": pytest.approx(0.3)}
    assert "exceeds" in sector_limit_block_reason(positions, 1_000)

    with pytest.raises(SectorMetadataUnavailable, match="classification unavailable"):
        sector_exposure_by_name(
            {"LEGACY": {"quantity": 1, "ltp": 100}},
            1_000,
        )
