from __future__ import annotations

import csv
import io
import json
from datetime import date

import pytest

from config import universe
from config.settings import settings


def _constituent_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Symbol", "Company Name"])
    writer.writeheader()
    for index in range(50):
        writer.writerow({"Symbol": f"STOCK{index:02d}", "Company Name": f"Company {index}"})
    return output.getvalue().encode("utf-8")


def test_refresh_validates_and_caches_official_membership(tmp_path, monkeypatch):
    class Response:
        content = _constituent_csv()

        def raise_for_status(self):
            return None

    calls = []
    monkeypatch.setattr(universe, "NIFTY50_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        universe.httpx, "get",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )

    snapshot = universe.refresh_nifty50_membership(date(2026, 6, 1))

    assert len(snapshot.symbols) == 50
    assert snapshot.observed_on == date(2026, 6, 1)
    assert calls == [(universe.NIFTY50_SOURCE_URL, {
        "timeout": 20.0, "follow_redirects": True,
        "headers": {"User-Agent": "Mozilla/5.0"},
    })]
    assert universe.load_nifty50_membership(date(2026, 6, 30)) == snapshot
    with pytest.raises(universe.UniverseUnavailableError, match="on or before"):
        universe.load_nifty50_membership(date(2026, 5, 31))


def test_universe_rejects_stale_membership_and_never_uses_custom_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "NIFTY50_CACHE_DIR", tmp_path)
    (tmp_path / "2026-01-01.json").write_text(
        json.dumps({
            "observed_on": "2026-01-01",
            "source_url": universe.NIFTY50_SOURCE_URL,
            "symbols": [f"STOCK{index:02d}" for index in range(50)],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "trading_universe_mode", "nifty50")
    monkeypatch.setattr(settings, "default_watchlist", ["FALLBACK"])

    with pytest.raises(universe.UniverseUnavailableError, match="stale"):
        universe.get_active_watchlist(date(2026, 2, 2))
    assert universe.get_active_watchlist(date(2026, 1, 15))[0] == "STOCK00"


def test_custom_universe_uses_only_configured_symbols(monkeypatch):
    monkeypatch.setattr(settings, "trading_universe_mode", "custom")
    monkeypatch.setattr(settings, "default_watchlist", ["AAA", "BBB"])

    assert universe.get_active_watchlist() == ["AAA", "BBB"]


def test_refresh_rejects_incomplete_constituent_lists(tmp_path, monkeypatch):
    class Response:
        content = b"Symbol,Company Name\nONLYONE,One\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(universe, "NIFTY50_CACHE_DIR", tmp_path)
    monkeypatch.setattr(universe.httpx, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(ValueError, match="exactly 50"):
        universe.refresh_nifty50_membership(date(2026, 6, 1))
    assert not list(tmp_path.iterdir())
