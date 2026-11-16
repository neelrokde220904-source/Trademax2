import importlib
from types import SimpleNamespace

import pytest

from operations.cash_flow_statement import parse_angelone_cash_flow_statement
from operations.control_state import ControlStateStore
from tests.test_dashboard_llm_mode import HEADERS, client
from tests.test_llm_runtime import llm_database


STATEMENT = (
    "transaction_id,timestamp,direction,amount,status\n"
    "deposit-1,2026-11-14T10:15:00+05:30,DEPOSIT,1250.00,SUCCESS\n"
    "withdraw-2,2026-11-14T11:00:00+05:30,WITHDRAWAL,250.00,COMPLETED\n"
)
BASELINE = "2026-11-13T00:00:00+00:00"


def test_statement_parser_normalizes_signed_transfers_and_hashes_source():
    digest, entries = parse_angelone_cash_flow_statement(STATEMENT)

    assert len(digest) == 64
    assert [(entry.reference, entry.amount) for entry in entries] == [
        ("deposit-1", 1250.0),
        ("withdraw-2", -250.0),
    ]
    assert entries[0].occurred_at == "2026-11-14T04:45:00+00:00"


@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("transaction_id,amount\nx,10\n", "columns"),
        (STATEMENT.replace("DEPOSIT,1250.00", "TRANSFER,1250.00"), "direction"),
        (STATEMENT.replace("10:15:00+05:30", "10:15:00"), "timezone"),
        (STATEMENT.replace("1250.00", "1250.001"), "amount"),
        (STATEMENT.replace("SUCCESS", "PENDING"), "completed"),
        (STATEMENT + "deposit-1,2026-11-14T12:00:00+05:30,DEPOSIT,5.00,SUCCESS\n", "repeats"),
    ],
)
def test_statement_parser_rejects_ambiguous_or_non_cash_rows(statement, error):
    with pytest.raises(ValueError, match=error):
        parse_angelone_cash_flow_statement(statement)


def test_statement_parser_enforces_upload_size():
    with pytest.raises(ValueError, match="1 MB"):
        parse_angelone_cash_flow_statement("x" * 1_000_001)


def test_statement_import_is_atomic_idempotent_and_preserves_overlapping_evidence(tmp_path):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    first = store.import_angelone_cash_flow_statement(
        "a" * 64,
        [("deposit-1", 1_000.0, "2026-11-14T04:45:00+00:00")],
        BASELINE,
    )
    assert first["imported"] == 1
    assert store.external_cash_flow_snapshot() == (1_000.0, 1, 0)

    overlap = store.import_angelone_cash_flow_statement(
        "b" * 64,
        [
            ("deposit-1", 1_000.0, "2026-11-14T04:45:00+00:00"),
            ("withdraw-1", -250.0, "2026-11-14T05:00:00+00:00"),
        ],
        BASELINE,
    )
    assert overlap["imported"] == 1
    assert overlap["duplicates"] == 1
    assert store.external_cash_flow_snapshot() == (750.0, 2, 0)

    replay = store.import_angelone_cash_flow_statement(
        "b" * 64,
        [
            ("deposit-1", 1_000.0, "2026-11-14T04:45:00+00:00"),
            ("withdraw-1", -250.0, "2026-11-14T05:00:00+00:00"),
        ],
        BASELINE,
    )
    assert replay["imported"] == 0
    assert replay["duplicates"] == 2
    with pytest.raises(ValueError, match="conflicts"):
        store.import_angelone_cash_flow_statement(
            "c" * 64,
            [
                ("deposit-1", 2_000.0, "2026-11-14T04:45:00+00:00"),
                ("deposit-2", 500.0, "2026-11-14T06:00:00+00:00"),
            ],
            BASELINE,
        )
    assert store.external_cash_flow_snapshot() == (750.0, 2, 0)


def test_statement_import_rejects_transfers_before_persisted_capital_baseline(tmp_path):
    from broker.portfolio_tracker import PortfolioTracker

    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = PortfolioTracker(store)
    tracker._cash_flow_baseline_at = "2026-11-15T00:00:00+00:00"

    with pytest.raises(ValueError, match="before the paper cash-flow baseline"):
        tracker.import_angelone_cash_flow_statement(
            "d" * 64,
            [("old-deposit", 1_000.0, "2026-11-14T04:45:00+00:00")],
        )
    assert store.external_cash_flow_snapshot() == (0.0, 0, 0)


def test_statement_import_rejects_possible_duplicate_of_legacy_manual_event(tmp_path):
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    store.record_external_cash_flow(
        "legacy-ref",
        1_000.0,
        "broker_statement",
        "2026-11-14T04:45:00+00:00",
    )

    with pytest.raises(ValueError, match="duplicate a prior manual event"):
        store.import_angelone_cash_flow_statement(
            "e" * 64,
            [("deposit-1", 1_000.0, "2026-11-14T04:45:00+00:00")],
            BASELINE,
        )
    assert store.external_cash_flow_snapshot() == (1_000.0, 1, 0)


def test_statement_import_endpoint_is_local_paused_paper_only_and_idempotent(
    client, monkeypatch, tmp_path,
):
    from broker.portfolio_tracker import PortfolioTracker
    from config.settings import settings

    class StaleMarkTracker(PortfolioTracker):
        def update_unrealised_pnl(self):
            return False

    dashboard = importlib.import_module("dashboard.backend.app")
    scheduler_module = importlib.import_module("scheduler.market_scheduler")
    scheduler = SimpleNamespace(is_paused=True, scan_in_progress=False)
    store = ControlStateStore(tmp_path / "controls.sqlite3")
    tracker = StaleMarkTracker(store)
    tracker._cash_flow_baseline_at = BASELINE
    monkeypatch.setattr(scheduler_module, "market_scheduler", scheduler)
    monkeypatch.setattr(dashboard, "portfolio_tracker", tracker)
    monkeypatch.setattr(settings, "paper_trading", True)

    payload = {"csv_text": STATEMENT, "reviewed": True}
    path = "/api/operations/cash-flows/angelone-statement"

    first = client.post(path, headers=HEADERS, json=payload)
    assert first.status_code == 200
    assert first.json()["imported"] == 2
    assert first.json()["reconciliation_pending"]
    assert not first.json()["portfolio_marks_fresh"]
    assert store.read("portfolio_drawdown")["cash_flow_baseline_at"] == BASELINE
    assert tracker.has_hit_portfolio_drawdown_limit()

    replay = client.post(path, headers=HEADERS, json=payload)
    assert replay.status_code == 200
    assert replay.json()["imported"] == 0
    assert replay.json()["duplicates"] == 2
    assert replay.json()["reconciliation_pending"]

    assert client.post(path, json=payload).status_code == 403
    monkeypatch.setattr(settings, "paper_trading", False)
    assert client.post(path, headers=HEADERS, json=payload).status_code == 409
    monkeypatch.setattr(settings, "paper_trading", True)
    scheduler.is_paused = False
    assert client.post(path, headers=HEADERS, json=payload).status_code == 409
    scheduler.is_paused = True
    scheduler.scan_in_progress = True
    assert client.post(path, headers=HEADERS, json=payload).status_code == 409
    scheduler.scan_in_progress = False
    assert client.post(
        path, headers=HEADERS, json={"csv_text": STATEMENT, "reviewed": False},
    ).status_code == 400
    assert client.post(
        path, headers=HEADERS, json={"csv_text": "bad", "reviewed": True},
    ).status_code == 400
