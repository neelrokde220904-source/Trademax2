"""Research commands, confirmations, and failure statuses without provider calls."""

import json
import sys

import pytest

import main
from config.settings import settings
from research import comparison, tracking


@pytest.fixture
def command(monkeypatch):
    monkeypatch.setattr(main, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(settings, "paper_trading", True)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["main.py", *args])
        return main.main()

    return run


@pytest.mark.parametrize(
    "flag,function,result",
    [
        ("--research-snapshots", "list_snapshots", [{"id": "scan-1", "status": "complete"}]),
        ("--forecasts", "forecast_summary", {"counts": {"pending": 1}, "forecasts": []}),
    ],
)
def test_research_read_commands_print_json(command, monkeypatch, capsys, flag, function, result):
    monkeypatch.setattr(tracking, function, lambda: result)
    command(flag)
    assert json.loads(capsys.readouterr().out) == result


def test_paper_comparison_requires_explicit_cost_confirmation(command, monkeypatch, capsys):
    monkeypatch.setattr(comparison, "run_comparison", lambda _ids: pytest.fail("unconfirmed comparison ran"))
    with pytest.raises(SystemExit) as exc:
        command("--paper-compare", "one", "two")
    assert exc.value.code == 2
    assert "--confirm-research-cost" in capsys.readouterr().err


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 1)])
def test_paper_comparison_prints_result_and_uses_its_exit_status(command, monkeypatch, capsys, status, exit_code):
    seen = []

    def compare(ids):
        seen.append(ids)
        return {"id": "result", "status": status, "variants": []}

    monkeypatch.setattr(comparison, "run_comparison", compare)
    with pytest.raises(SystemExit) as exc:
        command("--paper-compare", "one", "two", "--confirm-research-cost")
    assert exc.value.code == exit_code
    assert seen == [["one", "two"]]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_settlement_errors_produce_a_nonzero_exit(command, monkeypatch, capsys):
    monkeypatch.setattr(tracking, "settle_forecasts", lambda: {"settled": 0, "errors": ["price unavailable"]})
    with pytest.raises(SystemExit) as exc:
        command("--settle-forecasts")
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["price unavailable"]


def test_settlement_rejects_live_mode(command, monkeypatch):
    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(tracking, "settle_forecasts", lambda: pytest.fail("live settlement ran"))
    with pytest.raises(SystemExit) as exc:
        command("--settle-forecasts")
    assert exc.value.code == 2


def test_atr_comparison_does_not_require_model_cost_confirmation(command, monkeypatch, capsys):
    from research import atr_exit_comparison

    seen = []
    monkeypatch.setattr(
        atr_exit_comparison, "run_atr_exit_comparison",
        lambda ids: seen.append(ids) or {"status": "completed", "variants": []},
    )
    command("--atr-exit-compare", "one", "two")
    assert seen == [["one", "two"]]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_atr_comparison_invalid_input_exits_nonzero(command, monkeypatch):
    from research import atr_exit_comparison

    def reject(_ids):
        raise ValueError("incomplete captured inputs")

    monkeypatch.setattr(atr_exit_comparison, "run_atr_exit_comparison", reject)
    with pytest.raises(SystemExit) as exc:
        command("--atr-exit-compare", "one", "two")
    assert exc.value.code == 1
