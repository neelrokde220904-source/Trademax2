"""T22 calibrated signal gate and T23 half-Kelly reducer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from config.settings import settings
from database import db
from database.models import ClosedTrade, ResearchForecast, Trade
from models.signal_gate import gate as gate_module
from models.signal_gate.gate import (
    FEATURE_NAMES, MIN_TOTAL_SAMPLES, GateUnavailable, Sample, SignalGate, break_even_probability,
    brier_score, chronological_splits, log_loss_score, reliability_bins, train_gate,
)
from risk.position_sizer import PositionSizer

BASE = datetime(2026, 1, 5, 10, 0)


def make_samples(n=800, seed=1, shift=True):
    """Outcomes depend on rr_ratio; the cal/test blocks drift away from the training block."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        rr = float(rng.uniform(0.5, 4.0))
        confidence = float(rng.uniform(40, 90))
        z = 1.6 * (rr - 2.0)
        if shift and i >= int(n * 0.6):
            z = 0.5 * (rr - 2.0) - 0.6  # the world changed after training
        win = rng.random() < 1 / (1 + np.exp(-z))
        features = {
            "confidence": confidence, "rr_ratio": rr, "risk_pct": float(rng.uniform(0.005, 0.03)),
            "atr_pct": float(rng.uniform(0.005, 0.02)), "entry_hour": float(rng.integers(10, 15)),
            "is_short": 0.0,
        }
        r = rr if win else -1.0
        samples.append(Sample(
            when=BASE + timedelta(hours=i), features=features, r_net=r - 0.1, r_gross=r, cost_r=0.1,
        ))
    return samples


@pytest.fixture
def trained(tmp_path):
    path = tmp_path / "gate_model.json"
    report = train_gate(make_samples(), path)
    return path, report


# ── maths ────────────────────────────────────────────────────────────────────

def test_break_even_probability_matches_the_formula():
    assert break_even_probability([2.0, 2.0], [-1.0, -1.0], 0.1) == pytest.approx((1.0 + 0.1) / 3.0 + 0.02)
    with pytest.raises(ValueError):
        break_even_probability([], [-1.0], 0.1)


def test_scores_and_reliability():
    y = np.array([1, 0, 1, 0], dtype=float)
    assert brier_score(np.array([1, 0, 1, 0.0]), y) == 0.0
    assert brier_score(np.full(4, 0.5), y) == 0.25
    assert log_loss_score(np.full(4, 0.5), y) == pytest.approx(np.log(2))
    p = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    outcomes = np.concatenate([np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]])
    table, ece = reliability_bins(p, outcomes)
    assert ece == pytest.approx(0.0) and [row["count"] for row in table] == [100, 100]
    _, bad = reliability_bins(np.full(100, 0.9), np.zeros(100))
    assert bad == pytest.approx(0.9)
    _, edge = reliability_bins(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert edge == 0.0  # p = 1.0 lands in the top bin


def test_splits_are_chronological_with_a_purge_gap():
    train, cal, test = chronological_splits(1000)
    assert train.stop + gate_module.PURGE_ROWS == cal.start
    assert cal.stop + gate_module.PURGE_ROWS == test.start
    assert train.start == 0 and test.stop == 1000 and train.stop < cal.stop < test.stop


# ── training ─────────────────────────────────────────────────────────────────

def test_few_outcomes_report_insufficient_samples(tmp_path):
    path = tmp_path / "m.json"
    report = train_gate(make_samples(MIN_TOTAL_SAMPLES - 1), path)
    assert report["status"] == "insufficient_samples" and "insufficient samples" in report["message"]
    assert report["calibration_passed"] is False and not path.exists()


def test_todays_data_reports_insufficient_samples():
    report = gate_module.gate_report()
    assert report["status"] == "insufficient_samples" and report["samples"] < MIN_TOTAL_SAMPLES
    assert "insufficient samples" in report["message"]
    assert gate_module.conviction_report()["status"] == "insufficient_samples"


def test_calibration_improves_the_brier_score_on_a_drifted_synthetic_set(trained):
    _path, report = trained
    assert report["status"] == "trained" and report["calibration_method"] == "platt"
    assert report["brier"] < report["raw_brier"]
    assert {"brier", "log_loss", "ece", "reliability_bins", "p_star"} <= set(report)
    assert report["train_rows"] + report["calibration_rows"] + report["test_rows"] <= 800 - 2 * gate_module.PURGE_ROWS


def test_isotonic_is_used_only_with_enough_calibration_rows(tmp_path):
    report = train_gate(make_samples(3000, seed=4), tmp_path / "iso.json")
    assert report["calibration_rows"] >= gate_module.ISOTONIC_MIN_ROWS
    assert report["calibration_method"] == "isotonic"


def test_the_artefact_is_json_with_a_checksum_and_no_pickle(trained):
    path, _report = trained
    payload = json.loads(path.read_text())
    assert payload["features"] == list(FEATURE_NAMES) and len(payload["checksum"]) == 64
    assert payload["version"] == 1 and "calibration" in payload
    assert "pickle" not in Path(gate_module.__file__).read_text().replace("pickle.", "").lower().replace(
        "no pickle", "").replace("nothing here loads a pickle", "")


# ── scoring and fail-closed paths ────────────────────────────────────────────

SIGNAL = {"price": 100.0, "stop_loss": 98.0, "target": 106.0, "atr": 1.5, "confidence": 70, "action": "BUY"}


def test_a_good_model_scores_and_compares_with_p_star(trained):
    path, report = trained
    gate = SignalGate(path)
    ok, reason = gate.should_execute(SIGNAL, BASE)
    assert isinstance(ok, bool) and "p=" in reason
    weak = {**SIGNAL, "target": 100.4}  # rr 0.2: poor payoff
    strong = {**SIGNAL, "target": 108.0}
    assert gate.probability(strong, BASE) > gate.probability(weak, BASE)
    assert gate.calibration_passed == report["calibration_passed"]
    assert gate.payoff_inputs(SIGNAL, BASE)[1] == pytest.approx(3.0)


def test_every_failure_mode_blocks(trained, tmp_path):
    path, _ = trained
    assert SignalGate(tmp_path / "absent.json").should_execute(SIGNAL, BASE) == (
        False, "signal gate unavailable: model_missing",
    )
    tampered = json.loads(path.read_text())
    tampered["coef"][0] += 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered))
    assert SignalGate(bad).should_execute(SIGNAL, BASE) == (False, "signal gate unavailable: bad_checksum")

    mismatched = json.loads(path.read_text())
    mismatched["features"] = ["confidence"]
    mismatched["checksum"] = gate_module._checksum(mismatched)
    other = tmp_path / "feat.json"
    other.write_text(json.dumps(mismatched))
    assert SignalGate(other).should_execute(SIGNAL, BASE)[1].endswith("feature_mismatch")

    few = json.loads(path.read_text())
    few["trained_rows"] = 10
    few["checksum"] = gate_module._checksum(few)
    small = tmp_path / "small.json"
    small.write_text(json.dumps(few))
    assert SignalGate(small).should_execute(SIGNAL, BASE)[1].endswith("too_few_training_samples")

    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json")
    assert SignalGate(garbage).should_execute(SIGNAL, BASE)[0] is False

    gate = SignalGate(path)
    assert gate.should_execute({"price": 100.0}, BASE)[0] is False  # missing fields
    assert gate.should_execute({**SIGNAL, "stop_loss": 100.0}, BASE)[0] is False  # zero risk
    assert gate.payoff_inputs({"price": 100.0}, BASE) is None
    assert SignalGate(garbage).calibration_passed is False


def test_a_scoring_exception_blocks_instead_of_passing(trained, monkeypatch):
    path, _ = trained
    gate = SignalGate(path)
    monkeypatch.setattr(gate_module, "apply_calibration", lambda *_a: (_ for _ in ()).throw(RuntimeError("boom")))
    ok, reason = gate.should_execute(SIGNAL, BASE)
    assert ok is False and "error" in reason


# ── dataset from settled trades ──────────────────────────────────────────────

def test_samples_are_net_of_cost_r_multiples(tmp_path):
    with db.get_db() as session:
        entry = Trade(timestamp=datetime(2026, 11, 2, 10, 0), symbol="X", action="BUY", quantity=10, price=100.0,
                      stop_loss=98.0, target=106.0, atr=1.5, confidence=70, holding_period="INTRADAY",
                      status="EXECUTED", is_paper=True, order_id="e")
        exit_ = Trade(timestamp=datetime(2026, 11, 2, 11, 0), symbol="X", action="SELL", quantity=10, price=104.0,
                      holding_period="INTRADAY", status="EXECUTED", is_paper=True, order_id="x")
        session.add_all([entry, exit_])
        session.flush()
        session.add(ClosedTrade(
            entry_order_id=entry.id, exit_order_id=exit_.id, symbol="X", action="BUY", quantity=10,
            entry_price=100.0, exit_price=104.0, entry_timestamp=entry.timestamp, exit_timestamp=exit_.timestamp,
            gross_pnl=40.0, strategy="CompositeTrend", holding_period="INTRADAY", is_paper=True,
        ))
    [sample] = gate_module.load_closed_trade_samples()
    assert sample.r_gross == pytest.approx(40.0 / 20.0)  # risk = 2 x 10
    assert 0 < sample.cost_r < 1.0 and sample.r_net == pytest.approx(sample.r_gross - sample.cost_r)
    assert sample.features["rr_ratio"] == pytest.approx(3.0) and sample.features["entry_hour"] == 10.0


def test_conviction_report_measures_settled_forecasts_only():
    with db.get_db() as session:
        for index in range(MIN_TOTAL_SAMPLES):
            session.add(ResearchForecast(
                id=f"f{index}", run_id=f"r{index}", observed_at=datetime(2026, 11, 1), agent="technical",
                symbol="X", direction="UP", confidence=float(50 + index % 40), status="settled",
                direction_correct=index % 2 == 0,
            ))
        session.add(ResearchForecast(
            id="pending", run_id="r", observed_at=datetime(2026, 11, 1), agent="technical", symbol="Y",
            direction="UP", confidence=80.0, status="pending",
        ))
    report = gate_module.conviction_report()
    assert report["status"] == "measured" and report["samples"] == MIN_TOTAL_SAMPLES
    assert 0.0 <= report["ece"] <= 1.0


# ── off by default; wiring ───────────────────────────────────────────────────

def test_the_gate_is_off_by_default_and_never_consulted(monkeypatch, tmp_path):
    from config.settings import Settings
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state

    assert Settings.model_fields["signal_gate_enabled"].default is False and settings.signal_gate_enabled is False
    monkeypatch.setattr(SignalGate, "should_execute", lambda *_a, **_k: pytest.fail("gate consulted while off"))
    stub_persisted_review_verification(monkeypatch)
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    assert len(calls) == 0 and graph._order_router_node(state)["executed_orders"] and len(calls) == 1


def test_enabled_without_a_model_blocks_entries_but_not_exits(monkeypatch, tmp_path):
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state
    from risk import stop_loss_manager as slm

    stub_persisted_review_verification(monkeypatch)
    monkeypatch.setattr(settings, "signal_gate_enabled", True)
    monkeypatch.setattr(gate_module, "signal_gate", SignalGate(tmp_path / "absent.json"))
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    result = graph._order_router_node(state)
    assert calls == [] and any("Signal gate blocked OK" in e and "model_missing" in e for e in result["errors"])
    monkeypatch.setattr(slm.stop_loss_manager, "pending_exits", lambda: {})
    exit_state = {
        "execution_enabled": True,
        "current_positions": {"HELD": {"quantity": 2, "entry_trade_ids": [1], "holding_period": "INTRADAY"}},
        "trading_decisions": [{"symbol": "HELD", "action": "SELL", "quantity": 2, "price": 100,
                               "holding_period": "INTRADAY"}],
    }
    graph._order_router_node(exit_state)
    assert [(c["symbol"], c["is_exit"]) for c in calls] == [("HELD", True)]


def test_kelly_inputs_reach_the_sizer_only_when_the_gate_passed_calibration(monkeypatch, tmp_path):
    from risk import position_sizer as sizer_module
    from tests.sentiment_fixtures import stub_persisted_review_verification
    from tests.test_candidate_contract import _entry_state

    stub_persisted_review_verification(monkeypatch)
    monkeypatch.setattr(settings, "signal_gate_enabled", True)
    seen = []
    real = sizer_module.position_sizer.size

    def spy(signal, **kwargs):
        seen.append({k: kwargs.get(k) for k in ("calibrated_p", "payoff_b")})
        return real(signal, **kwargs)

    monkeypatch.setattr(sizer_module.position_sizer, "size", spy)

    class FakeGate:
        calibration_passed = True
        should_execute = staticmethod(lambda *_a, **_k: (True, "ok"))
        payoff_inputs = staticmethod(lambda *_a, **_k: (0.6, 2.0))

    monkeypatch.setattr(gate_module, "signal_gate", FakeGate())
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    graph._order_router_node(state)
    assert seen and seen[0] == {"calibrated_p": 0.6, "payoff_b": 2.0}

    seen.clear()
    FakeGate.calibration_passed = False
    graph, state, calls = _entry_state(monkeypatch, tmp_path)
    graph._order_router_node(state)
    assert seen and seen[0] == {"calibrated_p": None, "payoff_b": None}


# ── T23 half-Kelly ───────────────────────────────────────────────────────────

SIZING_SIGNAL = {"symbol": "X", "action": "BUY", "price": 100.0, "stop_loss": 98.0, "target": 106.0,
                 "holding_period": "INTRADAY"}


def size(**kwargs):
    return PositionSizer().size(
        SIZING_SIGNAL, portfolio_value=100_000, available_capital=100_000, current_exposure=0.0, **kwargs,
    )


def test_kelly_formula():
    assert PositionSizer.kelly_fraction(0.6, 2.0) == pytest.approx(0.4)
    assert PositionSizer.kelly_fraction(0.3, 1.0) == pytest.approx(-0.4)


def test_half_kelly_never_increases_the_quantity():
    base, reason = size()
    assert base > 0 and reason is None
    for p in (0.4, 0.5, 0.6, 0.8, 0.95):
        quantity, why = size(calibrated_p=p, payoff_b=3.0)
        assert quantity <= base, (p, quantity, base)
        assert why is None or why.startswith("kelly")


def test_a_small_edge_shrinks_and_a_large_edge_keeps_the_base_size():
    base, _ = size()
    # The 10% notional cap (99 shares) binds first unless half-Kelly risk is smaller: f must be tiny.
    small, _ = size(calibrated_p=0.2515, payoff_b=3.0)  # f = 0.2515 - 0.7485/3 = 0.002
    assert 0 < small < base
    large, _ = size(calibrated_p=0.95, payoff_b=3.0)
    assert large == base
    assert size(calibrated_p=0.2515, payoff_b=3.0)[0] > size(calibrated_p=0.2510, payoff_b=3.0)[0] > 0


@pytest.mark.parametrize("p,b", [(0.25, 3.0), (0.2, 3.0), (0.5, 1.0), (0.3, 2.0)])  # f <= 0 in each case
def test_non_positive_kelly_blocks_the_entry(p, b):
    assert size(calibrated_p=p, payoff_b=b) == (0, "kelly_non_positive")


@pytest.mark.parametrize("p,b", [(0.0, 3.0), (1.0, 3.0), (float("nan"), 3.0), (0.6, 0.0), (0.6, -1.0), (None, 3.0)])
def test_invalid_kelly_inputs_block_the_entry(p, b):
    quantity, reason = size(calibrated_p=p, payoff_b=b)
    assert quantity == 0 and reason == "kelly_inputs_invalid"


def test_default_sizing_is_unchanged_without_kelly_inputs():
    a, _ = size()
    b, _ = size(calibrated_p=None, payoff_b=None)
    assert a == b
