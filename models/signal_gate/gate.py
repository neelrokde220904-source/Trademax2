"""Calibrated signal gate: logistic regression with Platt scaling, JSON artefact, fails closed.

The label is the *net-of-cost R outcome* of a settled paper trade under the exit policy it
actually ran with (``ClosedTrade`` joined to its entry order). Splits are chronological with a
purge gap, probabilities are calibrated on a held-out block and judged on a later test block.

Nothing here loads a pickle. The artefact is JSON (coefficients and calibration) plus a SHA-256
checksum, and every failure mode (no model, bad checksum, bad features, an exception, too few
training samples) returns ``(False, reason)``: the gate only ever blocks, it never waves a trade
through by accident. It is wired to entries only when ``SIGNAL_GATE_ENABLED`` is true.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

MODEL_PATH = Path(__file__).parent / "gate_model.json"
FEATURE_NAMES = ("confidence", "rr_ratio", "risk_pct", "atr_pct", "entry_hour", "is_short")
MIN_TOTAL_SAMPLES = 300          # below this the report says "insufficient samples"
MIN_CALIBRATION_ROWS = 60
MIN_TEST_ROWS = 60
ISOTONIC_MIN_ROWS = 500
PURGE_ROWS = 5
P_STAR_MARGIN = 0.02
CALIBRATION_ECE_LIMIT = 0.10
RELIABILITY_BINS = 10


class GateUnavailable(RuntimeError):
    """The gate cannot score a signal (the caller must treat this as a block)."""


# ── features ─────────────────────────────────────────────────────────────────

def features_from_fields(
    *, price: float, stop: float, target: float, atr: float | None, confidence: float | None,
    action: str, hour: int,
) -> dict[str, float]:
    risk = abs(price - stop)
    if not all(math.isfinite(v) and v > 0 for v in (price, risk)):
        raise ValueError("price and stop distance must be positive and finite")
    reward = abs(target - price)
    return {
        "confidence": float(confidence if confidence is not None else 50.0),
        "rr_ratio": reward / risk,
        "risk_pct": risk / price,
        "atr_pct": float(atr) / price if atr else 0.0,
        "entry_hour": float(hour),
        "is_short": 1.0 if str(action).upper() == "SELL" else 0.0,
    }


def extract_features(signal: dict[str, Any], now: datetime | None = None) -> dict[str, float]:
    moment = now or datetime.now(timezone.utc).astimezone()
    return features_from_fields(
        price=float(signal["price"]), stop=float(signal["stop_loss"]), target=float(signal["target"]),
        atr=signal.get("atr"), confidence=signal.get("confidence"), action=str(signal["action"]),
        hour=moment.hour,
    )


# ── dataset ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sample:
    when: datetime
    features: dict[str, float]
    r_net: float
    r_gross: float
    cost_r: float


def load_closed_trade_samples() -> list[Sample]:
    """Settled paper trades with stop distances, in entry order."""
    from broker.portfolio_tracker import PortfolioTracker
    from database.db import get_db
    from database.models import ClosedTrade, Trade

    samples: list[Sample] = []
    with get_db() as db:
        rows = (
            db.query(ClosedTrade, Trade)
            .join(Trade, Trade.id == ClosedTrade.entry_order_id)
            .filter(ClosedTrade.is_paper == True)
            .order_by(ClosedTrade.entry_timestamp)
            .all()
        )
        exits = {t.id: t for t in db.query(Trade).filter(Trade.status == "EXECUTED").all()}
        for closed, entry in rows:
            try:
                if not entry.stop_loss or not entry.target or not entry.price:
                    continue
                risk_per_share = abs(float(entry.price) - float(entry.stop_loss))
                if risk_per_share <= 0:
                    continue
                exit_trade = exits.get(closed.exit_order_id)
                quantity = int(closed.quantity)
                cost = PortfolioTracker._trade_cost(entry) * quantity / max(int(entry.quantity), 1)
                if exit_trade is not None:
                    cost += PortfolioTracker._trade_cost(exit_trade) * quantity / max(int(exit_trade.quantity), 1)
                risk = risk_per_share * quantity
                features = features_from_fields(
                    price=float(entry.price), stop=float(entry.stop_loss), target=float(entry.target),
                    atr=entry.atr, confidence=entry.confidence, action=entry.action,
                    hour=entry.timestamp.hour,
                )
                samples.append(Sample(
                    when=closed.entry_timestamp, features=features,
                    r_net=(float(closed.gross_pnl) - cost) / risk, r_gross=float(closed.gross_pnl) / risk,
                    cost_r=cost / risk,
                ))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return samples


# ── maths ────────────────────────────────────────────────────────────────────

def _sigmoid(z: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(
    p: np.ndarray, y: np.ndarray, bins: int = RELIABILITY_BINS,
) -> tuple[list[dict[str, Any]], float]:
    """Equal-width reliability table and the expected calibration error (ECE)."""
    index = np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if not count:
            continue
        mean_p, observed = float(p[mask].mean()), float(y[mask].mean())
        ece += count / len(p) * abs(mean_p - observed)
        table.append({"low": round(b / bins, 2), "high": round((b + 1) / bins, 2), "count": count,
                      "mean_predicted": mean_p, "observed_rate": observed})
    return table, float(ece)


def break_even_probability(wins_r: list[float], losses_r: list[float], cost_r: float) -> float:
    """p* = (L + C) / (W + L) + margin: the win rate that makes net expectancy zero, plus a margin."""
    if not wins_r or not losses_r:
        raise ValueError("need both winning and losing trades to set a threshold")
    average_win, average_loss = float(np.mean(wins_r)), float(np.mean(np.abs(losses_r)))
    return min(0.99, (average_loss + cost_r) / (average_win + average_loss) + P_STAR_MARGIN)


def chronological_splits(n: int) -> tuple[slice, slice, slice]:
    """60/20/20 blocks in time order with ``PURGE_ROWS`` dropped between blocks."""
    train_end = int(n * 0.6)
    cal_start = train_end + PURGE_ROWS
    cal_end = int(n * 0.8)
    test_start = cal_end + PURGE_ROWS
    return slice(0, train_end), slice(cal_start, cal_end), slice(test_start, n)


def apply_calibration(raw: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    if calibration["method"] == "platt":
        return np.asarray(_sigmoid(calibration["a"] * _logit(raw) + calibration["b"]))
    if calibration["method"] == "isotonic":
        return np.interp(raw, calibration["x"], calibration["y"])
    raise GateUnavailable("unknown calibration method")


# ── training ─────────────────────────────────────────────────────────────────

def _checksum(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "checksum"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def train_gate(samples: list[Sample], path: Path | None = None, *, write: bool = True) -> dict[str, Any]:
    """Fit, calibrate and test the gate. Returns a report; the artefact is written only when usable."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    n = len(samples)
    report: dict[str, Any] = {
        "samples": n, "min_samples": MIN_TOTAL_SAMPLES, "status": "insufficient_samples",
        "calibration_passed": False,
    }
    if n < MIN_TOTAL_SAMPLES:
        report["message"] = f"insufficient samples: {n} settled outcomes, at least {MIN_TOTAL_SAMPLES} are needed"
        return report
    samples = sorted(samples, key=lambda s: s.when)
    train_s, cal_s, test_s = chronological_splits(n)
    x = np.array([[s.features[name] for name in FEATURE_NAMES] for s in samples], dtype=float)
    y = np.array([1.0 if s.r_net > 0 else 0.0 for s in samples])
    x_train, y_train = x[train_s], y[train_s]
    x_cal, y_cal = x[cal_s], y[cal_s]
    x_test, y_test = x[test_s], y[test_s]
    if min(len(y_cal), len(y_test)) < MIN_CALIBRATION_ROWS or len(set(y_train)) < 2 or len(set(y_cal)) < 2:
        report["message"] = "insufficient samples in a calibration or test block, or only one outcome class"
        return report

    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    model = LogisticRegression(C=1.0, max_iter=1000).fit((x_train - mean) / std, y_train)

    def raw_probability(rows: np.ndarray) -> np.ndarray:
        return np.asarray(_sigmoid(((rows - mean) / std) @ model.coef_[0] + model.intercept_[0]))

    raw_cal = raw_probability(x_cal)
    if len(y_cal) >= ISOTONIC_MIN_ROWS:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
        calibration: dict[str, Any] = {
            "method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
        }
    else:
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_cal).reshape(-1, 1), y_cal)
        calibration = {"method": "platt", "a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0])}

    p_test = apply_calibration(raw_probability(x_test), calibration)
    raw_brier = brier_score(raw_probability(x_test), y_test)
    brier = brier_score(p_test, y_test)
    baseline_brier = brier_score(np.full_like(y_test, float(y_train.mean())), y_test)
    bins, ece = reliability_bins(p_test, y_test)
    train_samples = samples[train_s]
    wins = [s.r_gross for s in train_samples if s.r_net > 0]
    losses = [s.r_gross for s in train_samples if s.r_net <= 0]
    try:
        p_star = break_even_probability(wins, losses, float(np.mean([s.cost_r for s in train_samples])))
    except ValueError as exc:
        report["message"] = str(exc)
        return report
    passed = brier < baseline_brier and ece <= CALIBRATION_ECE_LIMIT and len(y_test) >= MIN_TEST_ROWS
    report.update({
        "status": "trained", "calibration_passed": bool(passed),
        "brier": brier, "raw_brier": raw_brier, "baseline_brier": baseline_brier, "log_loss": log_loss_score(p_test, y_test),
        "ece": ece, "reliability_bins": bins, "p_star": p_star, "calibration_method": calibration["method"],
        "train_rows": int(len(y_train)), "calibration_rows": int(len(y_cal)), "test_rows": int(len(y_test)),
    })
    if write:
        artefact = {
            "version": 1, "features": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(),
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "calibration": calibration, "p_star": p_star, "trained_rows": n,
            "calibration_passed": bool(passed), "metrics": {"brier": brier, "ece": ece},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        artefact["checksum"] = _checksum(artefact)
        target = path or MODEL_PATH
        target.write_text(json.dumps(artefact, indent=2))
        report["artefact"] = str(target)
    logger.info("[SignalGate] Evaluated {} outcomes; calibration_passed={}", n, passed)
    return report


# ── scoring ──────────────────────────────────────────────────────────────────

class SignalGate:
    """Scores one signal; every failure is a block."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or MODEL_PATH
        self._artefact: dict[str, Any] | None = None
        self._error: str | None = None
        self._loaded = False

    def reload(self) -> None:
        self._artefact, self._error, self._loaded = None, None, False

    def _load(self) -> dict[str, Any]:
        if self._loaded:
            if self._artefact is None:
                raise GateUnavailable(self._error or "model unavailable")
            return self._artefact
        self._loaded = True
        try:
            if not self._path.exists():
                raise GateUnavailable("model_missing")
            payload = json.loads(self._path.read_text())
            if payload.get("checksum") != _checksum(payload):
                raise GateUnavailable("bad_checksum")
            if payload.get("features") != list(FEATURE_NAMES):
                raise GateUnavailable("feature_mismatch")
            if int(payload.get("trained_rows", 0)) < MIN_TOTAL_SAMPLES:
                raise GateUnavailable("too_few_training_samples")
            self._artefact = payload
            return payload
        except GateUnavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:
            self._error = f"load_error:{type(exc).__name__}"
            raise GateUnavailable(self._error) from exc

    @property
    def calibration_passed(self) -> bool:
        try:
            return self._load().get("calibration_passed") is True
        except GateUnavailable:
            return False

    def probability(self, signal: dict[str, Any], now: datetime | None = None) -> float:
        artefact = self._load()
        features = extract_features(signal, now)
        row = np.array([features[name] for name in artefact["features"]], dtype=float)
        z = float(
            ((row - np.array(artefact["mean"])) / np.array(artefact["std"])) @ np.array(artefact["coef"])
            + artefact["intercept"]
        )
        probability = float(apply_calibration(np.array([_sigmoid(z)]), artefact["calibration"])[0])
        if not math.isfinite(probability):
            raise GateUnavailable("non_finite_probability")
        return probability

    def should_execute(self, signal: dict[str, Any], now: datetime | None = None) -> tuple[bool, str]:
        try:
            probability = self.probability(signal, now)
            threshold = float(self._load()["p_star"])
        except GateUnavailable as exc:
            return False, f"signal gate unavailable: {exc}"
        except Exception as exc:
            return False, f"signal gate error: {type(exc).__name__}"
        if probability >= threshold:
            return True, f"calibrated p={probability:.3f} >= p*={threshold:.3f}"
        return False, f"calibrated p={probability:.3f} < p*={threshold:.3f}"

    def payoff_inputs(self, signal: dict[str, Any], now: datetime | None = None) -> tuple[float, float] | None:
        """(calibrated p, payoff b = reward/risk) for the Kelly reducer, or None when unusable."""
        try:
            probability = self.probability(signal, now)
            risk = abs(float(signal["price"]) - float(signal["stop_loss"]))
            reward = abs(float(signal["target"]) - float(signal["price"]))
        except Exception:
            return None
        return (probability, reward / risk) if risk > 0 and reward > 0 else None


def gate_report() -> dict[str, Any]:
    """Operator report from today's settled outcomes; evaluates only, never writes a live artefact."""
    return train_gate(load_closed_trade_samples(), write=False)


def conviction_report() -> dict[str, Any]:
    """Reliability of AI conviction on settled ResearchForecast rows (same calibration maths)."""
    from database.db import get_db
    from database.models import ResearchForecast

    with get_db() as db:
        rows = db.query(ResearchForecast).filter(
            ResearchForecast.direction_correct.isnot(None), ResearchForecast.confidence.isnot(None),
        ).all()
        data = [(float(r.confidence) / 100.0, 1.0 if r.direction_correct else 0.0) for r in rows]
    if len(data) < MIN_TOTAL_SAMPLES:
        return {
            "samples": len(data), "min_samples": MIN_TOTAL_SAMPLES, "status": "insufficient_samples",
            "message": f"insufficient samples: {len(data)} settled forecasts, at least {MIN_TOTAL_SAMPLES} are needed",
        }
    p = np.clip(np.array([item[0] for item in data]), 0.0, 1.0)
    y = np.array([item[1] for item in data])
    bins, ece = reliability_bins(p, y)
    return {"samples": len(data), "status": "measured", "brier": brier_score(p, y),
            "log_loss": log_loss_score(p, y), "ece": ece, "reliability_bins": bins}


signal_gate = SignalGate()
