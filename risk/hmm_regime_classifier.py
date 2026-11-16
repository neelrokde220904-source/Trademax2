"""Hidden Markov Model regime classifier — the global entry switch for the trading system.

Regimes:
  0: LOW_VOL_TREND        — Deploy momentum + breakout strategies
  1: HIGH_VOL_MEAN_REVERT — Deploy bollinger reversion, range strategies
  2: SYSTEMIC_PANIC       — ALL CASH + hedge only
  3: BULL_EUPHORIA        — Momentum but tighten stops (potential top)

The model is trained on completed daily Nifty 50 and India VIX candles and stored
as validated JSON parameters (never pickle). Without a valid fitted model the
classifier reports ``UNAVAILABLE`` and new entries are blocked.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
from loguru import logger

from config.settings import settings
from strategies.contracts import COMPOSITE_TREND_REVIEW

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IST = pytz.timezone("Asia/Kolkata")

MODEL_FORMAT = "india-ai-trader/gaussian-hmm-regime/v1"
FEATURES = ("realized_vol_20d", "momentum_20d", "india_vix")
FEATURE_WINDOW = 20
MIN_TRAINING_ROWS = 252
INFERENCE_HISTORY_DAYS = 450
MAX_FEATURE_AGE_DAYS = 7
REGIME_HISTORY_SESSIONS = 120

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. HMM regime classifier is unavailable; entries stay blocked.")


class HMMModelError(RuntimeError):
    """Raised when a model cannot be trained, loaded or validated."""


def _daily_series(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or "close" not in df or "datetime" not in df:
        raise HMMModelError(f"{name} history is empty")
    stamps = pd.to_datetime(df["datetime"])
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(IST).dt.tz_localize(None)
    series = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(), index=stamps.dt.normalize())
    return series.groupby(level=0).last().sort_index()


def build_daily_features(nifty_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-day realised volatility, momentum and India VIX, indexed by IST date."""
    close = _daily_series(nifty_df, "Nifty 50")
    close = close.where(close > 0)
    log_returns = np.log(close).diff()
    frame = pd.DataFrame({
        "realized_vol_20d": log_returns.rolling(FEATURE_WINDOW).std(ddof=0) * math.sqrt(252),
        "momentum_20d": close / close.shift(FEATURE_WINDOW) - 1,
    })
    vix = _daily_series(vix_df, "India VIX")
    vix = vix.where((vix > 0) & (vix <= 200)).rename("india_vix")
    frame = frame.join(vix, how="inner")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()[list(FEATURES)]


def _array(payload: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(payload.get(key), dtype=float)
    if value.shape != shape:
        raise HMMModelError(f"{key} has shape {value.shape}, expected {shape}")
    if not np.all(np.isfinite(value)):
        raise HMMModelError(f"{key} contains non-finite values")
    return value


def _is_stochastic(value: np.ndarray) -> bool:
    return bool(np.all(value >= 0) and np.allclose(value.sum(axis=-1), 1.0, atol=1e-6))


class HMMRegimeClassifier:
    """4-state Gaussian HMM over daily market features."""

    REGIMES = {
        0: "LOW_VOL_TREND",
        1: "HIGH_VOL_MEAN_REVERT",
        2: "SYSTEMIC_PANIC",
        3: "BULL_EUPHORIA",
    }
    REGIME_IDS = {name: regime_id for regime_id, name in REGIMES.items()}

    def __init__(self, n_components: int | None = None, model_path: str | None = None) -> None:
        self.n_components = int(n_components or settings.hmm_n_states or 4)
        self.model_path = Path(
            model_path or settings.hmm_model_path
            or str(PROJECT_ROOT / "models" / "hmm_regime_classifier.json")
        )
        self.model: Any = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.regime_map: dict[int, str] = {}
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self._lock = threading.RLock()
        self._cache: tuple[date, str | None, dict[str, Any]] | None = None
        self._history: list[dict[str, Any]] = []
        self.reload()

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------ persistence

    def reload(self) -> bool:
        """Load the JSON model if present and valid; otherwise stay unavailable."""
        with self._lock:
            self.model = None
            self._cache = None
            self._history = []
            if not HMM_AVAILABLE:
                self.load_error = "hmmlearn is not installed"
            elif self.model_path.suffix.lower() != ".json":
                self.load_error = f"Refusing non-JSON HMM model file {self.model_path.name}"
            elif not self.model_path.is_file():
                self.load_error = f"No fitted HMM model at {self.model_path}"
            else:
                try:
                    self._apply_payload(json.loads(self.model_path.read_text(encoding="utf-8")))
                    self.load_error = None
                    logger.info(
                        "Loaded HMM model trained {} from {}",
                        self.metadata.get("trained_at"), self.model_path,
                    )
                except Exception as exc:
                    self.model = None
                    self.load_error = f"Invalid HMM model file: {exc}"
            if self.load_error:
                logger.warning("HMM unavailable: {}", self.load_error)
            return self.is_fitted

    def _payload(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "features": list(FEATURES),
            "n_components": self.n_components,
            "covariance_type": "full",
            "startprob": self.model.startprob_.tolist(),
            "transmat": self.model.transmat_.tolist(),
            "means": self.model.means_.tolist(),
            "covars": self.model.covars_.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "regime_map": {str(state): name for state, name in self.regime_map.items()},
            "metadata": self.metadata,
        }

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
            raise HMMModelError("unsupported model format")
        if payload.get("features") != list(FEATURES):
            raise HMMModelError(f"feature mismatch: {payload.get('features')} != {list(FEATURES)}")
        if payload.get("covariance_type") != "full":
            raise HMMModelError("only full covariance models are supported")
        n = int(payload.get("n_components", 0))
        if n != self.n_components:
            raise HMMModelError(f"model has {n} states, expected {self.n_components}")
        f = len(FEATURES)
        startprob = _array(payload, "startprob", (n,))
        transmat = _array(payload, "transmat", (n, n))
        means = _array(payload, "means", (n, f))
        covars = _array(payload, "covars", (n, f, f))
        feature_mean = _array(payload, "feature_mean", (f,))
        feature_scale = _array(payload, "feature_scale", (f,))
        if not _is_stochastic(startprob) or not _is_stochastic(transmat):
            raise HMMModelError("start or transition probabilities are not valid distributions")
        if np.any(feature_scale <= 0):
            raise HMMModelError("feature scale must be positive")
        for covar in covars:
            if not np.allclose(covar, covar.T, atol=1e-8):
                raise HMMModelError("covariance matrix is not symmetric")
            np.linalg.cholesky(covar)
        raw_map = payload.get("regime_map") or {}
        regime_map = {int(state): str(name) for state, name in raw_map.items()}
        if set(regime_map) != set(range(n)) or not set(regime_map.values()) <= set(self.REGIME_IDS):
            raise HMMModelError("regime map must label every state with a known regime")

        model = hmm.GaussianHMM(
            n_components=n, covariance_type="full", init_params="", params="stmc",
        )
        model.n_features = f
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars
        model._check()

        self.model = model
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.regime_map = regime_map
        self.metadata = dict(payload.get("metadata") or {})

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically write the fitted model as JSON."""
        if not self.is_fitted:
            raise HMMModelError("cannot save an unfitted model")
        target = Path(path or self.model_path)
        if target.suffix.lower() != ".json":
            raise HMMModelError("HMM models are stored as .json only")
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".hmm-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._payload(), stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return target

    # ------------------------------------------------------------------ training

    def fit(
        self,
        features: pd.DataFrame,
        *,
        n_restarts: int = 10,
        random_state: int = 42,
        min_occupancy: float = 0.01,
        n_iter: int = 500,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit with random restarts; keep the best converged candidate with no near-empty state."""
        if not HMM_AVAILABLE:
            raise HMMModelError("hmmlearn is not installed")
        frame = features[list(FEATURES)].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < MIN_TRAINING_ROWS:
            raise HMMModelError(f"need at least {MIN_TRAINING_ROWS} feature rows, got {len(frame)}")
        raw = frame.to_numpy(dtype=float)
        feature_mean = raw.mean(axis=0)
        feature_scale = raw.std(axis=0)
        if np.any(feature_scale <= 0):
            raise HMMModelError("a feature has zero variance")
        scaled = (raw - feature_mean) / feature_scale

        candidates: list[dict[str, Any]] = []
        best: tuple[float, Any, np.ndarray] | None = None
        for attempt in range(n_restarts):
            seed = random_state + attempt
            model = hmm.GaussianHMM(
                n_components=self.n_components, covariance_type="full", n_iter=n_iter,
                tol=1e-4, min_covar=1e-4, random_state=seed,
            )
            try:
                model.fit(scaled)
                score = float(model.score(scaled))
                states = model.predict(scaled)
            except Exception as exc:
                candidates.append({"seed": seed, "error": str(exc)})
                continue
            occupancy = np.bincount(states, minlength=self.n_components) / len(states)
            converged = bool(model.monitor_.converged)
            accepted = bool(converged and math.isfinite(score) and occupancy.min() >= min_occupancy)
            candidates.append({
                "seed": seed, "converged": converged, "log_likelihood": round(score, 4),
                "min_occupancy": round(float(occupancy.min()), 4), "accepted": accepted,
            })
            if accepted and (best is None or score > best[0]):
                best = (score, model, states)
        if best is None:
            raise HMMModelError(f"no converged candidate with every state above {min_occupancy:.0%} occupancy")

        score, model, states = best
        raw_means = model.means_ * feature_scale + feature_mean
        regime_map = self._label_states(raw_means, model.means_)
        occupancy = np.bincount(states, minlength=self.n_components) / len(states)
        profiles = [
            {
                "state": state,
                "regime": regime_map[state],
                "occupancy": round(float(occupancy[state]), 4),
                **{name: round(float(raw_means[state, i]), 6) for i, name in enumerate(FEATURES)},
            }
            for state in range(self.n_components)
        ]
        with self._lock:
            self.model = model
            self.feature_mean = feature_mean
            self.feature_scale = feature_scale
            self.regime_map = regime_map
            self.load_error = None
            self._cache = None
            self._history = []
            self.metadata = {
                "trained_at": datetime.now(IST).isoformat(),
                "data_start": frame.index[0].date().isoformat(),
                "data_end": frame.index[-1].date().isoformat(),
                "n_observations": int(len(frame)),
                "log_likelihood": round(score, 4),
                "log_likelihood_per_observation": round(score / len(frame), 6),
                "state_profiles": profiles,
                "candidates": candidates,
                "source": source or {},
            }
        return dict(self.metadata)

    def _label_states(self, raw_means: np.ndarray, scaled_means: np.ndarray) -> dict[int, str]:
        """Calibrate states by stress (standardised realised vol + VIX).

        Lowest stress is LOW_VOL_TREND, highest SYSTEMIC_PANIC and second-highest
        HIGH_VOL_MEAN_REVERT. Any calmer middle state is BULL_EUPHORIA only when its
        mean momentum is positive, so an elevated-volatility state is never called euphoric.
        """
        stress = (
            scaled_means[:, FEATURES.index("realized_vol_20d")]
            + scaled_means[:, FEATURES.index("india_vix")]
        )
        momentum = raw_means[:, FEATURES.index("momentum_20d")]
        order = [int(state) for state in np.argsort(stress)]
        labels = {order[0]: "LOW_VOL_TREND", order[-1]: "SYSTEMIC_PANIC"}
        if len(order) > 2:
            labels[order[-2]] = "HIGH_VOL_MEAN_REVERT"
        for state in order[1:-2]:
            labels[state] = "BULL_EUPHORIA" if momentum[state] > 0 else "HIGH_VOL_MEAN_REVERT"
        return labels

    # ------------------------------------------------------------------ inference

    def decode(self, features: pd.DataFrame) -> pd.Series:
        """Most likely regime name for each feature row (used for training validation)."""
        if not self.is_fitted:
            raise HMMModelError(self.load_error or "model is not fitted")
        frame = features[list(FEATURES)].dropna()
        states = self.model.predict((frame.to_numpy(dtype=float) - self.feature_mean) / self.feature_scale)
        return pd.Series([self.regime_map[int(state)] for state in states], index=frame.index)

    def predict_regime(self, features: pd.DataFrame, *, today: date | None = None) -> dict[str, Any]:
        """Classify the latest completed day from a history of daily features."""
        if not self.is_fitted:
            return self.unavailable(self.load_error or "HMM model is not fitted")
        frame = features[list(FEATURES)].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            return self.unavailable("no valid HMM feature rows")
        scaled = (frame.to_numpy(dtype=float) - self.feature_mean) / self.feature_scale
        posterior = self.model.predict_proba(scaled)[-1]
        raw_state = int(np.argmax(posterior))
        regime = self.regime_map[raw_state]
        regime_id = self.REGIME_IDS[regime]
        next_day = posterior @ self.model.transmat_

        def by_regime(probabilities: np.ndarray) -> dict[str, float]:
            totals: dict[str, float] = {}
            for state, probability in enumerate(probabilities):
                name = self.regime_map[state]
                totals[name] = totals.get(name, 0.0) + float(probability)
            return {name: round(value, 6) for name, value in totals.items()}

        regime_probs = by_regime(posterior)
        transition_probs = by_regime(next_day)
        panic_risk = transition_probs.get("SYSTEMIC_PANIC", 0.0)
        as_of = frame.index[-1].date()
        today = today or datetime.now(IST).date()
        feature_age = (today - as_of).days
        degraded = feature_age > MAX_FEATURE_AGE_DAYS
        directive = self._get_action_directive(regime_id, panic_risk)
        if degraded:
            directive["allow_new_positions"] = False
            directive["description"] += f" [DEGRADED: features are {feature_age} days old]"
        latest = frame.iloc[-1]
        result = {
            "regime": regime,
            "regime_id": regime_id,
            "confidence": round(regime_probs.get(regime, 0.0), 6),
            "regime_probs": regime_probs,
            "panic_risk": round(panic_risk, 6),
            "transition_probs": transition_probs,
            "action_directive": directive,
            "as_of": as_of.isoformat(),
            "features": {name: round(float(latest[name]), 6) for name in FEATURES},
            "degraded": degraded,
            "model_trained_at": self.metadata.get("trained_at"),
            "model_stale": self._model_stale(today),
        }
        if degraded:
            result["reason"] = f"latest HMM features are from {as_of}, older than {MAX_FEATURE_AGE_DAYS} days"
        return result

    def classify_latest(self, today: date | None = None) -> dict[str, Any]:
        """Classify using completed daily candles up to yesterday; cached per IST date."""
        today = today or datetime.now(IST).date()
        with self._lock:
            if not self.is_fitted:
                return self.unavailable(self.load_error or "HMM model is not fitted")
            trained_at = self.metadata.get("trained_at")
            if self._cache and self._cache[0] == today and self._cache[1] == trained_at:
                return json.loads(json.dumps(self._cache[2]))
        from data.market_data import fetch_index_daily_history

        end = today - timedelta(days=1)
        start = end - timedelta(days=INFERENCE_HISTORY_DAYS)
        try:
            nifty = fetch_index_daily_history(settings.nifty50_token, start, end)
            vix = fetch_index_daily_history(settings.india_vix_token, start, end)
            features = build_daily_features(nifty, vix)
            result = self.predict_regime(features, today=today)
        except Exception as exc:
            logger.error("HMM inference failed: {}", exc)
            return self.unavailable(f"HMM inference data unavailable: {exc}")
        try:
            history = self.regime_history(nifty, features)
        except Exception as exc:
            logger.warning("HMM regime history unavailable: {}", exc)
            history = []
        with self._lock:
            if not result.get("degraded"):
                self._cache = (today, trained_at, result)
                self._history = history
        logger.info(
            "HMM Regime: {} (confidence={:.0%}, panic_risk={:.1%}, as_of={})",
            result["regime"], result["confidence"], result["panic_risk"], result["as_of"],
        )
        return result

    def regime_history(
        self, nifty_df: pd.DataFrame, features: pd.DataFrame, sessions: int = REGIME_HISTORY_SESSIONS,
    ) -> list[dict[str, Any]]:
        """Most likely (Viterbi) regime per recent session with the Nifty close and VIX, for display."""
        decoded = self.decode(features).tail(sessions)
        close = _daily_series(nifty_df, "Nifty 50")
        return [
            {
                "date": stamp.date().isoformat(),
                "nifty_close": round(float(close.get(stamp, np.nan)), 2),
                "india_vix": round(float(features.at[stamp, "india_vix"]), 2),
                "regime": regime,
            }
            for stamp, regime in decoded.items()
            if np.isfinite(close.get(stamp, np.nan))
        ]

    def unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "regime": "UNAVAILABLE",
            "regime_id": -1,
            "confidence": 0.0,
            "panic_risk": 0.0,
            "transition_probs": {},
            "action_directive": {
                "allowed_strategies": [],
                "max_position_size_pct": 0.0,
                "allow_new_positions": False,
                "hedge_required": False,
                "description": f"HMM unavailable — new entries blocked: {reason}",
            },
            "degraded": True,
            "reason": reason,
        }

    def _model_stale(self, today: date | None = None) -> bool:
        trained_at = self.metadata.get("trained_at")
        if not trained_at:
            return True
        today = today or datetime.now(IST).date()
        return (today - datetime.fromisoformat(trained_at).date()).days > settings.hmm_retrain_days

    def status(self) -> dict[str, Any]:
        with self._lock:
            cached = self._cache[2] if self._cache else None
            try:
                display_path = str(self.model_path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                display_path = str(self.model_path)
            return {
                "library_available": HMM_AVAILABLE,
                "fitted": self.is_fitted,
                "model_path": display_path,
                "load_error": self.load_error,
                "trained_at": self.metadata.get("trained_at"),
                "data_start": self.metadata.get("data_start"),
                "data_end": self.metadata.get("data_end"),
                "n_observations": self.metadata.get("n_observations"),
                "model_stale": self._model_stale() if self.is_fitted else None,
                "latest": None if cached is None else {
                    key: cached.get(key)
                    for key in (
                        "regime", "confidence", "panic_risk", "as_of", "degraded",
                        "regime_probs", "transition_probs", "features",
                    )
                } | {
                    "directive": {
                        key: (cached.get("action_directive") or {}).get(key)
                        for key in (
                            "description", "allow_new_positions", "max_position_size_pct",
                            "allowed_strategies", "hedge_required",
                        )
                    },
                },
                "history": list(self._history),
            }

    def _get_action_directive(self, regime_id: int, panic_risk: float) -> dict[str, Any]:
        """Translate regime into concrete system-wide behavior changes."""
        directives: dict[int, dict[str, Any]] = {
            0: {  # LOW_VOL_TREND
                "allowed_strategies": ["momentum_breakout", "supertrend_adx", "trend_following"],
                "max_position_size_pct": 0.12,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 2.5,
                "allow_new_positions": True,
                "hedge_required": False,
                "max_portfolio_beta": 1.3,
                "description": "Trending market — deploy momentum strategies with standard sizing",
            },
            1: {  # HIGH_VOL_MEAN_REVERT
                "allowed_strategies": ["mean_reversion", "bollinger_reversion", "range_bound"],
                "max_position_size_pct": 0.08,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 1.5,
                "allow_new_positions": True,
                "hedge_required": False,
                "max_portfolio_beta": 1.0,
                "description": "Volatile mean-reverting market — use reversion strategies with tighter sizing",
            },
            2: {  # SYSTEMIC_PANIC
                "allowed_strategies": [],
                "max_position_size_pct": 0.0,
                "stop_loss_multiplier": 0.0,
                "take_profit_multiplier": 0.0,
                "allow_new_positions": False,
                "hedge_required": True,
                "max_portfolio_beta": 0.0,
                "description": "SYSTEMIC PANIC — NO new positions. Hedge existing. Move to cash.",
            },
            3: {  # BULL_EUPHORIA
                "allowed_strategies": ["momentum_breakout", "supertrend_adx"],
                "max_position_size_pct": 0.08,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 2.0,
                "allow_new_positions": True,
                "hedge_required": True,
                "max_portfolio_beta": 0.8,
                "description": "Euphoric market — ride momentum but tighten stops (potential top signal)",
            },
        }

        result = dict(directives[regime_id])
        if panic_risk > 0.30:
            result["panic_warning"] = True
            result["max_position_size_pct"] *= 0.5
            result["description"] += " [PANIC WARNING: elevated transition probability]"
        return result


def entry_regime_block_reason(regime: dict[str, Any] | None) -> str | None:
    """Return why the regime forbids new entries, or None when entries are allowed."""
    if not isinstance(regime, dict) or not regime:
        return "HMM regime missing"
    if regime.get("degraded", True):
        return f"HMM regime degraded: {regime.get('reason') or regime.get('regime', 'UNAVAILABLE')}"
    directive = regime.get("action_directive") or {}
    if not isinstance(directive, dict):
        return "HMM action directive is invalid"
    if directive.get("allow_new_positions") is not True:
        return f"HMM regime {regime.get('regime')} disallows new positions"
    return None


def entry_regime_position_cap(regime: dict[str, Any] | None) -> float:
    """Return a valid HMM position cap, or fail closed."""
    reason = entry_regime_block_reason(regime)
    if reason:
        raise ValueError(reason)
    directive = regime.get("action_directive") if isinstance(regime, dict) else None
    cap = directive.get("max_position_size_pct") if isinstance(directive, dict) else None
    if isinstance(cap, bool) or not isinstance(cap, (int, float)):
        raise ValueError("HMM position-size cap is missing or invalid")
    try:
        cap = float(cap)
    except (OverflowError, ValueError) as exc:
        raise ValueError("HMM position-size cap is missing or invalid") from exc
    if not math.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("HMM position-size cap is missing or invalid")
    return cap


def entry_regime_strategy_block_reason(regime: dict[str, Any] | None, strategy: Any) -> str | None:
    """Apply the declared strategy and hedge directives without inventing a hedge."""
    reason = entry_regime_block_reason(regime)
    if reason:
        return reason
    directive = regime.get("action_directive") if isinstance(regime, dict) else None
    if not isinstance(directive, dict):
        return "HMM action directive is invalid"
    if directive.get("hedge_required") is not False:
        return "HMM requires a hedge or has an invalid hedge directive"
    allowed = directive.get("allowed_strategies")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        return "HMM allowed-strategies directive is invalid"
    from strategies.contracts import strategy_contract

    contract = strategy_contract(strategy)
    if contract is None or not contract.is_enabled() or contract.hmm_tag not in allowed:
        return f"HMM does not permit strategy {strategy!r}"
    return None


# Singleton
hmm_classifier = HMMRegimeClassifier()
