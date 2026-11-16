"""Offline HMM training from completed Angel One daily index candles."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytz

from config.settings import settings
from risk.hmm_regime_classifier import (
    FEATURES,
    HMMModelError,
    HMMRegimeClassifier,
    build_daily_features,
)

IST = pytz.timezone("Asia/Kolkata")
# Angel One's Nifty 50 daily history is missing 2017-08-25..2017-09-12; start after it
# (the fetch refuses gaps) while still covering the March 2020 crash used for validation.
DEFAULT_TRAINING_START = date(2017, 10, 1)
# Offline training can afford to wait out Angel's historical-candle throttling.
TRAINING_RATE_LIMIT_BACKOFF = (15.0, 30.0, 60.0, 120.0, 180.0)
COVID_CRASH = (date(2020, 3, 12), date(2020, 3, 31))
MIN_PANIC_VIX = 22.0


def validate_trained_model(classifier: HMMRegimeClassifier, features: Any) -> dict[str, Any]:
    """Sanity checks that the learned panic state behaves like market stress."""
    profiles = classifier.metadata["state_profiles"]
    panic = [profile for profile in profiles if profile["regime"] == "SYSTEMIC_PANIC"]
    checks: list[dict[str, Any]] = []
    panic_vix = max((profile["india_vix"] for profile in panic), default=0.0)
    highest_vix = max(profile["india_vix"] for profile in profiles)
    checks.append({
        "name": "panic_state_has_highest_vix",
        "passed": bool(panic) and panic_vix >= highest_vix,
        "detail": f"panic mean VIX {panic_vix:.1f}; highest state mean VIX {highest_vix:.1f}",
    })
    checks.append({
        "name": "panic_state_vix_is_elevated",
        "passed": panic_vix >= MIN_PANIC_VIX,
        "detail": f"panic mean VIX {panic_vix:.1f} (minimum {MIN_PANIC_VIX:.0f})",
    })
    decoded = classifier.decode(features)
    crash = decoded[(decoded.index.date >= COVID_CRASH[0]) & (decoded.index.date <= COVID_CRASH[1])]
    if crash.empty:
        checks.append({
            "name": "march_2020_is_panic", "passed": True,
            "detail": "skipped: training data does not cover March 2020",
        })
    else:
        share = float((crash == "SYSTEMIC_PANIC").mean())
        checks.append({
            "name": "march_2020_is_panic",
            "passed": share >= 0.5,
            "detail": f"{share:.0%} of {len(crash)} sessions decoded as SYSTEMIC_PANIC",
        })
    occupancy = decoded.value_counts(normalize=True).round(4).to_dict()
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "regime_occupancy": occupancy}


def train_hmm_from_angel(
    *,
    start: date = DEFAULT_TRAINING_START,
    end: date | None = None,
    n_restarts: int = 10,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch history, fit, validate, save atomically and verify the saved JSON reloads identically."""
    from data.market_data import fetch_index_daily_history

    end = end or (datetime.now(IST).date() - timedelta(days=1))
    nifty = fetch_index_daily_history(
        settings.nifty50_token, start, end, pause_seconds=3.0, rate_limit_backoff=TRAINING_RATE_LIMIT_BACKOFF,
    )
    time.sleep(3.0)
    vix = fetch_index_daily_history(
        settings.india_vix_token, start, end, pause_seconds=3.0, rate_limit_backoff=TRAINING_RATE_LIMIT_BACKOFF,
    )
    features = build_daily_features(nifty, vix)

    classifier = HMMRegimeClassifier(model_path=str(model_path) if model_path else None)
    report = classifier.fit(features, n_restarts=n_restarts, source={
        "provider": "angel_one_historical_candles",
        "interval": "ONE_DAY",
        "nifty50_token": settings.nifty50_token,
        "india_vix_token": settings.india_vix_token,
        "nifty_rows": int(len(nifty)),
        "vix_rows": int(len(vix)),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "features": list(FEATURES),
    })
    validation = validate_trained_model(classifier, features)
    classifier.metadata["validation"] = validation
    if not validation["passed"]:
        failed = [check["name"] for check in validation["checks"] if not check["passed"]]
        raise HMMModelError(f"trained model failed validation ({', '.join(failed)}); not saved")

    path = classifier.save()
    reloaded = HMMRegimeClassifier(model_path=str(path))
    if not reloaded.is_fitted:
        raise HMMModelError(f"saved model did not reload: {reloaded.load_error}")
    today = datetime.now(IST).date()
    original = classifier.predict_regime(features, today=today)
    restored = reloaded.predict_regime(features, today=today)
    if original != restored:
        raise HMMModelError("reloaded model predictions differ from the trained model")
    return {**report, "validation": validation, "model_path": str(path), "latest": restored}
