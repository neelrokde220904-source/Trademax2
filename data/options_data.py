"""Options chain data fetcher for NFO segment."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import pandas as pd
import pytz
from loguru import logger

from data.instrument_master import find_nfo_options, load_instrument_master

IST = pytz.timezone("Asia/Kolkata")


def fetch_options_chain(
    underlying: str = "NIFTY",
    expiry_contains: str = "",
) -> pd.DataFrame:
    """Fetch the options chain for an underlying (NIFTY / BANKNIFTY).

    Returns:
        Metadata for the nearest unexpired matching expiry. Prices, OI and volume
        are missing until a verified options data provider is integrated.
    """
    instruments = load_instrument_master()
    ce_options = find_nfo_options(underlying, expiry_contains=expiry_contains, option_type="CE", instruments=instruments)
    pe_options = find_nfo_options(underlying, expiry_contains=expiry_contains, option_type="PE", instruments=instruments)

    all_options = ce_options + pe_options

    if not all_options:
        logger.warning("No options found for {} (expiry filter: '{}')", underlying, expiry_contains)
        return pd.DataFrame()

    dated = []
    for opt in all_options:
        try:
            expiry = datetime.strptime(opt["expiry"].upper(), "%d%b%Y").date()
        except (TypeError, ValueError):
            logger.warning("Ignoring option {} with invalid expiry", opt["symbol"])
            continue
        if expiry >= datetime.now(IST).date():
            dated.append((expiry, opt))
    if not dated:
        logger.warning("No current option expiry found for {}", underlying)
        return pd.DataFrame()
    nearest = min(expiry for expiry, _ in dated)
    rows = []
    for expiry, opt in dated:
        if expiry != nearest:
            continue
        rows.append({
            "strike": opt["strike"],
            "option_type": opt["option_type"],
            "symbol": opt["symbol"],
            "token": opt["token"],
            "expiry": opt["expiry"],
            "lot_size": opt["lot_size"],
            "ltp": None,
            "oi": None,
            "volume": None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["strike", "option_type"]).reset_index(drop=True)

    logger.warning(
        "Options OI and contract quotes unavailable for {}; returning {} metadata-only contracts for {}.",
        underlying, len(df), nearest,
    )
    return df


def _validated_oi_chain(chain_df: pd.DataFrame) -> pd.DataFrame | None:
    required = {"strike", "option_type", "oi", "expiry"}
    if not isinstance(chain_df, pd.DataFrame) or chain_df.empty or not required.issubset(chain_df.columns):
        return None
    expiries = chain_df["expiry"]
    if expiries.isna().any() or expiries.astype(str).str.strip().eq("").any() or expiries.nunique() != 1:
        return None
    sides = chain_df["option_type"]
    if set(sides) != {"CE", "PE"}:
        return None
    oi = pd.to_numeric(chain_df["oi"], errors="coerce")
    strikes = pd.to_numeric(chain_df["strike"], errors="coerce")
    if (
        not oi.map(math.isfinite).all()
        or not strikes.map(math.isfinite).all()
        or (oi < 0).any()
        or (strikes <= 0).any()
        or oi[sides == "CE"].sum() <= 0
        or oi[sides == "PE"].sum() <= 0
    ):
        return None
    return chain_df.assign(oi=oi, strike=strikes)


def calculate_pcr(chain_df: pd.DataFrame) -> float | None:
    """Calculate Put-Call Ratio from OI data.

    PCR > 1.2 = bearish sentiment (contrarian buy signal)
    PCR < 0.7 = bullish sentiment (contrarian sell signal)
    """
    chain_df = _validated_oi_chain(chain_df)
    if chain_df is None:
        return None

    ce_oi = chain_df[chain_df["option_type"] == "CE"]["oi"].sum()
    pe_oi = chain_df[chain_df["option_type"] == "PE"]["oi"].sum()

    if ce_oi == 0:
        return None

    return round(pe_oi / ce_oi, 3)


def calculate_max_pain(chain_df: pd.DataFrame, spot_price: float) -> float | None:
    """Calculate Max Pain strike — the price where option sellers profit most.

    Price tends to gravitate towards max pain at expiry.
    """
    chain_df = _validated_oi_chain(chain_df)
    if chain_df is None:
        return None

    strikes = sorted(chain_df["strike"].unique())
    min_pain = float("inf")
    max_pain_strike = None

    for target_strike in strikes:
        total_pain = 0.0

        for _, row in chain_df.iterrows():
            strike = row["strike"]
            oi = row.get("oi", 0)
            opt_type = row["option_type"]

            if opt_type == "CE":
                # Call buyers' pain = max(0, target - strike) × OI
                intrinsic = max(0, target_strike - strike)
                total_pain += intrinsic * oi
            elif opt_type == "PE":
                # Put buyers' pain = max(0, strike - target) × OI
                intrinsic = max(0, strike - target_strike)
                total_pain += intrinsic * oi

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = target_strike

    return max_pain_strike


def find_max_oi_strikes(chain_df: pd.DataFrame) -> dict[str, Any]:
    """Find strikes with maximum OI for CE and PE — these act as support/resistance."""
    chain_df = _validated_oi_chain(chain_df)
    if chain_df is None:
        return {"max_ce_oi_strike": None, "max_pe_oi_strike": None}

    ce_data = chain_df[chain_df["option_type"] == "CE"]
    pe_data = chain_df[chain_df["option_type"] == "PE"]

    max_ce_strike = None
    max_pe_strike = None

    if not ce_data.empty and ce_data["oi"].sum() > 0:
        max_ce_idx = ce_data["oi"].idxmax()
        max_ce_strike = float(ce_data.loc[max_ce_idx, "strike"])

    if not pe_data.empty and pe_data["oi"].sum() > 0:
        max_pe_idx = pe_data["oi"].idxmax()
        max_pe_strike = float(pe_data.loc[max_pe_idx, "strike"])

    return {
        "max_ce_oi_strike": max_ce_strike,  # Resistance
        "max_pe_oi_strike": max_pe_strike,  # Support
    }


def detect_unusual_oi(chain_df: pd.DataFrame, threshold_multiplier: float = 2.0) -> list[dict[str, Any]]:
    """Detect unusual OI buildup — potential institutional positioning signal.

    Args:
        chain_df: Options chain DataFrame.
        threshold_multiplier: Consider OI unusual if > mean * multiplier.

    Returns:
        List of dicts for options with unusual OI.
    """
    chain_df = _validated_oi_chain(chain_df)
    if chain_df is None:
        return []

    mean_oi = chain_df["oi"].mean()
    if mean_oi == 0:
        return []

    threshold = mean_oi * threshold_multiplier
    unusual = chain_df[chain_df["oi"] > threshold]

    return [
        {
            "strike": row["strike"],
            "option_type": row["option_type"],
            "oi": row["oi"],
            "ltp": row.get("ltp", 0),
            "symbol": row.get("symbol", ""),
        }
        for _, row in unusual.iterrows()
    ]
