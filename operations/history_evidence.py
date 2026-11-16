"""Fingerprint the exact supported DataFrame presented to technical analysis."""

import hashlib

import pandas as pd

from agents.checkpoint_serializer import TradingCheckpointSerializer


def history_digest(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Historical data is unavailable for fingerprinting")
    kind, encoded = TradingCheckpointSerializer().dumps_typed(frame)
    return hashlib.sha256(kind.encode("utf-8") + b"\0" + encoded).hexdigest()
