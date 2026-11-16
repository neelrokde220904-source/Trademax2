"""Failover routing that never duplicates orders after an ambiguous response."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from broker.base_broker import (
    BaseBrokerAdapter,
    BrokerRejectedError,
    OrderSubmissionUncertainError,
)


class DualBrokerExecutionRouter:
    """Try a secondary broker only after a confirmed non-acceptance rejection."""

    _DEFINITE_REJECTIONS = {"REJECTED", "MARGIN_REJECTED", "FAILED_VALIDATION"}

    def execute(
        self,
        *,
        primary: BaseBrokerAdapter,
        secondary: BaseBrokerAdapter,
        primary_order: dict[str, Any] | Callable[[], dict[str, Any]],
        secondary_order: dict[str, Any] | Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            primary_payload = primary_order() if callable(primary_order) else primary_order
            response = primary.place_order(primary_payload)
        except BrokerRejectedError:
            response = None
        except (TimeoutError, ConnectionError) as exc:
            raise OrderSubmissionUncertainError(
                "Primary broker outcome is unknown; reconcile its order book before retrying"
            ) from exc
        if response is not None and not self._definitively_rejected(response):
            return {**response, "routed_broker": "primary"}

        secondary_payload = secondary_order() if callable(secondary_order) else secondary_order
        fallback = secondary.place_order(secondary_payload)
        if self._definitively_rejected(fallback):
            raise BrokerRejectedError("Both primary and secondary brokers rejected the order")
        return {**fallback, "routed_broker": "secondary"}

    @classmethod
    def _definitively_rejected(cls, response: dict[str, Any]) -> bool:
        return str(response.get("status", "")).upper() in cls._DEFINITE_REJECTIONS


dual_broker_router = DualBrokerExecutionRouter()
