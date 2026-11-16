"""Normalized contract shared by Indian broker adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BrokerRejectedError(RuntimeError):
    """A broker has definitively rejected an order before accepting it."""


class OrderSubmissionUncertainError(RuntimeError):
    """The broker request outcome is unknown; automatic resubmission is unsafe."""


class BaseBrokerAdapter(ABC):
    """Broker capabilities used by the execution and reconciliation layers."""

    @abstractmethod
    def authenticate_totp(self) -> bool:
        """Establish or verify a broker session."""

    @abstractmethod
    def get_funds_and_margins(self) -> dict[str, Any]:
        """Return normalized available funds and margin data."""

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Return normalized open positions."""

    @abstractmethod
    def get_holdings(self) -> list[dict[str, Any]]:
        """Return normalized delivery holdings."""

    @abstractmethod
    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Submit one order and return broker/order identifiers."""

    @abstractmethod
    def place_bracket_or_gtt_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Submit a broker-supported contingent order."""

    @abstractmethod
    def modify_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Modify an existing order."""

    @abstractmethod
    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict[str, Any]:
        """Cancel an existing order."""

    @abstractmethod
    def flatten_all_intraday(self) -> list[dict[str, Any]]:
        """Submit closing orders for all open intraday positions."""

    @abstractmethod
    def subscribe_ticks(
        self, exchange: str, tokens: list[str], callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Subscribe to normalized market ticks."""
