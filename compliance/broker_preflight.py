"""Fail-closed local checks for broker routing and order-rate constraints."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address

from config.settings import settings


@dataclass(frozen=True)
class BrokerPreflight:
    """Broker-specific checks that can be validated without submitting orders."""

    passed: bool
    errors: list[str]
    warnings: list[str]


def check_broker_readiness() -> BrokerPreflight:
    """Validate declared static-IP setup and configured broker mode.

    This verifies local configuration only. It cannot confirm provider-side
    allowlisting; that must be checked in the broker's account console.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if settings.trademax_broker_mode not in {
        "zerodha_only", "angel_only", "dual_active_failover",
    }:
        errors.append("TRADEMAX_BROKER_MODE is invalid.")
    if not settings.paper_trading:
        if not settings.trademax_static_outbound_ip:
            errors.append("Live trading requires TRADEMAX_STATIC_OUTBOUND_IP.")
        else:
            try:
                ip_address(settings.trademax_static_outbound_ip)
            except ValueError:
                errors.append("TRADEMAX_STATIC_OUTBOUND_IP is not a valid IP address.")
        if not settings.broker_static_ip_allowlist_confirmed:
            errors.append(
                "Confirm the declared static IP is allowlisted by each active broker."
            )
    elif not settings.broker_static_ip_allowlist_confirmed:
        warnings.append("Broker-side static IP allowlisting is not confirmed.")
    if settings.trademax_broker_mode in {"zerodha_only", "dual_active_failover"}:
        if not settings.zerodha_api_key:
            errors.append("ZERODHA_API_KEY is required by the selected broker mode.")
        if not settings.zerodha_access_token:
            warnings.append(
                "ZERODHA_ACCESS_TOKEN is absent; official Kite Connect OAuth login is required."
            )
    if settings.trademax_broker_mode in {"angel_only", "dual_active_failover"}:
        if not settings.angel_api_key:
            warnings.append("ANGEL_API_KEY is absent; Angel One will be unavailable.")
    return BrokerPreflight(not errors, errors, warnings)
