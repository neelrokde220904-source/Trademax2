"""Alpaca PAPER adapter for US equities (disabled unless ``FOREIGN_US_PAPER_ENABLED``).

It wraps ``AlpacaClient`` behind the common broker contract and refuses to exist unless the
account is a paper account: both ``settings.alpaca_paper`` and the client's own endpoint flag must
say paper, and a live URL on the underlying client is rejected. Orders are LIMIT only, USD-denominated,
checked against the US session calendar (DST aware), and never subject to the Indian 15:15 IST rule.

Data limitation (shown wherever prices from this adapter are displayed): the free Alpaca feed is
IEX-only, so quotes and volume reflect a single venue and are not the consolidated tape.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from broker.base_broker import BaseBrokerAdapter, BrokerRejectedError
from config.settings import settings
from data.us_calendar import is_open

DATA_FEED_LABEL = "IEX free feed: single-venue quotes and volume, not the consolidated tape"


class AlpacaPaperRefused(RuntimeError):
    """The adapter will not start: it can only ever talk to a paper account."""


def readiness(client: Any | None = None) -> dict[str, Any]:
    """Non-throwing readiness summary for dashboards and the readiness documents."""
    if client is None:
        from broker.alpaca_api import alpaca_client as client  # noqa: PLC0415
    if not settings.foreign_us_paper_enabled:
        return {"status": "disabled", "reason": "FOREIGN_US_PAPER_ENABLED is false"}
    if not (getattr(client, "api_key", "") and getattr(client, "secret_key", "")):
        return {"status": "keys_missing", "reason": "Alpaca paper API keys are not configured"}
    if not settings.alpaca_paper or getattr(client, "paper", False) is not True:
        return {"status": "refused", "reason": "not a paper endpoint"}
    return {"status": "ready_for_paper", "reason": "paper keys present", "data": DATA_FEED_LABEL}


class AlpacaPaperAdapter(BaseBrokerAdapter):
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from broker.alpaca_api import alpaca_client as client  # noqa: PLC0415
        if not settings.foreign_us_paper_enabled:
            raise AlpacaPaperRefused("US paper trading is disabled (FOREIGN_US_PAPER_ENABLED=false)")
        if settings.alpaca_paper is not True or getattr(client, "paper", False) is not True:
            raise AlpacaPaperRefused("Refusing to start: the Alpaca endpoint is not a paper endpoint")
        base_url = str(getattr(getattr(client, "trading_client", None), "_base_url", "") or "")
        if base_url and "paper" not in base_url.lower():
            raise AlpacaPaperRefused("Refusing to start: the Alpaca client points at a live URL")
        if not (getattr(client, "api_key", "") and getattr(client, "secret_key", "")):
            raise AlpacaPaperRefused("keys_missing: Alpaca paper API keys are not configured")
        self._client = client

    # -- account ------------------------------------------------------------------
    def authenticate_totp(self) -> bool:
        account = self._client.get_account()
        return "error" not in account and account.get("status") in {"ACTIVE", "AccountStatus.ACTIVE", None}

    def get_funds_and_margins(self) -> dict[str, Any]:
        account = self._client.get_account()
        if "error" in account:
            return {"currency": "USD", "error": account["error"]}
        return {
            "currency": "USD", "available_cash": account["cash"], "buying_power": account["buying_power"],
            "equity": account["equity"], "data_label": DATA_FEED_LABEL,
        }

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "tradingsymbol": p["symbol"], "quantity": p["qty"], "average_price": p["avg_entry_price"],
                "ltp": p["current_price"], "currency": "USD", "side": str(p.get("side", "")),
            }
            for p in self._client.get_positions()
        ]

    def get_holdings(self) -> list[dict[str, Any]]:
        return self.get_positions()  # US cash equities have no separate delivery book

    # -- orders -------------------------------------------------------------------
    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        symbol = str(params.get("tradingsymbol") or params.get("symbol") or "").upper()
        side = str(params.get("transactiontype") or params.get("side") or "").upper()
        order_type = str(params.get("ordertype") or "LIMIT").upper()
        try:
            quantity = float(params["quantity"])
            limit_price = float(params.get("price") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerRejectedError("quantity and price must be numbers") from exc
        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0:
            raise BrokerRejectedError("symbol, side and a positive quantity are required")
        if order_type != "LIMIT" or limit_price <= 0:
            raise BrokerRejectedError("US paper orders must be LIMIT orders with a positive price")
        if not is_open(datetime.now(timezone.utc)):
            raise BrokerRejectedError("The US market is closed (or its holiday calendar is unknown)")
        order = self._client.place_limit_order(symbol, side.lower(), quantity, limit_price, fractional=False)
        if order.status in {"REJECTED", "ERROR", "NOT_CONFIGURED", "FEMA_BLOCKED"}:
            raise BrokerRejectedError(f"Alpaca paper order not accepted: {order.status}")
        logger.info("[AlpacaPaper] {} {} x {} @ {} submitted.", side, symbol, quantity, limit_price)
        return {
            "orderid": order.order_id, "status": order.status, "symbol": symbol, "currency": "USD",
            "filled_qty": order.filled_qty, "avg_price": order.filled_avg_price, "paper": True,
        }

    def place_bracket_or_gtt_order(self, params: dict[str, Any]) -> dict[str, Any]:
        raise BrokerRejectedError("Bracket and GTT orders are not supported by the US paper adapter")

    def modify_order(self, params: dict[str, Any]) -> dict[str, Any]:
        raise BrokerRejectedError("Order modification is not supported; cancel and resubmit")

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict[str, Any]:
        return {"cancelled": bool(self._client.cancel_order(order_id)), "orderid": order_id}

    def flatten_all_intraday(self) -> list[dict[str, Any]]:
        # The Indian 15:15 IST square-off does not apply: US positions follow their own horizon.
        return []

    def subscribe_ticks(
        self, exchange: str, tokens: list[str], callback: Callable[[dict[str, Any]], None],
    ) -> None:
        raise NotImplementedError(f"No streaming ticks for US paper trading. {DATA_FEED_LABEL}")
