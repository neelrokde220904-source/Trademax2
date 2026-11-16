"""Telegram Notifier — trade alerts, daily PnL summaries, error alerts."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytz
from loguru import logger
from telegram.error import TelegramError

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class TelegramNotifier:
    """Send notifications via Telegram Bot API.

    Message types:
    - Trade Executed (BUY/SELL with entry details)
    - Trade Closed (PnL)
    - Daily PnL Summary
    - Strategy Signal
    - Risk Alert
    - System Error
    """

    def __init__(self) -> None:
        self._bot = None
        self._enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self._delivery_status = "untested" if self._enabled else "not_configured"
        self._last_delivery_at: str | None = None
        if not self._enabled:
            logger.warning("[Telegram] Bot token or chat ID missing — notifications disabled.")

    async def _get_bot(self):
        """Lazy-init the telegram bot."""
        if self._bot is None and self._enabled:
            from telegram import Bot
            self._bot = Bot(token=settings.telegram_bot_token)
        return self._bot

    async def _deliver(self, text: str, parse_mode: str | None) -> tuple[bool, str | None]:
        try:
            bot = await self._get_bot()
            if bot:
                await bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                self._delivery_status = "delivered"
                self._last_delivery_at = datetime.now(IST).isoformat()
                return True, None
        except TelegramError as exc:
            self._delivery_status = type(exc).__name__
            logger.error("[Telegram] Delivery failed ({})", type(exc).__name__)
            return False, type(exc).__name__
        return False, "no_bot"

    async def send_message(
        self, text: str, parse_mode: str = "HTML", *,
        dedup_key: str | None = None, severity: str = "info",
    ) -> bool:
        """Store the alert in the outbox, then try to deliver it.

        ``dedup_key`` suppresses an identical alert sent within 30 minutes.
        """
        from notifications import outbox

        if not self._enabled:
            logger.debug("[Telegram] Disabled — would send: {}", text[:100])
            if severity == "critical":
                outbox.enqueue(
                    text, parse_mode=parse_mode, dedup_key=dedup_key,
                    severity=severity, deliverable=False,
                )
            return False

        row_id = outbox.enqueue(
            text, parse_mode=parse_mode, dedup_key=dedup_key, severity=severity,
        )
        if row_id is None and dedup_key:
            logger.debug("[Telegram] Duplicate alert suppressed: {}", dedup_key)
            return False
        delivered, error = await self._deliver(text, parse_mode)
        outbox.record_attempt(row_id, delivered=delivered, error=error)
        return delivered

    async def retry_outbox(self) -> int:
        """Re-send stored alerts that were not delivered; returns the number delivered."""
        if not self._enabled:
            return 0
        from notifications import outbox

        delivered_count = 0
        for item in outbox.due_for_retry():
            delivered, error = await self._deliver(item["text"], item["parse_mode"])
            outbox.record_attempt(item["id"], delivered=delivered, error=error)
            delivered_count += int(delivered)
        return delivered_count

    def get_status(self) -> dict[str, Any]:
        return {
            "configured": self._enabled, "delivery_status": self._delivery_status,
            "last_delivery_at": self._last_delivery_at,
        }

    async def test_delivery(self) -> bool:
        """Validate the recipient and send a real, non-sensitive test message."""
        bot = await self._get_bot()
        if bot is None:
            return False
        try:
            me = await bot.get_me()
            if settings.telegram_chat_id == str(me.id):
                self._delivery_status = "recipient_is_bot"
                logger.error("[Telegram] Recipient is the bot; configure a private chat or group")
                return False
            chat = await bot.get_chat(settings.telegram_chat_id)
            if chat.type not in {"private", "group", "supergroup", "channel"}:
                self._delivery_status = "unsupported_recipient"
                return False
            return await self.send_message(
                "India AI Trader: Telegram delivery test. Paper mode only; "
                "this message does not resume entries or clear a safety halt."
            )
        except TelegramError as exc:
            self._delivery_status = type(exc).__name__
            logger.error("[Telegram] Recipient validation failed ({})", type(exc).__name__)
            return False

    # ----- TRADE NOTIFICATIONS -----

    async def notify_trade_executed(self, trade: dict[str, Any]) -> None:
        """Send trade execution notification."""
        action = trade.get("action", "?")
        symbol = trade.get("symbol", "?")
        price = trade.get("price", 0)
        qty = trade.get("quantity", 0)
        sl = trade.get("stop_loss", 0)
        target = trade.get("target", 0)
        strategy = trade.get("strategy", "N/A")
        confidence = trade.get("confidence", 0)
        mode = "📝 PAPER" if settings.paper_trading else "🔴 LIVE"

        emoji = "🟢" if action == "BUY" else "🔴"

        msg = (
            f"{emoji} <b>Trade Executed</b> {mode}\n\n"
            f"<b>{action} {symbol}</b>\n"
            f"Price: ₹{price:,.2f} × {qty} shares\n"
            f"Value: ₹{price * qty:,.2f}\n"
            f"SL: ₹{sl:,.2f} | Target: ₹{target:,.2f}\n"
            f"Strategy: {strategy}\n"
            f"Confidence: {confidence}%\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg)

    async def notify_trade_closed(self, trade: dict[str, Any]) -> None:
        """Send trade closure notification."""
        symbol = trade.get("symbol", "?")
        entry = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl = trade.get("pnl", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        reason = trade.get("exit_reason", "?")

        emoji = "💰" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""

        msg = (
            f"{emoji} <b>Trade Closed</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"Entry: ₹{entry:,.2f} → Exit: ₹{exit_price:,.2f}\n"
            f"PnL: {sign}₹{pnl:,.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg)

    async def notify_daily_summary(self, summary: dict[str, Any]) -> None:
        """Send end-of-day summary."""
        pnl = summary.get("pnl", 0)
        gross_pnl = summary.get("gross_pnl", pnl)
        transaction_costs = summary.get("transaction_costs", 0)
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        portfolio_value = summary.get("portfolio_value", 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0

        emoji = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""

        msg = (
            f"{emoji} <b>Daily Summary</b> — {datetime.now(IST).strftime('%d %b %Y')}\n\n"
            f"Net PnL: {sign}₹{pnl:,.2f}\n"
            f"Gross PnL: ₹{gross_pnl:,.2f} | Est. costs: ₹{transaction_costs:,.2f}\n"
            f"Total Trades: {trades} (W:{wins} L:{losses})\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Portfolio Value: ₹{portfolio_value:,.2f}\n"
            f"Mode: {'📝 Paper' if settings.paper_trading else '🔴 Live'}"
        )
        await self.send_message(msg)

    async def notify_signal(self, signal: dict[str, Any]) -> None:
        """Send a raw signal alert."""
        action = signal.get("action", "?")
        symbol = signal.get("symbol", "?")
        confidence = signal.get("confidence", 0)
        strategy = signal.get("strategy", "")
        reasoning = signal.get("reasoning", "")

        msg = (
            f"🔔 <b>Signal: {action} {symbol}</b>\n"
            f"Confidence: {confidence}%\n"
            f"Strategy: {strategy}\n"
            f"Reason: {reasoning[:200]}"
        )
        await self.send_message(msg)

    async def notify_risk_alert(
        self, alert: str, *, dedup_key: str | None = None, severity: str = "warning",
    ) -> None:
        """Send a risk alert."""
        msg = f"⚠️ <b>RISK ALERT</b>\n\n{alert}"
        await self.send_message(msg, dedup_key=dedup_key, severity=severity)

    async def notify_error(
        self, error: str, *, dedup_key: str | None = None, severity: str = "critical",
    ) -> None:
        """Send a system error alert."""
        msg = (
            f"🚨 <b>System Error</b>\n\n"
            f"{error[:500]}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg, dedup_key=dedup_key, severity=severity)

    # ----- COMMAND HANDLERS -----

    async def handle_resume_trading(self) -> None:
        """Handle /resume_trading command — resumes after Dead Man's Switch halt."""
        from risk.dead_mans_switch import dead_mans_switch

        if not dead_mans_switch.is_halted:
            await self.send_message("ℹ️ Trading is already active — no halt to resume from.")
            return

        resumed = dead_mans_switch.resume_trading(reason="Telegram /resume_trading command")
        if resumed:
            await self.send_message("✅ Trading resumed successfully. All failure counters reset.")
        else:
            await self.send_message("⚠️ Failed to resume trading. Check logs.")

    async def handle_dms_status(self) -> None:
        """Handle /dms_status command — show Dead Man's Switch status."""
        from risk.dead_mans_switch import dead_mans_switch

        status = dead_mans_switch.get_status()
        halted = "⛔ HALTED" if status["halted"] else "✅ ACTIVE"

        msg = f"🛡️ <b>Dead Man's Switch Status: {halted}</b>\n\n"

        if status["halted"]:
            msg += f"<b>Reason:</b> {status['halt_reason']}\n"
            msg += f"<b>Halted at:</b> {status['halt_time']}\n\n"

        msg += "<b>Failure counts:</b>\n"
        for comp, cnt in status["failure_counts"].items():
            threshold = status["thresholds"].get(comp, 3)
            icon = "🔴" if cnt >= threshold else "🟡" if cnt > 0 else "🟢"
            msg += f"  {icon} {comp}: {cnt}/{threshold}\n"

        if not status["failure_counts"]:
            msg += "  All clear — no failures recorded.\n"

        await self.send_message(msg)


# Singleton
telegram_notifier = TelegramNotifier()
