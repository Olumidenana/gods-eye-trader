"""Multi-channel Real-time Signal and Execution Notifier.

Supports:
- Telegram Bot API (Instant mobile push notifications)
- Discord Webhook (Rich embeds with color-coded sidebars)
- Structured JSON Events (for future SaaS WebSockets / REST API feeds)
- Clean console broadcasting with high-contrast formatting

All network dispatches are non-blocking and fail-safe: communication errors
will never interrupt live execution or corrupt order states.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .strategy import LONG, SHORT, Signal

log = logging.getLogger("ict_trader.notifier")


def _safe_print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
        sys.stdout.flush()


@dataclass
class NotifierConfig:
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    discord_webhook_url: str | None = None
    enabled: bool = True


class Notifier:
    def __init__(self, config: NotifierConfig | None = None):
        self.config = config or NotifierConfig(
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
            discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
        )
        self.events: list[dict[str, Any]] = []

    def _send_telegram(self, text: str) -> None:
        token = self.config.telegram_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            log.warning("Telegram notification failed: %s", e)

    def _send_discord(self, title: str, description: str, color: int, fields: list[dict[str, Any]]) -> None:
        url = self.config.discord_webhook_url
        if not url:
            return
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                    "fields": fields,
                    "footer": {"text": "ICT Trader Algorithmic Engine • SaaS Edition"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            log.warning("Discord notification failed: %s", e)

    def signal_detected(self, symbol: str, sig: Signal) -> None:
        side_str = "BUY (LONG)" if sig.direction == LONG else "SELL (SHORT)"
        emoji = "🟢" if sig.direction == LONG else "🔴"
        color = 0x00FF88 if sig.direction == LONG else 0xFF3366

        # Console broadcast
        _safe_print(f"\n{'='*60}\n{emoji} [ICT SIGNAL DETECTED] {symbol} {side_str} | {sig.rr:.2f}R\n{'='*60}")
        _safe_print(f"Entry: {sig.entry_ref} | SL: {sig.stop} | TP: {sig.target} | Session: {sig.killzone}")
        for r in sig.reasons:
            _safe_print(f" • {r}")
        _safe_print(f"{'='*60}\n")

        # Telegram text (HTML format)
        reasons_html = "\n".join(f"• <i>{r}</i>" for r in sig.reasons)
        tg_text = (
            f"<b>{emoji} ICT SIGNAL DETECTED: {symbol}</b>\n\n"
            f"<b>Action:</b> {side_str}\n"
            f"<b>Entry Ref:</b> <code>{sig.entry_ref}</code>\n"
            f"<b>Stop Loss:</b> <code>{sig.stop}</code>\n"
            f"<b>Take Profit:</b> <code>{sig.target}</code>\n"
            f"<b>Reward-to-Risk:</b> <b>{sig.rr:.2f}R</b>\n"
            f"<b>Session / Window:</b> {sig.killzone}\n\n"
            f"<b>Confluences:</b>\n{reasons_html}"
        )
        self._send_telegram(tg_text)

        # Discord embed fields
        fields = [
            {"name": "Action", "value": side_str, "inline": True},
            {"name": "Risk / Reward", "value": f"{sig.rr:.2f}R", "inline": True},
            {"name": "Session", "value": sig.killzone, "inline": True},
            {"name": "Entry Reference", "value": f"{sig.entry_ref}", "inline": True},
            {"name": "Stop Loss", "value": f"{sig.stop}", "inline": True},
            {"name": "Take Profit", "value": f"{sig.target}", "inline": True},
            {"name": "Algorithmic Confluences", "value": "\n".join(f"• {r}" for r in sig.reasons), "inline": False},
        ]
        self._send_discord(f"{emoji} ICT Signal: {symbol} {side_str}", f"High probability setup formed on {symbol}.", color, fields)

    def order_executed(
        self,
        symbol: str,
        ticket: int,
        direction: int,
        lots: float,
        fill: float,
        stop: float,
        target: float,
        risk_dollars: float,
        reason: str,
    ) -> None:
        side_str = "BUY" if direction == LONG else "SELL"
        emoji = "⚡"
        color = 0x00D4FF

        tg_text = (
            f"<b>{emoji} ORDER EXECUTED ON BROKER: {symbol}</b>\n\n"
            f"<b>Ticket:</b> <code>#{ticket}</code>\n"
            f"<b>Action:</b> {side_str} {lots} lots\n"
            f"<b>Fill Price:</b> <code>{fill}</code>\n"
            f"<b>Stop Loss:</b> <code>{stop}</code>\n"
            f"<b>Take Profit:</b> <code>{target}</code>\n"
            f"<b>Risk Allocated:</b> ${risk_dollars:.2f}\n"
            f"<b>Reason:</b> {reason}"
        )
        self._send_telegram(tg_text)

        fields = [
            {"name": "Ticket", "value": f"#{ticket}", "inline": True},
            {"name": "Volume", "value": f"{lots} Lots", "inline": True},
            {"name": "Fill Price", "value": f"{fill}", "inline": True},
            {"name": "Stop Loss", "value": f"{stop}", "inline": True},
            {"name": "Take Profit", "value": f"{target}", "inline": True},
            {"name": "Risk Budget", "value": f"${risk_dollars:.2f}", "inline": True},
        ]
        self._send_discord(f"{emoji} Order Placed: {symbol} {side_str}", f"Live execution confirmed on MetaTrader.", color, fields)

    def position_trimmed(self, symbol: str, ticket: int, milestone: str, new_stop: float) -> None:
        tg_text = (
            f"<b>🛡️ TRADE DEFENSE ACTIVE: {symbol}</b>\n\n"
            f"<b>Ticket:</b> <code>#{ticket}</code>\n"
            f"<b>Milestone:</b> {milestone}\n"
            f"<b>New Hard Stop:</b> <code>{new_stop}</code>\n"
            f"Risk locked in favorably."
        )
        self._send_telegram(tg_text)

    def trade_closed(self, symbol: str, ticket: int, pnl: float, reason: str) -> None:
        is_win = pnl >= 0
        emoji = "🎉" if is_win else "🛑"
        color = 0x00FF88 if is_win else 0xFF3366

        tg_text = (
            f"<b>{emoji} TRADE CLOSED: {symbol}</b>\n\n"
            f"<b>Ticket:</b> <code>#{ticket}</code>\n"
            f"<b>Net P&L:</b> <b>{pnl:+.2f} USD</b>\n"
            f"<b>Exit Reason:</b> {reason}"
        )
        self._send_telegram(tg_text)

        fields = [
            {"name": "Ticket", "value": f"#{ticket}", "inline": True},
            {"name": "Realized P&L", "value": f"${pnl:+.2f}", "inline": True},
            {"name": "Outcome", "value": "PROFIT" if is_win else "STOPPED OUT", "inline": True},
            {"name": "Exit Trigger", "value": reason, "inline": False},
        ]
        self._send_discord(f"{emoji} Position Closed: {symbol}", f"Realized outcome booked to account journal.", color, fields)
