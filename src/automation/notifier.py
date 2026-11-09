"""
Telegram & WhatsApp Bot Alerts Dispatcher
Dispatches VIP booking codes, bet settlement reports, sharp money steam moves,
and live second-half sniper alerts via Telegram Bot or WhatsApp click-to-chat links.
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Notifier")
logger.setLevel(logging.INFO)


class Notifier:
    """
    Unified notification engine with Telegram bot integration and WhatsApp web generation.
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    def format_slip_message(
        self,
        booking_code: str,
        combined_odds: float,
        stake: float,
        matches: List[Dict[str, Any]]
    ) -> str:
        payout = round(stake * combined_odds, 2)
        txt = (
            f"🔥 *SPORTYBET VIP ACCA TICKET* 🔥\n"
            f"📋 Code: *{booking_code.upper()}*\n"
            f"⚡ Total Odds: *{combined_odds:.2f}x*\n"
            f"💰 Stake: {stake:,.0f} NGN | Est. Return: {payout:,.2f} NGN\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
        )
        for i, m in enumerate(matches):
            h = m.get("home_team", "")
            a = m.get("away_team", "")
            pick = m.get("selected_outcome") or m.get("selection") or "Home"
            odd = float(m.get("selected_odds") or m.get("odds") or 1.5)
            txt += f"[{i+1}] {h} vs {a}\n    👉 *{pick}* @ {odd:.2f}\n"

        txt += (
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Multi-Sport Predictor V4 (Dixon-Coles & Sharp Intelligence)*"
        )
        return txt

    def generate_whatsapp_share_url(self, message: str) -> str:
        """
        Generates a 1-click WhatsApp web/app link to share the message directly.
        """
        encoded = urllib.parse.quote(message)
        return f"https://api.whatsapp.com/send?text={encoded}"

    def send_telegram_message(self, message: str) -> Dict[str, Any]:
        """
        Sends message via Telegram Bot if token and chat ID are configured.
        """
        if not self.bot_token or not self.chat_id:
            return {
                "sent": False,
                "reason": "Telegram credentials not configured in environment (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)",
                "formatted_text": message,
                "whatsapp_url": self.generate_whatsapp_share_url(message)
            }

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=7) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return {
                    "sent": True,
                    "response": res_data,
                    "whatsapp_url": self.generate_whatsapp_share_url(message)
                }
        except Exception as e:
            logger.debug(f"Telegram dispatch notice: {e}")
            return {
                "sent": False,
                "error": str(e),
                "whatsapp_url": self.generate_whatsapp_share_url(message)
            }


_notifier_instance = None


def get_notifier() -> Notifier:
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = Notifier()
    return _notifier_instance
