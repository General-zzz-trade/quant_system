"""Unified notification dispatcher — Telegram + console fallback.

Sends alerts to Telegram Bot API when configured, falls back to console logging.
Used by health_watchdog, auto_retrain, runtime_health_check, etc.

Setup:
1. Create bot via @BotFather on Telegram → get BOT_TOKEN
2. Send /start to your bot → get CHAT_ID via https://api.telegram.org/bot<TOKEN>/getUpdates
3. Add to .env:
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id

Usage:
    from monitoring.notify import send_alert, AlertLevel
    send_alert(AlertLevel.WARNING, "Model IC decaying", details={"symbol": "BTCUSDT"})
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("notify")


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Emoji mapping
_EMOJI = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.CRITICAL: "🚨",
}


def _load_telegram_config() -> tuple[str, str]:
    """Load Telegram credentials from env or .env file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("TELEGRAM_CHAT_ID="):
                        chat_id = line.split("=", 1)[1].strip().strip('"').strip("'")

    return token, chat_id


def send_telegram(text: str, token: str = "", chat_id: str = "") -> bool:
    """Send a message via Telegram Bot API."""
    if not token or not chat_id:
        token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }).encode()

    try:
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=10)
        result = json.loads(resp.read())
        if not result.get("ok"):
            logger.warning("Telegram API error: %s", result)
            return False
        return True
    except URLError as e:
        logger.debug("Telegram send failed: %s", e)
        return False
    except Exception as e:
        logger.debug("Telegram error: %s", e)
        return False


def send_alert(
    level: AlertLevel,
    title: str,
    details: dict | None = None,
    source: str = "quant_system",
) -> bool:
    """Send an alert via all configured channels.

    Returns True if at least one channel succeeded.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = _EMOJI.get(level, "")

    # Format message
    lines = [f"{emoji} *[{level.value}]* {title}", f"📅 {now}", f"🖥 {source}"]
    if details:
        for k, v in details.items():
            lines.append(f"  • {k}: `{v}`")
    text = "\n".join(lines)

    # Console always
    log_fn = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.CRITICAL: logger.error,
    }.get(level, logger.info)
    log_fn("ALERT [%s] %s %s", level.value, title, details or "")

    # Telegram
    tg_ok = send_telegram(text)
    if not tg_ok:
        logger.debug("Telegram not configured or failed — console only")

    # Save to alert history
    try:
        history_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "runtime", "alert_history.jsonl"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "a") as f:
            f.write(json.dumps({
                "ts": now, "level": level.value, "title": title,
                "details": details, "telegram": tg_ok,
            }) + "\n")
    except Exception:
        pass

    return tg_ok


def test_notification() -> None:
    """Send a test notification to verify Telegram setup."""
    token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        print("Telegram not configured.")
        print("Add to .env:")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")
        print()
        print("To get these:")
        print("  1. Message @BotFather on Telegram → /newbot → get token")
        print("  2. Message your bot → /start")
        print("  3. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates → get chat_id")
        return

    ok = send_alert(
        AlertLevel.INFO,
        "Test notification — system is working",
        details={"status": "connected", "bot": token[:8] + "..."},
        source="health_watchdog",
    )
    if ok:
        print("✓ Telegram notification sent successfully!")
    else:
        print("✗ Telegram notification failed. Check token/chat_id.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_notification()
