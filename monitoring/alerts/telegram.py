"""Telegram alert sink — sends alerts via Telegram Bot API."""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

from monitoring.alerts.base import Alert, AlertSink

logger = logging.getLogger(__name__)


class TelegramAlertSink:
    """Sends alerts to a Telegram chat via Bot API (stdlib only, no deps)."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def emit(self, alert: Alert) -> None:
        text = f"*[{alert.severity.value.upper()}]* {alert.title}\n{alert.message}"
        if alert.source:
            text += f"\n_source: {alert.source}_"
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception:
            logger.warning("Telegram alert delivery failed for: %s", alert.title)
