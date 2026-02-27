"""Alert delivery channels: Telegram and generic webhooks."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict

from monitoring.alerts.base import Alert

logger = logging.getLogger(__name__)


# ── Telegram ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Configuration for Telegram bot alerts."""

    bot_token: str
    chat_id: str
    api_url: str = "https://api.telegram.org"


class TelegramAlertSink:
    """Send alerts via Telegram bot."""

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config

    def emit(self, alert: Alert) -> None:
        """Send alert as Telegram message."""
        sev = alert.severity.value.upper()
        text = f"[{sev}] {alert.title}\n{alert.message}"
        if alert.source:
            text += f"\nSource: {alert.source}"

        url = (
            f"{self._config.api_url}/bot{self._config.bot_token}/sendMessage"
        )
        data = urllib.parse.urlencode(
            {"chat_id": self._config.chat_id, "text": text, "parse_mode": "HTML"}
        ).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)


# ── Webhook ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WebhookConfig:
    """Configuration for generic HTTP webhook alerts."""

    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 10.0


class WebhookAlertSink:
    """Send alerts via HTTP webhook (POST JSON)."""

    def __init__(self, config: WebhookConfig) -> None:
        self._config = config

    def emit(self, alert: Alert) -> None:
        """POST alert payload as JSON to webhook URL."""
        data = json.dumps(alert.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            self._config.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **self._config.headers,
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=self._config.timeout_sec)
        except Exception as e:
            logger.warning("Webhook send failed: %s", e)
