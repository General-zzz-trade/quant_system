"""The Notifier port as a Telegram bot message (migration step 12).

A message that cannot be delivered is logged by exception type only - the bot token is part of the URL and
never written anywhere - and the caller carries on: an alert channel must never stop trading or exits.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TEXT = 4096                     # Telegram's limit for one message
TIMEOUT_SECONDS = 5.0


def _post(url: str, body: bytes) -> None:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response.read()


class TelegramNotifier:
    channel = "telegram"

    def __init__(self, token: str, chat_id: str, *, post: Callable[[str, bytes], None] = _post) -> None:
        self._url, self._chat_id, self._post = API.format(token=token), chat_id, post

    def alert(self, level: str, title: str, detail: str = "") -> None:
        text = f"[{level.upper()}] {title}" + (f"\n{detail}" if detail else "")
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT - 1] + "…"
        body = json.dumps({"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True}).encode()
        try:
            self._post(self._url, body)
        except Exception as exc:
            logger.warning("Telegram alert not delivered (%s): %s", type(exc).__name__, title)


__all__ = ["TelegramNotifier"]
