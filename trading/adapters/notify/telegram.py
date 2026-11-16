"""The Notifier port as a Telegram bot message (migration step 12).

A message that cannot be delivered is logged by exception type only - the bot token is part of the URL and
never written anywhere - and the caller carries on: an alert channel must never stop trading or exits.

With ``background`` (how the runtime uses it) messages go through a bounded queue to one sending thread, so a
slow or unreachable Telegram never holds up the trading loop; when the queue is full a new alert is dropped
and that is logged (audit after step 12b, finding 9).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TEXT = 4096                     # Telegram's limit for one message
TIMEOUT_SECONDS = 5.0
QUEUE_SIZE = 100


def _post(url: str, body: bytes) -> None:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response.read()


def _request(url: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


class TelegramNotifier:
    channel = "telegram"

    def __init__(self, token: str, chat_id: str, *, post: Callable[[str, bytes], None] = _post,
                 request: Callable[[str, bytes, float], bytes] = _request,
                 background: bool = False, queue_size: int = QUEUE_SIZE) -> None:
        self._url, self._chat_id, self._post, self._request = API.format(token=token), chat_id, post, request
        self._queue: queue.Queue[tuple[str, str, str]] | None = queue.Queue(queue_size) if background else None
        self._worker: threading.Thread | None = None

    def alert(self, level: str, title: str, detail: str = "") -> None:
        if self._queue is None:
            self._send(level, title, detail)
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._drain, name="telegram-alerts", daemon=True)
            self._worker.start()
        try:
            self._queue.put_nowait((level, title, detail))
        except queue.Full:
            logger.warning("Telegram alert dropped (queue full): %s", title)

    def deliver(self, title: str, detail: str = "", *, timeout_s: float) -> bool:
        """Send now, beside any queued alerts, and wait (ADR-0016 point 2): True only when Telegram's answer says
        ``"ok": true`` within ``timeout_s``. A failure is logged by exception type only (the token is in the URL)."""
        try:
            answer = json.loads(self._request(self._url, self._body("info", title, detail), timeout_s))
        except Exception as exc:
            logger.warning("Telegram message not confirmed (%s): %s", type(exc).__name__, title)
            return False
        if not isinstance(answer, dict) or answer.get("ok") is not True:
            logger.warning("Telegram message not confirmed (answer not ok): %s", title)
            return False
        return True

    @property
    def background(self) -> bool:
        return self._queue is not None

    def flush(self, timeout: float) -> None:
        """Wait until every queued alert has been handed to Telegram (for tests and shutdown)."""
        deadline = time.monotonic() + timeout
        while self._queue is not None and self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

    def _drain(self) -> None:
        assert self._queue is not None
        while True:
            level, title, detail = self._queue.get()
            try:
                self._send(level, title, detail)
            finally:
                self._queue.task_done()

    def _body(self, level: str, title: str, detail: str) -> bytes:
        text = f"[{level.upper()}] {title}" + (f"\n{detail}" if detail else "")
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT - 1] + "…"
        return json.dumps({"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True}).encode()

    def _send(self, level: str, title: str, detail: str) -> None:
        body = self._body(level, title, detail)
        try:
            self._post(self._url, body)
        except Exception as exc:
            logger.warning("Telegram alert not delivered (%s): %s", type(exc).__name__, title)


__all__ = ["TelegramNotifier"]
