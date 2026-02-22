from __future__ import annotations

from .base import AlertSink


class ConsoleAlertSink:
    def notify(self, title: str, message: str) -> None:
        print(f"[ALERT] {title}: {message}")
