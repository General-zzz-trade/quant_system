from __future__ import annotations

from typing import Protocol


class AlertSink(Protocol):
    def notify(self, title: str, message: str) -> None:
        ...
