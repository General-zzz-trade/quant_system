"""The Notifier port as the process log: used when no alert channel is configured."""
from __future__ import annotations

import logging

logger = logging.getLogger("alerts")
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}


class LogNotifier:
    channel = "log only"

    def alert(self, level: str, title: str, detail: str = "") -> None:
        if detail:
            logger.log(_LEVELS.get(level, logging.ERROR), "%s: %s", title, detail)
        else:
            logger.log(_LEVELS.get(level, logging.ERROR), "%s", title)


__all__ = ["LogNotifier"]
