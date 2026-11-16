"""The Notifier port as lines appended to a file (owner, 2026-11-16: no Telegram; a Claude Code session reads the
file about once an hour, and the external heartbeat stays mandatory).

One JSON object per line: ``at`` (seconds), ``utc``, ``level``, ``title``, ``detail``. Each line is written and
synced at once, so there is no queue. A write that fails is logged and never raised: an alert channel must never
stop trading or exits. ``deliver`` is the confirmed send of the start-up alert: True only when the line is in the
file. The file's directory is made by the host installation, not here: a missing directory is a set-up error and
refuses the start.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("alerts")


class FileNotifier:
    channel = "file"
    background = False

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self._clock = Path(path), clock

    def alert(self, level: str, title: str, detail: str = "") -> None:
        try:
            self._append(level, title, detail)
        except OSError as exc:
            logger.error("Alert not written to the alert file (%s): %s: %s", type(exc).__name__, title, detail)

    def deliver(self, title: str, detail: str = "", *, timeout_s: float) -> bool:
        """Write now; True only when the line was written and synced (ADR-0016 point 2, for a file)."""
        try:
            self._append("info", title, detail)
        except OSError as exc:
            logger.warning("Alert file not writable (%s): %s", type(exc).__name__, title)
            return False
        return True

    def flush(self, timeout: float) -> None:
        """Nothing is queued."""

    def _append(self, level: str, title: str, detail: str) -> None:
        at = float(self._clock())
        utc = datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = json.dumps({"at": at, "utc": utc, "level": level, "title": title, "detail": detail},
                          ensure_ascii=False) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["FileNotifier"]
