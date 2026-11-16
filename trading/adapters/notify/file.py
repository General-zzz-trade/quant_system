"""The Notifier port as lines appended to a file (owner, 2026-11-16: no Telegram; a Claude Code session reads the
file about once an hour, and the external heartbeat stays mandatory).

One JSON object per line: ``at`` (seconds), ``utc``, ``level``, ``title``, ``detail``. Each line is written and
synced at once, so there is no queue. A write that fails for any reason is logged and never raised: an alert channel
must never stop trading or exits. ``deliver`` is the confirmed send (the start-up alert, a refused start, a failed
unit): True only when the whole line is in the file, and its lines carry the level ``notice`` so that a reader
ranking by level never puts them last.

Hardened after the audit of 2026-11-16:
- only a plain file is written: a symbolic link is not followed, a pipe or a directory is refused and never blocks;
- a short write (a full disk, a size limit) is a failure, not a confirmation, and the next line starts on its own
  line after a cut-off one;
- title and detail are cut (a venue's error body can be long) and text that cannot be encoded is escaped.
The file's directory is made by the host installation, not here: a missing directory is a set-up error and refuses
the start. A reader that wants to rotate the file moves it away; it must not create the new one (the runtime does,
as its own user).
"""
from __future__ import annotations

import errno
import json
import logging
import os
import stat
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("alerts")
MAX_TITLE, MAX_DETAIL = 300, 4000


def _cut(text: Any, limit: int) -> str:
    whole = text if isinstance(text, str) else str(text)
    return whole if len(whole) <= limit else whole[:limit - 1] + "…"


class FileNotifier:
    channel = "file"
    background = False

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self._clock = Path(path), clock

    def alert(self, level: str, title: str, detail: str = "") -> None:
        self.write(level, title, detail)

    def deliver(self, title: str, detail: str = "", *, timeout_s: float) -> bool:
        """Write now; True only when the whole line was written and synced (ADR-0016 point 2, for a file)."""
        return self.write("notice", title, detail)

    def write(self, level: str, title: Any, detail: Any = "") -> bool:
        """Append one line; False, logged and never raised, when it could not be written whole."""
        try:
            self._append(level, title, detail)
        except Exception as exc:
            try:                                           # the reason too: a link, a pipe, a full disk, no permission
                why = f"{type(exc).__name__} errno {exc.errno}" if isinstance(exc, OSError) else type(exc).__name__
                logger.error("Alert not written to the alert file (%s): %s: %s", why,
                             _cut(title, MAX_TITLE), _cut(detail, MAX_DETAIL))
            except Exception:                              # text that cannot even be turned into a string
                logger.error("Alert not written to the alert file, and its text could not be logged")
            return False
        return True

    def flush(self, timeout: float) -> None:
        """Nothing is queued."""

    def _append(self, level: str, title: Any, detail: Any) -> None:
        at = float(self._clock())
        utc = datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {"at": at, "utc": utc, "level": str(level), "title": _cut(title, MAX_TITLE),
                  "detail": _cut(detail, MAX_DETAIL)}
        data = json.dumps(record, ensure_ascii=False).encode("utf-8", errors="backslashreplace") + b"\n"
        # Never follow a link and never wait for a pipe's reader; what was opened must be a plain file.
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o640)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EINVAL, "the alert file is not a plain file")
            if info.st_size and os.pread(fd, 1, info.st_size - 1) != b"\n":
                data = b"\n" + data                        # the line before was cut off: start on a new one
            if os.write(fd, data) != len(data):
                raise OSError(errno.ENOSPC, "short write to the alert file")
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["FileNotifier", "MAX_DETAIL", "MAX_TITLE"]
