"""The external heartbeat (ADR-0016 point 3, S7): a ping to a monitoring address at most once a minute.

The address is a secret: it is never logged, never in the configuration record and not in this object's text.
A ping goes out on its own thread with a timeout, so a slow or unreachable monitor never holds up the loop;
a ping still in flight is not doubled. A failure is logged by exception type only and never reaches the caller:
the heartbeat is a bystander and must not become a trading failure.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)
MIN_INTERVAL_SECONDS = 60.0
TIMEOUT_SECONDS = 5.0


def _get(url: str, timeout: float) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        response.read()


class HttpHeartbeat:
    def __init__(self, url: str, *, get: Callable[[str, float], object] = _get,
                 clock: Callable[[], float] = time.monotonic, min_interval: float = MIN_INTERVAL_SECONDS,
                 timeout: float = TIMEOUT_SECONDS) -> None:
        self._url, self._get, self._clock = url, get, clock
        self._min_interval, self._timeout = min_interval, timeout
        self._last: float | None = None
        self._worker: threading.Thread | None = None

    def __repr__(self) -> str:
        return "HttpHeartbeat(<address hidden>)"

    def beat(self) -> None:
        """Ping if a minute has passed since the last ping and none is in flight; never raises."""
        try:
            now = self._clock()
            if self._last is not None and now - self._last < self._min_interval:
                return
            if self._worker is not None and self._worker.is_alive():
                return
            self._last = now
            self._worker = threading.Thread(target=self._send, name="heartbeat", daemon=True)
            self._worker.start()
        except Exception as exc:                      # a bystander: nothing it does may stop the caller
            logger.warning("heartbeat not sent (%s)", type(exc).__name__)

    def flush(self, timeout: float) -> None:
        """Wait for a ping in flight (tests and shutdown)."""
        if self._worker is not None:
            self._worker.join(timeout)

    def _send(self) -> None:
        try:
            self._get(self._url, self._timeout)
        except Exception as exc:
            logger.warning("heartbeat not delivered (%s)", type(exc).__name__)


__all__ = ["HttpHeartbeat"]
