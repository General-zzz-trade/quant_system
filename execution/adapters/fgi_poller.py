"""Lightweight daemon thread that polls the Fear & Greed Index."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_URL = "https://api.alternative.me/fng/?limit=1"


class FGIPoller:
    """Polls alternative.me Fear & Greed Index and caches the latest value.

    Returns the raw index (0-100) via get_value().
    Polling interval defaults to 300s (API updates daily, no need to hammer).
    """

    def __init__(self, interval_sec: float = 300.0) -> None:
        self._interval = interval_sec
        self._value: Optional[float] = None
        self._last_updated: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("FGIPoller started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_value(self) -> Optional[float]:
        return self._value

    def age_seconds(self) -> Optional[float]:
        if self._last_updated is None:
            return None
        return time.monotonic() - self._last_updated

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("FGIPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        import json
        import urllib.request

        req = urllib.request.Request(_URL, headers={"User-Agent": "quant-system/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        entries = data.get("data", [])
        if entries:
            val = entries[0].get("value")
            if val is not None:
                self._value = float(val)
                self._last_updated = time.monotonic()
                logger.debug("FGIPoller value=%s", self._value)
