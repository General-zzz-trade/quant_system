# execution/adapters/mempool_poller.py
"""Daemon thread that polls mempool.space API for BTC mempool/fee data."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_FEES_URL = "https://mempool.space/api/v1/fees/recommended"
_MEMPOOL_URL = "https://mempool.space/api/mempool"


class MempoolPoller:
    """Polls mempool.space for fee rates and mempool size."""

    def __init__(self, interval_sec: float = 600.0) -> None:
        self._interval = interval_sec
        self._data: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_updated: Optional[float] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MempoolPoller started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_current(self) -> Optional[Dict[str, float]]:
        with self._lock:
            return dict(self._data) if self._data is not None else None

    def age_seconds(self) -> Optional[float]:
        """Seconds since last successful fetch, or None if never fetched."""
        if self._last_updated is None:
            return None
        return time.monotonic() - self._last_updated

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("MempoolPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        headers = {"User-Agent": "quant-system/1.0"}

        # Fetch fee rates
        req = urllib.request.Request(_FEES_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            fees = json.loads(resp.read())

        # Fetch mempool stats
        req2 = urllib.request.Request(_MEMPOOL_URL, headers=headers)
        with urllib.request.urlopen(req2, timeout=5) as resp:
            mempool = json.loads(resp.read())

        result = {
            "fastest_fee": float(fees.get("fastestFee", 0)),
            "half_hour_fee": float(fees.get("halfHourFee", 0)),
            "hour_fee": float(fees.get("hourFee", 0)),
            "economy_fee": float(fees.get("economyFee", 0)),
            "minimum_fee": float(fees.get("minimumFee", 0)),
            "mempool_size": float(mempool.get("vsize", 0)),
            "mempool_count": float(mempool.get("count", 0)),
        }

        with self._lock:
            self._data = result
        self._last_updated = time.monotonic()
        logger.debug("MempoolPoller: fastest=%.0f economy=%.0f size=%.0f",
                      result["fastest_fee"], result["economy_fee"], result["mempool_size"])
