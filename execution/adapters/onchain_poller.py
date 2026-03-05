# execution/adapters/onchain_poller.py
"""Lightweight daemon thread that polls Coin Metrics Community API for on-chain metrics."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_API_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_METRICS = "FlowInExUSD,FlowOutExUSD,SplyExNtv,AdrActCnt,TxTfrCnt,HashRate"


class OnchainPoller:
    """Polls Coin Metrics for on-chain metrics and caches latest values."""

    def __init__(
        self,
        asset: str = "btc",
        interval_sec: float = 3600.0,
    ) -> None:
        self._asset = asset
        self._interval = interval_sec
        self._data: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("OnchainPoller started for %s (interval=%.0fs)", self._asset, self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_current(self) -> Optional[Dict[str, float]]:
        """Return latest on-chain metrics dict, or None if not yet fetched."""
        with self._lock:
            return dict(self._data) if self._data is not None else None

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("OnchainPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        url = (
            f"{_API_BASE}?assets={self._asset}&metrics={_METRICS}"
            f"&page_size=1&sort=time"
        )
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "quant-system/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())

        rows = body.get("data", [])
        if not rows:
            return

        row = rows[0]
        parsed: Dict[str, float] = {}
        for key in ("FlowInExUSD", "FlowOutExUSD", "SplyExNtv", "AdrActCnt", "TxTfrCnt", "HashRate"):
            val = row.get(key)
            if val is not None and val != "":
                parsed[key] = float(val)

        if parsed:
            with self._lock:
                self._data = parsed
            logger.debug(
                "OnchainPoller %s: %d metrics fetched",
                self._asset,
                len(parsed),
            )
