"""Lightweight daemon thread that polls Binance for current Open Interest."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_URL_PROD = "https://fapi.binance.com/fapi/v1/openInterest"
_URL_TESTNET = "https://testnet.binancefuture.com/fapi/v1/openInterest"


class BinanceOIPoller:
    """Polls Binance openInterest endpoint and caches the latest OI value."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval_sec: float = 60.0,
        testnet: bool = False,
    ) -> None:
        self._symbol = symbol
        self._interval = interval_sec
        self._base_url = _URL_TESTNET if testnet else _URL_PROD
        self._oi: Optional[float] = None
        self._prev_oi: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("OIPoller started for %s (interval=%.0fs)", self._symbol, self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_oi(self) -> Optional[float]:
        return self._oi

    def get_oi_change_pct(self) -> Optional[float]:
        if self._oi is None or self._prev_oi is None or self._prev_oi == 0:
            return None
        return (self._oi - self._prev_oi) / self._prev_oi

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("OIPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        import json
        import urllib.request

        url = f"{self._base_url}?symbol={self._symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "quant-system/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        oi_str = data.get("openInterest")
        if oi_str is not None:
            self._prev_oi = self._oi
            self._oi = float(oi_str)
            logger.debug("OIPoller %s oi=%.2f", self._symbol, self._oi)
