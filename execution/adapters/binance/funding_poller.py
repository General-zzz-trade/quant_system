# execution/adapters/binance/funding_poller.py
"""Lightweight daemon thread that polls Binance for the current funding rate."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_URL_PROD = "https://fapi.binance.com/fapi/v1/premiumIndex"
_URL_TESTNET = "https://testnet.binancefuture.com/fapi/v1/premiumIndex"


class BinanceFundingPoller:
    """Polls Binance premiumIndex endpoint and caches the latest funding rate."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval_sec: float = 60.0,
        testnet: bool = False,
    ) -> None:
        self._symbol = symbol
        self._interval = interval_sec
        self._base_url = _URL_TESTNET if testnet else _URL_PROD
        self._rate: Optional[float] = None
        self._last_updated: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("FundingPoller started for %s (interval=%.0fs)", self._symbol, self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_rate(self) -> Optional[float]:
        return self._rate

    def age_seconds(self) -> Optional[float]:
        if self._last_updated is None:
            return None
        return time.monotonic() - self._last_updated

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("FundingPoller fetch error")
            # Sleep in small increments so stop() is responsive
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        import urllib.request
        import json

        url = f"{self._base_url}?symbol={self._symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "quant-system/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        rate_str = data.get("lastFundingRate")
        if rate_str is not None:
            self._rate = float(rate_str)
            self._last_updated = time.monotonic()
            logger.debug("FundingPoller %s rate=%.6f", self._symbol, self._rate)
