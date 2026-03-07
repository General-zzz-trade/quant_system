# execution/adapters/sentiment_poller.py
"""Daemon thread that polls social sentiment data.

Uses CoinGecko trending API (free, no auth) as a proxy for social volume,
since LunarCrush requires API key.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# CoinGecko trending endpoint (free, no auth)
_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
# Alternative Fear & Greed (already have FGI poller, but this gives social signal)
_CRYPTO_PANIC_URL = "https://cryptopanic.com/api/free/v1/posts/?auth_token=free&public=true&currencies=BTC"


class SentimentPoller:
    """Polls public APIs for social sentiment signals."""

    def __init__(self, interval_sec: float = 1800.0) -> None:
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
        logger.info("SentimentPoller started (interval=%.0fs)", self._interval)

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
                logger.exception("SentimentPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        result: Dict[str, float] = {}

        # Try CoinGecko trending for social volume proxy
        try:
            req = urllib.request.Request(_TRENDING_URL, headers={
                "User-Agent": "quant-system/1.0",
            })
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())

            coins = data.get("coins", [])
            # Social volume proxy: BTC's rank in trending (lower = more social activity)
            btc_score = 0.0
            for i, coin in enumerate(coins):
                item = coin.get("item", {})
                if item.get("symbol", "").upper() == "BTC":
                    # Score: higher when BTC is trending higher
                    btc_score = max(0, 10 - i) / 10.0
                    break
            result["social_volume"] = btc_score * 100  # scale to 0-100
            result["sentiment_score"] = btc_score  # 0-1 range
        except Exception:
            logger.debug("CoinGecko trending fetch failed")

        if result:
            with self._lock:
                self._data = result
            self._last_updated = time.monotonic()
            logger.debug("SentimentPoller: %s", result)
