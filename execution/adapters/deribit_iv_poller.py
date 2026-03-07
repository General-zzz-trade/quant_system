# execution/adapters/deribit_iv_poller.py
"""Lightweight daemon thread that polls Deribit for implied volatility and put/call ratio."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_API_BASE = "https://www.deribit.com/api/v2/public"


class DeribitIVPoller:
    """Polls Deribit public API and caches IV + put/call ratio."""

    def __init__(
        self,
        currency: str = "BTC",
        interval_sec: float = 300.0,
    ) -> None:
        self._currency = currency
        self._interval = interval_sec
        self._implied_vol: Optional[float] = None
        self._put_call_ratio: Optional[float] = None
        self._last_updated: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("DeribitIVPoller started for %s (interval=%.0fs)", self._currency, self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_current(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (implied_vol, put_call_ratio) or (None, None) if not yet fetched."""
        return self._implied_vol, self._put_call_ratio

    def age_seconds(self) -> Optional[float]:
        if self._last_updated is None:
            return None
        return time.monotonic() - self._last_updated

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("DeribitIVPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        # Fetch latest historical volatility (last entry)
        url_hv = f"{_API_BASE}/get_historical_volatility?currency={self._currency}"
        req = urllib.request.Request(url_hv)
        req.add_header("User-Agent", "quant-system/1.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        result = data.get("result", [])
        if result:
            last = result[-1]
            if isinstance(last, list) and len(last) >= 2:
                self._implied_vol = float(last[1]) / 100.0

        # Fetch options book summary for put/call ratio
        url_bs = f"{_API_BASE}/get_book_summary_by_currency?currency={self._currency}&kind=option"
        req2 = urllib.request.Request(url_bs)
        req2.add_header("User-Agent", "quant-system/1.0")
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            data2 = json.loads(resp2.read())
        summaries = data2.get("result", [])
        put_oi = 0.0
        call_oi = 0.0
        for s in summaries:
            name = s.get("instrument_name", "")
            oi = float(s.get("open_interest", 0))
            if "-P" in name:
                put_oi += oi
            elif "-C" in name:
                call_oi += oi
        if call_oi > 1e-8:
            self._put_call_ratio = put_oi / call_oi

        self._last_updated = time.monotonic()
        logger.debug(
            "DeribitIV %s: iv=%.4f pcr=%.4f",
            self._currency,
            self._implied_vol or 0.0,
            self._put_call_ratio or 0.0,
        )
