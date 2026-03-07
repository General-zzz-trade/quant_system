# execution/adapters/macro_poller.py
"""Daemon thread that polls Yahoo Finance for DXY, SPX, VIX daily data."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Yahoo Finance v8 chart API — no auth required
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_TICKERS = {
    "dxy": "DX-Y.NYB",
    "spx": "%5EGSPC",   # ^GSPC
    "vix": "%5EVIX",    # ^VIX
}


class MacroPoller:
    """Polls Yahoo Finance for DXY, SPX, VIX and caches latest values."""

    def __init__(self, interval_sec: float = 3600.0) -> None:
        self._interval = interval_sec
        self._data: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_success_ts: Optional[float] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MacroPoller started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_current(self) -> Optional[Dict[str, float]]:
        with self._lock:
            return dict(self._data) if self._data is not None else None

    def is_fresh(self, max_age_sec: float = 7200.0) -> bool:
        if self._last_success_ts is None:
            return False
        return (time.monotonic() - self._last_success_ts) < max_age_sec

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch()
            except Exception:
                logger.exception("MacroPoller fetch error")
            deadline = time.monotonic() + self._interval
            while self._running and time.monotonic() < deadline:
                time.sleep(1.0)

    def _fetch(self) -> None:
        result: Dict[str, float] = {}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result["date"] = today

        for key, ticker in _TICKERS.items():
            try:
                url = f"{_YF_BASE}/{ticker}?range=5d&interval=1d"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (quant-system/1.0)",
                })
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())

                chart = data.get("chart", {}).get("result", [{}])[0]
                closes = chart.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                if closes:
                    # Use last non-null close
                    for c in reversed(closes):
                        if c is not None:
                            result[key] = float(c)
                            break
            except Exception:
                logger.warning("MacroPoller: failed to fetch %s (%s)", key, ticker)

        if len(result) > 1:  # at least date + one ticker
            with self._lock:
                self._data = result
            self._last_success_ts = time.monotonic()
            logger.debug("MacroPoller: %s", {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in result.items()})
