"""Data fetching utilities for the Bybit alpha runner."""
from __future__ import annotations

import json
import logging
import threading
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class BinanceOICache:
    """Background thread that refreshes Binance OI/LS/Taker data every 55s.

    Prevents inline REST calls from blocking the bar-processing thread
    (4 sequential calls, up to 12s each) which would stop stop-loss checks.
    """

    _REFRESH_INTERVAL = 55  # seconds (just under the 1m candlestick cadence)
    _NAN_DEFAULTS = {
        "open_interest": float("nan"),
        "ls_ratio": float("nan"),
        "top_trader_ls_ratio": float("nan"),
        "taker_buy_vol": 0.0,
        "taker_sell_vol": 0.0,
    }

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._lock = threading.Lock()
        self._data: dict = dict(self._NAN_DEFAULTS)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background refresh thread (daemon — exits with main process)."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"oi-cache-{self._symbol}", daemon=True
        )
        self._thread.start()
        logger.info("BinanceOICache %s started", self._symbol)

    def stop(self) -> None:
        """Signal background thread to stop."""
        self._stop_event.set()

    def get(self) -> dict:
        """Return latest cached OI data (thread-safe, never blocks)."""
        with self._lock:
            return dict(self._data)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                fresh = _fetch_binance_oi_data(self._symbol)
                with self._lock:
                    self._data = fresh
            except Exception as exc:
                logger.warning("BinanceOICache %s refresh error: %s", self._symbol, exc)
            self._stop_event.wait(timeout=self._REFRESH_INTERVAL)


def _fetch_binance_oi_data(symbol: str) -> dict:
    """Fetch latest OI, Long/Short ratio, Taker ratio, Top trader ratio from Binance.

    Returns dict with keys: open_interest, ls_ratio, top_trader_ls_ratio,
    taker_buy_vol, taker_sell_vol. Values are float or NaN if unavailable.
    Uses Binance public API (no auth required).
    """
    result = {
        "open_interest": float("nan"),
        "ls_ratio": float("nan"),
        "top_trader_ls_ratio": float("nan"),
        "taker_buy_vol": 0.0,
        "taker_sell_vol": 0.0,
    }
    base = "https://fapi.binance.com"
    headers = {"Accept": "application/json"}
    timeout = 3  # fast timeout, non-critical

    # OI
    try:
        url = f"{base}/fapi/v1/openInterest?symbol={symbol}"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        result["open_interest"] = float(data.get("openInterest", 0))
    except Exception as e:
        logger.warning("Failed to fetch OI for %s: %s", symbol, e)

    # Global Long/Short ratio
    try:
        url = f"{base}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["ls_ratio"] = float(data[0].get("longShortRatio", 1))
    except Exception as e:
        logger.warning("Failed to fetch LS ratio for %s: %s", symbol, e)

    # Top trader position ratio
    try:
        url = f"{base}/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["top_trader_ls_ratio"] = float(data[0].get("longShortRatio", 1))
    except Exception as e:
        logger.warning("Failed to fetch top trader LS ratio for %s: %s", symbol, e)

    # Taker buy/sell volume
    try:
        url = f"{base}/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["taker_buy_vol"] = float(data[0].get("buyVol", 0))
            result["taker_sell_vol"] = float(data[0].get("sellVol", 0))
    except Exception as e:
        logger.warning("Failed to fetch taker volume for %s: %s", symbol, e)

    return result
