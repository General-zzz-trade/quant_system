# execution/adapters/binance/btc_kline_poller.py
"""Lightweight BTC kline poller for cross-asset feature computation.

Fetches BTC 1m klines and pushes close/high/low to CrossAssetComputer
so that altcoin features (btc_ret_24, btc_rsi_14, etc.) stay current.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
import json
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BASE_URL = "https://fapi.binance.com"
_TESTNET_URL = "https://testnet.binancefuture.com"


class BtcKlinePoller:
    """Polls BTC kline data and pushes to a CrossAssetComputer."""

    def __init__(
        self,
        cross_asset_computer: Any,
        *,
        symbol: str = "BTCUSDT",
        interval_sec: float = 60.0,
        testnet: bool = False,
        funding_source: Any = None,
    ) -> None:
        self._cross = cross_asset_computer
        self._symbol = symbol
        self._interval = interval_sec
        self._base = _TESTNET_URL if testnet else _BASE_URL
        self._funding_source = funding_source
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _fetch_latest(self) -> Optional[dict]:
        url = f"{self._base}/fapi/v1/klines?symbol={self._symbol}&interval=1m&limit=1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quant/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data and len(data) > 0:
                    k = data[0]
                    return {
                        "close": float(k[4]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                    }
        except Exception as e:
            logger.debug("BtcKlinePoller fetch error: %s", e)
        return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            kline = self._fetch_latest()
            if kline is not None:
                fr = None
                if self._funding_source is not None:
                    try:
                        fr = self._funding_source()
                    except Exception:
                        pass
                self._cross.on_bar(
                    self._symbol,
                    close=kline["close"],
                    high=kline["high"],
                    low=kline["low"],
                    funding_rate=fr,
                )
                logger.debug(
                    "BtcKlinePoller: %s close=%.2f high=%.2f low=%.2f",
                    self._symbol, kline["close"], kline["high"], kline["low"],
                )
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="btc-kline-poller")
        self._thread.start()
        logger.info("BtcKlinePoller started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
