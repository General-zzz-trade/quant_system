"""Simple Binance REST feed for BTC 1m klines -> RSI computation.

5-minute maker loop only needs 1m klines every ~60s. REST is sufficient;
no WebSocket needed at this granularity.
"""
from __future__ import annotations

import json
import logging
import time
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
_USER_AGENT = "quant-system/1.0"


def fetch_btc_1m_closes(
    symbol: str = "BTCUSDT",
    limit: int = 30,
    timeout_s: float = 10.0,
) -> List[Tuple[int, float]]:
    """Fetch recent 1-minute kline close prices from Binance.

    Returns list of (timestamp_ms, close_price) tuples, oldest first.
    """
    url = f"{_BINANCE_KLINE_URL}?symbol={symbol}&interval=1m&limit={limit}"
    req = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    })
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
        return [(int(k[0]), float(k[4])) for k in data]
    except Exception:
        logger.warning("Failed to fetch Binance 1m klines", exc_info=True)
        return []


def fetch_btc_price(
    symbol: str = "BTCUSDT",
    timeout_s: float = 5.0,
) -> Optional[float]:
    """Fetch current BTC price from Binance ticker."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    req = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    })
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
        return float(data["price"])
    except Exception:
        logger.warning("Failed to fetch Binance ticker", exc_info=True)
        return None


class BinanceFeed:
    """Stateful Binance 1m feed with RSI computation.

    Caches klines and only fetches new data when stale (>55s since last fetch).
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        cache_ttl_s: float = 55.0,
    ) -> None:
        self._symbol = symbol
        self._cache_ttl_s = cache_ttl_s
        self._last_fetch_ts: float = 0.0
        self._closes: List[float] = []

    def refresh(self) -> List[float]:
        """Fetch fresh 1m closes if cache is stale. Returns close prices."""
        now = time.time()
        if now - self._last_fetch_ts < self._cache_ttl_s and self._closes:
            return self._closes
        klines = fetch_btc_1m_closes(symbol=self._symbol, limit=30)
        if klines:
            self._closes = [c for _, c in klines]
            self._last_fetch_ts = now
        return self._closes

    def get_latest_price(self) -> Optional[float]:
        """Return latest close price from cache, or fetch."""
        closes = self.refresh()
        return closes[-1] if closes else None

    def compute_rsi(self, period: int = 5) -> Optional[float]:
        """Compute RSI(period) from cached 1m closes.

        Returns None if insufficient data.
        """
        closes = self.refresh()
        if len(closes) < period + 1:
            return None
        return _rsi(closes, period)


def _rsi(closes: List[float], period: int) -> float:
    """Compute RSI from a list of close prices."""
    if len(closes) < period + 1:
        return 50.0

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    # Use Wilder smoothing (same as RSI5mStrategy)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
