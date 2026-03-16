"""Data fetching utilities for the Bybit alpha runner."""
from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


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
