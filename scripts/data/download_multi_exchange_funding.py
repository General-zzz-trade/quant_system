"""Download funding rates from multiple exchanges for spread computation.

Usage:
    python3 -m scripts.data.download_multi_exchange_funding --symbols ETHUSDT BTCUSDT
"""
from __future__ import annotations
import argparse
import logging

_log = logging.getLogger(__name__)


def fetch_binance_funding(symbol: str) -> float | None:
    """Fetch current funding rate from Binance USDT-M."""
    try:
        import urllib.request
        import json
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return float(data["lastFundingRate"])
    except Exception as e:
        _log.warning("Binance funding fetch failed: %s", e)
        return None


def fetch_bybit_funding(symbol: str) -> float | None:
    """Fetch current funding rate from Bybit V5."""
    try:
        import urllib.request
        import json
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            items = data.get("result", {}).get("list", [])
            if items:
                return float(items[0].get("fundingRate", 0))
        return None
    except Exception as e:
        _log.warning("Bybit funding fetch failed: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    args = parser.parse_args()

    for symbol in args.symbols:
        rates = {
            "binance": fetch_binance_funding(symbol),
            "bybit": fetch_bybit_funding(symbol),
        }
        _log.info("%s funding rates: %s", symbol, rates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
