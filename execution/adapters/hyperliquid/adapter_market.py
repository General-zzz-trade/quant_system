"""Market data and account methods for HyperliquidAdapter.

Extracted from adapter.py to keep main file under 500 lines.
"""
from __future__ import annotations

import logging
from typing import Any

from execution.adapters.hyperliquid.mapper import (
    coin_to_symbol,
    map_orderbook,
    normalize_coin,
)

logger = logging.getLogger(__name__)


def get_ticker(adapter, symbol: str) -> dict:
    """Get latest ticker (mid price, bid, ask, funding rate)."""
    coin = normalize_coin(symbol)

    book_data = adapter._client.info_request({"type": "l2Book", "coin": coin})
    if not isinstance(book_data, dict) or "levels" not in book_data:
        return {}

    book = map_orderbook(book_data)
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask

    funding_rate = 0.0
    if not adapter._asset_ctxs:
        adapter._refresh_meta()
    if coin.upper() in adapter._asset_index:
        idx = adapter._asset_index[coin.upper()]
        if idx < len(adapter._asset_ctxs):
            ctx = adapter._asset_ctxs[idx]
            funding_rate = float(ctx.get("funding", 0))

    return {
        "symbol": coin_to_symbol(coin),
        "lastPrice": mid,
        "bid1Price": best_bid,
        "ask1Price": best_ask,
        "volume24h": 0.0,
        "fundingRate": funding_rate,
    }


def get_orderbook(adapter, symbol: str) -> dict:
    """Get L2 orderbook."""
    coin = normalize_coin(symbol)
    data = adapter._client.info_request({"type": "l2Book", "coin": coin})
    if not isinstance(data, dict) or "levels" not in data:
        return {"bids": [], "asks": [], "coin": coin, "ts_ms": 0}
    return map_orderbook(data)


def get_funding_rates(adapter) -> list[dict]:
    """Get current funding rates for all assets."""
    if not adapter._asset_ctxs:
        adapter._refresh_meta()

    rates = []
    for i, ctx in enumerate(adapter._asset_ctxs):
        if i < len(adapter._universe):
            coin = adapter._universe[i].get("name", "")
            rates.append({
                "symbol": coin_to_symbol(coin),
                "fundingRate": float(ctx.get("funding", 0)),
                "markPx": float(ctx.get("markPx", 0)),
                "openInterest": float(ctx.get("openInterest", 0)),
                "volume24h": float(ctx.get("dayNtlVlm", 0)),
            })
    return rates


def get_account_info(adapter) -> dict:
    """Get detailed account information."""
    bal = adapter.get_balances()
    result: dict[str, Any] = {
        "venue": "hyperliquid",
        "is_testnet": adapter._config.is_testnet,
    }
    for b in bal.balances:
        result[f"{b.asset}_free"] = float(b.free)
        result[f"{b.asset}_locked"] = float(b.locked)
        result[f"{b.asset}_total"] = float(b.total)
    return result


def parse_order_response(data: Any, coin: str, side: str, qty: float) -> dict:
    """Parse exchange response for order submission."""
    if isinstance(data, dict):
        if data.get("status") == "ok":
            response = data.get("response", {})
            if response.get("type") == "order":
                statuses = response.get("data", {}).get("statuses", [])
                if statuses:
                    s = statuses[0]
                    if "resting" in s:
                        order_info = s["resting"]
                        return {
                            "orderId": str(order_info.get("oid", "")),
                            "status": "submitted",
                        }
                    elif "filled" in s:
                        fill_info = s["filled"]
                        return {
                            "orderId": str(fill_info.get("oid", "")),
                            "status": "filled",
                            "avgPrice": fill_info.get("avgPx"),
                            "filledQty": fill_info.get("totalSz"),
                        }
                    elif "error" in s:
                        return {"status": "error", "msg": s["error"]}
            logger.info(
                "Hyperliquid %s %s %.6f submitted", side, coin, qty,
            )
            return {"status": "submitted", "data": data}
        elif data.get("status") == "error":
            return {"status": "error", "msg": data.get("msg", str(data))}
    return {"status": "error", "msg": str(data)}
