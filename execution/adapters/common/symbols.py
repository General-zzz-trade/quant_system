# execution/adapters/common/symbols.py
"""Symbol normalization utilities."""
from __future__ import annotations

from typing import Optional


def normalize_symbol(symbol: str) -> str:
    """标准化交易对名称 — 去空格、大写化。"""
    return symbol.strip().upper()


def split_symbol(symbol: str) -> tuple[str, str]:
    """
    尝试拆分交易对为 base + quote。

    支持：BTCUSDT → (BTC, USDT), BTC/USDT → (BTC, USDT)
    """
    s = symbol.strip().upper()
    for sep in ("/", "-", "_"):
        if sep in s:
            parts = s.split(sep, 1)
            return parts[0], parts[1]

    known_quotes = ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "USD")
    for q in known_quotes:
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)], q
    return s, ""


def normalize_side(side: str) -> str:
    """标准化买卖方向：BUY/buy/b/long → "buy"。"""
    v = str(side).strip().lower()
    if v in ("buy", "b", "long", "bid"):
        return "buy"
    if v in ("sell", "s", "short", "ask"):
        return "sell"
    return v


def normalize_order_type(order_type: str) -> str:
    """标准化订单类型。"""
    return str(order_type).strip().lower()


def normalize_tif(tif: Optional[str]) -> Optional[str]:
    """标准化有效期类型。"""
    if tif is None:
        return None
    v = str(tif).strip().lower()
    aliases = {"gtx": "post_only"}
    return aliases.get(v, v)
