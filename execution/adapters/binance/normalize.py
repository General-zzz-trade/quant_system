# execution/adapters/binance/normalize.py
"""Binance field normalization helpers."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional


def normalize_side(side: str) -> str:
    """BUY/SELL → buy/sell。"""
    return side.lower() if side else ""


def normalize_order_type(order_type: str) -> str:
    """LIMIT/MARKET/STOP_MARKET → limit/market/stop_market。"""
    return order_type.lower() if order_type else ""


def normalize_tif(tif: str) -> str:
    """GTC/IOC/FOK → gtc/ioc/fok。"""
    return tif.lower() if tif else ""


def normalize_order_status(status: str) -> str:
    """Binance status → canonical status。

    Binance: NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED
    """
    mapping = {
        "NEW": "new",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "cancelled",
        "CANCELLED": "cancelled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
    }
    return mapping.get(status.upper(), status.lower())


def normalize_symbol(symbol: str) -> str:
    """标准化品种名称（大写，去空格）。"""
    return symbol.strip().upper()


def to_binance_side(side: str) -> str:
    """buy/sell → BUY/SELL。"""
    return side.upper()


def to_binance_order_type(order_type: str) -> str:
    """limit/market → LIMIT/MARKET。"""
    return order_type.upper()
