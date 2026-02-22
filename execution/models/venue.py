# execution/models/venue.py
"""Venue metadata and capability description."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class VenueType(str, Enum):
    """交易所类型。"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"


class VenueFeature(str, Enum):
    """交易所支持的特性。"""
    WEBSOCKET = "websocket"
    REST = "rest"
    MARGIN = "margin"
    REDUCE_ONLY = "reduce_only"
    POST_ONLY = "post_only"
    STOP_ORDER = "stop_order"
    OCO = "oco"
    HEDGE_MODE = "hedge_mode"


@dataclass(frozen=True, slots=True)
class VenueInfo:
    """
    交易所元数据。

    描述交易所的基本信息、支持的功能和限速参数。
    """
    name: str                          # "binance", "okx"
    venue_type: VenueType = VenueType.PERPETUAL
    features: tuple[VenueFeature, ...] = ()

    # 限速
    rest_rate_limit: float = 10.0      # 每秒请求数
    ws_rate_limit: float = 5.0         # 每秒消息数
    order_rate_limit: float = 10.0     # 每秒下单数

    # 连接
    rest_base_url: str = ""
    ws_base_url: str = ""

    # 精度默认值
    default_price_precision: int = 8
    default_qty_precision: int = 8

    testnet: bool = False

    def has_feature(self, feature: VenueFeature) -> bool:
        return feature in self.features

    @property
    def supports_websocket(self) -> bool:
        return self.has_feature(VenueFeature.WEBSOCKET)

    @property
    def supports_reduce_only(self) -> bool:
        return self.has_feature(VenueFeature.REDUCE_ONLY)
