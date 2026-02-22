# execution/adapters/registry.py
"""Venue adapter registry — central lookup for all registered venue adapters."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from execution.adapters.base import VenueAdapter


class AdapterNotFoundError(KeyError):
    """请求的交易所适配器未注册。"""


class AdapterRegistry:
    """
    交易所适配器注册中心。

    在启动时注册所有可用的 adapter，运行时按 venue 名查找。
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, VenueAdapter] = {}

    def register(self, venue: str, adapter: VenueAdapter) -> None:
        """注册一个交易所适配器。"""
        self._adapters[venue.lower()] = adapter

    def get(self, venue: str) -> VenueAdapter:
        """获取指定交易所的适配器，不存在则抛异常。"""
        v = venue.lower()
        adapter = self._adapters.get(v)
        if adapter is None:
            raise AdapterNotFoundError(
                f"no adapter registered for venue {venue!r}, "
                f"available: {list(self._adapters.keys())}"
            )
        return adapter

    def get_optional(self, venue: str) -> Optional[VenueAdapter]:
        """获取指定交易所的适配器，不存在返回 None。"""
        return self._adapters.get(venue.lower())

    @property
    def venues(self) -> Sequence[str]:
        """所有已注册的交易所名称。"""
        return list(self._adapters.keys())

    def __contains__(self, venue: str) -> bool:
        return venue.lower() in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)
