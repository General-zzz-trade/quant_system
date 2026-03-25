"""Ingestion — collector protocol for live market data sources.

The Collector protocol defines start/stop lifecycle and liveness tracking.
Concrete collectors (WS kline, tick, orderbook) implement this protocol.
"""
from __future__ import annotations

from data.collectors.base import Collector

__all__ = ["Collector"]
