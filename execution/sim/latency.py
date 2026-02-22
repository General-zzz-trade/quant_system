# execution/sim/latency.py
"""Latency simulation for paper trading."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class LatencyModel(Protocol):
    """延迟模型协议。"""
    def sample_ms(self) -> float:
        """采样一个延迟值（毫秒）。"""
        ...


@dataclass(frozen=True, slots=True)
class FixedLatency:
    """固定延迟模型。"""
    latency_ms: float = 0.0

    def sample_ms(self) -> float:
        return self.latency_ms


@dataclass(frozen=True, slots=True)
class UniformLatency:
    """均匀分布延迟模型。"""
    min_ms: float = 1.0
    max_ms: float = 10.0

    def sample_ms(self) -> float:
        return random.uniform(self.min_ms, self.max_ms)


@dataclass(frozen=True, slots=True)
class GaussianLatency:
    """高斯分布延迟模型。"""
    mean_ms: float = 5.0
    std_ms: float = 2.0
    min_ms: float = 0.5

    def sample_ms(self) -> float:
        return max(self.min_ms, random.gauss(self.mean_ms, self.std_ms))


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    """延迟配置。"""
    order_latency_ms: float = 5.0
    cancel_latency_ms: float = 3.0
    fill_latency_ms: float = 1.0
    market_data_latency_ms: float = 1.0

    def order_model(self) -> FixedLatency:
        return FixedLatency(self.order_latency_ms)

    def cancel_model(self) -> FixedLatency:
        return FixedLatency(self.cancel_latency_ms)
