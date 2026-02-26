# execution/sim/slippage.py
"""Slippage models for paper trading simulation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


class SlippageModel(Protocol):
    """滑点模型协议。"""

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        """
        计算实际成交价格。

        Args:
            price: 触发/预期价格
            side: "buy" / "sell"
            qty: 成交数量

        Returns:
            滑点后的实际价格
        """
        ...


@dataclass(frozen=True, slots=True)
class NoSlippage:
    """无滑点 — 按原价成交。"""

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        return price


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    """
    固定基点滑点模型。

    buy 方向价格上滑，sell 方向价格下滑。
    """
    bps: Decimal = Decimal("1.0")  # 基点，1 bps = 0.01%

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        factor = self.bps / Decimal("10000")
        if side == "buy":
            return price * (Decimal("1") + factor)
        else:
            return price * (Decimal("1") - factor)


@dataclass(frozen=True, slots=True)
class VolumeImpactSlippage:
    """
    数量影响滑点模型。

    滑点与成交量正相关：impact = base_bps + volume_factor * qty
    """
    base_bps: Decimal = Decimal("0.5")
    volume_factor: Decimal = Decimal("0.001")   # 每单位qty额外滑点(bps)

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        total_bps = self.base_bps + self.volume_factor * qty
        factor = total_bps / Decimal("10000")
        if side == "buy":
            return price * (Decimal("1") + factor)
        else:
            return price * (Decimal("1") - factor)


@dataclass(frozen=True, slots=True)
class RandomSlippage:
    """随机滑点（用于蒙特卡洛模拟）。"""
    max_bps: Decimal = Decimal("2.0")

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        bps = Decimal(str(random.uniform(0, float(self.max_bps))))
        factor = bps / Decimal("10000")
        if side == "buy":
            return price * (Decimal("1") + factor)
        else:
            return price * (Decimal("1") - factor)


@dataclass(frozen=True, slots=True)
class SqrtImpactSlippage:
    """Square-root market impact model (Almgren-Chriss).

    impact_bps = eta * sigma_daily * sqrt(qty / adv)

    where:
    - eta: impact coefficient (typically 0.1-1.0 for crypto)
    - sigma_daily: daily volatility (e.g. 0.02 = 2%)
    - adv: average daily volume in base asset units

    This produces sub-linear impact: doubling order size increases impact by ~41%.
    """
    eta: Decimal = Decimal("0.5")
    sigma_daily: Decimal = Decimal("0.02")     # 2% daily vol
    adv: Decimal = Decimal("1000")             # average daily volume (base units)

    def apply(self, *, price: Decimal, side: str, qty: Decimal) -> Decimal:
        if self.adv <= 0 or qty <= 0:
            return price

        participation = float(qty / self.adv)
        impact_frac = float(self.eta) * float(self.sigma_daily) * math.sqrt(participation)
        impact = Decimal(str(impact_frac))

        if side == "buy":
            return price * (Decimal("1") + impact)
        else:
            return price * (Decimal("1") - impact)
