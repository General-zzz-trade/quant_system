# execution/models/balances.py
"""Canonical balance snapshot from venue."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class CanonicalBalance:
    """
    标准化余额快照 — 交易所返回的资产余额。

    free:   可用于下单的数量
    locked: 被挂单或仓位占用的数量
    total:  free + locked
    """
    venue: str
    asset: str

    free: Decimal
    locked: Decimal
    total: Decimal

    ts_ms: int = 0
    raw: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_free_locked(
        cls,
        *,
        venue: str,
        asset: str,
        free: Decimal,
        locked: Decimal,
        ts_ms: int = 0,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> CanonicalBalance:
        return cls(
            venue=venue,
            asset=asset.upper(),
            free=free,
            locked=locked,
            total=free + locked,
            ts_ms=ts_ms,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class BalanceSnapshot:
    """一次完整的账户余额快照（多个资产）。"""
    venue: str
    balances: tuple[CanonicalBalance, ...]
    ts_ms: int = 0

    def get(self, asset: str) -> Optional[CanonicalBalance]:
        asset_upper = asset.upper()
        for b in self.balances:
            if b.asset == asset_upper:
                return b
        return None

    @property
    def assets(self) -> Sequence[str]:
        return [b.asset for b in self.balances]


# 兼容别名 — sim_venue.py 等模块使用
AssetBalance = CanonicalBalance
