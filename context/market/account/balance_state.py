# context/market/account/balance_state.py
"""Balance state — per-asset balance tracking."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class AssetBalanceSnapshot:
    """单个资产余额快照。"""
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal


class BalanceState:
    """资产余额状态管理。"""

    def __init__(self) -> None:
        self._balances: Dict[str, dict] = {}

    def update(self, asset: str, *, free: Decimal, locked: Decimal) -> None:
        self._balances[asset.upper()] = {
            "free": free, "locked": locked, "total": free + locked,
        }

    def get(self, asset: str) -> Optional[AssetBalanceSnapshot]:
        b = self._balances.get(asset.upper())
        if b is None:
            return None
        return AssetBalanceSnapshot(
            asset=asset.upper(), free=b["free"],
            locked=b["locked"], total=b["total"],
        )

    @property
    def all_assets(self) -> list[str]:
        return list(self._balances.keys())
