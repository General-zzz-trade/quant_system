"""VPIN — Volume-Synchronized Probability of Informed Trading.

Delegates to RustVPINCalculator. Keeps Python dataclass for API compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, Sequence

from _quant_hotpath import RustVPINCalculator


@dataclass(frozen=True, slots=True)
class VPINResult:
    """Result of VPIN calculation."""

    vpin: float  # 0 to 1, higher = more toxic
    buy_volume: Decimal
    sell_volume: Decimal
    bucket_count: int


class VPINCalculator:
    """VPIN calculator — thin wrapper over RustVPINCalculator."""

    def __init__(
        self,
        *,
        bucket_volume: Decimal = Decimal("100"),
        n_buckets: int = 50,
    ) -> None:
        self._rust = RustVPINCalculator(
            bucket_volume=float(bucket_volume),
            n_buckets=n_buckets,
        )

    def calculate(self, ticks: Sequence[Any]) -> VPINResult:
        """Calculate VPIN from tick data."""
        r = self._rust.calculate(list(ticks))
        return VPINResult(
            vpin=r.vpin,
            buy_volume=Decimal(str(r.buy_volume)),
            sell_volume=Decimal(str(r.sell_volume)),
            bucket_count=r.bucket_count,
        )

    def classify_tick(
        self,
        tick: Any,
        prev_price: Optional[Decimal] = None,
    ) -> str:
        """Classify tick as buy/sell using tick rule if side not available."""
        side = getattr(tick, "side", "")
        if side in ("buy", "sell"):
            return side
        price = getattr(tick, "price", Decimal("0"))
        if prev_price is not None:
            return "buy" if price >= prev_price else "sell"
        return "buy"
