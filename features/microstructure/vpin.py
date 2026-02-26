"""VPIN — Volume-Synchronized Probability of Informed Trading.

Measures order flow toxicity by comparing buy/sell volume imbalance
across fixed-volume buckets.

Reference: Easley, Lopez de Prado, O'Hara (2012).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VPINResult:
    """Result of VPIN calculation."""

    vpin: float  # 0 to 1, higher = more toxic
    buy_volume: Decimal
    sell_volume: Decimal
    bucket_count: int


@dataclass(frozen=True, slots=True)
class _VolumeBucket:
    """Internal: a single fixed-volume bucket."""

    buy_volume: Decimal
    sell_volume: Decimal


class VPINCalculator:
    """Volume-Synchronized Probability of Informed Trading.

    VPIN measures order flow toxicity by comparing buy/sell volume imbalance
    across volume buckets.

    Algorithm:
    1. Classify ticks as buy/sell using tick rule (or use side field)
    2. Aggregate into volume buckets of fixed size
    3. VPIN = mean(|V_buy - V_sell| / V_total) across last n_buckets
    """

    def __init__(
        self,
        *,
        bucket_volume: Decimal = Decimal("100"),
        n_buckets: int = 50,
    ) -> None:
        self._bucket_volume = bucket_volume
        self._n_buckets = n_buckets

    def calculate(self, ticks: Sequence[Any]) -> VPINResult:
        """Calculate VPIN from tick data.

        Walks through ticks, accumulates into fixed-volume buckets,
        and computes VPIN as the average absolute order imbalance.
        """
        if not ticks:
            return VPINResult(
                vpin=0.0,
                buy_volume=Decimal("0"),
                sell_volume=Decimal("0"),
                bucket_count=0,
            )

        buckets: deque[_VolumeBucket] = deque(maxlen=self._n_buckets)
        current_buy = Decimal("0")
        current_sell = Decimal("0")
        current_total = Decimal("0")
        total_buy = Decimal("0")
        total_sell = Decimal("0")
        prev_price: Optional[Decimal] = None

        for tick in ticks:
            side = self.classify_tick(tick, prev_price)
            qty = getattr(tick, "qty", Decimal("0"))
            prev_price = getattr(tick, "price", prev_price)

            remaining = qty
            while remaining > Decimal("0"):
                space = self._bucket_volume - current_total
                fill = min(remaining, space)

                if side == "buy":
                    current_buy += fill
                else:
                    current_sell += fill
                current_total += fill
                remaining -= fill

                if current_total >= self._bucket_volume:
                    buckets.append(_VolumeBucket(
                        buy_volume=current_buy,
                        sell_volume=current_sell,
                    ))
                    total_buy += current_buy
                    total_sell += current_sell
                    current_buy = Decimal("0")
                    current_sell = Decimal("0")
                    current_total = Decimal("0")

        n = len(buckets)
        if n == 0:
            return VPINResult(
                vpin=0.0,
                buy_volume=total_buy,
                sell_volume=total_sell,
                bucket_count=0,
            )

        # VPIN = sum(|buy_i - sell_i|) / (n * bucket_volume)
        imbalance_sum = sum(
            abs(b.buy_volume - b.sell_volume) for b in buckets
        )
        vpin = float(imbalance_sum / (n * self._bucket_volume))

        # Recompute totals from the buckets we actually use
        used_buy = sum(b.buy_volume for b in buckets)
        used_sell = sum(b.sell_volume for b in buckets)

        return VPINResult(
            vpin=vpin,
            buy_volume=used_buy,
            sell_volume=used_sell,
            bucket_count=n,
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
        # Tick rule: price up = buy, price down = sell
        price = getattr(tick, "price", Decimal("0"))
        if prev_price is not None:
            return "buy" if price >= prev_price else "sell"
        return "buy"  # default when no prior reference
