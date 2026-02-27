# execution/algos/adaptive_twap.py
"""Adaptive TWAP — adjusts slice sizes based on real-time market conditions."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, List, Optional

from execution.algos.twap import TWAPSlice, TWAPOrder
from execution.algos.volume_profile import IntraDayVolumeProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Lightweight market snapshot for adaptive decisions."""
    bid: float
    ask: float
    spread_bps: float
    recent_volatility: float  # e.g. 1-min realized vol
    volume_ratio: float = 1.0  # current vs expected volume


@dataclass
class AdaptiveTWAPConfig:
    """Configuration for adaptive TWAP behavior."""
    base_slices: int = 10
    base_duration_sec: float = 600
    spread_threshold_bps: float = 5.0    # widen → slow down
    vol_threshold: float = 0.005          # high vol → reduce slice size
    volume_acceleration: float = 1.5      # volume > expected → accelerate
    min_slice_fraction: float = 0.5       # min slice as fraction of base
    max_slice_fraction: float = 2.0       # max slice as fraction of base


@dataclass
class AdaptiveTWAPAlgo:
    """Adaptive TWAP that adjusts execution speed based on market conditions.

    Adaptations:
    - Wide spreads → slow down (reduce slice size)
    - High volatility → reduce slice size to minimize impact
    - High volume → accelerate execution (larger slices)
    - Volume profile → weight slices by expected intraday volume
    """

    submit_fn: Callable[[str, str, Decimal], Optional[Decimal]]
    cfg: AdaptiveTWAPConfig = field(default_factory=AdaptiveTWAPConfig)
    profile: IntraDayVolumeProfile = field(
        default_factory=IntraDayVolumeProfile.crypto_24h,
    )

    def create(
        self,
        symbol: str,
        side: str,
        total_qty: Decimal,
        *,
        n_slices: Optional[int] = None,
        duration_sec: Optional[float] = None,
        start_hour: int = 0,
    ) -> TWAPOrder:
        """Create an adaptive TWAP order with volume-weighted scheduling."""
        n = n_slices or self.cfg.base_slices
        dur = duration_sec or self.cfg.base_duration_sec
        duration_hours = dur / 3600.0

        # Get volume weights for the execution period
        weights = self.profile.get_weights(
            n_slices=n,
            start_hour=start_hour,
            duration_hours=max(duration_hours, 1.0),
        )

        now = time.monotonic()
        interval = dur / n

        slices = []
        allocated = Decimal("0")
        for i in range(n):
            if i == n - 1:
                qty = total_qty - allocated
            else:
                qty = (total_qty * Decimal(str(weights[i]))).quantize(Decimal("0.00000001"))
                allocated += qty

            slices.append(TWAPSlice(
                slice_idx=i,
                qty=qty,
                scheduled_at=now + i * interval,
            ))

        order = TWAPOrder(
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            n_slices=n,
            duration_sec=dur,
            slices=slices,
            start_time=now,
        )

        logger.info(
            "AdaptiveTWAP created: %s %s %s in %d slices over %ds",
            side, total_qty, symbol, n, dur,
        )
        return order

    def adaptive_tick(
        self,
        order: TWAPOrder,
        market: Optional[MarketSnapshot] = None,
    ) -> Optional[TWAPSlice]:
        """Tick with optional market-based adaptation."""
        now = time.monotonic()

        for i, s in enumerate(order.slices):
            if s.status != "pending":
                continue
            if now < s.scheduled_at:
                continue

            # Adapt slice size based on market conditions
            adjusted_qty = self._adapt_qty(s.qty, market)

            try:
                fill_price = self.submit_fn(order.symbol, order.side, adjusted_qty)
                order.slices[i] = TWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=adjusted_qty,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    fill_price=fill_price,
                    status="executed" if fill_price else "failed",
                )
                return order.slices[i]
            except Exception as e:
                logger.warning("AdaptiveTWAP slice %d failed: %s", s.slice_idx, e)
                order.slices[i] = TWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=adjusted_qty,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    status="failed",
                )
                return order.slices[i]

        return None

    def _adapt_qty(
        self,
        base_qty: Decimal,
        market: Optional[MarketSnapshot],
    ) -> Decimal:
        """Adjust slice quantity based on market conditions."""
        if market is None:
            return base_qty

        multiplier = 1.0

        # Wide spread → reduce size
        if market.spread_bps > self.cfg.spread_threshold_bps:
            spread_factor = self.cfg.spread_threshold_bps / max(market.spread_bps, 0.01)
            multiplier *= max(spread_factor, self.cfg.min_slice_fraction)

        # High volatility → reduce size
        if market.recent_volatility > self.cfg.vol_threshold:
            vol_factor = self.cfg.vol_threshold / max(market.recent_volatility, 1e-8)
            multiplier *= max(vol_factor, self.cfg.min_slice_fraction)

        # High volume → increase size (accelerate)
        if market.volume_ratio > 1.0:
            vol_accel = min(market.volume_ratio, self.cfg.volume_acceleration)
            multiplier *= vol_accel

        # Clamp
        multiplier = max(self.cfg.min_slice_fraction, min(self.cfg.max_slice_fraction, multiplier))

        adjusted = (base_qty * Decimal(str(multiplier))).quantize(Decimal("0.00000001"))
        return max(adjusted, Decimal("0.00000001"))
