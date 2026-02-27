# execution/algos/volume_profile.py
"""Volume profile models for execution algorithms."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VolumeBar:
    """Single volume bar in a profile."""
    hour: int          # 0-23
    weight: float      # normalized weight


@dataclass
class IntraDayVolumeProfile:
    """Real intra-day volume distribution curve.

    Models the typical U-shaped or W-shaped volume pattern
    observed in crypto/traditional markets.

    Usage:
        profile = IntraDayVolumeProfile.crypto_24h()
        weights = profile.get_weights(n_slices=10, start_hour=9, end_hour=17)
    """

    bars: List[VolumeBar] = field(default_factory=list)

    @classmethod
    def crypto_24h(cls) -> IntraDayVolumeProfile:
        """Default 24h crypto profile with peaks at 8-10 UTC and 14-16 UTC."""
        raw = []
        for h in range(24):
            # Base level
            w = 1.0
            # Asia session peak (1-3 UTC)
            w += 0.4 * math.exp(-0.5 * ((h - 2) / 1.5) ** 2)
            # Europe open peak (8-10 UTC)
            w += 0.8 * math.exp(-0.5 * ((h - 9) / 2.0) ** 2)
            # US session peak (14-16 UTC)
            w += 1.0 * math.exp(-0.5 * ((h - 15) / 2.0) ** 2)
            # Low overnight (20-24 UTC)
            w += -0.3 * math.exp(-0.5 * ((h - 22) / 2.0) ** 2)
            raw.append(max(w, 0.1))

        total = sum(raw)
        bars = [VolumeBar(hour=h, weight=w / total) for h, w in enumerate(raw)]
        return cls(bars=bars)

    @classmethod
    def uniform(cls) -> IntraDayVolumeProfile:
        """Flat profile for testing."""
        bars = [VolumeBar(hour=h, weight=1.0 / 24) for h in range(24)]
        return cls(bars=bars)

    @classmethod
    def from_historical(cls, hourly_volumes: Sequence[float]) -> IntraDayVolumeProfile:
        """Build profile from historical hourly volume data."""
        if len(hourly_volumes) != 24:
            raise ValueError("Expected 24 hourly volume values")
        total = sum(hourly_volumes) or 1.0
        bars = [
            VolumeBar(hour=h, weight=v / total)
            for h, v in enumerate(hourly_volumes)
        ]
        return cls(bars=bars)

    def get_weight(self, hour: int) -> float:
        """Get volume weight for a specific hour."""
        for bar in self.bars:
            if bar.hour == hour:
                return bar.weight
        return 1.0 / 24

    def get_weights(
        self,
        n_slices: int,
        start_hour: int = 0,
        duration_hours: int = 24,
    ) -> List[float]:
        """Generate normalized weights for N slices spanning given hours."""
        if n_slices <= 0:
            return []

        hours_per_slice = duration_hours / n_slices
        weights = []

        for i in range(n_slices):
            # Center hour of this slice
            center = start_hour + (i + 0.5) * hours_per_slice
            h = int(center) % 24
            weights.append(self.get_weight(h))

        total = sum(weights) or 1.0
        return [w / total for w in weights]

    @property
    def peak_hours(self) -> List[int]:
        """Hours with above-average volume."""
        if not self.bars:
            return []
        avg = 1.0 / max(len(self.bars), 1)
        return [b.hour for b in self.bars if b.weight > avg * 1.2]
