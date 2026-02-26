# portfolio/optimizer/views.py
"""Generate Black-Litterman views from signal strengths."""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

from portfolio.optimizer.black_litterman import ViewSpec

logger = logging.getLogger(__name__)


class ViewGenerator:
    """Generate Black-Litterman views from signal strengths.

    Converts numeric signals (in [-1, 1]) to ViewSpec objects
    compatible with the BlackLittermanModel.
    """

    def __init__(
        self,
        *,
        base_confidence: float = 0.5,
        signal_scaling: float = 0.1,
    ) -> None:
        self._base_confidence = base_confidence
        self._signal_scaling = signal_scaling

    @property
    def base_confidence(self) -> float:
        return self._base_confidence

    @property
    def signal_scaling(self) -> float:
        return self._signal_scaling

    def from_signals(
        self,
        signals: Mapping[str, float],
        symbols: Sequence[str],
    ) -> list[ViewSpec]:
        """Convert signal strengths to absolute return views.

        Each non-zero signal becomes an absolute view on that asset.
        signal > 0 -> positive expected return (bullish).
        signal < 0 -> negative expected return (bearish).
        Confidence is proportional to |signal|.
        """
        views: list[ViewSpec] = []
        for sym in symbols:
            strength = signals.get(sym, 0.0)
            if abs(strength) < 1e-10:
                continue

            expected_return = strength * self._signal_scaling
            confidence = self._base_confidence * abs(strength)
            # Clamp confidence to a reasonable floor
            confidence = max(confidence, 1e-4)

            views.append(ViewSpec(
                assets=(sym,),
                weights=(1.0,),
                expected_return=expected_return,
                confidence=confidence,
            ))

        return views

    def relative_view(
        self,
        long_asset: str,
        short_asset: str,
        expected_diff: float,
        confidence: float = 1.0,
    ) -> ViewSpec:
        """Create a relative view: long_asset outperforms short_asset by expected_diff.

        This encodes the view: E[r_long] - E[r_short] = expected_diff.
        """
        return ViewSpec(
            assets=(long_asset, short_asset),
            weights=(1.0, -1.0),
            expected_return=expected_diff,
            confidence=confidence,
        )

    def multi_asset_view(
        self,
        assets: Sequence[str],
        weights: Sequence[float],
        expected_return: float,
        confidence: float = 1.0,
    ) -> ViewSpec:
        """Create a general multi-asset view.

        The view asserts that the weighted combination of asset returns
        equals expected_return.
        """
        return ViewSpec(
            assets=tuple(assets),
            weights=tuple(weights),
            expected_return=expected_return,
            confidence=confidence,
        )
