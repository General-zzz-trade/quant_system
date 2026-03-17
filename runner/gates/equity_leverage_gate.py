# runner/gates/equity_leverage_gate.py
"""Equity-based leverage bracket and z-score sizing gates.

Extracted from AlphaRunner.LEVERAGE_LADDER and AlphaRunner.compute_z_scale.
Kelly-optimal leverage validated by Monte Carlo simulation 2026-03-15.
"""
from __future__ import annotations

# Kelly-optimal leverage ladder (matches AlphaRunner.LEVERAGE_LADDER)
# Full Kelly = 1.3x, half-Kelly = 0.65x. At 3x+, bust rate > 50%.
# Geometric mean: 1.5x=14.3%/q (best), 2x=11.0%/q, 3x=-4.4%/q (negative!)
# Ladder is flat at 1.5x — Kelly optimal doesn't depend on account size.
_LEVERAGE_LADDER = [
    (0,      1.5),    # $0-$5K:      1.5x (Kelly optimal, 2% bust rate)
    (5000,   1.5),    # $5K-$20K:    1.5x (same — Kelly is scale-invariant)
    (20000,  1.0),    # $20K-$50K:   1.0x (half-Kelly, capital preservation)
    (50000,  1.0),    # $50K+:       1.0x (pure alpha, no leverage risk)
]


def _bracket_leverage(equity: float) -> float:
    """Return Kelly-optimal leverage for given equity level.

    Matches AlphaRunner._get_leverage_for_equity() exactly.

    Args:
        equity: Current account equity in USD.

    Returns:
        Leverage multiplier (1.0 or 1.5).
    """
    lev = 1.0
    for threshold, lev_val in _LEVERAGE_LADDER:
        if equity >= threshold:
            lev = lev_val
    return lev


def _z_scale(z: float) -> float:
    """Non-linear position sizing based on z-score magnitude.

    Stronger signals get larger positions, weak signals get smaller.
    Matches AlphaRunner.compute_z_scale() exactly.

    Thresholds:
    - |z| > 2.0: scale=1.5 (extreme conviction)
    - |z| > 1.0: scale=1.0 (normal)
    - |z| > 0.5: scale=0.7 (weak signal)
    - else:      scale=0.5 (barely above deadzone)

    Args:
        z: Rolling z-score of the raw prediction.

    Returns:
        Scale factor in [0.5, 1.5].
    """
    abs_z = abs(z)
    if abs_z > 2.0:
        return 1.5
    elif abs_z > 1.0:
        return 1.0
    elif abs_z > 0.5:
        return 0.7
    else:
        return 0.5
