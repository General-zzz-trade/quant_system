# runner/gates/consensus_scaling_gate.py
"""Cross-symbol consensus scaling gate.

Extracted from AlphaRunner._get_consensus_scale().

Research finding: consensus is CONTRARIAN — when all 4 symbols agree
bearish, market actually goes UP (+28bp). This gate applies:
- All others disagree (contrarian): +30% boost (1.3x)
- 3/4+ of others agree: 1.0x (no boost; don't amplify herd)
- 1-2 out of 3-4 agree: 0.7x (lower conviction)
- Nobody else agrees (but not full contrarian): 0.5x
"""
from __future__ import annotations

from typing import Dict


def _consensus_scale(
    symbol: str,
    my_signal: int,
    other_signals: Dict[str, int],
) -> float:
    """Compute position scale from cross-symbol signal consensus.

    Matches AlphaRunner._get_consensus_scale() exactly.

    Args:
        symbol: Current symbol key (excluded from others count).
        my_signal: Current signal for this symbol (+1/-1/0).
        other_signals: Mapping of other symbol keys to their signals.
                       Keys matching ``symbol`` are ignored.

    Returns:
        Scale factor in [0.5, 1.3].
    """
    if my_signal == 0:
        return 1.0  # flat — no scaling needed

    # Count signals from other symbols (exclude self)
    n_bull = 0
    n_bear = 0
    n_total = 0
    for rkey, sig in other_signals.items():
        if rkey == symbol:
            continue
        if sig > 0:
            n_bull += 1
        elif sig < 0:
            n_bear += 1
        n_total += 1

    if n_total == 0:
        return 1.0  # no data from other symbols

    same_dir = n_bull if my_signal > 0 else n_bear
    opposite_dir = n_bear if my_signal > 0 else n_bull

    # Contrarian boost: ALL others disagree → historically profitable
    if opposite_dir == n_total and n_total >= 2:
        return 1.3

    # Fraction of others that agree
    agree_frac = same_dir / n_total if n_total > 0 else 0

    if agree_frac >= 0.75:   # 3/4+ agree (unanimous or near)
        return 1.0
    elif agree_frac >= 0.25:  # 1-2 out of 3-4 agree
        return 0.7
    else:                     # nobody agrees (but not full contrarian)
        return 0.5
