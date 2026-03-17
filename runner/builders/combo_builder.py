# runner/builders/combo_builder.py
"""Dual-alpha combo signal combiner (AGREE ONLY mode).

Extracted from PortfolioCombiner in scripts/ops/portfolio_combiner.py.

Backtest result: AGREE ONLY Sharpe=5.48 vs weighted COMBO Sharpe=3.18 (+72%).
Both alphas must agree direction for a trade; any disagreement → flat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ComboConfig:
    """Configuration for the combo signal combiner.

    Attributes:
        mode:             Combination mode. Only "agree" is currently supported.
        conviction_both:  Position scale when both alphas agree (default 1.0 = full).
        conviction_single: Position scale when only one alpha has a signal (default 0.5).
        per_symbol_cap:   Max fraction of equity × leverage per symbol (default 0.3 = 30%).
    """
    mode: str = "agree"
    conviction_both: float = 1.0
    conviction_single: float = 0.5
    per_symbol_cap: float = 0.3


@dataclass
class ComboResult:
    """Result of combining two alpha signals.

    Attributes:
        direction:  Net signal: +1 (long), -1 (short), 0 (flat).
        conviction: Position scale in [0.0, 1.0]; 0.0 = flat.
    """
    direction: int
    conviction: float


def combine_signals(
    signal_a: int,
    signal_b: int,
    cfg: Optional[ComboConfig] = None,
) -> ComboResult:
    """Combine two alpha signals under AGREE ONLY mode.

    Rules (AGREE ONLY):
    - Both long  (+1, +1) → direction=+1, conviction=cfg.conviction_both
    - Both short (-1, -1) → direction=-1, conviction=cfg.conviction_both
    - Disagree   (+1, -1) → direction=0,  conviction=0.0 (flat)
    - One flat   (+1,  0) → direction=+1, conviction=cfg.conviction_single
    - Both flat  ( 0,  0) → direction=0,  conviction=0.0

    Matches PortfolioCombiner AGREE ONLY logic exactly.

    Args:
        signal_a: First alpha signal in {-1, 0, +1}.
        signal_b: Second alpha signal in {-1, 0, +1}.
        cfg:      ComboConfig (uses defaults if None).

    Returns:
        ComboResult with direction and conviction.
    """
    if cfg is None:
        cfg = ComboConfig()

    if cfg.mode != "agree":
        raise ValueError(f"Unsupported combo mode: {cfg.mode!r}. Only 'agree' is supported.")

    signals = [signal_a, signal_b]
    n_long = sum(1 for s in signals if s > 0)
    n_short = sum(1 for s in signals if s < 0)
    n_total = len(signals)

    if n_long == n_total:
        # Unanimous long
        return ComboResult(direction=1, conviction=cfg.conviction_both)
    elif n_short == n_total:
        # Unanimous short
        return ComboResult(direction=-1, conviction=cfg.conviction_both)
    elif n_long > 0 and n_short > 0:
        # Explicit disagreement → flat
        return ComboResult(direction=0, conviction=0.0)
    elif n_long == 1:
        # One long, one flat → partial conviction
        return ComboResult(direction=1, conviction=cfg.conviction_single)
    elif n_short == 1:
        # One short, one flat → partial conviction
        return ComboResult(direction=-1, conviction=cfg.conviction_single)
    else:
        # Both flat
        return ComboResult(direction=0, conviction=0.0)
