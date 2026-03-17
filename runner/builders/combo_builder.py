"""Combo builder: wires PortfolioCombiner for dual-alpha AGREE mode.

When enabled, wraps the decision module so that 1h and 15m signals
must agree before generating an order. Conviction is scaled based
on agreement level.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

_log = logging.getLogger(__name__)


class ComboConfig(NamedTuple):
    mode: str  # "agree" or "any"
    conviction_both: float
    conviction_single: float
    per_symbol_cap: float


class CombinedSignal(NamedTuple):
    direction: int  # -1, 0, +1
    conviction: float  # 0.0 to 1.0
    source: str  # "both_agree", "single_1h", "single_15m", "disagree"


def combine_signals(
    signal_1h: int,
    signal_15m: int,
    config: ComboConfig,
) -> CombinedSignal:
    """Combine two timeframe signals using AGREE logic.

    Args:
        signal_1h: 1h signal (-1, 0, +1)
        signal_15m: 15m signal (-1, 0, +1)
        config: combo configuration

    Returns:
        CombinedSignal with direction, conviction, and source
    """
    if config.mode == "agree":
        if signal_1h == signal_15m and signal_1h != 0:
            return CombinedSignal(signal_1h, config.conviction_both, "both_agree")
        elif signal_1h != 0 and signal_15m == 0:
            return CombinedSignal(signal_1h, config.conviction_single, "single_1h")
        elif signal_15m != 0 and signal_1h == 0:
            return CombinedSignal(signal_15m, config.conviction_single, "single_15m")
        else:
            # Both non-zero but disagree, or both flat
            return CombinedSignal(0, 0.0, "disagree")
    else:
        # "any" mode: either signal triggers
        if signal_1h != 0:
            return CombinedSignal(signal_1h, config.conviction_both, "signal_1h")
        if signal_15m != 0:
            return CombinedSignal(signal_15m, config.conviction_both, "signal_15m")
        return CombinedSignal(0, 0.0, "flat")


def build_combo(config) -> Optional[ComboConfig]:
    """Build combo configuration from LiveRunnerConfig."""
    if not config.enable_combo:
        _log.info("Combo mode disabled")
        return None

    combo = ComboConfig(
        mode=config.combo_mode,
        conviction_both=config.combo_conviction_both,
        conviction_single=config.combo_conviction_single,
        per_symbol_cap=config.combo_per_symbol_cap,
    )
    _log.info("Combo mode enabled: %s", combo)
    return combo
