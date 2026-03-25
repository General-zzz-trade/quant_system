# runner/gates/equity_leverage_gate.py
"""Equity-bracket leverage + z-score position scaling gate."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from strategy.gates.types import GateResult

_log = logging.getLogger(__name__)

# Kelly-optimal leverage brackets.
# Research: 1.5x → 14.3%/quarter (best), 2x → 11.0%/q, 3x → -4.4%/q (negative!)
# Ladder is flat until $20K then steps down to half-Kelly for capital preservation.
_DEFAULT_BRACKETS = [
    (0,      5_000,         1.5),
    (5_000,  20_000,        1.5),
    (20_000, 50_000,        1.0),
    (50_000, float("inf"),  1.0),
]


def _bracket_leverage(equity: float, brackets=_DEFAULT_BRACKETS) -> float:
    """Return Kelly-optimal leverage for equity bracket."""
    for lo, hi, lev in brackets:
        if lo <= equity < hi:
            return lev
    return 1.0


def _z_scale(z: float) -> float:
    """Non-linear z-score scaling.

    Stronger signals get larger positions, weak signals get smaller:
    - |z| > 2.0: scale=1.5 (extreme conviction)
    - |z| > 1.0: scale=1.0 (normal)
    - |z| > 0.5: scale=0.7 (weak signal)
    - else:       scale=0.5 (barely above deadzone)

    Returns scale factor in [0.5, 1.5].
    """
    az = abs(z)
    if az > 2.0:
        return 1.5
    if az > 1.0:
        return 1.0
    if az > 0.5:
        return 0.7
    return 0.5


class EquityLeverageGate:
    """Gate: scale order qty by equity-bracket leverage × z-score conviction.

    Combines two independent scaling signals:
    1. Equity bracket → Kelly-optimal leverage (1.5x small, 1.0x large accounts)
    2. Z-score magnitude → conviction scaling (0.5x weak to 1.5x extreme)

    The combined multiplier can exceed 1.0 (e.g. $500 equity + |z|=2.5 → 2.25x).
    GateChain only downscales via ``_apply_scale`` when scale < 1.0, so callers
    that want upscaling should read ``result.scale`` directly.

    Always returns ``allowed=True`` — this is a sizing gate, not a rejection gate.
    """

    name = "EquityLeverage"

    def __init__(
        self,
        get_equity: Optional[Callable[[], float]] = None,
        brackets=None,
    ) -> None:
        """
        Args:
            get_equity: callable returning current account equity (float).
                        Falls back to ``context["equity"]`` if None.
            brackets:   custom leverage brackets; defaults to _DEFAULT_BRACKETS.
        """
        self._get_equity = get_equity
        self._brackets = brackets if brackets is not None else _DEFAULT_BRACKETS

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        # --- Equity lookup ---
        equity = 0.0
        if self._get_equity is not None:
            try:
                equity = float(self._get_equity())
            except Exception as exc:
                _log.warning("EquityLeverageGate: get_equity() failed: %s", exc)
        if equity <= 0.0:
            equity = float(context.get("equity", 0.0))

        lev = _bracket_leverage(equity, self._brackets)

        # --- Z-score lookup (event metadata first, then context) ---
        z = 0.0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            z = float(meta.get("z_score", meta.get("z", 0.0)))
        if z == 0.0:
            z = float(context.get("z_score", context.get("z", 0.0)))

        zs = _z_scale(z)
        scale = lev * zs

        _log.debug(
            "EquityLeverageGate: equity=%.2f lev=%.2f z=%.3f z_scale=%.2f → scale=%.3f",
            equity, lev, z, zs, scale,
        )
        return GateResult(allowed=True, scale=scale)
