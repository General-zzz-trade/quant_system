# runner/gates/consensus_scaling_gate.py
"""Cross-symbol consensus scaling gate."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from strategy.gates.types import GateResult

_log = logging.getLogger(__name__)


def _consensus_scale(symbol: str, signal: int, consensus: dict) -> float:
    """Compute scale factor based on cross-symbol signal agreement.

    Research finding: consensus is CONTRARIAN — when all symbols agree bearish,
    market tends to go UP. So this gate *boosts* the contrarian signal.

    Scale logic:
    - No others with active signals → 1.0 (no opinion)
    - All others disagree (contrarian) → 1.3 (+30% boost)
    - ≥75% of others agree → 1.0 (consensus, no extra sizing)
    - 25%–74% of others agree → 0.7 (mixed, reduce size)
    - <25% but not all disagree → 0.5 (near-contrarian, reduce more)

    Args:
        symbol:    current symbol key
        signal:    current symbol's signal (-1, 0, +1)
        consensus: dict of {symbol: signal} for ALL symbols (including self)

    Returns:
        float scale factor in [0.5, 1.3]
    """
    others = {s: sig for s, sig in consensus.items() if s != symbol and sig != 0}
    if not others:
        return 1.0

    total = len(others)
    agree = sum(1 for sig in others.values() if sig == signal)

    if agree == 0:
        # Every active other disagrees → contrarian boost
        return 1.3

    ratio = agree / total
    if ratio >= 0.75:
        return 1.0
    if ratio >= 0.25:
        return 0.7
    return 0.5


class ConsensusScalingGate:
    """Gate: scale order qty by cross-symbol signal consensus.

    Uses a shared consensus dict (symbol → signal) to detect whether the
    current symbol is trading with or against the crowd.  Because research
    shows consensus is mean-reverting (contrarian), going against the crowd
    gets a 1.3x boost while going with a weak majority gets a 0.7x reduction.

    Always returns ``allowed=True`` — this is a sizing gate, not a rejection gate.

    If the current signal is 0 (flat) or there are no other active signals,
    the gate is a no-op (scale=1.0).
    """

    name = "ConsensusScaling"

    def __init__(
        self,
        get_consensus: Optional[Callable[[], Dict[str, int]]] = None,
        consensus: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            get_consensus: callable returning {symbol: signal} dict (live, refreshed each call).
                           Takes priority over ``consensus``.
            consensus:     static or externally-managed dict reference.
                           Falls back to ``context["consensus"]`` if neither is given.
        """
        self._get_consensus = get_consensus
        self._consensus = consensus

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        sym = getattr(ev, "symbol", "") or ""

        # --- Signal lookup ---
        signal = int(getattr(ev, "signal", 0) or context.get("signal", 0))
        if signal == 0:
            return GateResult(allowed=True, scale=1.0, reason="flat_signal")

        # --- Consensus dict lookup ---
        consensus: Dict[str, int] = {}
        if self._get_consensus is not None:
            try:
                consensus = self._get_consensus()
            except Exception as exc:
                _log.warning("ConsensusScalingGate: get_consensus() failed: %s", exc)
        elif self._consensus is not None:
            consensus = self._consensus
        else:
            consensus = context.get("consensus", {})

        scale = _consensus_scale(sym, signal, consensus)

        _log.debug(
            "ConsensusScalingGate: sym=%s signal=%d consensus=%s → scale=%.2f",
            sym, signal, consensus, scale,
        )
        return GateResult(allowed=True, scale=scale)
