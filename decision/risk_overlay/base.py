"""Risk overlay protocols and composite implementation.

Risk overlays act as soft gates in the decision pipeline — they can
block or allow decisions before order generation. Unlike the hard
risk interceptor (core/interceptors.py), overlays are advisory and
run within the decision engine itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Tuple

from state.snapshot import StateSnapshot


class RiskOverlay(Protocol):
    """Protocol for decision-level risk gates."""
    def allow(self, snapshot: StateSnapshot) -> Tuple[bool, Sequence[str]]:
        """Return (allowed, reasons).

        If allowed is False, ``reasons`` should explain why.
        """
        ...


@dataclass(frozen=True, slots=True)
class AlwaysAllow:
    """No-op overlay that always allows."""
    def allow(self, snapshot: StateSnapshot) -> Tuple[bool, Sequence[str]]:
        return True, ()


@dataclass(frozen=True, slots=True)
class CompositeOverlay:
    """Chains multiple overlays — all must allow for the composite to allow.

    Collects all rejection reasons from all overlays.
    """
    overlays: Tuple[RiskOverlay, ...] = ()

    def allow(self, snapshot: StateSnapshot) -> Tuple[bool, Sequence[str]]:
        all_reasons: list[str] = []
        allowed = True

        for overlay in self.overlays:
            ok, reasons = overlay.allow(snapshot)
            if not ok:
                allowed = False
                all_reasons.extend(reasons)

        return allowed, tuple(all_reasons)
