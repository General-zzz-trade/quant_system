from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class BasicKillOverlay:
    def allow(self, snapshot: StateSnapshot) -> tuple[bool, Sequence[str]]:
        r = snapshot.risk
        if r is None:
            return True, ()
        if getattr(r, "halted", False):
            return False, ("risk_halted",)
        if getattr(r, "blocked", False):
            return False, ("risk_blocked",)
        return True, ()
