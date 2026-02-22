from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class UniverseSelector:
    symbols: Optional[Sequence[str]] = None

    def select(self, snapshot: StateSnapshot) -> List[str]:
        if self.symbols:
            return list(dict.fromkeys([str(s) for s in self.symbols]))
        syms: Set[str] = set()
        if getattr(snapshot, "symbol", None):
            syms.add(str(snapshot.symbol))
        pos = getattr(snapshot, "positions", None)
        if isinstance(pos, dict):
            for k in pos.keys():
                syms.add(str(k))
        return sorted(syms)
