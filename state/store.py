from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from state.snapshot import StateSnapshot
from state._util import ensure_utc


@dataclass(frozen=True, slots=True)
class StateCheckpoint:
    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    bar_index: int
    snapshot: StateSnapshot


class InMemoryStateStore:
    """Minimal state checkpoint store.

    Route B will often pair this with an append-only event log. Persisting events
    is preferable; checkpoints are accelerators for rebuild.
    """

    def __init__(self) -> None:
        self._latest: Dict[str, StateCheckpoint] = {}

    def save(self, snapshot: StateSnapshot) -> None:
        cp = StateCheckpoint(
            symbol=snapshot.symbol,
            ts=ensure_utc(snapshot.ts) if snapshot.ts is not None else None,
            event_id=snapshot.event_id,
            bar_index=snapshot.bar_index,
            snapshot=snapshot,
        )
        self._latest[snapshot.symbol] = cp

    def latest(self, symbol: str) -> Optional[StateCheckpoint]:
        return self._latest.get(symbol)
