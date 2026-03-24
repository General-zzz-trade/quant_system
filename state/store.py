from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from state.snapshot import StateSnapshot  # noqa: E402
from state._util import ensure_utc  # noqa: E402

# Import serialization helpers (canonical home: state.serialization)
# Re-export for backward compatibility — existing callers can still do
#   from state.store import _dc_to_dict, _state_decoder_hook, _SCALE, ...
from state.serialization import (  # noqa: E402, F401
    _SCALE,
    _dc_to_dict,
    _StateEncoder,
    _state_decoder_hook,
    _serialize_snapshot,
    _deserialize_snapshot,
    _reconstruct_snapshot,
)


@dataclass(frozen=True, slots=True)
class StateCheckpoint:
    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    bar_index: int
    snapshot: StateSnapshot


def _build_checkpoint(snapshot: StateSnapshot) -> StateCheckpoint:
    return StateCheckpoint(
        symbol=snapshot.symbol,
        ts=ensure_utc(snapshot.ts) if snapshot.ts is not None else None,
        event_id=snapshot.event_id,
        bar_index=snapshot.bar_index,
        snapshot=snapshot,
    )


class InMemoryStateStore:
    """Minimal state checkpoint store.

    Route B will often pair this with an append-only event log. Persisting events
    is preferable; checkpoints are accelerators for rebuild.
    """

    def __init__(self) -> None:
        self._latest: Dict[str, StateCheckpoint] = {}

    def save(self, snapshot: StateSnapshot) -> None:
        self._latest[snapshot.symbol] = _build_checkpoint(snapshot)

    def latest(self, symbol: str) -> Optional[StateCheckpoint]:
        return self._latest.get(symbol)


# ---------------------------------------------------------------------------
# SqliteStateStore — persistent checkpoint store using stdlib sqlite3
# ---------------------------------------------------------------------------

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    symbol      TEXT    NOT NULL,
    bar_index   INTEGER NOT NULL,
    event_id    TEXT,
    ts          TEXT,
    snapshot    TEXT    NOT NULL,
    saved_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (symbol)
);
"""

_CREATE_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS checkpoint_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    bar_index   INTEGER NOT NULL,
    event_id    TEXT,
    ts          TEXT,
    snapshot    TEXT    NOT NULL,
    saved_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_history_symbol_bar ON checkpoint_history(symbol, bar_index);
"""


class SqliteStateStore:
    """Persistent state checkpoint store backed by SQLite with WAL mode.

    Same interface as InMemoryStateStore: save(snapshot) / latest(symbol).
    Uses Python stdlib sqlite3 only (zero external deps).

    Features:
    - WAL mode for concurrent read/write safety
    - Upsert semantics: latest checkpoint per symbol
    - Optional history table for audit trail
    - Atomic writes via transactions
    """

    def __init__(
        self,
        path: str | Path,
        *,
        keep_history: bool = False,
        history_limit: int = 1000,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._keep_history = keep_history
        self._history_limit = history_limit
        self._lock = threading.Lock()
        self._closed = False

        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_CREATE_SQL)
            if keep_history:
                self._conn.executescript(_CREATE_HISTORY_SQL)
            self._conn.commit()

    def save(self, snapshot: StateSnapshot) -> None:
        blob = _serialize_snapshot(snapshot)
        ts_str = snapshot.ts.isoformat() if snapshot.ts else None

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """INSERT INTO checkpoints (symbol, bar_index, event_id, ts, snapshot)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(symbol) DO UPDATE SET
                           bar_index = excluded.bar_index,
                           event_id  = excluded.event_id,
                           ts        = excluded.ts,
                           snapshot  = excluded.snapshot,
                           saved_at  = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    """,
                    (snapshot.symbol, snapshot.bar_index, snapshot.event_id, ts_str, blob),
                )
                if self._keep_history:
                    self._conn.execute(
                        """INSERT INTO checkpoint_history (symbol, bar_index, event_id, ts, snapshot)
                           VALUES (?, ?, ?, ?, ?)""",
                        (snapshot.symbol, snapshot.bar_index, snapshot.event_id, ts_str, blob),
                    )

    def latest(self, symbol: str) -> Optional[StateCheckpoint]:
        with self._lock:
            row = self._conn.execute(
                "SELECT symbol, bar_index, event_id, ts, snapshot FROM checkpoints WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        snap = _deserialize_snapshot(row[4])
        return StateCheckpoint(
            symbol=row[0],
            ts=snap.ts,
            event_id=row[2],
            bar_index=row[1],
            snapshot=snap,
        )

    def all_symbols(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT symbol FROM checkpoints ORDER BY symbol").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "SqliteStateStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
