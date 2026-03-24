"""State checkpoint stores with JSON serialization (Rust-native to_dict/from_dict).

Serialization uses Rust types' built-in to_dict()/from_dict() methods.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from _quant_hotpath import (  # type: ignore[import-untyped]
    RustMarketState,
    RustPositionState,
    RustAccountState,
    RustPortfolioState,
    RustRiskState,
    RustReducerResult,
)

from state.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

# RustReducerResult is the typed result wrapper returned by individual reducer
# operations in the Rust state layer.  Exposed here for type annotations when
# inspecting reducer outputs during serialization / deserialization audits.
_ReducerResultType = RustReducerResult


def _ensure_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        raise TypeError(f"ts must be datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# Fixed-point scale: 10^8 (matches Rust Fd8)
_SCALE = 100_000_000

# ---------------------------------------------------------------------------
# Serialization helpers (Rust types ↔ JSON)
# ---------------------------------------------------------------------------

_RUST_TYPES = (RustMarketState, RustPositionState, RustAccountState,
               RustPortfolioState, RustRiskState)


def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert Rust state types / dataclasses / Decimal / datetime to plain dicts."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, _RUST_TYPES):
        return dict(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dc_to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "items") and callable(getattr(obj, "items")):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dc_to_dict(v) for v in obj]
    return obj


class _StateEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return {"__decimal__": str(obj)}
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, _RUST_TYPES):
            return dict(obj.to_dict())
        if hasattr(obj, "__dataclass_fields__"):
            return _dc_to_dict(obj)
        if hasattr(obj, "items") and callable(getattr(obj, "items")):
            return {k: _dc_to_dict(v) for k, v in obj.items()}
        return super().default(obj)


def _state_decoder_hook(d: dict[str, Any]) -> Any:
    if "__decimal__" in d:
        return Decimal(d["__decimal__"])
    if "__datetime__" in d:
        dt = datetime.fromisoformat(d["__datetime__"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return d


def _serialize_snapshot(snap: StateSnapshot) -> str:
    return json.dumps(snap, cls=_StateEncoder, ensure_ascii=False)


def _deserialize_snapshot(blob: str) -> StateSnapshot:
    raw = json.loads(blob, object_hook=_state_decoder_hook)
    return _reconstruct_snapshot(raw)


def _reconstruct_snapshot(d: dict[str, Any]) -> StateSnapshot:
    """Reconstruct a StateSnapshot from a plain dict (deserialized JSON)."""
    # Multi-symbol compat: new format "markets" dict, old format "market" single object
    markets: Dict[str, Any] = {}
    if "markets" in d and isinstance(d["markets"], dict):
        for sym, md in d["markets"].items():
            if isinstance(md, dict):
                md.pop("__dc__", None)
                if "symbol" not in md:
                    md["symbol"] = sym
                markets[sym] = RustMarketState.from_dict(md)
    elif "market" in d:
        market_d = d["market"]
        if isinstance(market_d, dict):
            market_d.pop("__dc__", None)
            sym = market_d.get("symbol", d.get("symbol", "UNKNOWN"))
            if "symbol" not in market_d:
                market_d["symbol"] = sym
            markets[sym] = RustMarketState.from_dict(market_d)

    account_d = d["account"]
    if isinstance(account_d, dict):
        account_d.pop("__dc__", None)
        account = RustAccountState.from_dict(account_d)
    else:
        account = account_d  # Already a Rust type

    positions: Dict[str, Any] = {}
    raw_pos = d.get("positions") or {}
    for sym, pos_d in raw_pos.items():
        if isinstance(pos_d, dict):
            pos_d.pop("__dc__", None)
            if "symbol" not in pos_d:
                pos_d["symbol"] = sym
            positions[sym] = RustPositionState.from_dict(pos_d)

    portfolio = None
    if d.get("portfolio") is not None:
        pd = d["portfolio"]
        if isinstance(pd, dict):
            pd.pop("__dc__", None)
            portfolio = RustPortfolioState.from_dict(pd)
        else:
            portfolio = pd

    risk = None
    if d.get("risk") is not None:
        rd = d["risk"]
        if isinstance(rd, dict):
            rd.pop("__dc__", None)
            risk = RustRiskState.from_dict(rd)
        else:
            risk = rd

    return StateSnapshot.of(
        symbol=d["symbol"],
        ts=d.get("ts"),
        event_id=d.get("event_id"),
        event_type=d.get("event_type", ""),
        bar_index=d.get("bar_index", 0),
        markets=markets,
        positions=positions,
        account=account,
        portfolio=portfolio,
        risk=risk,
    )


# ---------------------------------------------------------------------------
# State checkpoint types and stores
# ---------------------------------------------------------------------------

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
        ts=_ensure_utc(snapshot.ts) if snapshot.ts is not None else None,
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
