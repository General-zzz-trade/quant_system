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

logger = logging.getLogger(__name__)

from state.snapshot import StateSnapshot  # noqa: E402
from state._util import ensure_utc  # noqa: E402


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
# JSON serialization helpers for frozen dataclass state objects
# ---------------------------------------------------------------------------

def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert frozen dataclasses to plain dicts, handling MappingProxyType."""
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    try:
        from _quant_hotpath import (  # type: ignore[import-untyped]
            RustAccountState,
            RustMarketState,
            RustPortfolioState,
            RustPositionState,
            RustRiskState,
        )
        from state.rust_adapters import (
            account_from_rust,
            market_from_rust,
            portfolio_from_rust,
            position_from_rust,
            risk_from_rust,
        )

        if isinstance(obj, RustMarketState):
            return _dc_to_dict(market_from_rust(obj))
        if isinstance(obj, RustPositionState):
            return _dc_to_dict(position_from_rust(obj))
        if isinstance(obj, RustAccountState):
            return _dc_to_dict(account_from_rust(obj))
        if isinstance(obj, RustPortfolioState):
            return _dc_to_dict(portfolio_from_rust(obj))
        if isinstance(obj, RustRiskState):
            return _dc_to_dict(risk_from_rust(obj))
    except Exception as e:
        logger.debug("Failed to convert Rust state object: %s", e)
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dc_to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    # MappingProxyType and other Mapping-like objects
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
    from state.market import MarketState
    from state.account import AccountState
    from state.position import PositionState
    from state.portfolio import PortfolioState
    from state.risk import RiskState

    # 多品种兼容：新格式 "markets" dict，旧格式 "market" 单对象
    markets: Dict[str, MarketState] = {}
    if "markets" in d and isinstance(d["markets"], dict):
        for sym, md in d["markets"].items():
            if isinstance(md, dict):
                md.pop("__dc__", None)
                markets[sym] = MarketState(**md)
    elif "market" in d:
        market_d = d["market"]
        if isinstance(market_d, dict):
            market_d.pop("__dc__", None)
            sym = market_d.get("symbol", d.get("symbol", "UNKNOWN"))
            markets[sym] = MarketState(**market_d)

    account_d = d["account"]
    account_d.pop("__dc__", None)
    account = AccountState(**account_d)

    positions: Dict[str, PositionState] = {}
    raw_pos = d.get("positions") or {}
    for sym, pos_d in raw_pos.items():
        if isinstance(pos_d, dict):
            pos_d.pop("__dc__", None)
            positions[sym] = PositionState(**pos_d)

    portfolio = None
    if d.get("portfolio") is not None:
        pd = d["portfolio"]
        pd.pop("__dc__", None)
        if isinstance(pd.get("symbols"), list):
            pd["symbols"] = tuple(pd["symbols"])
        portfolio = PortfolioState(**pd)

    risk = None
    if d.get("risk") is not None:
        rd = d["risk"]
        rd.pop("__dc__", None)
        if isinstance(rd.get("flags"), list):
            rd["flags"] = tuple(rd["flags"])
        risk = RiskState(**rd)

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
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SqliteStateStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
