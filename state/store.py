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

from _quant_hotpath import (  # type: ignore[import-untyped]  # noqa: E402
    RustMarketState,
    RustPositionState,
    RustAccountState,
    RustPortfolioState,
    RustRiskState,
)

# Fixed-point scale: 10^8 (matches Rust Fd8)
_SCALE = 100_000_000


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
# JSON serialization helpers — Rust state types → plain dicts
# ---------------------------------------------------------------------------

def _rust_market_to_dict(obj: RustMarketState) -> dict:
    return {
        "symbol": obj.symbol,
        "last_price": obj.last_price,
        "open": obj.open,
        "high": obj.high,
        "low": obj.low,
        "close": obj.close,
        "volume": obj.volume,
        "last_ts": obj.last_ts,
    }


def _rust_position_to_dict(obj: RustPositionState) -> dict:
    return {
        "symbol": obj.symbol,
        "qty": obj.qty,
        "avg_price": obj.avg_price,
        "last_price": obj.last_price,
        "last_ts": obj.last_ts,
    }


def _rust_account_to_dict(obj: RustAccountState) -> dict:
    return {
        "currency": obj.currency,
        "balance": obj.balance,
        "margin_used": obj.margin_used,
        "margin_available": obj.margin_available,
        "realized_pnl": obj.realized_pnl,
        "unrealized_pnl": obj.unrealized_pnl,
        "fees_paid": obj.fees_paid,
        "last_ts": obj.last_ts,
    }


def _rust_portfolio_to_dict(obj: RustPortfolioState) -> dict:
    return {
        "total_equity": obj.total_equity,
        "cash_balance": obj.cash_balance,
        "realized_pnl": obj.realized_pnl,
        "unrealized_pnl": obj.unrealized_pnl,
        "fees_paid": obj.fees_paid,
        "gross_exposure": obj.gross_exposure,
        "net_exposure": obj.net_exposure,
        "leverage": obj.leverage,
        "margin_used": obj.margin_used,
        "margin_available": obj.margin_available,
        "margin_ratio": obj.margin_ratio,
        "symbols": list(obj.symbols),
        "last_ts": obj.last_ts,
    }


def _rust_risk_to_dict(obj: RustRiskState) -> dict:
    return {
        "blocked": obj.blocked,
        "halted": obj.halted,
        "level": obj.level,
        "message": obj.message,
        "flags": list(obj.flags),
        "equity_peak": obj.equity_peak,
        "drawdown_pct": obj.drawdown_pct,
        "last_ts": obj.last_ts,
    }


def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert Rust state types / dataclasses to plain dicts."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, RustMarketState):
        return _rust_market_to_dict(obj)
    if isinstance(obj, RustPositionState):
        return _rust_position_to_dict(obj)
    if isinstance(obj, RustAccountState):
        return _rust_account_to_dict(obj)
    if isinstance(obj, RustPortfolioState):
        return _rust_portfolio_to_dict(obj)
    if isinstance(obj, RustRiskState):
        return _rust_risk_to_dict(obj)
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
        if isinstance(obj, (RustMarketState, RustPositionState, RustAccountState,
                            RustPortfolioState, RustRiskState)):
            return _dc_to_dict(obj)
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


def _to_opt_i64(val: Any) -> Optional[int]:
    """Convert a value to Optional[int] for Rust i64 fields.

    Handles: None, int, Decimal (from old checkpoints), float.
    """
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, Decimal):
        return int(val * _SCALE)
    if isinstance(val, float):
        return int(val * _SCALE)
    return int(val)


def _to_i64(val: Any, default: int = 0) -> int:
    """Convert a value to int for Rust i64 fields (non-optional)."""
    if val is None:
        return default
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, Decimal):
        return int(val * _SCALE)
    if isinstance(val, float):
        return int(val * _SCALE)
    return int(val)


def _reconstruct_market(md: dict) -> RustMarketState:
    return RustMarketState(
        symbol=str(md["symbol"]),
        last_price=_to_opt_i64(md.get("last_price")),
        open=_to_opt_i64(md.get("open")),
        high=_to_opt_i64(md.get("high")),
        low=_to_opt_i64(md.get("low")),
        close=_to_opt_i64(md.get("close")),
        volume=_to_opt_i64(md.get("volume")),
        last_ts=md.get("last_ts"),
    )


def _reconstruct_position(pd: dict) -> RustPositionState:
    return RustPositionState(
        symbol=str(pd["symbol"]),
        qty=_to_i64(pd.get("qty", 0)),
        avg_price=_to_opt_i64(pd.get("avg_price")),
        last_price=_to_opt_i64(pd.get("last_price")),
        last_ts=pd.get("last_ts"),
    )


def _reconstruct_account(ad: dict) -> RustAccountState:
    return RustAccountState(
        currency=str(ad.get("currency", "USDT")),
        balance=_to_i64(ad.get("balance", 0)),
        margin_used=_to_i64(ad.get("margin_used", 0)),
        margin_available=_to_i64(ad.get("margin_available", 0)),
        realized_pnl=_to_i64(ad.get("realized_pnl", 0)),
        unrealized_pnl=_to_i64(ad.get("unrealized_pnl", 0)),
        fees_paid=_to_i64(ad.get("fees_paid", 0)),
        last_ts=ad.get("last_ts"),
    )


def _to_str(val: Any, default: str = "0") -> str:
    """Convert a value to string for Rust String fields (Portfolio/Risk)."""
    if val is None:
        return default
    return str(val)


def _to_opt_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    return str(val)


def _reconstruct_portfolio(pd: dict) -> RustPortfolioState:
    symbols = pd.get("symbols", [])
    if isinstance(symbols, tuple):
        symbols = list(symbols)
    return RustPortfolioState(
        total_equity=_to_str(pd.get("total_equity")),
        cash_balance=_to_str(pd.get("cash_balance")),
        realized_pnl=_to_str(pd.get("realized_pnl")),
        unrealized_pnl=_to_str(pd.get("unrealized_pnl")),
        fees_paid=_to_str(pd.get("fees_paid")),
        gross_exposure=_to_str(pd.get("gross_exposure")),
        net_exposure=_to_str(pd.get("net_exposure")),
        leverage=_to_opt_str(pd.get("leverage")),
        margin_used=_to_str(pd.get("margin_used")),
        margin_available=_to_str(pd.get("margin_available")),
        margin_ratio=_to_opt_str(pd.get("margin_ratio")),
        symbols=symbols,
        last_ts=pd.get("last_ts"),
    )


def _reconstruct_risk(rd: dict) -> RustRiskState:
    flags = rd.get("flags", [])
    if isinstance(flags, tuple):
        flags = list(flags)
    return RustRiskState(
        blocked=bool(rd.get("blocked", False)),
        halted=bool(rd.get("halted", False)),
        level=rd.get("level"),
        message=rd.get("message"),
        flags=flags,
        equity_peak=_to_str(rd.get("equity_peak")),
        drawdown_pct=_to_str(rd.get("drawdown_pct")),
        last_ts=rd.get("last_ts"),
    )


def _reconstruct_snapshot(d: dict[str, Any]) -> StateSnapshot:
    # Multi-symbol compat: new format "markets" dict, old format "market" single object
    markets: Dict[str, Any] = {}
    if "markets" in d and isinstance(d["markets"], dict):
        for sym, md in d["markets"].items():
            if isinstance(md, dict):
                md.pop("__dc__", None)
                if "symbol" not in md:
                    md["symbol"] = sym
                markets[sym] = _reconstruct_market(md)
    elif "market" in d:
        market_d = d["market"]
        if isinstance(market_d, dict):
            market_d.pop("__dc__", None)
            sym = market_d.get("symbol", d.get("symbol", "UNKNOWN"))
            if "symbol" not in market_d:
                market_d["symbol"] = sym
            markets[sym] = _reconstruct_market(market_d)

    account_d = d["account"]
    if isinstance(account_d, dict):
        account_d.pop("__dc__", None)
        account = _reconstruct_account(account_d)
    else:
        account = account_d  # Already a Rust type

    positions: Dict[str, Any] = {}
    raw_pos = d.get("positions") or {}
    for sym, pos_d in raw_pos.items():
        if isinstance(pos_d, dict):
            pos_d.pop("__dc__", None)
            if "symbol" not in pos_d:
                pos_d["symbol"] = sym
            positions[sym] = _reconstruct_position(pos_d)

    portfolio = None
    if d.get("portfolio") is not None:
        pd = d["portfolio"]
        if isinstance(pd, dict):
            pd.pop("__dc__", None)
            portfolio = _reconstruct_portfolio(pd)
        else:
            portfolio = pd  # Already a Rust type

    risk = None
    if d.get("risk") is not None:
        rd = d["risk"]
        if isinstance(rd, dict):
            rd.pop("__dc__", None)
            risk = _reconstruct_risk(rd)
        else:
            risk = rd  # Already a Rust type

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
