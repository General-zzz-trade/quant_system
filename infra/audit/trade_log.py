"""Immutable trade audit log — append-only, traceable record of all trades.

Every fill, order, and cancel is recorded with full context for compliance
and post-trade analysis.

Uses JSONL format for easy streaming and analysis.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Single trade audit entry."""
    seq: int
    ts: datetime
    event_type: str        # "fill", "order_new", "order_cancel", "reject"
    symbol: str
    side: str              # "buy" or "sell"
    qty: str
    price: str
    fee: str = ""
    realized_pnl: str = ""
    order_id: str = ""
    correlation_id: str = ""
    venue: str = ""
    actor: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class TradeAuditLog:
    """Append-only trade audit log with JSONL persistence."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._load_last_seq()

    def _load_last_seq(self) -> int:
        if not self._path.exists():
            return 0
        seq = 0
        with self._path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    seq = max(seq, record.get("seq", 0))
                except json.JSONDecodeError:
                    continue
        return seq

    def append(
        self,
        *,
        event_type: str,
        symbol: str,
        side: str,
        qty: str,
        price: str,
        fee: str = "",
        realized_pnl: str = "",
        order_id: str = "",
        correlation_id: str = "",
        venue: str = "",
        actor: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        """Append a trade record. Thread-safe."""
        with self._lock:
            self._seq += 1
            record = TradeRecord(
                seq=self._seq,
                ts=datetime.now(timezone.utc),
                event_type=event_type,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                fee=fee,
                realized_pnl=realized_pnl,
                order_id=order_id,
                correlation_id=correlation_id,
                venue=venue,
                actor=actor,
                meta=meta or {},
            )

            data = asdict(record)
            data["ts"] = record.ts.isoformat()

            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")

        return record

    def query(
        self,
        *,
        after_seq: int = 0,
        event_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 1000,
    ) -> List[TradeRecord]:
        """Query trade records with filters."""
        if not self._path.exists():
            return []

        results: List[TradeRecord] = []
        with self._path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                seq = data.get("seq", 0)
                if seq <= after_seq:
                    continue
                if event_type and data.get("event_type") != event_type:
                    continue
                if symbol and data.get("symbol") != symbol:
                    continue

                ts = datetime.fromisoformat(data["ts"])
                meta = data.get("meta", {})
                results.append(TradeRecord(
                    seq=seq,
                    ts=ts,
                    event_type=data.get("event_type", ""),
                    symbol=data.get("symbol", ""),
                    side=data.get("side", ""),
                    qty=data.get("qty", ""),
                    price=data.get("price", ""),
                    fee=data.get("fee", ""),
                    realized_pnl=data.get("realized_pnl", ""),
                    order_id=data.get("order_id", ""),
                    correlation_id=data.get("correlation_id", ""),
                    venue=data.get("venue", ""),
                    actor=data.get("actor", ""),
                    meta=meta,
                ))

                if len(results) >= limit:
                    break

        return results

    @property
    def count(self) -> int:
        return self._seq
