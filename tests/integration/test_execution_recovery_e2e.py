"""Integration tests for execution recovery across replay, restart, and reconcile."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib
import json

from engine.dispatcher import EventDispatcher, Route
from engine.replay import EventReplay, ReplayConfig
from execution.observability.incidents import reconcile_report_to_alert, synthetic_fill_to_alert
from execution.reconcile.controller import ReconcileController


@dataclass(frozen=True, slots=True)
class _Header:
    event_id: str
    event_index: int


@dataclass(frozen=True, slots=True)
class _ExecutionFillEvent:
    header: _Header
    symbol: str
    order_id: str
    fill_id: str
    fill_seq: int
    side: str
    qty: float
    price: float
    fee: float = 0.0
    ts: int = 0

    EVENT_TYPE: str = "execution.fill"


@dataclass
class _ExecState:
    position_qty: float = 0.0
    cash_delta: float = 0.0
    fills_applied: int = 0

    def digest(self) -> str:
        payload = {
            "position_qty": round(self.position_qty, 12),
            "cash_delta": round(self.cash_delta, 12),
            "fills_applied": self.fills_applied,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_qty": self.position_qty,
            "cash_delta": self.cash_delta,
            "fills_applied": self.fills_applied,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_ExecState":
        return cls(
            position_qty=float(data["position_qty"]),
            cash_delta=float(data["cash_delta"]),
            fills_applied=int(data["fills_applied"]),
        )


class _IdempotencyStore:
    def __init__(self) -> None:
        self._seen: Dict[str, str] = {}

    @staticmethod
    def _fp(event: _ExecutionFillEvent) -> str:
        payload = {
            "symbol": event.symbol,
            "order_id": event.order_id,
            "fill_id": event.fill_id,
            "fill_seq": event.fill_seq,
            "side": event.side,
            "qty": round(float(event.qty), 12),
            "price": round(float(event.price), 12),
            "fee": round(float(event.fee), 12),
            "ts": int(event.ts),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def accept_once(self, event: _ExecutionFillEvent) -> bool:
        fp = self._fp(event)
        prev = self._seen.get(event.fill_id)
        if prev is None:
            self._seen[event.fill_id] = fp
            return True
        if prev == fp:
            return False
        raise ValueError(f"duplicate fill payload mismatch: {event.fill_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {"seen": dict(self._seen)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_IdempotencyStore":
        inst = cls()
        inst._seen = dict(data.get("seen", {}))
        return inst

    @property
    def fill_ids(self) -> Set[str]:
        return set(self._seen.keys())


class _ReorderBuffer:
    def __init__(self) -> None:
        self._next_seq: Dict[str, int] = {}
        self._pending: Dict[str, Dict[int, _ExecutionFillEvent]] = {}

    def push(self, event: _ExecutionFillEvent) -> List[_ExecutionFillEvent]:
        oid = event.order_id
        if oid not in self._next_seq:
            self._next_seq[oid] = 1
            self._pending[oid] = {}
        self._pending[oid][event.fill_seq] = event
        ready: List[_ExecutionFillEvent] = []
        nxt = self._next_seq[oid]
        while nxt in self._pending[oid]:
            ready.append(self._pending[oid].pop(nxt))
            nxt += 1
        self._next_seq[oid] = nxt
        return ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_seq": dict(self._next_seq),
            "pending": {
                oid: {str(k): _event_to_dict(v) for k, v in bucket.items()}
                for oid, bucket in self._pending.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_ReorderBuffer":
        inst = cls()
        inst._next_seq = {str(k): int(v) for k, v in data.get("next_seq", {}).items()}
        pending = {}
        for oid, bucket in data.get("pending", {}).items():
            pending[oid] = {int(k): _event_from_dict(v) for k, v in bucket.items()}
        inst._pending = pending
        return inst


class _FillProcessor:
    def __init__(
        self,
        state: Optional[_ExecState] = None,
        idem: Optional[_IdempotencyStore] = None,
        reorder: Optional[_ReorderBuffer] = None,
    ) -> None:
        self.state = state or _ExecState()
        self.idem = idem or _IdempotencyStore()
        self.reorder = reorder or _ReorderBuffer()

    def on_fill_fact(self, event: Any) -> None:
        if not isinstance(event, _ExecutionFillEvent):
            raise TypeError(f"unexpected fill event: {type(event)!r}")
        if not self.idem.accept_once(event):
            return
        for ready in self.reorder.push(event):
            self._apply_one(ready)

    def _apply_one(self, event: _ExecutionFillEvent) -> None:
        qty = float(event.qty)
        px = float(event.price)
        fee = float(event.fee)
        if event.side == "buy":
            self.state.position_qty += qty
            self.state.cash_delta -= qty * px + fee
        else:
            self.state.position_qty -= qty
            self.state.cash_delta += qty * px - fee
        self.state.fills_applied += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "idem": self.idem.to_dict(),
            "reorder": self.reorder.to_dict(),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "_FillProcessor":
        return cls(
            state=_ExecState.from_dict(data["state"]),
            idem=_IdempotencyStore.from_dict(data["idem"]),
            reorder=_ReorderBuffer.from_dict(data["reorder"]),
        )


class _CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, snapshot: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))

    def load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())


def _event_to_dict(event: _ExecutionFillEvent) -> Dict[str, Any]:
    return {
        "header": {"event_id": event.header.event_id, "event_index": event.header.event_index},
        "symbol": event.symbol,
        "order_id": event.order_id,
        "fill_id": event.fill_id,
        "fill_seq": event.fill_seq,
        "side": event.side,
        "qty": event.qty,
        "price": event.price,
        "fee": event.fee,
        "ts": event.ts,
        "EVENT_TYPE": event.EVENT_TYPE,
    }


def _event_from_dict(data: Dict[str, Any]) -> _ExecutionFillEvent:
    return _ExecutionFillEvent(
        header=_Header(
            event_id=data["header"]["event_id"],
            event_index=int(data["header"]["event_index"]),
        ),
        symbol=data["symbol"],
        order_id=data["order_id"],
        fill_id=data["fill_id"],
        fill_seq=int(data["fill_seq"]),
        side=data["side"],
        qty=float(data["qty"]),
        price=float(data["price"]),
        fee=float(data.get("fee", 0.0)),
        ts=int(data.get("ts", 0)),
        EVENT_TYPE=data.get("EVENT_TYPE", "execution.fill"),
    )


def _make_fill(
    event_id: str,
    event_index: int,
    *,
    order_id: str,
    fill_id: str,
    fill_seq: int,
    side: str,
    qty: float,
    price: float,
    symbol: str = "BTCUSDT",
    fee: float = 0.0,
    ts: int = 1_700_000_000,
) -> _ExecutionFillEvent:
    return _ExecutionFillEvent(
        header=_Header(event_id=event_id, event_index=event_index),
        symbol=symbol,
        order_id=order_id,
        fill_id=fill_id,
        fill_seq=fill_seq,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        ts=ts,
    )


def _run(processor: _FillProcessor, events: List[_ExecutionFillEvent]) -> None:
    dispatcher = EventDispatcher()
    dispatcher.register(route=Route.PIPELINE, handler=processor.on_fill_fact)
    EventReplay(
        dispatcher=dispatcher,
        source=events,
        sink=None,
        config=ReplayConfig(strict_order=False, actor="replay", stop_on_error=True),
    ).run()


def _incident_categories_for_replay(
    *,
    fills: List[_ExecutionFillEvent],
    report: Optional[Any] = None,
) -> set[str]:
    categories = {
        str((synthetic_fill_to_alert(fill).meta or {}).get("category", ""))
        for fill in fills
    }
    if report is not None:
        categories.add(str((reconcile_report_to_alert(report).meta or {}).get("category", "")))
    return categories


def test_restart_after_out_of_order_and_duplicate_stream_reconciles_cleanly(tmp_path: Path) -> None:
    ckpt = _CheckpointStore(tmp_path / "exec_recovery.json")

    f1 = _make_fill("e1", 0, order_id="o-1", fill_id="f1", fill_seq=1, side="buy", qty=1.0, price=100.0)
    f2 = _make_fill("e2", 1, order_id="o-1", fill_id="f2", fill_seq=2, side="buy", qty=2.0, price=101.0)
    f2_dup = _make_fill("e3", 2, order_id="o-1", fill_id="f2", fill_seq=2, side="buy", qty=2.0, price=101.0)
    f3 = _make_fill("e4", 3, order_id="o-1", fill_id="f3", fill_seq=3, side="sell", qty=1.5, price=102.0)
    stream = [f2, f1, f2_dup, f3]

    one_shot = _FillProcessor()
    _run(one_shot, stream)

    half_then_restart = _FillProcessor()
    _run(half_then_restart, stream[:2])
    ckpt.save(half_then_restart.snapshot())
    restored = _FillProcessor.from_snapshot(ckpt.load())
    _run(restored, stream[2:])

    assert restored.state.digest() == one_shot.state.digest()
    assert restored.state.position_qty == 1.5
    assert restored.state.fills_applied == 3

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal(str(restored.state.position_qty))},
        venue_positions={"BTCUSDT": Decimal("1.5")},
        local_fill_ids=restored.idem.fill_ids,
        venue_fill_ids={"f1", "f2", "f3"},
        fill_symbol="BTCUSDT",
    )
    assert report.ok


def test_restart_reconcile_detects_missing_late_fill_on_venue(tmp_path: Path) -> None:
    ckpt = _CheckpointStore(tmp_path / "exec_recovery_late_fill.json")

    base = [
        _make_fill("e10", 0, order_id="o-2", fill_id="lf1", fill_seq=1, side="buy", qty=1.0, price=200.0),
        _make_fill("e11", 1, order_id="o-2", fill_id="lf2", fill_seq=2, side="buy", qty=1.0, price=201.0),
    ]
    late_fill = _make_fill("e12", 2, order_id="o-2", fill_id="lf3", fill_seq=3, side="sell", qty=0.5, price=202.0)

    proc = _FillProcessor()
    _run(proc, base)
    ckpt.save(proc.snapshot())
    restored = _FillProcessor.from_snapshot(ckpt.load())
    _run(restored, [late_fill])

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal(str(restored.state.position_qty))},
        venue_positions={"BTCUSDT": Decimal("0.5")},
        local_fill_ids=restored.idem.fill_ids,
        venue_fill_ids={"lf1", "lf2"},
        fill_symbol="BTCUSDT",
    )

    assert not report.ok
    assert len(report.all_drifts) >= 1


def test_replay_incident_categories_match_fill_and_reconcile_semantics(tmp_path: Path) -> None:
    ckpt = _CheckpointStore(tmp_path / "exec_replay_incidents.json")

    base = [
        _make_fill("e20", 0, order_id="o-3", fill_id="rf1", fill_seq=1, side="buy", qty=1.0, price=300.0),
        _make_fill("e21", 1, order_id="o-3", fill_id="rf2", fill_seq=2, side="buy", qty=1.0, price=301.0),
    ]
    late_fill = _make_fill("e22", 2, order_id="o-3", fill_id="rf3", fill_seq=3, side="sell", qty=0.5, price=302.0)

    proc = _FillProcessor()
    _run(proc, base)
    ckpt.save(proc.snapshot())
    restored = _FillProcessor.from_snapshot(ckpt.load())
    _run(restored, [late_fill])

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal(str(restored.state.position_qty))},
        venue_positions={"BTCUSDT": Decimal("0.5")},
        local_fill_ids=restored.idem.fill_ids,
        venue_fill_ids={"rf1", "rf2"},
        fill_symbol="BTCUSDT",
    )

    categories = _incident_categories_for_replay(fills=[late_fill], report=report)

    assert "execution_fill" in categories
    assert "execution_reconcile" in categories
