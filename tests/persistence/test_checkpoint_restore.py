# tests/persistence/test_checkpoint_restart_consistency.py
# 机构级最终门槛：Checkpoint / Restart 一致性测试
#
# 目标（必须满足）：
# 1) 同一事件流（含乱序/重复）：
#    - 方案 A：一次性完整处理
#    - 方案 B：处理中途 checkpoint -> “崩溃重启” -> 从 checkpoint 恢复 -> 继续处理
#    两种方案最终 state digest 必须完全一致
#
# 2) 重启后幂等仍成立：
#    - 重启后再次收到已处理过的 fill（同 fill_id 同 payload）必须被忽略
#    - 重启后收到同 fill_id 但 payload 不一致必须 fail fast
#
# 你当前系统语义（与 engine/dispatcher.py 一致）：
# - 含 "fill" 的事件会被路由到 Route.PIPELINE（事实事件）
#
# 运行：
#   pytest -q tests/persistence/test_checkpoint_restart_consistency.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json

import pytest

from engine.dispatcher import EventDispatcher, Route
from engine.replay import EventReplay, ReplayConfig


# ============================================================
# Minimal Fill Event Model
# ============================================================

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


# ============================================================
# Deterministic Execution State (persistable)
# ============================================================

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
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_qty": float(self.position_qty),
            "cash_delta": float(self.cash_delta),
            "fills_applied": int(self.fills_applied),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ExecState":
        return cls(
            position_qty=float(d["position_qty"]),
            cash_delta=float(d["cash_delta"]),
            fills_applied=int(d["fills_applied"]),
        )


# ============================================================
# Idempotency + Reorder Buffer (persistable)
# ============================================================

class _IdempotencyStore:
    def __init__(self) -> None:
        self._fp: Dict[str, str] = {}

    @staticmethod
    def _fingerprint(e: _ExecutionFillEvent) -> str:
        payload = {
            "symbol": e.symbol,
            "order_id": e.order_id,
            "fill_id": e.fill_id,
            "fill_seq": int(e.fill_seq),
            "side": e.side,
            "qty": round(float(e.qty), 12),
            "price": round(float(e.price), 12),
            "fee": round(float(e.fee), 12),
            "ts": int(e.ts),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def accept_once(self, e: _ExecutionFillEvent) -> bool:
        fp = self._fingerprint(e)
        prev = self._fp.get(e.fill_id)
        if prev is None:
            self._fp[e.fill_id] = fp
            return True
        if prev == fp:
            return False
        raise ValueError(f"Duplicate fill_id {e.fill_id!r} payload mismatch (data corruption).")

    def to_dict(self) -> Dict[str, Any]:
        # 保存所有已见 fill_id 的指纹（重启后幂等仍然成立）
        return {"fp": dict(self._fp)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_IdempotencyStore":
        inst = cls()
        inst._fp = dict(d.get("fp", {}))
        return inst


class _OrderFillReorderBuffer:
    def __init__(self) -> None:
        self._next_seq: Dict[str, int] = {}
        self._pending: Dict[str, Dict[int, _ExecutionFillEvent]] = {}

    def push(self, e: _ExecutionFillEvent) -> List[_ExecutionFillEvent]:
        oid = e.order_id
        if oid not in self._next_seq:
            self._next_seq[oid] = 1
            self._pending[oid] = {}
        self._pending[oid][int(e.fill_seq)] = e

        out: List[_ExecutionFillEvent] = []
        nxt = self._next_seq[oid]
        pend = self._pending[oid]
        while nxt in pend:
            out.append(pend.pop(nxt))
            nxt += 1
        self._next_seq[oid] = nxt
        return out

    def to_dict(self) -> Dict[str, Any]:
        # 注意：pending 里存 event，需要可序列化
        pending_ser: Dict[str, Dict[str, Any]] = {}
        for oid, mp in self._pending.items():
            pending_ser[oid] = {str(k): _event_to_dict(v) for k, v in mp.items()}
        return {
            "next_seq": dict(self._next_seq),
            "pending": pending_ser,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_OrderFillReorderBuffer":
        inst = cls()
        inst._next_seq = {str(k): int(v) for k, v in d.get("next_seq", {}).items()}
        pend_in: Dict[str, Dict[str, Any]] = d.get("pending", {})
        inst._pending = {}
        for oid, mp in pend_in.items():
            inst._pending[oid] = {int(k): _event_from_dict(v) for k, v in mp.items()}
        return inst


# ============================================================
# Fill Processor (persistable)
# ============================================================

class _FillProcessor:
    def __init__(
        self,
        state: _ExecState,
        idem: Optional[_IdempotencyStore] = None,
        reorder: Optional[_OrderFillReorderBuffer] = None,
    ) -> None:
        self.state = state
        self.idem = idem or _IdempotencyStore()
        self.reorder = reorder or _OrderFillReorderBuffer()

    def _apply_one(self, e: _ExecutionFillEvent) -> None:
        side = e.side.lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {e.side!r}")

        qty = float(e.qty)
        px = float(e.price)
        fee = float(e.fee)

        if qty <= 0 or px <= 0:
            raise ValueError("qty/price must be positive")

        if side == "buy":
            self.state.position_qty += qty
            self.state.cash_delta -= qty * px + fee
        else:
            self.state.position_qty -= qty
            self.state.cash_delta += qty * px - fee

        self.state.fills_applied += 1

    def on_fill_fact(self, event: Any) -> None:
        if not isinstance(event, _ExecutionFillEvent):
            raise TypeError(f"Unexpected fill event type: {type(event)!r}")

        if not self.idem.accept_once(event):
            return

        ready = self.reorder.push(event)
        for e in ready:
            self._apply_one(e)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "idem": self.idem.to_dict(),
            "reorder": self.reorder.to_dict(),
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "_FillProcessor":
        st = _ExecState.from_dict(snap["state"])
        idem = _IdempotencyStore.from_dict(snap["idem"])
        reorder = _OrderFillReorderBuffer.from_dict(snap["reorder"])
        return cls(state=st, idem=idem, reorder=reorder)


# ============================================================
# Checkpoint Store (file-based, deterministic)
# ============================================================

class _CheckpointStore:
    def __init__(self, path: "pytest.PathLike[str]") -> None:
        self.path = path

    def save(self, snap: Dict[str, Any]) -> None:
        raw = json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(raw)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.loads(f.read())


# ============================================================
# Helpers
# ============================================================

def _event_to_dict(e: _ExecutionFillEvent) -> Dict[str, Any]:
    return {
        "header": {"event_id": e.header.event_id, "event_index": e.header.event_index},
        "symbol": e.symbol,
        "order_id": e.order_id,
        "fill_id": e.fill_id,
        "fill_seq": e.fill_seq,
        "side": e.side,
        "qty": e.qty,
        "price": e.price,
        "fee": e.fee,
        "ts": e.ts,
        "EVENT_TYPE": e.EVENT_TYPE,
    }


def _event_from_dict(d: Dict[str, Any]) -> _ExecutionFillEvent:
    h = d["header"]
    return _ExecutionFillEvent(
        header=_Header(event_id=h["event_id"], event_index=int(h["event_index"])),
        symbol=d["symbol"],
        order_id=d["order_id"],
        fill_id=d["fill_id"],
        fill_seq=int(d["fill_seq"]),
        side=d["side"],
        qty=float(d["qty"]),
        price=float(d["price"]),
        fee=float(d.get("fee", 0.0)),
        ts=int(d.get("ts", 0)),
        EVENT_TYPE=d.get("EVENT_TYPE", "execution.fill"),
    )


def _build_dispatcher(proc: _FillProcessor) -> EventDispatcher:
    disp = EventDispatcher()
    disp.register(route=Route.PIPELINE, handler=proc.on_fill_fact)
    return disp


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
    ts: int = 1700000000,
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


def _run_with_processor(proc: _FillProcessor, events: List[_ExecutionFillEvent], strict_order: bool) -> str:
    disp = _build_dispatcher(proc)
    r = EventReplay(
        dispatcher=disp,
        source=events,
        sink=None,
        config=ReplayConfig(strict_order=strict_order, actor="replay", stop_on_error=True),
    )
    r.run()
    return proc.state.digest()


# ============================================================
# Tests
# ============================================================

def test_checkpoint_restart_final_state_matches_one_shot(tmp_path: pytest.TempPathFactory) -> None:
    """
    事件流包含乱序与重复，但我们让 replay 层允许乱序进入（strict_order=False），
    然后验证：
      one-shot digest == checkpoint/restart digest
    """
    ckpt_file = tmp_path / "ckpt.json"
    store = _CheckpointStore(ckpt_file)

    # 构造同一 order 的 fill，逻辑 seq: 1,2,3
    # 输入刻意乱序 + 重复（但 payload 一致）
    f1 = _make_fill("e000001", 0, order_id="o-1", fill_id="f1", fill_seq=1, side="buy", qty=1.0, price=100.0)
    f2 = _make_fill("e000002", 1, order_id="o-1", fill_id="f2", fill_seq=2, side="buy", qty=2.0, price=101.0)
    f2_dup = _make_fill("e000003", 2, order_id="o-1", fill_id="f2", fill_seq=2, side="buy", qty=2.0, price=101.0)
    f3 = _make_fill("e000004", 3, order_id="o-1", fill_id="f3", fill_seq=3, side="sell", qty=1.5, price=102.0)

    stream = [f2, f1, f2_dup, f3]  # 乱序 + 重复

    # A) one-shot
    proc_a = _FillProcessor(state=_ExecState())
    digest_a = _run_with_processor(proc_a, stream, strict_order=False)

    # B) checkpoint/restart in middle
    proc_b1 = _FillProcessor(state=_ExecState())
    first_half = stream[:2]  # [f2, f1]
    second_half = stream[2:]  # [f2_dup, f3]

    _run_with_processor(proc_b1, first_half, strict_order=False)

    # checkpoint snapshot（包含 state + idem + reorder pending/next_seq）
    store.save(proc_b1.snapshot())

    # 模拟崩溃：重新构造 processor
    snap = store.load()
    proc_b2 = _FillProcessor.from_snapshot(snap)

    digest_b = _run_with_processor(proc_b2, second_half, strict_order=False)

    assert digest_b == digest_a


def test_restart_preserves_idempotency_for_already_seen_fill(tmp_path: pytest.TempPathFactory) -> None:
    """
    重启后再次收到“已处理过的 fill（相同 fill_id + 相同 payload）”必须被忽略。
    """
    ckpt_file = tmp_path / "ckpt2.json"
    store = _CheckpointStore(ckpt_file)

    f1 = _make_fill("e100001", 0, order_id="o-2", fill_id="x1", fill_seq=1, side="buy", qty=1.0, price=200.0)
    f2 = _make_fill("e100002", 1, order_id="o-2", fill_id="x2", fill_seq=2, side="buy", qty=1.0, price=201.0)

    proc1 = _FillProcessor(state=_ExecState())
    _run_with_processor(proc1, [f1, f2], strict_order=False)
    store.save(proc1.snapshot())

    # restart
    proc2 = _FillProcessor.from_snapshot(store.load())

    # 重启后重复来一遍 f2（payload 完全一致）=> 不应增加 fills_applied
    before = proc2.state.fills_applied
    _run_with_processor(proc2, [f2], strict_order=False)
    after = proc2.state.fills_applied

    assert after == before  # 幂等仍然生效


def test_restart_payload_mismatch_duplicate_must_fail(tmp_path: pytest.TempPathFactory) -> None:
    """
    重启后收到同 fill_id 但 payload 不一致：必须 fail fast。
    """
    ckpt_file = tmp_path / "ckpt3.json"
    store = _CheckpointStore(ckpt_file)

    f1 = _make_fill("e200001", 0, order_id="o-3", fill_id="z1", fill_seq=1, side="buy", qty=1.0, price=300.0)
    f2 = _make_fill("e200002", 1, order_id="o-3", fill_id="z2", fill_seq=2, side="buy", qty=1.0, price=301.0)

    proc1 = _FillProcessor(state=_ExecState())
    _run_with_processor(proc1, [f1, f2], strict_order=False)
    store.save(proc1.snapshot())

    proc2 = _FillProcessor.from_snapshot(store.load())

    # 同 fill_id z2，但 qty 被篡改
    dup_bad = _make_fill("e200003", 2, order_id="o-3", fill_id="z2", fill_seq=2, side="buy", qty=9.0, price=301.0)

    disp = _build_dispatcher(proc2)
    r = EventReplay(
        dispatcher=disp,
        source=[dup_bad],
        sink=None,
        config=ReplayConfig(strict_order=False, actor="replay", stop_on_error=True),
    )

    with pytest.raises(ValueError):
        r.run()
