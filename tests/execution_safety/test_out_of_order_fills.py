# tests/execution_safety/test_out_of_order_fills.py
# 机构级实盘安全线：Out-of-Order Fills（乱序成交回报）一致性测试
#
# 目标（必须满足）：
# - 成交回报到达顺序可能乱（fill_index / ts 乱序）
# - 系统最终的“可审计结果”必须与“按逻辑顺序处理”一致
#
# 你当前系统语义（与 engine/dispatcher.py 一致）：
# - 名称包含 "fill" 的事件会被路由到 Route.PIPELINE（事实事件）
# - 因此本测试将 fill handler 注册在 Route.PIPELINE
#
# 运行：
#   pytest -q tests/execution_safety/test_out_of_order_fills.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
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
    """
    事实事件：包含 fill -> Route.PIPELINE
    order_id: 订单
    fill_id : 幂等键
    fill_seq: 逻辑序（交易所/撮合层应有的序列号；没有也可以用 ts+fill_id 复合）
    """
    header: _Header
    symbol: str
    order_id: str
    fill_id: str
    fill_seq: int          # 逻辑顺序号（关键：乱序到达时用它重排）
    side: str              # "buy" / "sell"
    qty: float
    price: float
    fee: float = 0.0
    ts: int = 0

    EVENT_TYPE: str = "execution.fill"


# ============================================================
# Deterministic Execution State
# ============================================================

@dataclass
class _ExecState:
    """
    我们比较的不是“中间过程”，而是最终一致性：
    - position_qty
    - cash_delta
    - fills_applied
    """
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


# ============================================================
# Idempotency + Reorder Buffer (institutional)
# ============================================================

class _IdempotencyStore:
    """
    幂等：fill_id 唯一
    """
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


class _OrderFillReorderBuffer:
    """
    机构常见做法：对“同一 order_id”的 fill 做序列重排（以 fill_seq 为准）。
    - 乱序到达：先缓存
    - 当 next_seq 到齐：按序依次释放 apply
    - 不强求必须连续（真实交易所有时给 gap）；本测试用“连续”来验证正确性
    """
    def __init__(self) -> None:
        self._next_seq: Dict[str, int] = {}
        self._pending: Dict[str, Dict[int, _ExecutionFillEvent]] = {}

    def push(self, e: _ExecutionFillEvent) -> List[_ExecutionFillEvent]:
        order_id = e.order_id
        if order_id not in self._next_seq:
            # 机构策略：首次看到 order_id，next_seq 从最小 seq 开始
            # 为了测试可控，我们假设 fill_seq 从 1 开始
            self._next_seq[order_id] = 1
            self._pending[order_id] = {}

        self._pending[order_id][int(e.fill_seq)] = e

        out: List[_ExecutionFillEvent] = []
        nxt = self._next_seq[order_id]
        pend = self._pending[order_id]

        while nxt in pend:
            out.append(pend.pop(nxt))
            nxt += 1

        self._next_seq[order_id] = nxt
        return out


# ============================================================
# Fill Processor (idempotent + reorder)
# ============================================================

class _FillProcessor:
    def __init__(self, state: _ExecState) -> None:
        self.state = state
        self.idem = _IdempotencyStore()
        self.reorder = _OrderFillReorderBuffer()

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

        # 幂等：重复一致 fill 忽略
        if not self.idem.accept_once(event):
            return

        # 乱序：缓存并按 fill_seq 释放
        ready = self.reorder.push(event)
        for e in ready:
            self._apply_one(e)


# ============================================================
# Helpers
# ============================================================

def _build_dispatcher_and_state() -> tuple[EventDispatcher, _ExecState]:
    disp = EventDispatcher()
    st = _ExecState()
    proc = _FillProcessor(st)

    # 关键：fill 事件 -> Route.PIPELINE
    disp.register(route=Route.PIPELINE, handler=proc.on_fill_fact)
    return disp, st


def _make_fill(
    event_id: str,
    event_index: int,
    *,
    symbol: str = "BTCUSDT",
    order_id: str = "o-001",
    fill_id: str,
    fill_seq: int,
    side: str,
    qty: float,
    price: float,
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


def _run(events: List[_ExecutionFillEvent]) -> str:
    disp, st = _build_dispatcher_and_state()
    r = EventReplay(
        dispatcher=disp,
        source=events,
        sink=None,
        config=ReplayConfig(strict_order=False, actor="replay", stop_on_error=True, allow_drop=False),
    )
    r.run()
    return st.digest()


# ============================================================
# Tests
# ============================================================

def test_out_of_order_fills_final_state_matches_logical_order() -> None:
    """
    核心测试：
    - 构造同一 order 的 3 个 fill，逻辑序为 1,2,3
    - 输入顺序故意打乱：2,1,3
    - 最终 digest 必须等于“按逻辑序输入”的 digest
    """
    logical = [
        _make_fill("e000001", 0, order_id="o-777", fill_id="f1", fill_seq=1, side="buy", qty=1.0, price=100.0),
        _make_fill("e000002", 1, order_id="o-777", fill_id="f2", fill_seq=2, side="buy", qty=2.0, price=101.0),
        _make_fill("e000003", 2, order_id="o-777", fill_id="f3", fill_seq=3, side="sell", qty=1.5, price=102.0),
    ]
    out_of_order = [logical[1], logical[0], logical[2]]  # 2,1,3

    digest_logical = _run(logical)
    digest_ooo = _run(out_of_order)

    assert digest_ooo == digest_logical


def test_out_of_order_with_duplicates_still_idempotent_and_consistent() -> None:
    """
    乱序 + 重复混合：
    - 2,1,2(dup),3
    - 结果仍必须与逻辑序一致
    """
    f1 = _make_fill("e100001", 0, order_id="o-888", fill_id="x1", fill_seq=1, side="buy", qty=1.0, price=200.0)
    f2 = _make_fill("e100002", 1, order_id="o-888", fill_id="x2", fill_seq=2, side="buy", qty=1.0, price=201.0)
    f3 = _make_fill("e100003", 2, order_id="o-888", fill_id="x3", fill_seq=3, side="sell", qty=0.5, price=202.0)

    logical = [f1, f2, f3]
    mixed = [
        f2,
        f1,
        _make_fill("e100004", 3, order_id="o-888", fill_id="x2", fill_seq=2, side="buy", qty=1.0, price=201.0),  # dup identical
        f3,
    ]

    assert _run(mixed) == _run(logical)


def test_out_of_order_payload_mismatch_duplicate_must_fail() -> None:
    """
    fill_id 重复但 payload 不一致：必须 fail fast（数据损坏）。
    """
    f1 = _make_fill("e200001", 0, order_id="o-999", fill_id="z1", fill_seq=1, side="buy", qty=1.0, price=300.0)
    f2 = _make_fill("e200002", 1, order_id="o-999", fill_id="z2", fill_seq=2, side="buy", qty=1.0, price=301.0)
    dup_bad = _make_fill("e200003", 2, order_id="o-999", fill_id="z2", fill_seq=2, side="buy", qty=9.0, price=301.0)  # qty mismatch

    disp, _st = _build_dispatcher_and_state()
    r = EventReplay(
        dispatcher=disp,
        source=[f2, f1, dup_bad],  # 乱序 + 坏重复
        sink=None,
        config=ReplayConfig(strict_order=False, actor="replay", stop_on_error=True),
    )

    with pytest.raises(ValueError):
        r.run()
