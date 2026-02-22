# tests/execution_safety/test_duplicate_fills.py
# 机构级实盘安全线：Duplicate Fills（重复成交回报）幂等测试
#
# 重要说明（与你的 engine/dispatcher.py 路由规则一致）：
# - dispatcher 会把包含 "fill" 的事件路由到 Route.PIPELINE（事实事件）
# - 所以这里 fill handler 必须注册在 Route.PIPELINE，而不是 Route.EXECUTION
#
# 目标（必须满足）：
# 1) 同一笔 fill（相同 fill_id）重复回报多少次，状态只能被更新一次
# 2) 若相同 fill_id 却携带不同内容（qty/price/side等不一致），必须硬失败
#
# 运行：
#   pytest -q tests/execution_safety/test_duplicate_fills.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import json

import pytest

from engine.dispatcher import EventDispatcher, Route
from engine.replay import EventReplay, ReplayConfig


# ============================================================
# Minimal Fill Event Model (for tests only)
# ============================================================

@dataclass(frozen=True, slots=True)
class _Header:
    event_id: str
    event_index: int


@dataclass(frozen=True, slots=True)
class _ExecutionFillEvent:
    """
    你的 dispatcher 规则：name 里包含 "fill" -> Route.PIPELINE
    所以 EVENT_TYPE 只要包含 fill 即可（execution.fill / order.fill 都会走 PIPELINE）
    """
    header: _Header
    symbol: str
    order_id: str
    fill_id: str
    side: str            # "buy" / "sell"
    qty: float
    price: float
    fee: float = 0.0
    ts: int = 0

    EVENT_TYPE: str = "execution.fill"


# ============================================================
# Minimal Deterministic Execution State
# ============================================================

@dataclass
class _ExecState:
    """
    用最小 state 来验证“不会重复算仓位/不会重复算现金”
    """
    position_qty: float = 0.0
    cash_delta: float = 0.0  # buy 为负，sell 为正（净变化）
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
# Idempotency Store + Fill Processor (institutional policy)
# ============================================================

class _IdempotencyStore:
    """
    机构标准：fill_id 是幂等键。
    - 首次出现：记录 payload 指纹并放行
    - 再次出现且 payload 完全一致：忽略（idempotent）
    - 再次出现但 payload 不一致：硬失败（数据一致性破坏）
    """
    def __init__(self) -> None:
        self._fingerprints: Dict[str, str] = {}

    @staticmethod
    def _fp(e: _ExecutionFillEvent) -> str:
        payload = {
            "symbol": e.symbol,
            "order_id": e.order_id,
            "fill_id": e.fill_id,
            "side": e.side,
            "qty": round(float(e.qty), 12),
            "price": round(float(e.price), 12),
            "fee": round(float(e.fee), 12),
            "ts": int(e.ts),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def should_apply(self, e: _ExecutionFillEvent) -> bool:
        fp = self._fp(e)
        prev = self._fingerprints.get(e.fill_id)
        if prev is None:
            self._fingerprints[e.fill_id] = fp
            return True
        if prev == fp:
            return False
        raise ValueError(f"Fill id {e.fill_id!r} payload mismatch (non-idempotent duplicate).")


class _FillProcessor:
    def __init__(self, state: _ExecState, idem: _IdempotencyStore) -> None:
        self.state = state
        self.idem = idem

    def on_fill_fact(self, event: Any) -> None:
        # 在你的系统语义里：fill 是事实事件（pipeline）
        if not isinstance(event, _ExecutionFillEvent):
            raise TypeError(f"Unexpected fill event type: {type(event)!r}")

        if not self.idem.should_apply(event):
            return  # 幂等重复 -> 不做任何变更

        side = event.side.lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {event.side!r}")

        qty = float(event.qty)
        px = float(event.price)
        fee = float(event.fee)

        if qty <= 0 or px <= 0:
            raise ValueError("qty/price must be positive")

        # 最小可审计的资金/仓位变更：
        # buy : position += qty; cash -= qty*px + fee
        # sell: position -= qty; cash += qty*px - fee
        if side == "buy":
            self.state.position_qty += qty
            self.state.cash_delta -= qty * px + fee
        else:
            self.state.position_qty -= qty
            self.state.cash_delta += qty * px - fee

        self.state.fills_applied += 1


# ============================================================
# Helpers
# ============================================================

def _build_dispatcher_and_exec_state() -> tuple[EventDispatcher, _ExecState]:
    disp = EventDispatcher()
    st = _ExecState()
    idem = _IdempotencyStore()
    proc = _FillProcessor(st, idem)

    # ✅ 关键：fill 事件在你的 dispatcher 规则下会路由到 PIPELINE
    disp.register(route=Route.PIPELINE, handler=proc.on_fill_fact)
    return disp, st


def _make_fill(
    event_id: str,
    event_index: int,
    *,
    symbol: str = "BTCUSDT",
    order_id: str = "o-001",
    fill_id: str = "f-001",
    side: str = "buy",
    qty: float = 1.0,
    price: float = 100.0,
    fee: float = 0.1,
    ts: int = 1700000000,
) -> _ExecutionFillEvent:
    return _ExecutionFillEvent(
        header=_Header(event_id=event_id, event_index=event_index),
        symbol=symbol,
        order_id=order_id,
        fill_id=fill_id,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        ts=ts,
    )


# ============================================================
# Tests
# ============================================================

def test_duplicate_fill_same_payload_applies_once() -> None:
    """
    同一 fill 回报重复两次（payload 完全相同）：
    - fills_applied 只能 +1
    - position/cash 只能变化一次
    """
    e1 = _make_fill("e000001", 0, fill_id="f-dup", side="buy", qty=2.0, price=200.0, fee=0.2)
    e2 = _make_fill("e000002", 1, fill_id="f-dup", side="buy", qty=2.0, price=200.0, fee=0.2)

    disp, st = _build_dispatcher_and_exec_state()
    r = EventReplay(
        dispatcher=disp,
        source=[e1, e2],
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    processed = r.run()

    assert processed == 2
    assert st.fills_applied == 1
    assert st.position_qty == pytest.approx(2.0)
    assert st.cash_delta == pytest.approx(-(2.0 * 200.0 + 0.2))


def test_duplicate_fill_payload_mismatch_must_fail_fast() -> None:
    """
    同一 fill_id 却出现不同 qty/price/side 等内容：必须硬失败。
    """
    e1 = _make_fill("e000010", 0, fill_id="f-bad", side="buy", qty=1.0, price=100.0, fee=0.1)
    e2 = _make_fill("e000011", 1, fill_id="f-bad", side="buy", qty=3.0, price=100.0, fee=0.1)  # qty changed

    disp, _ = _build_dispatcher_and_exec_state()
    r = EventReplay(
        dispatcher=disp,
        source=[e1, e2],
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )

    with pytest.raises(ValueError):
        r.run()


def test_duplicate_fill_not_double_count_across_long_stream() -> None:
    """
    长事件流里混入重复 fill，最终 digest 必须等同于“去重后”的结果。
    """
    base = [
        _make_fill("e100001", 0, fill_id="f1", side="buy", qty=1.0, price=100.0, fee=0.0),
        _make_fill("e100002", 1, fill_id="f2", side="buy", qty=2.0, price=110.0, fee=0.0),
        _make_fill("e100003", 2, fill_id="f3", side="sell", qty=1.5, price=120.0, fee=0.0),
    ]

    stream = [
        base[0],
        base[1],
        _make_fill("e100004", 3, fill_id="f2", side="buy", qty=2.0, price=110.0, fee=0.0),  # dup
        base[2],
        _make_fill("e100005", 4, fill_id="f1", side="buy", qty=1.0, price=100.0, fee=0.0),  # dup
    ]

    disp1, st1 = _build_dispatcher_and_exec_state()
    r1 = EventReplay(
        dispatcher=disp1,
        source=base,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    r1.run()
    digest_base = st1.digest()

    disp2, st2 = _build_dispatcher_and_exec_state()
    r2 = EventReplay(
        dispatcher=disp2,
        source=stream,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    r2.run()
    digest_stream = st2.digest()

    assert digest_stream == digest_base
    assert st2.fills_applied == 3  # 只应用 f1,f2,f3 三次


def test_duplicate_fill_side_case_sell_duplicate() -> None:
    """
    覆盖 sell 的幂等：重复 sell fill 不能重复减仓/加现金。
    """
    e1 = _make_fill("e200001", 0, fill_id="f-sell", side="sell", qty=1.0, price=150.0, fee=0.5)
    e2 = _make_fill("e200002", 1, fill_id="f-sell", side="sell", qty=1.0, price=150.0, fee=0.5)

    disp, st = _build_dispatcher_and_exec_state()
    r = EventReplay(
        dispatcher=disp,
        source=[e1, e2],
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    r.run()

    assert st.fills_applied == 1
    assert st.position_qty == pytest.approx(-1.0)
    assert st.cash_delta == pytest.approx(1.0 * 150.0 - 0.5)
