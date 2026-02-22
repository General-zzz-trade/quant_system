# tests/replay/test_replay_determinism.py
# 机构级最小闭环：Replay 确定性（Determinism）测试
#
# 目标：
# - 同一份输入事件序列
# - 用相同的 engine.dispatcher + engine.replay.EventReplay 跑两次
# - 最终“状态摘要”必须完全一致（hash 相同）
#
# 说明：
# - 这里故意不依赖你业务侧的 state/account/position 实现，避免被其它层阻塞
# - 只证明 engine/replay/dispatcher 的制度性：顺序、路由、处理、无随机性
#
# 运行：
#   pytest -q tests/replay/test_replay_determinism.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json

import pytest

from engine.dispatcher import EventDispatcher, Route
from engine.replay import EventReplay, ReplayConfig


# ============================================================
# Minimal Event Model (for tests only)
# ============================================================

@dataclass(frozen=True, slots=True)
class _Header:
    event_id: str
    event_index: int  # replay strict_order 依赖这个字段（可选，但建议给）


@dataclass(frozen=True, slots=True)
class _MarketBarEvent:
    """
    用于触发 dispatcher -> PIPELINE 路由：
    dispatcher 会用 EVENT_TYPE 字符串判定 route
    """
    header: _Header
    symbol: str
    ts: int
    close: float

    EVENT_TYPE: str = "market.bar"  # 必须包含 "market" 才会路由到 PIPELINE


# ============================================================
# Deterministic State + Handler
# ============================================================

@dataclass
class _DeterministicState:
    """
    一个可稳定序列化/哈希的状态容器。
    真实系统里会是你的 AccountState / PositionState / PortfolioState 等。
    """
    closes_sum: float = 0.0
    bars_count: int = 0
    last_ts: int = -1
    last_close: float = 0.0
    per_symbol_count: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_symbol_count is None:
            self.per_symbol_count = {}

    def apply_bar(self, e: _MarketBarEvent) -> None:
        # 严格确定性：不使用时间/随机/哈希 set 顺序等不稳定因素
        self.closes_sum += float(e.close)
        self.bars_count += 1
        self.last_ts = int(e.ts)
        self.last_close = float(e.close)
        self.per_symbol_count[e.symbol] = int(self.per_symbol_count.get(e.symbol, 0) + 1)

    def digest(self) -> str:
        """
        稳定哈希：json + sort_keys，避免 dict 顺序问题
        """
        payload = {
            "closes_sum": round(self.closes_sum, 12),
            "bars_count": self.bars_count,
            "last_ts": self.last_ts,
            "last_close": round(self.last_close, 12),
            "per_symbol_count": dict(sorted(self.per_symbol_count.items(), key=lambda kv: kv[0])),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _build_dispatcher_and_state() -> tuple[EventDispatcher, _DeterministicState]:
    disp = EventDispatcher()
    st = _DeterministicState()

    # 只注册 PIPELINE 路由：模拟“事实事件 -> state 更新”
    def on_pipeline(event: Any) -> None:
        # 你的真实系统里，通常是 pipeline.apply(event) 或 reducer(event)
        if isinstance(event, _MarketBarEvent):
            st.apply_bar(event)
        else:
            # 测试应当显式失败，避免吞掉异常导致“假确定性”
            raise TypeError(f"Unexpected event type: {type(event)!r}")

    disp.register(route=Route.PIPELINE, handler=on_pipeline)
    return disp, st


# ============================================================
# Event Source
# ============================================================

def _make_events(symbol: str = "BTCUSDT", n: int = 500) -> List[_MarketBarEvent]:
    # 固定生成逻辑，确保输入确定
    events: List[_MarketBarEvent] = []
    for i in range(n):
        # event_id 与 event_index 都严格递增
        hdr = _Header(event_id=f"e{i:06d}", event_index=i)
        # ts、close 采用可复现序列
        ts = 1700000000 + i
        close = 100.0 + (i % 17) * 0.1
        events.append(_MarketBarEvent(header=hdr, symbol=symbol, ts=ts, close=close))
    return events


# ============================================================
# Tests
# ============================================================

def test_replay_determinism_same_input_same_digest() -> None:
    """
    机构级硬标准：
    - 同一份事件输入
    - 两次 replay
    - 最终状态 digest 必须完全一致
    """
    events = _make_events(n=1000)

    # run #1
    disp1, st1 = _build_dispatcher_and_state()
    r1 = EventReplay(
        dispatcher=disp1,
        source=events,  # list 本身就是 EventSource（可迭代）
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    processed1 = r1.run()
    digest1 = st1.digest()

    # run #2（必须重新构建 dispatcher/state，避免“共享对象”造成假阳性）
    disp2, st2 = _build_dispatcher_and_state()
    r2 = EventReplay(
        dispatcher=disp2,
        source=events,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )
    processed2 = r2.run()
    digest2 = st2.digest()

    assert processed1 == len(events)
    assert processed2 == len(events)
    assert digest1 == digest2


def test_replay_strict_order_rejects_out_of_order() -> None:
    """
    strict_order=True 时：
    - event_index 必须单调递增（或至少不逆序，取决于你的 replay 校验实现）
    - 输入逆序必须抛错
    """
    events = _make_events(n=10)
    # 制造逆序：把最后一个放到前面
    bad = [events[-1]] + events[:-1]

    disp, _ = _build_dispatcher_and_state()
    r = EventReplay(
        dispatcher=disp,
        source=bad,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )

    with pytest.raises(Exception):
        r.run()


def test_replay_route_drop_not_counted_processed() -> None:
    """
    replay.run() 返回的 processed 不含 DROP 事件（engine/replay.py 的定义）。
    这里造一个无法路由的事件，让它被 DROP，然后校验计数规则。
    """

    @dataclass(frozen=True, slots=True)
    class _UnknownEvent:
        header: _Header
        # 不提供 EVENT_TYPE / event_type -> dispatcher 会 Route.DROP

    events: List[Any] = []
    # 9 个 market.bar + 1 个 unknown
    bars = _make_events(n=9)
    events.extend(bars)
    events.append(_UnknownEvent(header=_Header(event_id="unknown000001", event_index=9)))

    disp, st = _build_dispatcher_and_state()
    r = EventReplay(
        dispatcher=disp,
        source=events,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True, allow_drop=True),
    )

    processed = r.run()

    # processed 只统计非 DROP 的事件：9
    assert processed == 9
    assert st.bars_count == 9


def test_replay_duplicate_event_id_fails_fast() -> None:
    """
    dispatcher 有 event_id 去重制度（DuplicateEventError）。
    replay 在 stop_on_error=True 时应当立刻失败，避免“带病继续”。
    """
    events = _make_events(n=5)
    # 复制一个 event_id，制造重复
    dup = list(events)
    dup.append(_MarketBarEvent(
        header=_Header(event_id=events[2].header.event_id, event_index=5),
        symbol="BTCUSDT",
        ts=1700000000 + 999,
        close=123.45,
    ))

    disp, _ = _build_dispatcher_and_state()
    r = EventReplay(
        dispatcher=disp,
        source=dup,
        sink=None,
        config=ReplayConfig(strict_order=True, actor="replay", stop_on_error=True),
    )

    with pytest.raises(Exception):
        r.run()
