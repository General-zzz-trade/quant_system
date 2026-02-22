# quant_system/context/context.py
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Dict
import time
import uuid

from event.bus import EventBus, Subscription
from event.clock import Clock
from event.types import BaseEvent, EventType, MarketEvent
from context.market.market_state import MarketState, MarketSnapshot

# 重要：先引入“硬接口”，哪怕 validate_context 目前是空实现也必须存在
# 你需要在 context/validators.py 里提供 validate_context(context: Context) -> None
from context.validators import validate_context  # type: ignore


class ContextError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """
    Context 的轻量快照（用于 debug / 审计 / 回放对齐）

    顶级机构级硬约束：
    - Snapshot 必须真正不可变（不能把可变 dict/list 直接塞进来）
    - Snapshot 必须可追责：能回答“它从哪里来、属于哪个 Context、父快照是谁”
    """
    # 身份链路（审计/追责/回放对齐核心）
    context_id: str
    snapshot_id: str
    parent_snapshot_id: Optional[str]

    # 时钟与轻量元信息
    ts: Any
    bar_index: int
    meta: Mapping[str, Any]  # 强制存在，但可为空映射

    # 可选：生成时间（用于审计日志对齐；不是交易时间）
    created_at_ms: int


class Context:
    """
    顶级机构级 Context（最小可封版骨架）

    设计铁律：
    1) Context 是全系统唯一真相（Single Source of Truth）
    2) 任何状态更新必须由事件驱动（Event-driven）
    3) 对外只暴露 Snapshot / View，不暴露可变 State
    4) Context 不做策略、不做风控、不做执行，只做“事实更新 + 视图隔离”
    """

    def __init__(
        self,
        *,
        clock: Clock,
        market_state: Optional[MarketState] = None,
        meta: Optional[Mapping[str, Any]] = None,
        context_id: Optional[str] = None,
    ) -> None:
        self._clock = clock
        self._market = market_state or MarketState()

        # 内部 meta 保持 dict 方便内部更新，但对外必须冻结
        self._meta: Dict[str, Any] = dict(meta or {})

        # Context 身份（用于审计 / lineage）
        self._context_id: str = context_id or f"ctx_{uuid.uuid4().hex}"

        # Snapshot lineage：用于回答“当前 snapshot 的父亲是谁”
        self._last_snapshot_id: Optional[str] = None

        # 最近事件信息（debug/审计辅助）
        self._last_event_id: Optional[str] = None
        self._last_trace_id: Optional[str] = None
        self._last_source: Optional[str] = None

        # 统计被忽略事件（最小可观测性）
        self._ignored_events: int = 0

    # ============================================================
    # 只读属性（审计用途）
    # ============================================================

    @property
    def context_id(self) -> str:
        """Context 的稳定身份标识（审计/追责用）。"""
        return self._context_id

    # ============================================================
    # 对外只读：不建议直接暴露可变 Clock
    # ============================================================

    def clock_snapshot(self) -> Mapping[str, Any]:
        """只读时钟快照，避免外部误改 clock。"""
        return {
            "ts": self._clock.ts,
            "bar_index": self._clock.bar_index,
        }

    def snapshot(self) -> ContextSnapshot:
        """
        生成 ContextSnapshot：
        - meta 使用 MappingProxyType 冻结成只读映射，避免“冻结假象”
        - snapshot_id / parent_snapshot_id 用于 lineage（审计必备）
        """
        frozen_meta = MappingProxyType(dict(self._meta))

        snap_id = f"snap_{uuid.uuid4().hex}"
        parent_id = self._last_snapshot_id

        snap = ContextSnapshot(
            context_id=self._context_id,
            snapshot_id=snap_id,
            parent_snapshot_id=parent_id,
            ts=self._clock.ts,
            bar_index=self._clock.bar_index,
            meta=frozen_meta,
            created_at_ms=int(time.time() * 1000),
        )

        # 更新 lineage 指针
        self._last_snapshot_id = snap_id
        return snap

    # ============================================================
    # 对外只读视图（严禁返回可变 state）
    # ============================================================

    def get_market(self, *, symbol: str, venue: str) -> Optional[MarketSnapshot]:
        # 前置约束：MarketSnapshot 应该是不可变对象（建议 MarketSnapshot 用 frozen dataclass）
        return self._market.get_snapshot(symbol=symbol, venue=venue)

    def require_market(self, *, symbol: str, venue: str) -> MarketSnapshot:
        return self._market.require_snapshot(symbol=symbol, venue=venue)

    # ============================================================
    # 核心：事件应用（唯一写入口）
    # ============================================================

    def apply(self, event: BaseEvent) -> None:
        """
        将事件应用到 Context（唯一写入口）

        顶级机构级约束：
        - apply 内必须是“确定性逻辑”：同事件序列 => 同状态结果
        - 禁止在 apply 内做 IO / 网络调用
        - 必须 fail-fast：不自洽就立刻抛异常，不允许“继续跑”
        """
        self._last_event_id = event.header.event_id
        self._last_trace_id = event.header.trace_id
        self._last_source = event.header.source

        # 最小版本：只消费 MarketEvent
        if event.type != EventType.MARKET:
            self._ignored_events += 1
            return

        if not isinstance(event, MarketEvent):
            raise ContextError("event.type=MARKET 但事件对象不是 MarketEvent")

        # 记录更新前时钟，用于“时间不可倒流”的二次防线
        prev_ts = self._clock.ts
        prev_bar_index = self._clock.bar_index

        # 1) 推进 Clock：优先 header.event_time，其次 bar.ts
        et = event.header.event_time
        if et is None and getattr(event, "bar", None) is not None:
            et = event.bar.ts
        if et is None:
            raise ContextError("MarketEvent 缺少 event_time（header.event_time 或 bar.ts）")

        # 2) 是否推进 bar_index：更稳健的判定（兼容 enum / str）
        kind = getattr(event, "kind", None)
        kind_value = getattr(kind, "value", kind)  # enum -> value，str -> str
        is_bar = (kind_value == "bar")

        self._clock.update_from_event_time(et, is_bar=is_bar)

        # 2.1) 顶级机构二次防线：时间不可倒流 / bar_index 不可回退
        # 注意：如果你的业务允许乱序事件，这里需要改成“缓冲排序 + watermark”。
        # 但在“最小可封版”里，先严格禁止时间回退。
        if prev_ts is not None and self._clock.ts is not None:
            try:
                if self._clock.ts < prev_ts:
                    raise ContextError(
                        f"Clock 时间倒流：prev_ts={prev_ts} -> new_ts={self._clock.ts}"
                    )
            except TypeError:
                # ts 类型不可比较时，说明 Clock/事件时间类型不统一，必须立刻修正
                raise ContextError(
                    f"Clock ts 不可比较：prev_ts={type(prev_ts)} new_ts={type(self._clock.ts)}"
                )

        if self._clock.bar_index < prev_bar_index:
            raise ContextError(
                f"bar_index 回退：prev={prev_bar_index} -> new={self._clock.bar_index}"
            )

        # 3) 写入 MarketState（ts/bar_index 必须来自 Clock）
        self._market.update_from_market_event(
            event,
            ts=self._clock.ts,
            bar_index=self._clock.bar_index,
        )

        # 4) 顶级机构硬要求：apply 后必须校验一致性（fail-fast）
        validate_context(self)

    # ============================================================
    # EventBus 绑定（Context 作为订阅者）
    # ============================================================

    def bind_to_bus(
        self,
        bus: EventBus,
        *,
        priority: int = 10,
        enabled: bool = True,
        name: str = "context.apply",
    ) -> None:
        """
        将 Context 绑定到 EventBus，让 Context 自动消费事件。
        """
        bus.subscribe(
            Subscription(
                name=name,
                handler=self.apply,
                event_type=EventType.MARKET,
                priority=priority,
                enabled=enabled,
            )
        )

    # ============================================================
    # Debug/审计辅助（可选）
    # ============================================================

    def last_event_info(self) -> Mapping[str, Any]:
        return {
            "context_id": self._context_id,
            "last_snapshot_id": self._last_snapshot_id,
            "last_event_id": self._last_event_id,
            "last_trace_id": self._last_trace_id,
            "last_source": self._last_source,
            "ignored_events": self._ignored_events,
        }
