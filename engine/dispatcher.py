# engine/dispatcher.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time as _time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
import threading

from _quant_hotpath import DuplicateGuard as _RustDedupGuard
from _quant_hotpath import rust_route_event_type as _rust_route_event_type

# event 侧（你的 event 层）
try:
    from event.types import EventType  # Enum 风格
except Exception:  # pragma: no cover
    EventType = None  # type: ignore


# ============================================================
# Dispatcher Errors
# ============================================================

class DispatcherError(RuntimeError):
    pass


class DuplicateEventError(DispatcherError):
    pass


# ============================================================
# Routing Types
# ============================================================

class Route(Enum):
    """
    dispatcher 的路由决策结果（制度枚举）
    """
    PIPELINE = "pipeline"      # 事实事件 → state
    DECISION = "decision"      # 意见事件 → decision_bridge
    EXECUTION = "execution"    # 执行命令 → execution_bridge
    DROP = "drop"              # 明确丢弃（无副作用）


@dataclass(frozen=True, slots=True)
class DispatchContext:
    """
    单次 dispatch 的只读上下文
    """
    event: Any
    route: Route
    actor: str                # live / replay / system
    seq: int                  # dispatcher 内部顺序号


# ============================================================
# Dispatcher (frozen v1.0)
# ============================================================

class EventDispatcher:
    """
    EventDispatcher —— engine 的“交通警察”（冻结版 v1.0）

    冻结铁律：
    1) dispatcher 不修改 event
    2) dispatcher 不关心 reducer / state
    3) dispatcher 不做 replay / time travel
    4) dispatcher 只负责顺序、路由、失败定位
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: Dict[Route, List[Callable[[Any], None]]] = {
            Route.PIPELINE: [],
            Route.DECISION: [],
            Route.EXECUTION: [],
        }
        self._seq: int = 0
        self._rust_dedup = _RustDedupGuard(ttl_sec=86400.0, max_size=500_000)

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    def register(
        self,
        *,
        route: Route,
        handler: Callable[[Any], None],
    ) -> None:
        """
        注册 handler 到指定 route
        """
        if route not in self._handlers:
            raise DispatcherError(f"未知 route: {route}")
        with self._lock:
            self._handlers[route].append(handler)

    # --------------------------------------------------------
    # Dispatch
    # --------------------------------------------------------

    def dispatch(self, *, event: Any, actor: str = "system") -> None:
        """
        dispatcher 的唯一入口

        - 保证事件顺序
        - 路由到 0~N 个 handler
        """
        with self._lock:
            self._seq += 1
            seq = self._seq

            # 事件去重（仅当 event_id 存在），带 TTL + 容量上限
            header = getattr(event, "header", None)
            event_id = getattr(header, "event_id", None)
            if isinstance(event_id, str):
                now = _time.monotonic()
                if not self._rust_dedup.check_and_insert(event_id, now):
                    raise DuplicateEventError(f"重复 event_id: {event_id}")

            route = self._route_for(event)
            ctx = DispatchContext(
                event=event,
                route=route,
                actor=actor,
                seq=seq,
            )

        # 锁外执行 handlers，避免阻塞 dispatcher
        self._invoke(ctx)

    # --------------------------------------------------------
    # Routing Rules (制度核心)
    # --------------------------------------------------------

    def _route_for(self, event: Any) -> Route:
        """
        决定 event 该走哪条路

        规则：
        - 事实事件 → PIPELINE
        - 意见事件 → DECISION
        - 命令事件 → EXECUTION
        - 其余 → DROP
        """
        et = getattr(event, "event_type", None)
        if et is None:
            header = getattr(event, "header", None)
            et = getattr(header, "event_type", None)

        # 风格 A：EventType Enum / header.event_type
        if et is not None:
            try:
                et_val = et.value if hasattr(et, "value") else et
                return self._route_from_label(str(et_val))
            except Exception:
                pass

        # 风格 B：EVENT_TYPE 字符串
        name = getattr(event, "EVENT_TYPE", None)
        if isinstance(name, str):
            return self._route_from_label(name)

        return Route.DROP

    @staticmethod
    def _route_from_label(label: str) -> Route:
        routed = _rust_route_event_type(label)
        if routed == "pipeline":
            return Route.PIPELINE
        if routed == "decision":
            return Route.DECISION
        if routed == "execution":
            return Route.EXECUTION
        return Route.DROP

    @staticmethod
    def _route_from_type(et: str) -> Route:
        return EventDispatcher._route_from_label(et)

    @staticmethod
    def _route_from_name(name: str) -> Route:
        return EventDispatcher._route_from_label(name)


    # --------------------------------------------------------
    # Invoke
    # --------------------------------------------------------

    def _invoke(self, ctx: DispatchContext) -> None:
        """
        执行对应 route 的 handlers
        """
        handlers = self._handlers.get(ctx.route, [])
        for h in handlers:
            try:
                h(ctx.event)
            except Exception:
                # dispatcher 只负责定位，不吞掉异常
                raise
