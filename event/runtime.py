# event/runtime.py
from __future__ import annotations

from typing import Optional, Protocol

from event.types import BaseEvent
from event.schema import SchemaRegistry
from event.store import EventStore
from event.security import Actor, EventSecurity
from event.errors import (
    EventFatalError,
    EventValidationError,
)


class EventTracer(Protocol):
    """Tracing hook for the event system."""

    def on_emit(self, event: BaseEvent) -> None: ...
    def on_error(self, error: Exception, *, context: object = None) -> None: ...
    def on_reject(self, event: BaseEvent, reason: str) -> None: ...

from event.metrics import EventMetrics
from event.policy import EventPolicy


class EventRuntime:
    """
    EventRuntime —— 事件系统的唯一执行入口（制度执法者）

    核心职责：
    1. 强制事件事实形态（BaseEvent）
    2. 调用 schema 进行制度校验
    3. 将“事实事件”写入 store
    """

    def __init__(
        self,
        *,
        schema_registry: SchemaRegistry,
        store: EventStore,
        tracer: Optional[EventTracer] = None,
        metrics: Optional[EventMetrics] = None,
        policy: Optional[EventPolicy] = None,
        security: Optional[EventSecurity] = None,
        actor: Optional[Actor] = None,
    ) -> None:
        self._schema_registry = schema_registry
        self._store = store
        self._tracer = tracer
        self._metrics = metrics
        self._policy = policy
        self._security = security
        self._actor = actor

    # ============================================================
    # 核心入口：emit
    # ============================================================

    def emit(self, event: BaseEvent, *, actor: Optional[Actor] = None) -> None:
        """
        发射一个事件（唯一合法入口）

        制度约束：
        - 只接受 BaseEvent
        - runtime 不做编码（codec 不属于这里）
        """

        # -------- 1. 事实形态校验（铁门） --------
        if not isinstance(event, BaseEvent):
            raise EventFatalError(
                f"EventRuntime.emit only accepts BaseEvent, got {type(event)}"
            )

        # -------- 2. 策略前置检查（可选） --------
        if self._policy is not None:
            self._policy.before_emit(event)

        # -------- 3. 安全边界校验（可选） --------
        if self._security is not None:
            effective_actor = actor or self._actor
            if effective_actor is None:
                raise EventFatalError("EventSecurity is enabled but runtime actor is not configured")
            self._security.check(event=event, actor=effective_actor)

        # -------- 4. Schema 校验 --------
        try:
            schema = self._schema_registry.get(
                event.event_type,
                event.version,
            )
            schema.validate(event)
        except EventValidationError:
            # 业务级校验失败，允许上抛
            raise
        except Exception as e:
            # 制度级异常（schema 缺失 / 不一致）
            raise EventFatalError(str(e)) from e

        # -------- 5. 写入事件事实 --------
        self._store.append(event)

        # -------- 6. Trace / Metrics --------
        if self._tracer is not None:
            self._tracer.on_emit(event)

        if self._metrics is not None:
            self._metrics.on_emit(event)
