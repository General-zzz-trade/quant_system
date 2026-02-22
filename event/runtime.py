# event/runtime.py
from __future__ import annotations

from typing import Optional

from event.types import BaseEvent
from event.schema import SchemaRegistry
from event.store import EventStore
from event.errors import (
    EventFatalError,
    EventValidationError,
)
from event.trace import EventTracer
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
    ) -> None:
        self._schema_registry = schema_registry
        self._store = store
        self._tracer = tracer
        self._metrics = metrics
        self._policy = policy

    # ============================================================
    # 核心入口：emit
    # ============================================================

    def emit(self, event: BaseEvent) -> None:
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

        # -------- 3. Schema 校验 --------
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

        # -------- 4. 写入事件事实 --------
        self._store.append(event)

        # -------- 5. Trace / Metrics --------
        if self._tracer is not None:
            self._tracer.on_emit(event)

        if self._metrics is not None:
            self._metrics.on_emit(event)
