# event/store.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from event.types import BaseEvent
from event.errors import EventFatalError


# ============================================================
# Event Store Interface
# ============================================================

class EventStore(ABC):
    """
    EventStore —— 事件事实仓库（BaseEvent-only）

    铁律：
    - 只存 BaseEvent（事实）
    - 不编码、不解码
    - 不接触 dict / payload
    """

    @abstractmethod
    def append(self, event: BaseEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def iter_events(self) -> Iterable[BaseEvent]:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


# ============================================================
# In-Memory Store (冻结参考实现)
# ============================================================

class InMemoryEventStore(EventStore):
    """
    InMemoryEventStore —— 内存事件事实仓库

    用途：
    - 回测
    - Replay
    - 单元测试
    - event 层制度封顶阶段
    """

    def __init__(self) -> None:
        self._events: List[BaseEvent] = []

    def append(self, event: BaseEvent) -> None:
        if not isinstance(event, BaseEvent):
            raise EventFatalError(
                f"EventStore.append only accepts BaseEvent, got {type(event)}"
            )
        self._events.append(event)

    def iter_events(self) -> Iterable[BaseEvent]:
        # 返回不可变快照，防止外部篡改事实
        return tuple(self._events)

    def size(self) -> int:
        return len(self._events)


# ============================================================
# Utilities（可选）
# ============================================================

def assert_store_integrity(store: EventStore) -> None:
    """
    开发期校验：确认 store 内部只存 BaseEvent
    """
    for e in store.iter_events():
        if not isinstance(e, BaseEvent):
            raise EventFatalError(
                f"Store integrity violation: {type(e)} is not BaseEvent"
            )
