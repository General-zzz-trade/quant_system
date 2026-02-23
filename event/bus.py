# event/bus.py
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Iterable, List, Optional, Type

from event.types import BaseEvent


# ============================================================
# 类型定义
# ============================================================

# 事件处理函数签名
EventHandler = Callable[[BaseEvent], None]


@dataclass(frozen=True)
class Subscription:
    """
    订阅定义（冻结）

    约定：
    - 通过 event_type 或 event_cls 进行路由
    - 二者至少提供一个
    """
    handler: EventHandler
    event_type: Optional[str] = None
    event_cls: Optional[Type[BaseEvent]] = None


# ============================================================
# EventBus
# ============================================================

class EventBus:
    """
    EventBus —— 事件路由器（机构级最小实现）

    职责（且仅有这些）：
    1. 注册 handler
    2. 根据 event.event_type / event.__class__ 路由
    3. 顺序调用 handler

    明确不做：
    - 不做异步（dispatcher 负责）
    - 不做校验（runtime 负责）
    - 不做统计（metrics 负责）
    - 不做失败策略（dispatcher / runtime 负责）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_type: DefaultDict[str, List[EventHandler]] = defaultdict(list)
        self._by_cls: DefaultDict[Type[BaseEvent], List[EventHandler]] = defaultdict(list)
        self._any: List[EventHandler] = []

    # --------------------------------------------------------
    # 注册
    # --------------------------------------------------------

    def subscribe(
        self,
        *,
        handler: EventHandler,
        event_type: Optional[str] = None,
        event_cls: Optional[Type[BaseEvent]] = None,
    ) -> None:
        """
        注册事件处理器

        规则：
        - event_type / event_cls 至少一个
        - 两者都提供时，handler 会被注册到两条路由
        """
        if event_type is None and event_cls is None:
            raise ValueError("subscribe 至少需要 event_type 或 event_cls")

        with self._lock:
            if event_type is not None:
                self._by_type[event_type].append(handler)

            if event_cls is not None:
                self._by_cls[event_cls].append(handler)

    def subscribe_any(self, handler: EventHandler) -> None:
        """
        订阅所有事件（谨慎使用）
        """
        with self._lock:
            self._any.append(handler)

    def unsubscribe(
        self,
        *,
        handler: EventHandler,
        event_type: Optional[str] = None,
        event_cls: Optional[Type[BaseEvent]] = None,
    ) -> None:
        """
        注销事件处理器

        规则：
        - event_type / event_cls 至少一个
        - 如果 handler 未注册，静默忽略
        """
        if event_type is None and event_cls is None:
            raise ValueError("unsubscribe 至少需要 event_type 或 event_cls")

        with self._lock:
            if event_type is not None:
                handlers = self._by_type.get(event_type, [])
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass

            if event_cls is not None:
                handlers = self._by_cls.get(event_cls, [])
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass

    def unsubscribe_any(self, handler: EventHandler) -> None:
        """
        注销 subscribe_any 注册的处理器
        """
        with self._lock:
            try:
                self._any.remove(handler)
            except ValueError:
                pass

    # --------------------------------------------------------
    # 发布（由 dispatcher 调用）
    # --------------------------------------------------------

    def publish(self, event: BaseEvent) -> None:
        """
        发布事件到所有匹配的 handler

        注意：
        - handler 顺序 = 注册顺序
        - handler 异常由 dispatcher/runtime 处理
        """
        # 快照 handler 列表（锁内复制，锁外执行，避免迭代中修改）
        with self._lock:
            any_handlers = list(self._any)
            type_handlers = list(self._by_type.get(event.event_type, ()))
            cls_handlers = [
                (cls, list(hs)) for cls, hs in self._by_cls.items()
            ]

        # 1) 任意订阅
        for h in any_handlers:
            h(event)

        # 2) 按 event_type 路由
        for h in type_handlers:
            h(event)

        # 3) 按 event class 路由（支持继承）
        for cls, hs in cls_handlers:
            if isinstance(event, cls):
                for h in hs:
                    h(event)
