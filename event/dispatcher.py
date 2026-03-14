# event/dispatcher.py
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable, Optional

from event.errors import EventDispatchError
from event.lifecycle import EventLifecycle
from event.types import BaseEvent


# ============================================================
# Dispatcher
# ============================================================

class EventDispatcher:
    """
    EventDispatcher —— 事件异步分发器（机构级）

    职责（冻结）：
    1. 异步消费事件
    2. 顺序调用 handlers
    3. 标记生命周期（不直接操作 metrics）
    4. 背压 + retry

    明确不做：
    - 不做路由（EventBus 负责）
    - 不做校验（Runtime 负责）
    - 不做统计（Lifecycle / Metrics 负责）
    """

    def __init__(
        self,
        *,
        handlers: Iterable[Callable[[BaseEvent], None]],
        lifecycle: EventLifecycle,
        queue_size: int = 10000,
        worker_name: str = "event-dispatcher",
        retry_on_exception: bool = False,
    ) -> None:
        self._handlers = list(handlers)
        self._lifecycle = lifecycle
        self._queue: queue.Queue[BaseEvent] = queue.Queue(maxsize=queue_size)

        self._retry_on_exception = retry_on_exception
        self._worker_name = worker_name

        self._started = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name=self._worker_name,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
        self._started = False

    # --------------------------------------------------------
    # Submit
    # --------------------------------------------------------

    def submit(self, event: BaseEvent) -> None:
        if not self._started:
            raise EventDispatchError("EventDispatcher 尚未启动")

        try:
            self._queue.put(event, block=False)
        except queue.Full as exc:
            # 背压：直接标记 dropped
            self._lifecycle.mark_dropped(event)
            raise EventDispatchError("Dispatcher queue full") from exc

    # --------------------------------------------------------
    # Worker loop
    # --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._dispatch_one(event)
            self._queue.task_done()

    def _dispatch_one(self, event: BaseEvent) -> None:
        """
        分发单个事件
        """
        # 生命周期：开始分发
        self._lifecycle.mark_dispatching(event)

        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                # 生命周期：失败
                self._lifecycle.mark_failed(event)

                if self._retry_on_exception:
                    try:
                        self._queue.put(event, block=False)
                    except queue.Full:
                        self._lifecycle.mark_dropped(event)
                return

        # 生命周期：成功分发
        self._lifecycle.mark_dispatched(event)
