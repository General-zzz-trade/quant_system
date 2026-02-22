# engine/loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple
import queue
import threading
import time

from engine.coordinator import EngineCoordinator
from engine.errors import EngineErrorContext
from engine.guards import Guard, GuardAction, GuardDecision, build_basic_guard, GuardConfig


class RuntimeLike(Protocol):
    def subscribe(self, handler: Callable[[Any], None]) -> None: ...
    def unsubscribe(self, handler: Callable[[Any], None]) -> None: ...


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """
    生产级最小 Loop 配置（v1.0）

    - inbox_maxsize：背压边界
    - drop_on_full：inbox 满时是否丢弃（True=保护进程；False=阻塞调用方）
    - retry_limit：RETRY 的最大重试次数（超过后 DROP）
    - idle_sleep_s：空转时 sleep，避免 busy loop
    """
    inbox_maxsize: int = 100_000
    drop_on_full: bool = True
    retry_limit: int = 3
    idle_sleep_s: float = 0.002


@dataclass(slots=True)
class _Envelope:
    event: Any
    actor: str
    retries: int = 0
    received_at: float = 0.0


def _extract_ctx(event: Any, *, actor: str, stage: str) -> EngineErrorContext:
    header = getattr(event, "header", None)
    ts = getattr(header, "ts", None)
    event_id = getattr(header, "event_id", None)

    # event_type / EVENT_TYPE（兼容两种风格）
    et = getattr(event, "event_type", None)
    if et is not None and hasattr(et, "value"):
        event_type = str(getattr(et, "value"))
    else:
        event_type = getattr(event, "EVENT_TYPE", None)
        if event_type is not None:
            event_type = str(event_type)
        else:
            event_type = None

    # symbol 常见位置
    symbol = getattr(event, "symbol", None)
    if symbol is None:
        symbol = getattr(header, "symbol", None)

    return EngineErrorContext(
        ts=ts,
        actor=actor,
        event_id=event_id if isinstance(event_id, str) else None,
        event_type=event_type,
        symbol=str(symbol) if symbol is not None else None,
        stage=stage,
        details=None,
    )


class EngineLoop:
    """
    EngineLoop（生产骨架 v1.0）

    核心职责：
    1) IO/回调线程只负责 submit() 进 inbox
    2) 单线程 drain() 串行处理 -> coordinator.emit()（保证确定性）
    3) 统一 Guard 执行点：before_event / on_error / after_event
    4) 提供背压（Queue maxsize）

    重要：Loop 不替你“做 IO”，它只隔离 IO 的并发风险。
    """

    def __init__(
        self,
        *,
        coordinator: EngineCoordinator,
        guard: Optional[Guard] = None,
        cfg: Optional[LoopConfig] = None,
    ) -> None:
        self._coord = coordinator
        self._guard: Guard = guard or build_basic_guard(GuardConfig())
        self._cfg = cfg or LoopConfig()

        self._inbox: "queue.Queue[_Envelope]" = queue.Queue(maxsize=int(self._cfg.inbox_maxsize))
        self._running = False

        # 可选：后台线程模式（你也可以不用，手动 step/drain）
        self._thread: Optional[threading.Thread] = None

        # runtime 订阅（可选）
        self._runtime: Optional[Any] = None
        self._runtime_handler: Optional[Callable[[Any], None]] = None

    @property
    def coordinator(self) -> EngineCoordinator:
        return self._coord

    @property
    def guard(self) -> Guard:
        return self._guard

    # -------------------------
    # Ingress (thread-safe)
    # -------------------------

    def submit(self, event: Any, *, actor: str = "live") -> bool:
        """
        线程安全：把事件放入 inbox。

        返回：
        - True：成功入队
        - False：inbox 满且 drop_on_full=True，被丢弃
        """
        env = _Envelope(event=event, actor=actor, retries=0, received_at=time.time())
        try:
            if self._cfg.drop_on_full:
                self._inbox.put_nowait(env)
            else:
                self._inbox.put(env)  # block
            return True
        except queue.Full:
            return False

    # -------------------------
    # Processing (single-thread)
    # -------------------------

    def step(self, *, max_events: int = 1) -> int:
        """
        单步处理（推荐在你的主循环里调用）。
        - 处理至多 max_events 个事件
        """
        if max_events <= 0:
            return 0

        processed = 0
        for _ in range(max_events):
            try:
                env = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._process_one(env)
            processed += 1
        return processed

    def drain(self, *, max_events: int = 10_000) -> int:
        """
        批量 drain（适合主循环每 tick 调一次）。
        """
        return self.step(max_events=max_events)

    def _process_one(self, env: _Envelope) -> None:
        event = env.event
        actor = env.actor

        # 1) before_event
        ctx = _extract_ctx(event, actor=actor, stage="before_event")
        d0 = self._guard.before_event(event, actor=actor, ctx=ctx)
        if d0.action == GuardAction.DROP:
            return
        if d0.action == GuardAction.STOP:
            self._coord.stop()
            self.stop_background()
            return
        # RETRY 在 before_event 通常不用；这里按 DROP 处理
        if d0.action == GuardAction.RETRY:
            self._retry_or_drop(env, d0)
            return

        # 2) main processing
        try:
            self._coord.emit(event, actor=actor)
        except BaseException as exc:
            ctx_err = _extract_ctx(event, actor=actor, stage="on_error")
            d1 = self._guard.on_error(exc, actor=actor, ctx=ctx_err)
            self._apply_error_decision(env, d1)
            return

        # 3) after_event
        ctx2 = _extract_ctx(event, actor=actor, stage="after_event")
        d2 = self._guard.after_event(event, actor=actor, ctx=ctx2)
        if d2.action == GuardAction.STOP:
            self._coord.stop()
            self.stop_background()
            return
        if d2.action == GuardAction.DROP:
            return
        if d2.action == GuardAction.RETRY:
            self._retry_or_drop(env, d2)
            return

    def _apply_error_decision(self, env: _Envelope, d: GuardDecision) -> None:
        if d.action == GuardAction.STOP:
            self._coord.stop()
            self.stop_background()
            return
        if d.action == GuardAction.DROP:
            return
        if d.action == GuardAction.RETRY:
            self._retry_or_drop(env, d)
            return
        # ALLOW：虽然发生异常但 guard 决定允许继续 → 重新提交事件进行处理
        if d.action == GuardAction.ALLOW:
            try:
                self._coord.emit(env.event, actor=env.actor)
            except BaseException:
                pass  # 二次失败不再重试
            return

    def _retry_or_drop(self, env: _Envelope, d: GuardDecision) -> None:
        if env.retries >= int(self._cfg.retry_limit):
            return
        # 等待一小会儿再重试（避免 tight loop）
        if d.retry_after_s is not None and d.retry_after_s > 0:
            time.sleep(float(d.retry_after_s))
        env.retries += 1
        env.received_at = time.time()
        # 重试不改变 actor
        try:
            self._inbox.put_nowait(env)
        except queue.Full:
            # inbox 满：按 drop_on_full 策略
            if not self._cfg.drop_on_full:
                self._inbox.put(env)

    # -------------------------
    # Background mode (optional)
    # -------------------------

    def start_background(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self.run_forever, name="engine-loop", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._running = False
        t = self._thread
        self._thread = None
        # 不强 join：daemon 线程允许自然退出

    def run_forever(self) -> None:
        """
        后台线程模式：一直处理 inbox。
        生产建议：更常见的是你自己写主循环（scheduler.poll + loop.drain）。
        """
        self._running = True
        while self._running and self._coord.phase.value != "stopped":
            n = self.drain(max_events=10_000)
            if n == 0:
                time.sleep(self._cfg.idle_sleep_s)

    # -------------------------
    # Runtime attach (IO -> inbox)
    # -------------------------

    def attach_runtime(self, runtime: Any) -> None:
        """
        让 runtime 回调只做 submit（不直接触发 dispatch），保证并发隔离。
        """
        self._runtime = runtime

        def _handler(ev: Any) -> None:
            self.submit(ev, actor="live")

        self._runtime_handler = _handler

        if hasattr(runtime, "subscribe") and callable(getattr(runtime, "subscribe")):
            runtime.subscribe(_handler)
        elif hasattr(runtime, "on") and callable(getattr(runtime, "on")):
            runtime.on(_handler)
        else:
            # runtime 不支持订阅：忽略
            self._runtime_handler = None

    def detach_runtime(self) -> None:
        runtime = self._runtime
        handler = self._runtime_handler
        self._runtime = None
        self._runtime_handler = None

        if runtime is None or handler is None:
            return

        if hasattr(runtime, "unsubscribe") and callable(getattr(runtime, "unsubscribe")):
            runtime.unsubscribe(handler)
        elif hasattr(runtime, "off") and callable(getattr(runtime, "off")):
            runtime.off(handler)
