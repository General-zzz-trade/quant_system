# engine/execution_bridge.py
"""ExecutionBridge -- simple adapter-to-dispatcher bridge for the engine layer.

This is the production default bridge used by EngineCoordinator.  It delegates
directly to the adapter (send_order) and re-emits results to the dispatcher.

For advanced features (retry, circuit breaker, rate limiting, ack store),
see execution/bridge/execution_bridge.py which is used by specific adapters.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

_log = logging.getLogger(__name__)


# ============================================================
# Contracts（执行层制度，非常关键）
# ============================================================

class ExecutionAdapter(Protocol):
    """
    ExecutionAdapter：交易所 / Broker 适配器契约

    设计铁律：
    - 只接收"订单类事件"
    - 不修改 state
    - 不产生 snapshot
    - 成交结果必须以 Event 形式返回（FillEvent）
    """

    def send_order(self, order_event: Any) -> Iterable[Any]:
        """
        输入：OrderEvent
        输出：0~N 个 Event（通常是 FillEvent / RejectEvent）
        """
        ...


# ============================================================
# Errors
# ============================================================

class ExecutionBridgeError(RuntimeError):
    pass


# ============================================================
# Execution Bridge（冻结版 v1.0）
# ============================================================

@dataclass(slots=True)
class ExecutionBridge:
    """
    ExecutionBridge —— "意见 → 现实"的唯一合法出口（冻结版 v1.0）

    职责：
    - 接收 dispatcher 路由来的 EXECUTION 事件（通常是 OrderEvent）
    - 调用 execution adapter（真实 or mock）
    - 将执行结果（Fill / Reject）重新注入 dispatcher

    冻结铁律：
    1) ExecutionBridge 永远不修改 state
    2) ExecutionBridge 永远不生成 snapshot
    3) ExecutionBridge 永远不绕过 dispatcher

    Retry + circuit breaker：
    - max_retries: 单次 handle_event 最多重试次数（默认 2）
    - retry_base_delay: 指数退避基础延迟，单位秒（默认 0.5s）
    - cb_failure_threshold: 累计失败次数触发熔断（默认 5）
    - cb_cooldown_seconds: 熔断冷却时间，超时后自动半开（默认 30s）
    """

    adapter: ExecutionAdapter
    dispatcher_emit: Callable[[Any], None]
    risk_gate: Optional[Any] = None
    # Retry parameters
    max_retries: int = 2
    retry_base_delay: float = 0.5
    # Circuit breaker parameters
    cb_failure_threshold: int = 5
    cb_cooldown_seconds: float = 30.0
    # Mutable state — not part of __init__ signature
    _failure_count: int = field(default=0, init=False, repr=False)
    _circuit_opened_at: Optional[float] = field(default=None, init=False, repr=False)

    # --------------------------------------------------------
    # Circuit breaker property
    # --------------------------------------------------------

    @property
    def circuit_open(self) -> bool:
        if self._circuit_opened_at is None:
            return False
        if time.monotonic() - self._circuit_opened_at > self.cb_cooldown_seconds:
            # Cooldown elapsed — half-open: allow next attempt
            return False
        return True

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    def handle_event(self, event: Any) -> None:
        """
        dispatcher 路由到 Route.EXECUTION 时的 handler
        """
        # Second defense: RiskGate check at execution boundary
        if self.risk_gate is not None:
            check = self.risk_gate.check(event)
            if not check.allowed:
                _log.warning("RiskGate (execution bridge) REJECTED: %s", check.reason)
                return

        # Circuit breaker: fast-fail when open
        if self.circuit_open:
            raise ExecutionBridgeError(
                "Circuit breaker open — execution adapter requests are suspended"
            )

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                results = self.adapter.send_order(event)
                # Success — reset failure counter (half-open recovery)
                self._failure_count = 0
                self._circuit_opened_at = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._failure_count += 1
                _log.warning(
                    "ExecutionBridge adapter error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if self._failure_count >= self.cb_failure_threshold:
                    self._circuit_opened_at = time.monotonic()
                    _log.error(
                        "ExecutionBridge circuit breaker OPENED after %d failures",
                        self._failure_count,
                    )
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
        else:
            # All attempts exhausted
            raise ExecutionBridgeError("Execution adapter failed") from last_exc

        if not results:
            return

        for ev in results:
            try:
                # 重新注入 dispatcher：
                # - FillEvent → PIPELINE
                # - RejectEvent → DECISION or DROP
                self.dispatcher_emit(ev)
            except Exception as e:
                raise ExecutionBridgeError("Failed to emit execution result") from e
