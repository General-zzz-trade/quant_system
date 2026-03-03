# engine/execution_bridge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol


# ============================================================
# Contracts（执行层制度，非常关键）
# ============================================================

class ExecutionAdapter(Protocol):
    """
    ExecutionAdapter：交易所 / Broker 适配器契约

    设计铁律：
    - 只接收“订单类事件”
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
    ExecutionBridge —— “意见 → 现实”的唯一合法出口（冻结版 v1.0）

    职责：
    - 接收 dispatcher 路由来的 EXECUTION 事件（通常是 OrderEvent）
    - 调用 execution adapter（真实 or mock）
    - 将执行结果（Fill / Reject）重新注入 dispatcher

    冻结铁律：
    1) ExecutionBridge 永远不修改 state
    2) ExecutionBridge 永远不生成 snapshot
    3) ExecutionBridge 永远不绕过 dispatcher
    """

    adapter: ExecutionAdapter
    dispatcher_emit: Callable[[Any], None]
    risk_gate: Optional[Any] = None

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
                import logging
                logging.getLogger(__name__).warning(
                    "RiskGate (execution bridge) REJECTED: %s", check.reason)
                return

        try:
            results = self.adapter.send_order(event)
        except Exception as e:
            raise ExecutionBridgeError("Execution adapter failed") from e

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
