# execution/ingress/submitter.py
"""Order submitter — converts ExecutionIntents into bridge commands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from execution.bridge.execution_bridge import Ack, ExecutionBridge
from execution.models.intents import ExecutionIntent, IntentStatus


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """提交结果。"""
    intent_id: str
    success: bool
    ack: Optional[Ack] = None
    error: Optional[str] = None


class OrderSubmitter:
    """
    订单提交器 — 将 ExecutionIntent 转化为命令并提交到 ExecutionBridge。

    职责：
    1. 从 intent 构建 SubmitOrderCommand
    2. 通过 bridge 提交
    3. 返回 SubmitResult
    """

    def __init__(self, bridge: ExecutionBridge) -> None:
        self._bridge = bridge

    def submit_intent(self, intent: ExecutionIntent, *, actor: str = "submitter") -> SubmitResult:
        """提交一个执行意图。"""
        if intent.is_terminal:
            return SubmitResult(
                intent_id=intent.intent_id,
                success=False,
                error=f"intent already terminal: {intent.status.value}",
            )

        try:
            # 构建简单的 command 对象
            cmd = _IntentCommand(
                command_id=f"cmd-{intent.intent_id}",
                venue=intent.venue,
                symbol=intent.symbol,
                idempotency_key=f"idem-{intent.intent_id}",
                side=intent.side,
                qty=intent.qty,
                price=intent.price,
                order_type=intent.order_type,
            )
            ack = self._bridge.submit(cmd)
            return SubmitResult(
                intent_id=intent.intent_id,
                success=ack.ok,
                ack=ack,
                error=ack.error if not ack.ok else None,
            )
        except Exception as e:
            return SubmitResult(
                intent_id=intent.intent_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )


@dataclass(frozen=True, slots=True)
class _IntentCommand:
    """轻量级 command 对象，用于将 intent 提交到 bridge。"""
    command_id: str
    venue: str
    symbol: str
    idempotency_key: str
    side: str
    qty: Any
    price: Any
    order_type: str = "limit"
