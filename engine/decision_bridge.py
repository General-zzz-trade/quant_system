# engine/decision_bridge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Protocol

# engine 侧
from engine.pipeline import PipelineOutput

# dispatcher 侧
from engine.dispatcher import Route


# ============================================================
# Contracts (非常关键：只读、无副作用)
# ============================================================

class DecisionModule(Protocol):
    """
    Decision 模块契约（risk / portfolio / strategy 都遵守）

    设计铁律：
    - 只读 state / snapshot
    - 返回“意见”，不是行为
    - 不允许修改任何 state
    """

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """
        输入：StateSnapshot 或最小 snapshot dict
        输出：DecisionEvent / IntentEvent / SignalEvent 等“意见事件”
        """
        ...


# ============================================================
# Errors
# ============================================================

class DecisionBridgeError(RuntimeError):
    pass


# ============================================================
# Decision Bridge (frozen v1.0)
# ============================================================

@dataclass(slots=True)
class DecisionBridge:
    """
    DecisionBridge —— “state → 意见”的唯一合法出口（冻结版 v1.0）

    职责：
    - 接收 pipeline 产出的 snapshot
    - 调用 decision 模块
    - 将 decision 结果重新注入 dispatcher（作为新 event）

    冻结铁律：
    1) DecisionBridge 永远不修改 state
    2) DecisionBridge 永远不调用 execution
    3) DecisionBridge 只处理 snapshot，不处理原始 state
    """

    dispatcher_emit: Callable[[Any], None]

    # 可插拔 decision 模块（顺序即优先级）
    modules: List[DecisionModule]

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    def on_pipeline_output(self, out: PipelineOutput) -> None:
        """
        pipeline 输出钩子（由 core 在 snapshot 生成后调用）

        注意：
        - 没有 snapshot → 直接返回
        - snapshot 是 decision 的唯一输入
        """
        snapshot = out.snapshot
        if snapshot is None:
            return

        for module in self.modules:
            try:
                decisions = module.decide(snapshot)
            except Exception as e:
                raise DecisionBridgeError(f"Decision module failed: {module}") from e

            if not decisions:
                continue

            for ev in decisions:
                # 再次注入 dispatcher
                # 由 dispatcher 决定 Route.DECISION / Route.EXECUTION / DROP
                try:
                    self.dispatcher_emit(ev)
                except Exception as e:
                    raise DecisionBridgeError("Failed to emit decision event") from e

    # --------------------------------------------------------
    # Hot reload support
    # --------------------------------------------------------

    def swap_modules(self, new_modules: List[DecisionModule]) -> List[DecisionModule]:
        """Atomically replace decision modules. Returns the old modules."""
        old = self.modules
        self.modules = list(new_modules)
        return old
