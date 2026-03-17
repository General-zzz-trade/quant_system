# event/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from event.errors import EventError, EventFatalError


# ============================================================
# 决策常量（冻结）
# ============================================================

DECISION_DROP = "drop"
DECISION_RETRY = "retry"
DECISION_FAIL = "fail"
DECISION_RAISE = "raise"


# ============================================================
# 配置（冻结）
# ============================================================

@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """
    Policy 行为配置（制度级）

    说明：
    - 所有决策只返回字符串，不做副作用
    - Dispatcher / Runtime 根据返回值执行动作
    """
    max_retries: int = 1

    # 当发生“处理失败”（非致命）时的默认决策
    failure_decision: str = DECISION_RETRY

    # 达到最大重试次数后的决策
    max_retry_exceeded_decision: str = DECISION_FAIL

    # 未知异常（已被 Runtime 归类为失败）时的决策
    unknown_failure_decision: str = DECISION_FAIL


# ============================================================
# Policy
# ============================================================

class EventPolicy:
    """
    EventPolicy —— 纯决策模块（机构级）

    冻结约定：
    - Policy 只做“决策”，不直接操作 dispatcher / lifecycle / metrics
    - 不依赖具体异常类型（避免与 Dispatcher 实现耦合）
    - 不处理 queue full（Dispatcher 已制度化处理）
    """

    def __init__(self, cfg: Optional[PolicyConfig] = None) -> None:
        self._cfg = cfg or PolicyConfig()
        # event_id -> retry_count
        self._retry_counts: Dict[str, int] = {}

    # --------------------------------------------------------
    # 对外入口
    # --------------------------------------------------------

    def on_failure(self, *, event: Any, error: EventError) -> str:
        """
        处理失败时的统一决策入口

        规则：
        1) 致命错误：FAIL
        2) 非致命错误：按重试策略
        """
        # 1) 致命错误：立即失败
        if isinstance(error, EventFatalError):
            return DECISION_FAIL

        # 2) 非致命：进入重试/失败判定
        return self._decide_retry_or_fail(event)

    # --------------------------------------------------------
    # 内部决策
    # --------------------------------------------------------

    def _decide_retry_or_fail(self, event: Any) -> str:
        key = self._event_key(event)
        cnt = self._retry_counts.get(key, 0)

        if cnt < self._cfg.max_retries:
            self._retry_counts[key] = cnt + 1
            return self._cfg.failure_decision

        return self._cfg.max_retry_exceeded_decision

    # --------------------------------------------------------
    # 可选：成功/终态清理（P2 优化，保留接口）
    # --------------------------------------------------------

    def clear(self, *, event: Any) -> None:
        """
        在事件进入终态（DISPATCHED / DROPPED / FAILED）后调用，
        清理 retry 计数，避免长期累积。
        """
        key = self._event_key(event)
        self._retry_counts.pop(key, None)

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------

    def _event_key(self, event: Any) -> str:
        try:
            key = event.header.event_id
            return str(key)
        except Exception:
            return str(id(event))
