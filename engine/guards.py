# engine/guards.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from engine.errors import (
    ClassifiedError,
    EngineErrorContext,
    ErrorSeverity,
    ErrorDomain,
    classify_exception,
)


class GuardAction(str, Enum):
    ALLOW = "allow"          # 放行（继续处理）
    DROP = "drop"            # 丢弃当前 event（不中断 engine）
    RETRY = "retry"          # 建议重试（由上层 loop 执行）
    STOP = "stop"            # 停止 engine（kill-switch）


@dataclass(frozen=True, slots=True)
class GuardDecision:
    action: GuardAction
    reason: str
    classified: Optional[ClassifiedError] = None
    retry_after_s: Optional[float] = None


class Guard(Protocol):
    def before_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext]) -> GuardDecision: ...
    def on_error(self, exc: BaseException, *, actor: str, ctx: Optional[EngineErrorContext]) -> GuardDecision: ...
    def after_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext]) -> GuardDecision: ...


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """
    v1.0 冻结配置：只做最基本的“止损开关”与“错误阈值”。
    metrics/tracing 后续接入时可以增强。
    """
    # 任何 INVARIANT/FATAL 直接 STOP
    stop_on_fatal: bool = True

    # 连续错误上限（达到后 STOP）
    max_consecutive_errors: int = 5

    # 同 domain 的连续错误上限
    max_consecutive_domain_errors: int = 5

    # execution 错误更敏感（比如连续 2 次就停）
    max_consecutive_execution_errors: int = 2

    # 对“未知异常”是否直接停机（建议 True：避免 silent corruption）
    stop_on_unknown_exception: bool = True

    # 可选：遇到 Retryable 建议等待
    default_retry_after_s: float = 0.2


class BasicGuard:
    """
    机构级最低配 Guard（冻结版 v1.0）：
    - before_event：可扩展（目前默认放行）
    - on_error：统一 classify，基于阈值与 severity 决定 STOP/RETRY/DROP
    - after_event：可扩展（目前默认放行）

    注意：Guard 不负责真正 stop/retry，Guard 只给决策。
    stop/retry 的执行点应在 coordinator 主循环（后续会把 coordinator 引入 guard）
    """

    def __init__(self, cfg: GuardConfig) -> None:
        self._cfg = cfg
        self._consecutive_errors: int = 0
        self._domain_consecutive: Dict[ErrorDomain, int] = {}
        self._execution_consecutive: int = 0

    def before_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        return GuardDecision(action=GuardAction.ALLOW, reason="ok")

    def after_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        # 事件成功处理后：清空连续错误计数（机构级默认策略）
        self._consecutive_errors = 0
        self._domain_consecutive.clear()
        self._execution_consecutive = 0
        return GuardDecision(action=GuardAction.ALLOW, reason="ok")

    def on_error(self, exc: BaseException, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        classified = classify_exception(exc, ctx=ctx)

        # 更新计数
        self._consecutive_errors += 1
        self._domain_consecutive[classified.domain] = self._domain_consecutive.get(classified.domain, 0) + 1
        if classified.domain == ErrorDomain.EXECUTION:
            self._execution_consecutive += 1

        # 1) FATAL / INVARIANT：直接 STOP
        if classified.severity == ErrorSeverity.FATAL:
            if self._cfg.stop_on_fatal:
                return GuardDecision(
                    action=GuardAction.STOP,
                    reason=f"fatal: {classified.domain}/{classified.code}",
                    classified=classified,
                )

        if classified.domain == ErrorDomain.INVARIANT:
            return GuardDecision(
                action=GuardAction.STOP,
                reason=f"invariant: {classified.code}",
                classified=classified,
            )

        # 2) execution 错误阈值更严格
        if classified.domain == ErrorDomain.EXECUTION and self._execution_consecutive >= self._cfg.max_consecutive_execution_errors:
            return GuardDecision(
                action=GuardAction.STOP,
                reason=f"execution errors >= {self._cfg.max_consecutive_execution_errors}",
                classified=classified,
            )

        # 3) domain 连续错误阈值
        dom_n = self._domain_consecutive.get(classified.domain, 0)
        if dom_n >= self._cfg.max_consecutive_domain_errors:
            return GuardDecision(
                action=GuardAction.STOP,
                reason=f"{classified.domain} errors >= {self._cfg.max_consecutive_domain_errors}",
                classified=classified,
            )

        # 4) 全局连续错误阈值
        if self._consecutive_errors >= self._cfg.max_consecutive_errors:
            return GuardDecision(
                action=GuardAction.STOP,
                reason=f"consecutive errors >= {self._cfg.max_consecutive_errors}",
                classified=classified,
            )

        # 5) Retryable / timeout：建议 RETRY
        if classified.code in ("RETRYABLE", "TIMEOUT"):
            return GuardDecision(
                action=GuardAction.RETRY,
                reason=f"retry: {classified.domain}/{classified.code}",
                classified=classified,
                retry_after_s=self._cfg.default_retry_after_s,
            )

        # 6) 未知异常（第三方）：可选直接 STOP
        if self._cfg.stop_on_unknown_exception and not isinstance(exc, Exception):
            return GuardDecision(
                action=GuardAction.STOP,
                reason="unknown baseexception",
                classified=classified,
            )

        # 默认：DROP 当前 event（保守，防止污染）
        return GuardDecision(
            action=GuardAction.DROP,
            reason=f"drop on error: {classified.domain}/{classified.code}",
            classified=classified,
        )


def build_basic_guard(cfg: Optional[GuardConfig] = None) -> BasicGuard:
    return BasicGuard(cfg or GuardConfig())
