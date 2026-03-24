# engine/guards.py
"""Engine guards — Rust-backed error classification and threshold logic.

BasicGuard delegates counting and threshold checks to RustBasicGuard,
keeping only Python-side exception classification (classify_exception).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from _quant_hotpath import RustBasicGuard, RustGuardConfig

from engine.errors import (
    ClassifiedError,
    EngineErrorContext,
    ErrorDomain,
    classify_exception,
)


class GuardAction(str, Enum):
    ALLOW = "allow"
    DROP = "drop"
    RETRY = "retry"
    STOP = "stop"


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
    stop_on_fatal: bool = True
    max_consecutive_errors: int = 5
    max_consecutive_domain_errors: int = 5
    max_consecutive_execution_errors: int = 2
    stop_on_unknown_exception: bool = True
    default_retry_after_s: float = 0.2


class BasicGuard:
    """Error guard — Python classifies exceptions, Rust handles counting/thresholds.

    Shadow counters (_consecutive_errors etc.) are maintained Python-side
    for observability and test assertions; the authoritative decision
    comes from RustBasicGuard.
    """

    def __init__(self, cfg: GuardConfig) -> None:
        self._cfg = cfg
        self._rust = RustBasicGuard(RustGuardConfig(
            max_consecutive_errors=cfg.max_consecutive_errors,
            max_consecutive_domain_errors=cfg.max_consecutive_domain_errors,
            max_consecutive_execution_errors=cfg.max_consecutive_execution_errors,
            stop_on_fatal=cfg.stop_on_fatal,
            stop_on_unknown_exception=cfg.stop_on_unknown_exception,
            default_retry_after_s=cfg.default_retry_after_s,
        ))
        # Shadow counters (for observability / tests)
        self._consecutive_errors: int = 0
        self._domain_consecutive: dict[ErrorDomain, int] = {}
        self._execution_consecutive: int = 0

    def before_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        action, reason = self._rust.before_event()
        return GuardDecision(action=GuardAction(action), reason=reason)

    def after_event(self, event: Any, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        action, reason = self._rust.after_event()
        self._consecutive_errors = 0
        self._domain_consecutive.clear()
        self._execution_consecutive = 0
        return GuardDecision(action=GuardAction(action), reason=reason)

    def on_error(self, exc: BaseException, *, actor: str, ctx: Optional[EngineErrorContext] = None) -> GuardDecision:
        classified = classify_exception(exc, ctx=ctx)
        # Shadow counters
        self._consecutive_errors += 1
        self._domain_consecutive[classified.domain] = self._domain_consecutive.get(classified.domain, 0) + 1
        if classified.domain == ErrorDomain.EXECUTION:
            self._execution_consecutive += 1

        action, reason, retry_after = self._rust.on_error(
            classified.severity.value,
            classified.domain.value,
            classified.code,
        )
        return GuardDecision(
            action=GuardAction(action),
            reason=reason,
            classified=classified,
            retry_after_s=retry_after,
        )


def build_basic_guard(cfg: Optional[GuardConfig] = None) -> BasicGuard:
    return BasicGuard(cfg or GuardConfig())
