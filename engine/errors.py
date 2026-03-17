# engine/errors.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ErrorSeverity(str, Enum):
    INFO = "info"          # 记录即可
    WARNING = "warning"    # 记录 + 可降级
    ERROR = "error"        # 记录 + 重试/跳过
    FATAL = "fatal"        # 必须停止（kill-switch）


class ErrorDomain(str, Enum):
    ENGINE = "engine"
    PIPELINE = "pipeline"
    DISPATCHER = "dispatcher"
    DECISION = "decision"
    EXECUTION = "execution"
    IO = "io"
    RISK = "risk"
    DATA = "data"
    INVARIANT = "invariant"


@dataclass(frozen=True, slots=True)
class EngineErrorContext:
    """
    错误上下文（用于审计/追踪/metrics）。
    v1.0 只放“最小可用字段”，不要耦合具体 event 类型。
    """
    ts: Any = None
    actor: str = "unknown"
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    symbol: Optional[str] = None
    stage: Optional[str] = None     # pipeline/decision/execution/...
    details: Optional[dict] = None  # 额外结构化信息


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    """
    将任意异常“定性”后的结构化结果。
    guards/metrics 只应该依赖这个对象，而不是依赖异常文本。
    """
    severity: ErrorSeverity
    domain: ErrorDomain
    code: str
    message: str
    ctx: EngineErrorContext
    cause: Optional[BaseException] = None


# ------------------------------
# Base classes
# ------------------------------

class EngineException(RuntimeError):
    """
    engine 内部显式抛出的异常基类（用于区分第三方异常）。
    """
    domain: ErrorDomain = ErrorDomain.ENGINE
    code: str = "ENGINE_ERROR"
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __init__(self, message: str = "", *, ctx: Optional[EngineErrorContext] = None,
        cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.ctx = ctx or EngineErrorContext()
        self.cause = cause


class RecoverableError(EngineException):
    severity = ErrorSeverity.ERROR
    code = "RECOVERABLE"


class RetryableError(EngineException):
    severity = ErrorSeverity.ERROR
    code = "RETRYABLE"


class FatalError(EngineException):
    severity = ErrorSeverity.FATAL
    code = "FATAL"


class InvariantViolation(EngineException):
    domain = ErrorDomain.INVARIANT
    severity = ErrorSeverity.FATAL
    code = "INVARIANT_VIOLATION"


class ExecutionError(EngineException):
    domain = ErrorDomain.EXECUTION
    code = "EXECUTION_ERROR"
    severity = ErrorSeverity.ERROR


class DataError(EngineException):
    domain = ErrorDomain.DATA
    code = "DATA_ERROR"
    severity = ErrorSeverity.ERROR


# ------------------------------
# Classifier (policy-free)
# ------------------------------

def classify_exception(
    exc: BaseException,
    *,
    default_domain: ErrorDomain = ErrorDomain.ENGINE,
    default_severity: ErrorSeverity = ErrorSeverity.ERROR,
    ctx: Optional[EngineErrorContext] = None,
) -> ClassifiedError:
    """
    将异常映射为 ClassifiedError。
    原则：policy-free（不在这里决定重试/停机，只定性）。
    """
    c = ctx or EngineErrorContext()

    # 1) engine 自己抛的异常（最可信）
    if isinstance(exc, EngineException):
        return ClassifiedError(
            severity=getattr(exc, "severity", default_severity),
            domain=getattr(exc, "domain", default_domain),
            code=getattr(exc, "code", "ENGINE_EXCEPTION"),
            message=str(exc) or getattr(exc, "code", "ENGINE_EXCEPTION"),
            ctx=getattr(exc, "ctx", c) or c,
            cause=getattr(exc, "cause", None),
        )

    # 2) 常见 Python 异常：分类为 DATA/IO 等（保守）
    if isinstance(exc, (ValueError, TypeError)):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR,
            domain=ErrorDomain.DATA,
            code=exc.__class__.__name__,
            message=str(exc) or exc.__class__.__name__,
            ctx=c,
            cause=exc,
        )

    if isinstance(exc, TimeoutError):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR,
            domain=ErrorDomain.IO,
            code="TIMEOUT",
            message=str(exc) or "TimeoutError",
            ctx=c,
            cause=exc,
        )

    # Network/IO errors (must be after TimeoutError check since TimeoutError is subclass of OSError)
    if isinstance(exc, (ConnectionError, OSError)):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR, domain=ErrorDomain.IO,
            code=exc.__class__.__name__, message=str(exc) or exc.__class__.__name__,
            ctx=c, cause=exc,
        )

    # Data lookup errors
    if isinstance(exc, (KeyError, IndexError, AttributeError)):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR, domain=ErrorDomain.DATA,
            code=exc.__class__.__name__, message=str(exc) or exc.__class__.__name__,
            ctx=c, cause=exc,
        )

    # 3) 默认：ENGINE ERROR
    return ClassifiedError(
        severity=default_severity,
        domain=default_domain,
        code=exc.__class__.__name__,
        message=str(exc) or exc.__class__.__name__,
        ctx=c,
        cause=exc,
    )
