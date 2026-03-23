"""Pipeline interceptor chain — risk and validation as built-in pipeline stages.

Migrated from core/interceptors.py.

Interceptors run *before* and *after* state reduction.  They cannot be
bypassed — if you go through the pipeline, you go through the interceptors.

This replaces the loose coupling between Risk and Engine with a mandatory
interception chain (Interceptor Pattern / Chain of Responsibility).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, FrozenSet, Optional, Protocol, Sequence, Tuple

from event.core_types import Envelope


# -- Intercept result -----------------------------------------

class InterceptAction(Enum):
    CONTINUE = auto()    # proceed to next interceptor / reducer
    REJECT = auto()      # reject this event, don't reduce
    KILL = auto()        # trigger kill-switch, halt pipeline


@dataclass(frozen=True, slots=True)
class InterceptResult:
    """Outcome of a single interceptor evaluation."""
    action: InterceptAction
    interceptor: str          # name of the interceptor that produced this
    reason: str = ""
    adjustment: Optional[Any] = None  # e.g., reduced qty

    @classmethod
    def ok(cls, interceptor: str) -> InterceptResult:
        return cls(action=InterceptAction.CONTINUE, interceptor=interceptor)

    @classmethod
    def reject(cls, interceptor: str, reason: str) -> InterceptResult:
        return cls(action=InterceptAction.REJECT, interceptor=interceptor, reason=reason)

    @classmethod
    def kill(cls, interceptor: str, reason: str) -> InterceptResult:
        return cls(action=InterceptAction.KILL, interceptor=interceptor, reason=reason)


# -- Interceptor protocol ------------------------------------

class PipelineInterceptor(Protocol):
    """Protocol for pipeline interceptors.

    Implementations may be stateless (pure validation) or stateful
    (risk aggregation, rate limiting).
    """

    @property
    def name(self) -> str:
        """Unique name for logging and diagnostics."""
        ...

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        """Called *before* reducers run.  Return REJECT to skip reduction."""
        ...

    def after_reduce(
        self,
        envelope: Envelope[Any],
        old_state: Any,
        new_state: Any,
    ) -> InterceptResult:
        """Called *after* reducers run.  Return KILL to trigger emergency stop."""
        ...


# -- Interceptor chain ----------------------------------------

class InterceptorChain:
    """Ordered sequence of interceptors executed for every pipeline event.

    Evaluation stops on the first non-CONTINUE result (fail-fast).
    """

    def __init__(self, interceptors: Sequence[PipelineInterceptor] = ()) -> None:
        self._interceptors: Tuple[PipelineInterceptor, ...] = tuple(interceptors)

    @property
    def interceptors(self) -> Tuple[PipelineInterceptor, ...]:
        return self._interceptors

    def run_before(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        """Run all interceptors' ``before_reduce``.  Fail-fast on non-CONTINUE."""
        for ic in self._interceptors:
            result = ic.before_reduce(envelope, state)
            if result.action != InterceptAction.CONTINUE:
                return result
        return InterceptResult.ok("chain")

    def run_after(
        self,
        envelope: Envelope[Any],
        old_state: Any,
        new_state: Any,
    ) -> InterceptResult:
        """Run all interceptors' ``after_reduce``.  Fail-fast on KILL."""
        for ic in self._interceptors:
            result = ic.after_reduce(envelope, old_state, new_state)
            if result.action == InterceptAction.KILL:
                return result
        return InterceptResult.ok("chain")


# -- Built-in interceptors ------------------------------------

class PassthroughInterceptor:
    """No-op interceptor — useful as a placeholder or for testing."""

    @property
    def name(self) -> str:
        return "passthrough"

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        return InterceptResult.ok(self.name)

    def after_reduce(self, envelope: Envelope[Any], old_state: Any, new_state: Any) -> InterceptResult:
        return InterceptResult.ok(self.name)


class EventKindGate:
    """Rejects events of disallowed kinds — e.g., block ORDER when kill-switch active."""

    def __init__(self, *, blocked_kinds: FrozenSet[Any], reason: str = "blocked by gate") -> None:
        self._blocked = blocked_kinds
        self._reason = reason

    @property
    def name(self) -> str:
        return "event_kind_gate"

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        if envelope.kind in self._blocked:
            return InterceptResult.reject(self.name, self._reason)
        return InterceptResult.ok(self.name)

    def after_reduce(self, envelope: Envelope[Any], old_state: Any, new_state: Any) -> InterceptResult:
        return InterceptResult.ok(self.name)
