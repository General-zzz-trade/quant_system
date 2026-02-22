# event/trace.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from event.types import BaseEvent


class EventTracer(Protocol):
    """Tracing hook for the event system.

    Runtime will call these hooks opportunistically.
    Implementations must be non-throwing.
    """

    def on_emit(self, event: BaseEvent) -> None:
        ...

    def on_error(self, error: Exception, *, context: Optional[Any] = None) -> None:
        ...

    def on_reject(self, event: BaseEvent, reason: str) -> None:
        ...


@dataclass(slots=True)
class NoopTracer:
    def on_emit(self, event: BaseEvent) -> None:
        return None

    def on_error(self, error: Exception, *, context: Optional[Any] = None) -> None:
        return None

    def on_reject(self, event: BaseEvent, reason: str) -> None:
        return None


@dataclass(slots=True)
class MemoryTracer:
    """In-memory tracer for debugging / tests."""

    emitted: List[BaseEvent] = field(default_factory=list)
    rejected: List[tuple[BaseEvent, str]] = field(default_factory=list)
    errors: List[tuple[Exception, Optional[Any]]] = field(default_factory=list)

    def on_emit(self, event: BaseEvent) -> None:
        self.emitted.append(event)

    def on_error(self, error: Exception, *, context: Optional[Any] = None) -> None:
        self.errors.append((error, context))

    def on_reject(self, event: BaseEvent, reason: str) -> None:
        self.rejected.append((event, reason))
