from __future__ import annotations

from typing import Any, Protocol


class EventSink(Protocol):
    """Minimal sink contract required by execution ingress routers.

    The main project can provide an EngineCoordinator that implements this interface.
    """

    def emit(self, event: Any, *, actor: str) -> None: ...
