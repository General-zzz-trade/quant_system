"""Collector protocol for market data ingestion."""
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Collector(Protocol):
    """Protocol for data collectors that ingest market data."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def last_active_ts(self) -> Optional[float]: ...
