"""Metrics registry — simple counters for runners and pipeline stages.

Provides a standalone ``Metrics`` class for runners and monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Metrics:
    """Simple metrics container for runners and standalone use.

    Simple metrics container with increment/gauge/timing operations.
    """

    values: Dict[str, Any] = field(default_factory=dict)
    _delegate: Optional[Any] = field(default=None, repr=False)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value
        if self._delegate is not None:
            self._delegate.gauge(key, float(value))

    def inc(self, key: str, n: int = 1) -> None:
        self.values[key] = int(self.values.get(key, 0)) + int(n)
        if self._delegate is not None:
            self._delegate.counter(key, n)

    def get(self, key: str, default: Any = 0) -> Any:
        return self.values.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.values)

    def reset(self) -> None:
        self.values.clear()


def create_metrics(effects: Optional[Any] = None) -> Metrics:
    """Create a Metrics instance, optionally backed by core Effects.

    Parameters
    ----------
    effects : Effects, optional
        If provided, metrics are also forwarded to ``effects.metrics``.
    """
    delegate = getattr(effects, "metrics", None) if effects is not None else None
    return Metrics(_delegate=delegate)
