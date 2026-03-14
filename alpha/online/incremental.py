"""Incremental online learning — update models with new data without full retraining.

Supports SGD-based partial fitting for sklearn-compatible models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Protocol

logger = logging.getLogger(__name__)


class PartialFitModel(Protocol):
    """Protocol for models supporting incremental updates."""
    def partial_fit(self, X: Any, y: Any, **kwargs: Any) -> None: ...
    def predict(self, X: Any) -> Any: ...


@dataclass
class IncrementalUpdater:
    """Manages incremental model updates with a buffer of recent observations.

    Accumulates data and triggers partial_fit when buffer is full.
    """

    model: Any
    buffer_size: int = 100
    _X_buffer: List[List[float]] = field(default_factory=list)
    _y_buffer: List[float] = field(default_factory=list)
    _update_count: int = 0

    def add_observation(self, features: List[float], target: float) -> bool:
        """Add a single observation. Returns True if an update was triggered."""
        self._X_buffer.append(features)
        self._y_buffer.append(target)

        if len(self._X_buffer) >= self.buffer_size:
            self._do_update()
            return True
        return False

    def _do_update(self) -> None:
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy required for incremental updates")
            self._X_buffer.clear()
            self._y_buffer.clear()
            return

        X = np.array(self._X_buffer)
        y = np.array(self._y_buffer)

        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        elif hasattr(self.model, "fit"):
            self.model.fit(X, y)

        self._update_count += 1
        logger.info("Incremental update #%d: %d samples", self._update_count, len(X))

        self._X_buffer.clear()
        self._y_buffer.clear()

    def force_update(self) -> None:
        """Force an update with whatever is in the buffer."""
        if self._X_buffer:
            self._do_update()

    @property
    def update_count(self) -> int:
        return self._update_count
