"""Feature pipeline — chainable transform → normalize → store workflow.

Usage:
    pipeline = FeaturePipeline(store=feature_store)
    pipeline.add_step("sma_20", lambda bars: sma([b.close for b in bars], 20))
    pipeline.add_step("rsi_14", lambda bars: rsi([b.close for b in bars], 14))
    pipeline.run(bars, symbol="BTCUSDT", timeframe="1h")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from features.store import FeatureStore
from features.types import Bar, FeatureName, FeatureSeries

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Single feature computation step."""
    name: FeatureName
    compute_fn: Callable[[Sequence[Bar]], FeatureSeries]
    normalize_fn: Optional[Callable[[Sequence[float]], list[float]]] = None


class FeaturePipeline:
    """Chainable feature computation pipeline."""

    def __init__(self, store: Optional[FeatureStore] = None) -> None:
        self._steps: List[PipelineStep] = []
        self._store = store

    def add_step(
        self,
        name: str,
        compute_fn: Callable[[Sequence[Bar]], FeatureSeries],
        normalize_fn: Optional[Callable[[Sequence[float]], list[float]]] = None,
    ) -> "FeaturePipeline":
        """Add a computation step. Returns self for chaining."""
        self._steps.append(PipelineStep(
            name=name,
            compute_fn=compute_fn,
            normalize_fn=normalize_fn,
        ))
        return self

    def run(
        self,
        bars: Sequence[Bar],
        *,
        symbol: str,
        timeframe: str = "1h",
    ) -> Dict[str, FeatureSeries]:
        """Execute all pipeline steps and optionally store results."""
        results: Dict[str, FeatureSeries] = {}

        for step in self._steps:
            try:
                raw = step.compute_fn(bars)
            except Exception as e:
                logger.warning("Step %s failed: %s", step.name, e)
                continue

            if step.normalize_fn is not None:
                valid = [v for v in raw if v is not None]
                if valid:
                    normalized = step.normalize_fn(valid)
                    # Map back, preserving None positions
                    norm_iter = iter(normalized)
                    raw = [next(norm_iter) if v is not None else None for v in raw]

            results[step.name] = raw

            if self._store is not None:
                self._store.put(
                    symbol=symbol,
                    timeframe=timeframe,
                    name=step.name,
                    series=raw,
                )

            logger.debug("Computed %s: %d values", step.name, len(raw))

        return results

    def clear(self) -> None:
        """Remove all steps."""
        self._steps.clear()

    @property
    def step_names(self) -> List[str]:
        return [s.name for s in self._steps]
