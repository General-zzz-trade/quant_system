# alpha/inference
"""Alpha model inference — running predictions in production."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from alpha.base import AlphaModel, Signal


@dataclass
class InferenceResult:
    """单次推理结果。"""
    model_name: str
    symbol: str
    signal: Optional[Signal]
    latency_ms: float = 0.0
    error: Optional[str] = None


class InferenceEngine:
    """批量运行多个 alpha 模型的推理引擎。"""

    def __init__(self, models: Sequence[AlphaModel] | None = None) -> None:
        self._models: list[AlphaModel] = list(models or [])

    def add_model(self, model: AlphaModel) -> None:
        self._models.append(model)

    def run(
        self,
        symbol: str,
        ts: datetime,
        features: Dict[str, Any],
    ) -> list[InferenceResult]:
        results = []
        for model in self._models:
            import time
            t0 = time.monotonic()
            try:
                sig = model.predict(symbol=symbol, ts=ts, features=features)
                elapsed = (time.monotonic() - t0) * 1000
                results.append(InferenceResult(
                    model_name=model.name,
                    symbol=symbol,
                    signal=sig,
                    latency_ms=elapsed,
                ))
            except Exception as exc:
                elapsed = (time.monotonic() - t0) * 1000
                results.append(InferenceResult(
                    model_name=model.name,
                    symbol=symbol,
                    signal=None,
                    latency_ms=elapsed,
                    error=str(exc),
                ))
        return results
