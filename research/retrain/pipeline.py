"""End-to-end retraining pipeline: train, validate, register, promote/reject."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from research.retrain.scheduler import RetrainTrigger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrainResult:
    """Outcome of a retraining run."""
    model_id: str
    old_sharpe: Optional[float]
    new_sharpe: float
    promoted: bool
    reason: str


class RetrainPipeline:
    """End-to-end retraining: data -> train -> validate -> register -> promote/reject.

    The pipeline delegates actual training to a user-supplied ``train_fn``
    that accepts parameters and returns a metrics dict. Registration and
    promotion are handled via an optional ``ModelRegistry``.

    Usage:
        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": run_backtest(params)},
            registry=registry,
            trigger=trigger,
        )
        result = pipeline.run(params={"window": 20}, features=["sma_20"])
    """

    def __init__(
        self,
        *,
        train_fn: Callable[[dict[str, Any]], dict[str, float]],
        registry: Optional[Any] = None,
        trigger: Optional[RetrainTrigger] = None,
        model_name: str = "default",
        promotion_metric: str = "sharpe",
    ) -> None:
        self._train_fn = train_fn
        self._registry = registry
        self._trigger = trigger
        self._model_name = model_name
        self._promotion_metric = promotion_metric

    def run(
        self,
        *,
        params: dict[str, Any],
        features: Sequence[str],
    ) -> RetrainResult:
        """Execute retraining pipeline.

        Steps:
            1. Train model with params (via ``train_fn``).
            2. Extract the promotion metric from training results.
            3. Compare with current production model (if registry available).
            4. If better: register and promote.
            5. If worse: register but don't promote.
        """
        metrics = self._train_fn(params)
        new_sharpe = metrics.get(self._promotion_metric, 0.0)

        old_sharpe = self._get_production_metric()
        model_id = ""

        if self._registry is not None:
            mv = self._registry.register(
                name=self._model_name,
                params=params,
                features=features,
                metrics=metrics,
            )
            model_id = mv.model_id

        should_promote = self._should_promote(old_sharpe, new_sharpe)

        if should_promote and self._registry is not None and model_id:
            self._registry.promote(model_id)
            reason = f"Promoted: new {self._promotion_metric}={new_sharpe:.4f} > old={old_sharpe}"
            logger.info(reason)
        elif should_promote:
            reason = f"Would promote (no registry): new {self._promotion_metric}={new_sharpe:.4f}"
        else:
            reason = (
                f"Not promoted: new {self._promotion_metric}={new_sharpe:.4f} "
                f"<= old={old_sharpe}"
            )
            logger.info(reason)

        if self._trigger is not None:
            self._trigger.record_retrain(new_sharpe)

        return RetrainResult(
            model_id=model_id,
            old_sharpe=old_sharpe,
            new_sharpe=new_sharpe,
            promoted=should_promote,
            reason=reason,
        )

    def _get_production_metric(self) -> Optional[float]:
        if self._registry is None:
            return None
        prod = self._registry.get_production(self._model_name)
        if prod is None:
            return None
        return prod.metrics.get(self._promotion_metric)

    def _should_promote(
        self,
        old_sharpe: Optional[float],
        new_sharpe: float,
    ) -> bool:
        if old_sharpe is None:
            return True
        return new_sharpe > old_sharpe
