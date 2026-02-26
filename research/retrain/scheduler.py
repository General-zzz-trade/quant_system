"""Retrain scheduling based on time and performance degradation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEDULE_INTERVALS: dict[str, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
}


@dataclass(frozen=True, slots=True)
class RetrainConfig:
    """Configuration for retrain scheduling.

    Attributes:
        schedule: Time-based schedule (``daily``, ``weekly``, ``monthly``, ``on_degradation``).
        degradation_threshold: Absolute Sharpe drop that triggers retraining.
        min_training_samples: Minimum data points required for training.
        validation_split: Fraction of data reserved for validation.
    """
    schedule: str = "weekly"
    degradation_threshold: float = 0.5
    min_training_samples: int = 1000
    validation_split: float = 0.2


class RetrainTrigger:
    """Determines when model retraining should occur.

    Supports both time-based schedules (daily/weekly/monthly) and
    performance-based triggers (Sharpe degradation).

    Usage:
        trigger = RetrainTrigger(RetrainConfig(schedule="weekly"))
        if trigger.should_retrain(current_sharpe=0.8):
            run_retrain()
            trigger.record_retrain(sharpe=1.2)
    """

    def __init__(self, config: RetrainConfig) -> None:
        self._config = config
        self._last_retrain: Optional[datetime] = None
        self._last_sharpe: Optional[float] = None

    def should_retrain(
        self,
        *,
        current_sharpe: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if retraining should be triggered.

        Args:
            current_sharpe: Current live Sharpe ratio (for degradation check).
            now: Current timestamp. Defaults to UTC now.

        Returns:
            True if retraining should occur.
        """
        now = now or datetime.now(timezone.utc)

        if self._config.schedule == "on_degradation":
            return self._check_degradation(current_sharpe)

        if self._check_time_elapsed(now):
            return True

        return self._check_degradation(current_sharpe)

    def _check_time_elapsed(self, now: datetime) -> bool:
        if self._last_retrain is None:
            return True

        interval = _SCHEDULE_INTERVALS.get(self._config.schedule)
        if interval is None:
            logger.warning("Unknown schedule %r", self._config.schedule)
            return False

        elapsed = now - self._last_retrain
        if elapsed >= interval:
            logger.info(
                "Time-based retrain triggered: %s elapsed (schedule=%s)",
                elapsed, self._config.schedule,
            )
            return True
        return False

    def _check_degradation(self, current_sharpe: Optional[float]) -> bool:
        if current_sharpe is None or self._last_sharpe is None:
            return False

        drop = self._last_sharpe - current_sharpe
        if drop >= self._config.degradation_threshold:
            logger.info(
                "Degradation retrain triggered: Sharpe dropped %.3f (%.3f -> %.3f, threshold=%.3f)",
                drop, self._last_sharpe, current_sharpe, self._config.degradation_threshold,
            )
            return True
        return False

    def record_retrain(self, sharpe: float) -> None:
        """Record that a retrain occurred with the given resulting Sharpe."""
        self._last_retrain = datetime.now(timezone.utc)
        self._last_sharpe = sharpe

    @property
    def last_retrain(self) -> Optional[datetime]:
        return self._last_retrain

    @property
    def last_sharpe(self) -> Optional[float]:
        return self._last_sharpe

    @property
    def config(self) -> RetrainConfig:
        return self._config
