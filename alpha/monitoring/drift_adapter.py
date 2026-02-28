# alpha/monitoring/drift_adapter.py
"""Concept drift adapter — monitors rolling model performance and recommends actions."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class DriftState:
    is_drifting: bool
    severity: str  # "none", "warning", "critical"
    metrics: Dict[str, float]
    recommendation: str  # "continue", "reduce_size", "pause", "retrain"


class ConceptDriftAdapter:
    """Monitors model performance and adapts to concept drift.

    Tracks rolling hit rate, information coefficient (IC), and Sharpe ratio.
    Compares rolling window against baseline to detect degradation.
    """

    def __init__(
        self,
        window: int = 200,
        baseline_window: int = 500,
        sharpe_floor: float = 0.0,
        ic_floor: float = 0.02,
        hit_rate_floor: float = 0.48,
    ) -> None:
        self._window = window
        self._baseline_window = baseline_window
        self._sharpe_floor = sharpe_floor
        self._ic_floor = ic_floor
        self._hit_rate_floor = hit_rate_floor

        # Each entry: (predicted_side, actual_return)
        self._history: Deque[Tuple[str, float]] = deque(maxlen=max(window, baseline_window))
        self._baseline_hits: List[bool] = []
        self._baseline_returns: List[float] = []
        self._baseline_frozen = False

    def on_prediction(self, predicted_side: str, actual_return: float) -> None:
        """Record a prediction outcome."""
        self._history.append((predicted_side, actual_return))

        if not self._baseline_frozen:
            hit = self._is_hit(predicted_side, actual_return)
            self._baseline_hits.append(hit)
            self._baseline_returns.append(actual_return)
            if len(self._baseline_hits) >= self._baseline_window:
                self._baseline_frozen = True

    def check(self) -> DriftState:
        """Check if concept drift has occurred based on rolling performance."""
        n = len(self._history)
        if n < self._window:
            return DriftState(
                is_drifting=False,
                severity="none",
                metrics={},
                recommendation="continue",
            )

        # Use last `window` entries for rolling metrics
        recent = list(self._history)[-self._window:]
        hit_rate = self._calc_hit_rate(recent)
        ic = self._calc_ic(recent)
        sharpe = self._calc_sharpe(recent)

        metrics = {
            "rolling_hit_rate": hit_rate,
            "rolling_ic": ic,
            "rolling_sharpe": sharpe,
        }

        # Count how many metrics are below floor
        below = 0
        if hit_rate < self._hit_rate_floor:
            below += 1
        if ic < self._ic_floor:
            below += 1
        if sharpe < self._sharpe_floor:
            below += 1

        if below == 0:
            return DriftState(
                is_drifting=False,
                severity="none",
                metrics=metrics,
                recommendation="continue",
            )
        elif below == 1:
            return DriftState(
                is_drifting=True,
                severity="warning",
                metrics=metrics,
                recommendation="reduce_size",
            )
        elif below == 2:
            return DriftState(
                is_drifting=True,
                severity="critical",
                metrics=metrics,
                recommendation="pause",
            )
        else:
            return DriftState(
                is_drifting=True,
                severity="critical",
                metrics=metrics,
                recommendation="retrain",
            )

    def reset_baseline(self) -> None:
        """Reset baseline after model retraining."""
        self._baseline_hits.clear()
        self._baseline_returns.clear()
        self._baseline_frozen = False

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _is_hit(predicted_side: str, actual_return: float) -> bool:
        if predicted_side == "long":
            return actual_return > 0
        elif predicted_side == "short":
            return actual_return < 0
        return abs(actual_return) < 1e-9  # "flat" is correct if return ≈ 0

    def _calc_hit_rate(self, entries: List[Tuple[str, float]]) -> float:
        if not entries:
            return 0.0
        hits = sum(1 for side, ret in entries if self._is_hit(side, ret))
        return hits / len(entries)

    @staticmethod
    def _calc_ic(entries: List[Tuple[str, float]]) -> float:
        """Rank correlation (Spearman-like) between prediction direction and return."""
        if len(entries) < 2:
            return 0.0

        # Convert predicted side to numeric: long=1, flat=0, short=-1
        preds = []
        actuals = []
        for side, ret in entries:
            if side == "long":
                preds.append(1.0)
            elif side == "short":
                preds.append(-1.0)
            else:
                preds.append(0.0)
            actuals.append(ret)

        # Pearson correlation between prediction direction and actual return
        n = len(preds)
        mean_p = sum(preds) / n
        mean_a = sum(actuals) / n

        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(preds, actuals)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in preds) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)

        if std_p < 1e-12 or std_a < 1e-12:
            return 0.0

        return cov / (std_p * std_a)

    @staticmethod
    def _calc_sharpe(entries: List[Tuple[str, float]]) -> float:
        """Annualized Sharpe from per-prediction PnL (sign-adjusted returns)."""
        if len(entries) < 2:
            return 0.0

        pnls = []
        for side, ret in entries:
            if side == "long":
                pnls.append(ret)
            elif side == "short":
                pnls.append(-ret)
            else:
                pnls.append(0.0)

        n = len(pnls)
        mean = sum(pnls) / n
        var = sum((x - mean) ** 2 for x in pnls) / n
        std = math.sqrt(max(var, 0.0))

        if std < 1e-12:
            return 0.0 if abs(mean) < 1e-12 else float("inf") if mean > 0 else float("-inf")

        return mean / std
