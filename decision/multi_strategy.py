# decision/multi_strategy.py
"""Multi-strategy ensemble — dynamic weight allocation with performance tracking."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from _quant_hotpath import rust_rolling_sharpe, rust_max_drawdown
from decision.types import SignalResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Tracks rolling performance for a single strategy."""
    name: str
    lookback: int = 60
    _returns: Deque[float] = field(default=None, init=False)
    _equity: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        self._returns = deque(maxlen=self.lookback)

    def record_return(self, ret: float) -> None:
        self._returns.append(ret)
        self._equity *= (1.0 + ret)

    @property
    def n_observations(self) -> int:
        return len(self._returns)

    @property
    def cumulative_return(self) -> float:
        return self._equity - 1.0

    @property
    def rolling_sharpe(self) -> Optional[float]:
        return rust_rolling_sharpe(list(self._returns), window=self.lookback, min_obs=10)

    @property
    def rolling_max_drawdown(self) -> float:
        if not self._returns:
            return 0.0
        return rust_max_drawdown(list(self._returns), is_returns=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cumulative_return": round(self.cumulative_return, 6),
            "rolling_sharpe": round(self.rolling_sharpe, 4) if self.rolling_sharpe is not None else None,
            "rolling_max_dd": round(self.rolling_max_drawdown, 4),
            "n_observations": self.n_observations,
        }


@dataclass
class MultiStrategyModule:
    """DecisionModule that ensembles multiple strategy modules with dynamic weights.

    Weight allocation methods:
    - "equal": Equal weight to all strategies
    - "sharpe": Weight proportional to rolling Sharpe ratio
    - "inverse_vol": Weight inversely proportional to rolling volatility

    Implements DecisionModule protocol: decide(snapshot) -> Iterable[Event]
    """

    modules: Sequence[Any]  # Each has .decide(snapshot) -> Iterable[Event]
    module_names: Sequence[str] = ()
    allocation_method: str = "sharpe"  # equal | sharpe | inverse_vol
    min_weight: float = 0.05
    max_weight: float = 0.5
    warmup_bars: int = 30
    lookback: int = 60

    _trackers: Dict[str, StrategyPerformance] = field(default_factory=dict, init=False)
    _bar_count: int = field(default=0, init=False)
    _weights: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        names = list(self.module_names) if self.module_names else [
            f"strategy_{i}" for i in range(len(self.modules))
        ]
        for name in names:
            self._trackers[name] = StrategyPerformance(
                name=name, lookback=self.lookback,
            )
        # Initialize equal weights
        n = len(self.modules)
        for name in names:
            self._weights[name] = 1.0 / max(n, 1)

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """Run all strategies, combine by weight, emit top events."""
        self._bar_count += 1
        names = list(self._trackers.keys())

        # Collect all events from all modules
        all_events: Dict[str, List[Any]] = {}
        for name, mod in zip(names, self.modules):
            try:
                events = list(mod.decide(snapshot))
                all_events[name] = events
            except Exception as e:
                logger.warning("Strategy %s failed: %s", name, e)
                all_events[name] = []

        # Update weights after warmup
        if self._bar_count > self.warmup_bars:
            self._update_weights()

        # Weight-filter events: emit events from highest-weighted strategies
        combined: List[Any] = []
        for name in sorted(names, key=lambda n: self._weights.get(n, 0), reverse=True):
            w = self._weights.get(name, 0)
            if w < self.min_weight:
                continue
            combined.extend(all_events.get(name, []))

        return combined

    def record_strategy_return(self, strategy_name: str, ret: float) -> None:
        """Record a period return for a strategy (called externally after fills)."""
        tracker = self._trackers.get(strategy_name)
        if tracker is not None:
            tracker.record_return(ret)

    def _update_weights(self) -> None:
        """Recompute weights based on allocation method."""
        names = list(self._trackers.keys())
        n = len(names)

        if self.allocation_method == "equal" or n == 0:
            for name in names:
                self._weights[name] = 1.0 / max(n, 1)
            return

        if self.allocation_method == "sharpe":
            sharpes = {}
            for name, tracker in self._trackers.items():
                sr = tracker.rolling_sharpe
                sharpes[name] = max(sr, 0.0) if sr is not None else 0.0

            total = sum(sharpes.values())
            if total > 0:
                for name in names:
                    self._weights[name] = sharpes[name] / total
            else:
                for name in names:
                    self._weights[name] = 1.0 / n

        elif self.allocation_method == "inverse_vol":
            inv_vols = {}
            for name, tracker in self._trackers.items():
                if tracker.n_observations < 10:
                    inv_vols[name] = 1.0
                    continue
                rets = list(tracker._returns)
                mean = sum(rets) / len(rets)
                var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
                vol = sqrt(max(var, 0.0))
                inv_vols[name] = 1.0 / max(vol, 1e-8)

            total = sum(inv_vols.values())
            if total > 0:
                for name in names:
                    self._weights[name] = inv_vols[name] / total
            else:
                for name in names:
                    self._weights[name] = 1.0 / n

        # Apply min/max weight constraints
        self._clamp_weights()

    def _clamp_weights(self) -> None:
        """Enforce min/max weight bounds and renormalize."""
        for name in self._weights:
            self._weights[name] = max(self.min_weight, min(self.max_weight, self._weights[name]))
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    @property
    def strategy_stats(self) -> List[Dict[str, Any]]:
        return [
            {**t.to_dict(), "weight": round(self._weights.get(t.name, 0), 4)}
            for t in self._trackers.values()
        ]
