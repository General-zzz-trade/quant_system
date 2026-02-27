# monitoring/alpha_decay.py
"""Alpha decay monitoring — detects strategy performance degradation over time."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecayAlert:
    """Alert for detected alpha decay."""
    strategy_name: str
    current_sharpe: float
    baseline_sharpe: float
    decay_pct: float
    severity: str  # "warning" | "critical"


@dataclass
class AlphaDecayMonitor:
    """Monitors rolling Sharpe ratio for alpha decay.

    Compares recent Sharpe to a baseline (initial period or fixed target).
    Emits alerts when decay exceeds thresholds.

    Usage:
        monitor = AlphaDecayMonitor(
            warning_decay_pct=30.0,
            critical_decay_pct=60.0,
        )
        monitor.set_baseline("my_strategy", 1.5)

        # On each bar/period:
        monitor.record_return("my_strategy", 0.001)
        alerts = monitor.check()
    """

    short_window: int = 20
    long_window: int = 60
    warning_decay_pct: float = 30.0
    critical_decay_pct: float = 60.0
    on_alert: Optional[Callable[[DecayAlert], None]] = None

    _returns: Dict[str, Deque[float]] = field(default_factory=dict, init=False)
    _baselines: Dict[str, float] = field(default_factory=dict, init=False)
    _alerts_emitted: Dict[str, str] = field(default_factory=dict, init=False)

    def set_baseline(self, strategy_name: str, sharpe: float) -> None:
        """Set the baseline Sharpe ratio for a strategy."""
        self._baselines[strategy_name] = sharpe

    def record_return(self, strategy_name: str, ret: float) -> None:
        """Record a period return for a strategy."""
        if strategy_name not in self._returns:
            self._returns[strategy_name] = deque(maxlen=self.long_window)
        self._returns[strategy_name].append(ret)

    def check(self) -> List[DecayAlert]:
        """Check all strategies for alpha decay. Returns new alerts."""
        alerts: List[DecayAlert] = []

        for name, rets in self._returns.items():
            baseline = self._baselines.get(name)
            if baseline is None or baseline <= 0:
                continue

            current = self._rolling_sharpe(rets, self.short_window)
            if current is None:
                continue

            decay_pct = (1.0 - current / baseline) * 100.0

            if decay_pct >= self.critical_decay_pct:
                severity = "critical"
            elif decay_pct >= self.warning_decay_pct:
                severity = "warning"
            else:
                # No alert needed; clear previous alert state
                self._alerts_emitted.pop(name, None)
                continue

            # Avoid re-emitting same severity
            prev = self._alerts_emitted.get(name)
            if prev == severity:
                continue

            alert = DecayAlert(
                strategy_name=name,
                current_sharpe=current,
                baseline_sharpe=baseline,
                decay_pct=decay_pct,
                severity=severity,
            )
            alerts.append(alert)
            self._alerts_emitted[name] = severity

            logger.warning(
                "Alpha decay [%s] %s: Sharpe %.2f → %.2f (decay %.1f%%)",
                severity, name, baseline, current, decay_pct,
            )

            if self.on_alert is not None:
                try:
                    self.on_alert(alert)
                except Exception:
                    logger.exception("Alpha decay alert callback failed")

        return alerts

    def get_rolling_sharpe(self, strategy_name: str) -> Optional[float]:
        """Get current rolling Sharpe for a strategy."""
        rets = self._returns.get(strategy_name)
        if rets is None:
            return None
        return self._rolling_sharpe(rets, self.short_window)

    def get_long_sharpe(self, strategy_name: str) -> Optional[float]:
        """Get longer-term rolling Sharpe for a strategy."""
        rets = self._returns.get(strategy_name)
        if rets is None:
            return None
        return self._rolling_sharpe(rets, self.long_window)

    @staticmethod
    def _rolling_sharpe(returns: Deque[float], window: int) -> Optional[float]:
        if len(returns) < min(window, 10):
            return None
        recent = list(returns)[-window:]
        mean = sum(recent) / len(recent)
        var = sum((r - mean) ** 2 for r in recent) / max(len(recent) - 1, 1)
        std = sqrt(max(var, 0.0))
        if std < 1e-12:
            return 99.0 if mean > 0 else 0.0
        return mean / std * sqrt(252)

    @property
    def strategy_names(self) -> List[str]:
        return list(self._returns.keys())
