"""Alpha Health Monitor — live IC tracking with automatic risk response.

Tracks per-horizon IC in real time, exports to Prometheus, and triggers
automatic position reduction or halt when alpha decays.

Three response levels:
1. Warning: rolling IC < 0 for 5+ days → log warning, Prometheus alert
2. Reduce:  rolling IC < 0 for 10+ days → scale position to 50%
3. Halt:    rolling IC < -0.02 for 10+ days → stop trading, trigger retrain

Integrates with:
- alpha/ic_monitor.py (per-horizon ICMonitor)
- monitoring/metrics/prometheus.py (PrometheusExporter)
- monitoring/alpha_decay.py (AlphaDecayMonitor for Sharpe-based alerts)
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from alpha.ic_monitor import ICMonitor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaHealthAlert:
    """An alpha health alert."""
    symbol: str
    horizon: int
    alert_type: str       # "ic_warning" | "ic_reduce" | "ic_halt" | "ic_recovery"
    rolling_ic: float
    days_negative: int
    message: str


@dataclass
class AlphaHealthConfig:
    """Configuration for alpha health monitoring."""
    # IC thresholds
    ic_warning_days: int = 5           # days of IC < 0 before warning
    ic_reduce_days: int = 10           # days of IC < 0 before position reduction
    ic_halt_threshold: float = -0.02   # IC below this for reduce_days → halt
    ic_recovery_threshold: float = 0.01  # IC above this to clear alerts

    # Position scaling
    reduce_scale: float = 0.5         # position scale during "reduce" state

    # IC evaluation frequency (bars between IC checks)
    eval_interval_bars: int = 24      # evaluate IC once per day (24 1h bars)

    # Monitor window
    ic_window: int = 720              # bars for rolling IC computation

    # Retrain trigger
    auto_retrain_on_halt: bool = True


class AlphaHealthMonitor:
    """Monitors alpha health per symbol/horizon and triggers risk responses.

    Usage:
        monitor = AlphaHealthMonitor(config=AlphaHealthConfig())

        # Register symbols/horizons at startup
        monitor.register("ETHUSDT", horizons=[12, 24, 48])

        # On each bar, feed predictions and realized returns
        monitor.update("ETHUSDT", horizon=12, pred=0.3, actual_return=0.002)

        # Periodic check (once per day)
        alerts = monitor.check("ETHUSDT")
        scale = monitor.position_scale("ETHUSDT")  # 0.0, 0.5, or 1.0
    """

    def __init__(
        self,
        config: Optional[AlphaHealthConfig] = None,
        prometheus: Any = None,
    ):
        self._config = config or AlphaHealthConfig()
        self._prometheus = prometheus

        # Per (symbol, horizon) IC monitors
        self._monitors: Dict[str, Dict[int, ICMonitor]] = {}

        # Per symbol state
        self._state: Dict[str, _SymbolHealthState] = {}

        # Bar counter for eval scheduling
        self._bar_counts: Dict[str, int] = {}

    def register(self, symbol: str, horizons: List[int]) -> None:
        """Register a symbol with its horizons for monitoring."""
        self._monitors[symbol] = {
            h: ICMonitor(window=self._config.ic_window)
            for h in horizons
        }
        self._state[symbol] = _SymbolHealthState(horizons=horizons)
        self._bar_counts[symbol] = 0

    def update(
        self,
        symbol: str,
        horizon: int,
        pred: float,
        actual_return: float,
    ) -> None:
        """Feed a (prediction, realized_return) pair for one horizon."""
        monitors = self._monitors.get(symbol)
        if monitors is None:
            return
        monitor = monitors.get(horizon)
        if monitor is None:
            return
        monitor.update(pred, actual_return)

    def on_bar(self, symbol: str) -> List[AlphaHealthAlert]:
        """Called once per bar. Returns alerts if eval interval reached."""
        self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1
        if self._bar_counts[symbol] % self._config.eval_interval_bars != 0:
            return []
        return self.check(symbol)

    def check(self, symbol: str) -> List[AlphaHealthAlert]:
        """Evaluate IC health for a symbol. Returns new alerts."""
        monitors = self._monitors.get(symbol)
        state = self._state.get(symbol)
        if monitors is None or state is None:
            return []

        alerts: List[AlphaHealthAlert] = []
        cfg = self._config

        for h, monitor in monitors.items():
            if monitor.n_samples < 50:
                continue

            ic = monitor.rolling_ic
            h_state = state.horizon_states[h]

            # Export to Prometheus
            self._export_ic(symbol, h, ic)

            # Track consecutive negative days
            if ic < 0:
                h_state.negative_streak_bars += cfg.eval_interval_bars
            else:
                if h_state.negative_streak_bars > 0 and ic >= cfg.ic_recovery_threshold:
                    # Recovery
                    if h_state.alert_level != "ok":
                        alerts.append(AlphaHealthAlert(
                            symbol=symbol, horizon=h,
                            alert_type="ic_recovery",
                            rolling_ic=ic,
                            days_negative=0,
                            message=f"{symbol} h{h} IC recovered to {ic:.4f}",
                        ))
                        logger.info("Alpha recovery: %s h%d IC=%.4f", symbol, h, ic)
                    h_state.alert_level = "ok"
                h_state.negative_streak_bars = 0

            days_neg = h_state.negative_streak_bars // 24

            # Determine alert level
            new_level = "ok"
            if ic < cfg.ic_halt_threshold and days_neg >= cfg.ic_reduce_days:
                new_level = "halt"
            elif days_neg >= cfg.ic_reduce_days:
                new_level = "reduce"
            elif days_neg >= cfg.ic_warning_days:
                new_level = "warning"

            # Emit alert on level change
            if new_level != h_state.alert_level:
                alert_type = f"ic_{new_level}" if new_level != "ok" else "ic_recovery"
                alert = AlphaHealthAlert(
                    symbol=symbol, horizon=h,
                    alert_type=alert_type,
                    rolling_ic=ic,
                    days_negative=days_neg,
                    message=self._format_message(symbol, h, new_level, ic, days_neg),
                )
                alerts.append(alert)
                h_state.alert_level = new_level

                if new_level == "halt":
                    logger.critical(
                        "Alpha HALT: %s h%d IC=%.4f for %d days — stopping trading",
                        symbol, h, ic, days_neg,
                    )
                elif new_level == "reduce":
                    logger.warning(
                        "Alpha REDUCE: %s h%d IC=%.4f for %d days — scaling to %.0f%%",
                        symbol, h, ic, days_neg, cfg.reduce_scale * 100,
                    )
                elif new_level == "warning":
                    logger.warning(
                        "Alpha WARNING: %s h%d IC=%.4f negative for %d days",
                        symbol, h, ic, days_neg,
                    )

            h_state.latest_ic = ic

        # Update aggregate state
        state.update_aggregate()

        return alerts

    def position_scale(self, symbol: str) -> float:
        """Get position scale factor for a symbol (0.0=halt, 0.5=reduce, 1.0=ok).

        Uses the worst horizon's state to determine overall scale.
        """
        state = self._state.get(symbol)
        if state is None:
            return 1.0
        return state.aggregate_scale(self._config.reduce_scale)

    def get_horizon_ic(self, symbol: str, horizon: int) -> float:
        """Get current rolling IC for a specific horizon."""
        monitors = self._monitors.get(symbol)
        if monitors is None:
            return 0.0
        monitor = monitors.get(horizon)
        if monitor is None or monitor.n_samples < 50:
            return 0.0
        return monitor.rolling_ic

    def get_status(self, symbol: str) -> Dict[str, Any]:
        """Get full status report for a symbol."""
        state = self._state.get(symbol)
        monitors = self._monitors.get(symbol)
        if state is None or monitors is None:
            return {"symbol": symbol, "status": "unregistered"}

        horizon_status = {}
        for h, h_state in state.horizon_states.items():
            horizon_status[h] = {
                "ic": h_state.latest_ic,
                "alert_level": h_state.alert_level,
                "days_negative": h_state.negative_streak_bars // 24,
                "n_samples": monitors[h].n_samples,
            }

        return {
            "symbol": symbol,
            "position_scale": self.position_scale(symbol),
            "horizons": horizon_status,
            "needs_retrain": any(
                hs.alert_level == "halt" for hs in state.horizon_states.values()
            ),
        }

    def should_retrain(self, symbol: str) -> bool:
        """Whether this symbol needs model retraining."""
        state = self._state.get(symbol)
        if state is None:
            return False
        return (
            self._config.auto_retrain_on_halt
            and any(hs.alert_level == "halt" for hs in state.horizon_states.values())
        )

    def _export_ic(self, symbol: str, horizon: int, ic: float) -> None:
        """Export IC to Prometheus."""
        if self._prometheus is None:
            return
        try:
            self._prometheus.set_gauge(
                "alpha_rolling_ic",
                ic,
                labels={"symbol": symbol, "horizon": str(horizon)},
            )
        except Exception:
            pass  # Prometheus not critical

    @staticmethod
    def _format_message(
        symbol: str, horizon: int, level: str, ic: float, days_neg: int
    ) -> str:
        if level == "halt":
            return (
                f"HALT {symbol} h{horizon}: IC={ic:.4f} for {days_neg}d. "
                f"Trading stopped. Retrain required."
            )
        elif level == "reduce":
            return (
                f"REDUCE {symbol} h{horizon}: IC={ic:.4f} negative for {days_neg}d. "
                f"Position scaled down."
            )
        elif level == "warning":
            return (
                f"WARNING {symbol} h{horizon}: IC={ic:.4f} negative for {days_neg}d."
            )
        return f"{symbol} h{horizon}: IC={ic:.4f} — status {level}"


@dataclass
class _HorizonHealthState:
    """Per-horizon health tracking."""
    latest_ic: float = 0.0
    alert_level: str = "ok"        # "ok" | "warning" | "reduce" | "halt"
    negative_streak_bars: int = 0


@dataclass
class _SymbolHealthState:
    """Per-symbol aggregate health state."""
    horizons: List[int] = field(default_factory=list)
    horizon_states: Dict[int, _HorizonHealthState] = field(default_factory=dict)

    def __post_init__(self):
        for h in self.horizons:
            if h not in self.horizon_states:
                self.horizon_states[h] = _HorizonHealthState()

    def update_aggregate(self) -> None:
        """Update aggregate metrics from horizon states."""
        pass  # Currently position_scale() computes on the fly

    def aggregate_scale(self, reduce_scale: float) -> float:
        """Worst-case position scale across all horizons."""
        if not self.horizon_states:
            return 1.0

        worst = "ok"
        for hs in self.horizon_states.values():
            if hs.alert_level == "halt":
                worst = "halt"
                break
            elif hs.alert_level == "reduce" and worst != "halt":
                worst = "reduce"
            elif hs.alert_level == "warning" and worst == "ok":
                worst = "warning"

        if worst == "halt":
            return 0.0
        elif worst == "reduce":
            return reduce_scale
        return 1.0
