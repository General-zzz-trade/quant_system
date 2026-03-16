"""Live signal attribution tracker — real-time per-signal-origin PnL tracking.

Bridges the existing rust_attribute_by_signal() for continuous live monitoring,
with Prometheus gauge exports and rapid loss detection (48h consecutive loss → alert).

Usage:
    tracker = LiveSignalTracker(prometheus=exporter)
    # On each fill:
    tracker.on_fill(fill_event)
    # Periodic export (every 10 ticks via EngineMonitoringHook):
    tracker.export_metrics()
    # Status for health endpoint:
    tracker.get_status()
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SignalPnLSnapshot:
    """Point-in-time PnL for a signal origin."""
    origin: str
    realized_pnl: float
    trade_count: int
    win_count: int
    last_update_ts: float


@dataclass
class LiveSignalTracker:
    """Real-time per-signal PnL tracker.

    Accumulates fill-by-fill PnL attributed to signal origins,
    exports to Prometheus, and detects sustained losses for early alerting.
    """

    prometheus: Optional[Any] = None
    loss_alert_hours: float = 48.0
    _pnl_by_origin: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _trade_count_by_origin: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _win_count_by_origin: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_update_ts: Dict[str, float] = field(default_factory=dict)
    # Hourly PnL snapshots for loss detection: origin -> deque of (ts, cumulative_pnl)
    _hourly_pnl: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    _last_hourly_snapshot_ts: float = field(default=0.0)
    _HOURLY_WINDOW: int = 72  # keep 72 hours of snapshots

    def on_fill(self, fill: Any, origin: str | None = None) -> None:
        """Record a fill event attributed to a signal origin.

        Args:
            fill: Fill event with pnl, qty, price attributes.
            origin: Signal origin string (e.g., "h24_lgbm", "h12_xgb").
                    If None, tries to extract from fill.origin or fill.signal_origin.
        """
        if origin is None:
            origin = (
                getattr(fill, "origin", None)
                or getattr(fill, "signal_origin", None)
                or "unknown"
            )

        pnl = float(getattr(fill, "realized_pnl", 0) or getattr(fill, "pnl", 0) or 0)
        now = time.time()

        self._pnl_by_origin[origin] += pnl
        self._trade_count_by_origin[origin] += 1
        if pnl > 0:
            self._win_count_by_origin[origin] += 1
        self._last_update_ts[origin] = now

    def record_pnl(self, origin: str, pnl: float, is_win: bool = False) -> None:
        """Direct PnL recording (alternative to on_fill)."""
        now = time.time()
        self._pnl_by_origin[origin] += pnl
        self._trade_count_by_origin[origin] += 1
        if is_win:
            self._win_count_by_origin[origin] += 1
        self._last_update_ts[origin] = now

    def _snapshot_hourly(self) -> None:
        """Take hourly snapshot of cumulative PnL per origin for loss detection."""
        now = time.time()
        if now - self._last_hourly_snapshot_ts < 3600:
            return
        self._last_hourly_snapshot_ts = now

        for origin, pnl in self._pnl_by_origin.items():
            if origin not in self._hourly_pnl:
                self._hourly_pnl[origin] = deque(maxlen=self._HOURLY_WINDOW)
            self._hourly_pnl[origin].append((now, pnl))

    def export_metrics(self) -> None:
        """Export current PnL metrics to Prometheus."""
        self._snapshot_hourly()

        if self.prometheus is None:
            return

        for origin, pnl in self._pnl_by_origin.items():
            self.prometheus.set_gauge(
                "signal_pnl_by_origin", pnl, labels={"origin": origin},
            )
            trades = self._trade_count_by_origin.get(origin, 0)
            self.prometheus.set_gauge(
                "signal_trades_by_origin", float(trades), labels={"origin": origin},
            )
            wins = self._win_count_by_origin.get(origin, 0)
            win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
            self.prometheus.set_gauge(
                "signal_win_rate_by_origin", win_rate, labels={"origin": origin},
            )

    def check_sustained_losses(self) -> List[str]:
        """Check for origins with sustained losses over the alert window.

        Returns list of alert messages for origins losing money consistently.
        """
        alerts = []
        now = time.time()
        cutoff = now - self.loss_alert_hours * 3600

        for origin, snapshots in self._hourly_pnl.items():
            if len(snapshots) < 2:
                continue

            # Find the PnL at the cutoff point
            pnl_at_cutoff = None
            for ts, pnl in snapshots:
                if ts >= cutoff:
                    pnl_at_cutoff = pnl
                    break

            if pnl_at_cutoff is None:
                continue

            current_pnl = self._pnl_by_origin.get(origin, 0)
            period_pnl = current_pnl - pnl_at_cutoff

            if period_pnl < 0:
                # Check if ALL hourly snapshots in the window are declining
                window_snapshots = [(ts, pnl) for ts, pnl in snapshots if ts >= cutoff]
                if len(window_snapshots) >= 2:
                    all_declining = all(
                        window_snapshots[i][1] >= window_snapshots[i + 1][1]
                        for i in range(len(window_snapshots) - 1)
                    )
                    if all_declining:
                        msg = (
                            f"Signal '{origin}' has {self.loss_alert_hours:.0f}h "
                            f"sustained loss: {period_pnl:.2f} "
                            f"(trades={self._trade_count_by_origin.get(origin, 0)})"
                        )
                        alerts.append(msg)
                        logger.warning("LiveSignalTracker: %s", msg)

        return alerts

    def compute_rolling_sharpe(
        self, origin: str, window_hours: float = 168.0,
    ) -> Optional[float]:
        """Compute rolling Sharpe ratio for an origin over the specified window.

        Uses hourly PnL snapshots to compute returns and Sharpe.
        Returns None if insufficient data.
        """
        snapshots = self._hourly_pnl.get(origin)
        if snapshots is None or len(snapshots) < 2:
            return None

        now = time.time()
        cutoff = now - window_hours * 3600

        # Get hourly returns (diff of cumulative PnL)
        window_snaps = [(ts, pnl) for ts, pnl in snapshots if ts >= cutoff]
        if len(window_snaps) < 2:
            return None

        returns = [
            window_snaps[i][1] - window_snaps[i - 1][1]
            for i in range(1, len(window_snaps))
        ]
        if not returns:
            return None

        mean_ret = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0

        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = var ** 0.5

        if std < 1e-10:
            return 0.0

        # Annualize (hourly -> yearly: sqrt(8760))
        return (mean_ret / std) * (8760 ** 0.5)

    def compute_weight_recommendations(
        self,
        *,
        sharpe_zero_threshold_hours: float = 48.0,
        sharpe_negative_threshold: float = -0.5,
        alpha_health_monitor: Any = None,
    ) -> Dict[str, float]:
        """Compute recommended weight multipliers based on rolling performance.

        Rules:
          - Rolling Sharpe < -0.5 -> weight x 0.0
          - Rolling Sharpe < 0 for threshold hours -> weight x 0.5
          - If alpha_health confirms IC degradation -> apply reduction
          - Otherwise -> weight x 1.0 (no adjustment)

        Returns dict of origin -> weight multiplier (0.0-1.0).
        """
        recommendations: Dict[str, float] = {}

        for origin in self._pnl_by_origin:
            sharpe = self.compute_rolling_sharpe(origin, window_hours=168.0)

            if sharpe is None:
                recommendations[origin] = 1.0
                continue

            if sharpe < sharpe_negative_threshold:
                weight = 0.0
            elif sharpe < 0:
                short_sharpe = self.compute_rolling_sharpe(
                    origin, window_hours=sharpe_zero_threshold_hours,
                )
                if short_sharpe is not None and short_sharpe < 0:
                    weight = 0.5
                else:
                    weight = 1.0
            else:
                weight = 1.0

            # Cross-check with alpha health if available
            if alpha_health_monitor is not None and weight < 1.0:
                scale = 1.0
                try:
                    parts = origin.split("_")
                    if parts:
                        sym = parts[0]
                        scale = alpha_health_monitor.position_scale(sym)
                except Exception as e:
                    logger.warning("Failed to get alpha health scale for %s: %s", origin, e)
                if scale >= 1.0:
                    # Alpha health fine -> don't reduce (single indicator noise)
                    weight = max(weight, 0.5)

            recommendations[origin] = weight

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Return current attribution status for health endpoint."""
        origins = {}
        for origin in self._pnl_by_origin:
            trades = self._trade_count_by_origin.get(origin, 0)
            wins = self._win_count_by_origin.get(origin, 0)
            sharpe = self.compute_rolling_sharpe(origin)
            origins[origin] = {
                "pnl": round(self._pnl_by_origin[origin], 4),
                "trades": trades,
                "win_rate": round(wins / trades * 100, 1) if trades > 0 else 0.0,
                "last_update": self._last_update_ts.get(origin),
                "rolling_sharpe_7d": round(sharpe, 4) if sharpe is not None else None,
            }

        recommendations = self.compute_weight_recommendations()

        return {
            "origins": origins,
            "total_pnl": round(sum(self._pnl_by_origin.values()), 4),
            "total_trades": sum(self._trade_count_by_origin.values()),
            "weight_recommendations": recommendations,
        }
