"""Live signal attribution tracker — real-time per-signal-origin PnL tracking.

Bridges the existing rust_attribute_by_signal() for continuous live monitoring,
with Prometheus gauge exports and rapid loss detection (48h consecutive loss -> alert).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    """Real-time per-signal PnL tracker with Prometheus export and loss alerting."""

    prometheus: Optional[Any] = None
    loss_alert_hours: float = 48.0
    _pnl_by_origin: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _trade_count_by_origin: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _win_count_by_origin: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_update_ts: Dict[str, float] = field(default_factory=dict)
    _hourly_pnl: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    _last_hourly_snapshot_ts: float = field(default=0.0)
    _HOURLY_WINDOW: int = 72

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def on_fill(self, fill: Any, origin: str | None = None) -> None:
        """Record a fill event attributed to a signal origin."""
        if origin is None:
            origin = getattr(fill, "origin", None) or getattr(fill, "signal_origin", None) or "unknown"
        pnl = float(getattr(fill, "realized_pnl", 0) or getattr(fill, "pnl", 0) or 0)
        self._record(origin, pnl, pnl > 0)

    def record_pnl(self, origin: str, pnl: float, is_win: bool = False) -> None:
        """Direct PnL recording (alternative to on_fill)."""
        self._record(origin, pnl, is_win)

    def _record(self, origin: str, pnl: float, is_win: bool) -> None:
        self._pnl_by_origin[origin] += pnl
        self._trade_count_by_origin[origin] += 1
        if is_win:
            self._win_count_by_origin[origin] += 1
        self._last_update_ts[origin] = time.time()

    # ------------------------------------------------------------------
    # Metrics export
    # ------------------------------------------------------------------

    def export_metrics(self) -> None:
        """Export current PnL metrics to Prometheus."""
        self._snapshot_hourly()
        if self.prometheus is None:
            return
        for origin, pnl in self._pnl_by_origin.items():
            trades = self._trade_count_by_origin.get(origin, 0)
            wins = self._win_count_by_origin.get(origin, 0)
            labels = {"origin": origin}
            self.prometheus.set_gauge("signal_pnl_by_origin", pnl, labels=labels)
            self.prometheus.set_gauge("signal_trades_by_origin", float(trades), labels=labels)
            self.prometheus.set_gauge("signal_win_rate_by_origin",
                                     (wins / trades * 100.0) if trades > 0 else 0.0, labels=labels)

    # ------------------------------------------------------------------
    # Loss detection
    # ------------------------------------------------------------------

    def check_sustained_losses(self) -> List[str]:
        """Check for origins with sustained losses over the alert window."""
        alerts: List[str] = []
        cutoff = time.time() - self.loss_alert_hours * 3600
        for origin, snapshots in self._hourly_pnl.items():
            if len(snapshots) < 2:
                continue
            window = [(ts, pnl) for ts, pnl in snapshots if ts >= cutoff]
            if len(window) < 2:
                continue
            pnl_at_start = window[0][1]
            period_pnl = self._pnl_by_origin.get(origin, 0) - pnl_at_start
            if period_pnl < 0 and all(window[i][1] >= window[i + 1][1] for i in range(len(window) - 1)):
                msg = (f"Signal '{origin}' has {self.loss_alert_hours:.0f}h sustained loss: "
                       f"{period_pnl:.2f} (trades={self._trade_count_by_origin.get(origin, 0)})")
                alerts.append(msg)
                logger.warning("LiveSignalTracker: %s", msg)
        return alerts

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def compute_rolling_sharpe(self, origin: str, window_hours: float = 168.0) -> Optional[float]:
        """Compute rolling Sharpe ratio for an origin (annualised, hourly returns)."""
        snapshots = self._hourly_pnl.get(origin)
        if snapshots is None or len(snapshots) < 2:
            return None
        cutoff = time.time() - window_hours * 3600
        window = [(ts, pnl) for ts, pnl in snapshots if ts >= cutoff]
        if len(window) < 2:
            return None
        returns = [window[i][1] - window[i - 1][1] for i in range(1, len(window))]
        if not returns:
            return None
        mean_ret = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = var ** 0.5
        return float((mean_ret / std) * (8760 ** 0.5)) if std > 1e-10 else 0.0

    # ------------------------------------------------------------------
    # Weight recommendations
    # ------------------------------------------------------------------

    def compute_weight_recommendations(
        self, *, sharpe_zero_threshold_hours: float = 48.0,
        sharpe_negative_threshold: float = -0.5,
        alpha_health_monitor: Any = None,
    ) -> Dict[str, float]:
        """Compute recommended weight multipliers based on rolling performance."""
        recs: Dict[str, float] = {}
        for origin in self._pnl_by_origin:
            sharpe = self.compute_rolling_sharpe(origin, window_hours=168.0)
            if sharpe is None:
                recs[origin] = 1.0
                continue
            if sharpe < sharpe_negative_threshold:
                weight = 0.0
            elif sharpe < 0:
                short_sharpe = self.compute_rolling_sharpe(origin, window_hours=sharpe_zero_threshold_hours)
                weight = 0.5 if (short_sharpe is not None and short_sharpe < 0) else 1.0
            else:
                weight = 1.0
            # Cross-check with alpha health
            if alpha_health_monitor is not None and weight < 1.0:
                try:
                    sym = origin.split("_")[0] if origin else ""
                    if alpha_health_monitor.position_scale(sym) >= 1.0:
                        weight = max(weight, 0.5)
                except Exception as e:
                    logger.warning("Failed to get alpha health scale for %s: %s", origin, e)
            recs[origin] = weight
        return recs

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

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
        return {
            "origins": origins,
            "total_pnl": round(sum(self._pnl_by_origin.values()), 4),
            "total_trades": sum(self._trade_count_by_origin.values()),
            "weight_recommendations": self.compute_weight_recommendations(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _snapshot_hourly(self) -> None:
        now = time.time()
        if now - self._last_hourly_snapshot_ts < 3600:
            return
        self._last_hourly_snapshot_ts = now
        for origin, pnl in self._pnl_by_origin.items():
            if origin not in self._hourly_pnl:
                self._hourly_pnl[origin] = deque(maxlen=self._HOURLY_WINDOW)
            self._hourly_pnl[origin].append((now, pnl))
