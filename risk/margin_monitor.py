# risk/margin_monitor.py
"""MarginMonitor — monitors margin ratio and funding rates in a background thread."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

from monitoring.alerts.base import Alert, Severity
from risk.kill_switch import KillMode, KillScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarginConfig:
    """Thresholds for margin and funding rate monitoring."""
    check_interval_sec: float = 30.0
    warning_margin_ratio: float = 0.15   # 15% margin ratio
    critical_margin_ratio: float = 0.08  # 8% margin ratio
    extreme_funding_threshold: float = 0.001  # 0.1% per 8h
    auto_reduce_on_critical: bool = True


@dataclass
class MarginMonitor:
    """Monitors margin ratio and funding rates.

    Periodically calls fetch_margin to get the current margin state.
    When the margin ratio drops below warning or critical thresholds,
    it emits alerts and optionally triggers a KillSwitch REDUCE_ONLY.

    Also monitors funding rates for extreme values that could signal
    market stress or unfavorable positions.
    """

    config: MarginConfig
    fetch_margin: Callable[[], Dict[str, Any]]
    fetch_funding: Optional[Callable[[], Dict[str, float]]] = None
    kill_switch: Optional[Any] = None  # KillSwitch
    alert_sink: Optional[Any] = None   # AlertSink

    _running: bool = field(default=False, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _last_status: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="margin-monitor", daemon=True,
        )
        self._thread.start()
        logger.info("MarginMonitor started (interval=%.1fs)", self.config.check_interval_sec)

    def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.config.check_interval_sec + 2.0)
            self._thread = None
        logger.info("MarginMonitor stopped")

    def check_once(self) -> Dict[str, Any]:
        """Run margin and funding checks. Returns status dict."""
        status: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "margin_ok": True,
            "funding_ok": True,
            "alerts": [],
        }

        # 1. Fetch and check margin ratio
        try:
            margin_data = self.fetch_margin()
        except Exception:
            logger.exception("fetch_margin failed")
            status["margin_ok"] = False
            status["error"] = "fetch_margin_failed"
            self._last_status = status
            return status

        margin_ratio = float(margin_data.get("margin_ratio", 1.0))
        status["margin_ratio"] = margin_ratio
        status["total_margin"] = margin_data.get("total_margin")

        # 2. Critical margin check
        if margin_ratio <= self.config.critical_margin_ratio:
            status["margin_ok"] = False
            alert_msg = (
                f"CRITICAL margin ratio {margin_ratio:.4f} "
                f"below threshold {self.config.critical_margin_ratio}"
            )
            status["alerts"].append(alert_msg)
            logger.critical(alert_msg)

            self._emit_alert(
                title="Margin Critical",
                message=alert_msg,
                severity=Severity.CRITICAL,
                meta={"margin_ratio": margin_ratio, **margin_data},
            )

            if self.config.auto_reduce_on_critical and self.kill_switch is not None:
                self.kill_switch.trigger(
                    scope=KillScope.GLOBAL,
                    key="*",
                    mode=KillMode.REDUCE_ONLY,
                    reason=f"margin_ratio={margin_ratio:.4f} < critical={self.config.critical_margin_ratio}",
                    source="margin_monitor",
                    ttl_seconds=300,
                )
                logger.warning("KillSwitch REDUCE_ONLY triggered by critical margin")

        # 3. Warning margin check
        elif margin_ratio <= self.config.warning_margin_ratio:
            alert_msg = (
                f"Warning margin ratio {margin_ratio:.4f} "
                f"below threshold {self.config.warning_margin_ratio}"
            )
            status["alerts"].append(alert_msg)
            logger.warning(alert_msg)

            self._emit_alert(
                title="Margin Warning",
                message=alert_msg,
                severity=Severity.WARNING,
                meta={"margin_ratio": margin_ratio, **margin_data},
            )

        # 4. Check funding rates
        if self.fetch_funding is not None:
            try:
                funding_rates = self.fetch_funding()
            except Exception:
                logger.exception("fetch_funding failed")
                funding_rates = {}

            extreme_symbols = {
                sym: rate
                for sym, rate in funding_rates.items()
                if abs(rate) >= self.config.extreme_funding_threshold
            }

            if extreme_symbols:
                status["funding_ok"] = False
                status["extreme_funding"] = extreme_symbols
                alert_msg = f"Extreme funding rates detected: {extreme_symbols}"
                status["alerts"].append(alert_msg)
                logger.warning(alert_msg)

                self._emit_alert(
                    title="Extreme Funding Rate",
                    message=alert_msg,
                    severity=Severity.WARNING,
                    meta={"funding_rates": extreme_symbols},
                )

        self._last_status = status
        return status

    def _emit_alert(
        self,
        *,
        title: str,
        message: str,
        severity: Severity,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.alert_sink is None:
            return
        try:
            alert = Alert(
                title=title,
                message=message,
                severity=severity,
                source="margin_monitor",
                ts=datetime.now(timezone.utc),
                meta=meta,
            )
            self.alert_sink.emit(alert)
        except Exception:
            logger.exception("alert_sink.emit failed")

    def _run_loop(self) -> None:
        while self._running:
            time.sleep(self.config.check_interval_sec)
            if self._running:
                try:
                    self.check_once()
                except Exception:
                    logger.exception("MarginMonitor check_once failed unexpectedly")

    @property
    def last_status(self) -> Optional[Dict[str, Any]]:
        return self._last_status
