"""System health monitoring for live trading.

Tracks key operational metrics and emits alerts when thresholds are breached.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

from monitoring.alerts.base import Alert, AlertSink, Severity
from monitoring.alerts.console import ConsoleAlertSink


@dataclass(frozen=True, slots=True)
class HealthConfig:
    """Thresholds for health checks."""
    # Data freshness: alert if no market data for this many seconds
    stale_data_sec: float = 30.0
    # Balance: alert if USDT balance drops below this
    min_balance_usdt: Decimal = Decimal("100")
    # Drawdown: alert at warning level, critical at 2x
    drawdown_warning_pct: float = 10.0
    drawdown_critical_pct: float = 20.0
    # Check interval
    check_interval_sec: float = 5.0


@dataclass
class HealthStatus:
    """Current system health state."""
    last_market_ts: Optional[float] = None
    last_balance: Optional[Decimal] = None
    peak_equity: Optional[Decimal] = None
    current_equity: Optional[Decimal] = None
    is_connected: bool = False
    last_check_ts: Optional[float] = None

    @property
    def data_age_sec(self) -> Optional[float]:
        if self.last_market_ts is None:
            return None
        return time.monotonic() - self.last_market_ts

    @property
    def drawdown_pct(self) -> Optional[float]:
        if self.peak_equity is None or self.current_equity is None:
            return None
        if self.peak_equity <= 0:
            return None
        dd = float((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0
        return max(0.0, dd)


class SystemHealthMonitor:
    """Monitors system health and emits alerts for operational issues.

    Usage:
        monitor = SystemHealthMonitor(config=HealthConfig(), sink=ConsoleAlertSink())
        monitor.start()

        # Feed updates from your event loop:
        monitor.on_market_data(ts=datetime.now(timezone.utc))
        monitor.on_balance_update(balance=Decimal("9500"), equity=Decimal("9500"))
        monitor.on_connection_change(connected=True)

        monitor.stop()
    """

    def __init__(
        self,
        *,
        config: Optional[HealthConfig] = None,
        sink: Optional[AlertSink] = None,
        on_status: Optional[Callable[["HealthStatus"], None]] = None,
    ) -> None:
        self._cfg = config or HealthConfig()
        self._sink = sink or ConsoleAlertSink()
        self._on_status = on_status
        self._status = HealthStatus()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Update methods (called from event loop) ───────────────

    def on_market_data(self, ts: Optional[datetime] = None) -> None:
        with self._lock:
            self._status.last_market_ts = time.monotonic()

    def on_balance_update(
        self,
        *,
        balance: Optional[Decimal] = None,
        equity: Optional[Decimal] = None,
    ) -> None:
        with self._lock:
            if balance is not None:
                self._status.last_balance = balance
            if equity is not None:
                self._status.current_equity = equity
                if self._status.peak_equity is None or equity > self._status.peak_equity:
                    self._status.peak_equity = equity

    def on_connection_change(self, connected: bool) -> None:
        prev = self._status.is_connected
        with self._lock:
            self._status.is_connected = connected
        if prev and not connected:
            self._emit(Alert(
                title="WebSocket disconnected",
                message="Market data connection lost. Attempting reconnect.",
                severity=Severity.ERROR,
                source="health_monitor",
                ts=datetime.now(timezone.utc),
            ))
        elif not prev and connected:
            self._emit(Alert(
                title="WebSocket reconnected",
                message="Market data connection restored.",
                severity=Severity.INFO,
                source="health_monitor",
                ts=datetime.now(timezone.utc),
            ))

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._cfg.check_interval_sec * 2)

    # ── Internal check loop ───────────────────────────────────

    def _run_loop(self) -> None:
        while self._running:
            time.sleep(self._cfg.check_interval_sec)
            if self._running:
                self._run_checks()

    def _run_checks(self) -> None:
        with self._lock:
            status = HealthStatus(
                last_market_ts=self._status.last_market_ts,
                last_balance=self._status.last_balance,
                peak_equity=self._status.peak_equity,
                current_equity=self._status.current_equity,
                is_connected=self._status.is_connected,
                last_check_ts=time.monotonic(),
            )

        ts_now = datetime.now(timezone.utc)

        # Check 1: Data freshness
        age = status.data_age_sec
        if age is not None and age > self._cfg.stale_data_sec:
            self._emit(Alert(
                title="Stale market data",
                message=f"No market data for {age:.0f}s (threshold: {self._cfg.stale_data_sec}s)",
                severity=Severity.WARNING,
                source="health_monitor",
                ts=ts_now,
                meta={"age_sec": round(age, 1)},
            ))

        # Check 2: Low balance
        if status.last_balance is not None and status.last_balance < self._cfg.min_balance_usdt:
            self._emit(Alert(
                title="Low balance",
                message=f"Balance {status.last_balance} USDT below threshold {self._cfg.min_balance_usdt}",
                severity=Severity.WARNING,
                source="health_monitor",
                ts=ts_now,
                meta={"balance": str(status.last_balance), "threshold": str(self._cfg.min_balance_usdt)},
            ))

        # Check 3: Drawdown
        dd = status.drawdown_pct
        if dd is not None:
            if dd >= self._cfg.drawdown_critical_pct:
                self._emit(Alert(
                    title="Critical drawdown",
                    message=f"Drawdown {dd:.1f}% exceeds critical threshold {self._cfg.drawdown_critical_pct}%",
                    severity=Severity.CRITICAL,
                    source="health_monitor",
                    ts=ts_now,
                    meta={"drawdown_pct": round(dd, 2), "peak": str(status.peak_equity), "current": str(status.current_equity)},
                ))
            elif dd >= self._cfg.drawdown_warning_pct:
                self._emit(Alert(
                    title="Elevated drawdown",
                    message=f"Drawdown {dd:.1f}% exceeds warning threshold {self._cfg.drawdown_warning_pct}%",
                    severity=Severity.WARNING,
                    source="health_monitor",
                    ts=ts_now,
                    meta={"drawdown_pct": round(dd, 2)},
                ))

        if self._on_status is not None:
            self._on_status(status)

    def _emit(self, alert: Alert) -> None:
        try:
            self._sink.emit(alert)
        except Exception:
            pass

    def get_status(self) -> HealthStatus:
        with self._lock:
            return HealthStatus(
                last_market_ts=self._status.last_market_ts,
                last_balance=self._status.last_balance,
                peak_equity=self._status.peak_equity,
                current_equity=self._status.current_equity,
                is_connected=self._status.is_connected,
                last_check_ts=self._status.last_check_ts,
            )
