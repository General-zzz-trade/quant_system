"""SLO/SLI definitions and real-time tracking for the trading system."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from monitoring.alerts.base import Alert, AlertSink, Severity


@dataclass(frozen=True, slots=True)
class SLOConfig:
    """Service Level Objective definitions."""
    # Latency: pipeline P99 must be below this (seconds)
    pipeline_latency_p99_sec: float = 5.0
    # Availability: uptime target (fraction, e.g. 0.999 = 99.9%)
    availability_target: float = 0.999
    # Data freshness: market data must arrive within this interval (seconds)
    data_freshness_sec: float = 30.0
    # Order fill rate: fraction of orders that must be filled
    fill_rate_target: float = 0.95
    # Error budget window (seconds)
    window_sec: float = 3600.0
    # Check interval (seconds)
    check_interval_sec: float = 60.0


@dataclass
class SLISnapshot:
    """Point-in-time SLI measurements."""
    ts: datetime
    pipeline_latency_p99_ms: float = 0.0
    uptime_fraction: float = 1.0
    data_freshness_sec: float = 0.0
    fill_rate: float = 1.0
    total_orders: int = 0
    filled_orders: int = 0
    error_budget_remaining_pct: float = 100.0
    violations: tuple[str, ...] = ()


class SLOTracker:
    """Tracks SLIs against SLO targets and emits alerts on violations."""

    def __init__(
        self,
        config: Optional[SLOConfig] = None,
        sink: Optional[AlertSink] = None,
    ) -> None:
        self._cfg = config or SLOConfig()
        self._sink = sink
        self._lock = threading.Lock()

        max_samples = int(self._cfg.window_sec / 0.5) + 100
        self._latencies: deque[float] = deque(maxlen=max_samples)
        self._up_checks: deque[bool] = deque(maxlen=max_samples)
        self._orders_total = 0
        self._orders_filled = 0
        self._last_market_ts: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Recording methods ─────────────────────────────────────

    def record_latency(self, latency_sec: float) -> None:
        with self._lock:
            self._latencies.append(latency_sec)

    def record_uptime_check(self, is_up: bool) -> None:
        with self._lock:
            self._up_checks.append(is_up)

    def record_market_data(self) -> None:
        with self._lock:
            self._last_market_ts = time.monotonic()

    def record_order(self, filled: bool) -> None:
        with self._lock:
            self._orders_total += 1
            if filled:
                self._orders_filled += 1

    # ── Snapshot ──────────────────────────────────────────────

    def snapshot(self) -> SLISnapshot:
        with self._lock:
            latencies = sorted(self._latencies)
            p99 = latencies[int(len(latencies) * 0.99)] * 1000 if latencies else 0.0

            up_count = sum(1 for x in self._up_checks if x)
            total_checks = len(self._up_checks)
            uptime = up_count / total_checks if total_checks > 0 else 1.0

            freshness = 0.0
            if self._last_market_ts is not None:
                freshness = time.monotonic() - self._last_market_ts

            fill_rate = (
                self._orders_filled / self._orders_total
                if self._orders_total > 0
                else 1.0
            )

            # Error budget: fraction of allowed downtime remaining
            allowed_downtime = (1.0 - self._cfg.availability_target) * self._cfg.window_sec
            actual_downtime = (1.0 - uptime) * self._cfg.window_sec if total_checks > 0 else 0.0
            budget_remaining = max(0.0, (allowed_downtime - actual_downtime) / allowed_downtime * 100) if allowed_downtime > 0 else 100.0

            violations: list[str] = []
            if p99 > self._cfg.pipeline_latency_p99_sec * 1000:
                violations.append(f"latency_p99={p99:.0f}ms > {self._cfg.pipeline_latency_p99_sec * 1000:.0f}ms")
            if uptime < self._cfg.availability_target:
                violations.append(f"availability={uptime:.4f} < {self._cfg.availability_target}")
            if freshness > self._cfg.data_freshness_sec:
                violations.append(f"data_freshness={freshness:.0f}s > {self._cfg.data_freshness_sec}s")
            if fill_rate < self._cfg.fill_rate_target:
                violations.append(f"fill_rate={fill_rate:.3f} < {self._cfg.fill_rate_target}")

            return SLISnapshot(
                ts=datetime.now(timezone.utc),
                pipeline_latency_p99_ms=p99,
                uptime_fraction=uptime,
                data_freshness_sec=freshness,
                fill_rate=fill_rate,
                total_orders=self._orders_total,
                filled_orders=self._orders_filled,
                error_budget_remaining_pct=budget_remaining,
                violations=tuple(violations),
            )

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="slo-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._cfg.check_interval_sec * 2)

    def _run_loop(self) -> None:
        while self._running:
            time.sleep(self._cfg.check_interval_sec)
            if not self._running:
                break
            snap = self.snapshot()
            if snap.violations and self._sink is not None:
                self._sink.emit(Alert(
                    title="SLO violation",
                    message="; ".join(snap.violations),
                    severity=Severity.WARNING,
                    source="slo_tracker",
                    ts=snap.ts,
                    meta={"error_budget_remaining_pct": round(snap.error_budget_remaining_pct, 1)},
                ))
