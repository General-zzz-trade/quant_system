"""Continuous drawdown circuit breaker — monitors equity on every pipeline tick.

Unlike max_drawdown.py (which checks only at order time), this breaker runs
continuously and can trigger reduce-only or hard-kill on the KillSwitch
when drawdown thresholds are breached.

Velocity detection: if equity drops >5% in 15 minutes, triggers immediate hard-kill
to protect against flash crashes or fat-finger errors.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional, Tuple

from risk.kill_switch import KillMode, KillScope, KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class DrawdownBreakerConfig:
    """Thresholds for the drawdown circuit breaker."""
    warning_pct: float = 10.0       # 10% drawdown → warning log
    reduce_pct: float = 15.0        # 15% drawdown → reduce-only mode
    kill_pct: float = 20.0          # 20% drawdown → hard kill
    velocity_pct: float = 5.0       # 5% drop in velocity_window → hard kill
    velocity_window_sec: float = 900.0  # 15 minutes
    reduce_ttl_sec: int = 3600      # reduce-only mode lasts 1 hour by default


class DrawdownCircuitBreaker:
    """Continuous equity drawdown monitor.

    Call `on_equity_update(equity)` on every pipeline tick.
    Integrates with KillSwitch for automatic trade halting.
    """

    def __init__(
        self,
        kill_switch: KillSwitch,
        config: DrawdownBreakerConfig | None = None,
        alert_manager: Any = None,
    ) -> None:
        self._kill_switch = kill_switch
        self._config = config or DrawdownBreakerConfig()
        self._alert_manager = alert_manager
        self._equity_hwm: float = 0.0
        self._current_dd_pct: float = 0.0
        self._state: str = "normal"  # normal | warning | reduce_only | killed
        # Velocity tracking: (timestamp, equity) pairs
        self._equity_history: Deque[Tuple[float, float]] = deque(maxlen=1000)
        self._last_warning_ts: float = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_drawdown_pct(self) -> float:
        return self._current_dd_pct

    @property
    def equity_hwm(self) -> float:
        return self._equity_hwm

    def on_equity_update(self, equity: float, now_ts: float | None = None) -> str:
        """Process an equity update. Returns the current state.

        States: "normal", "warning", "reduce_only", "killed"
        """
        if equity <= 0:
            return self._state

        now = now_ts if now_ts is not None else time.time()
        cfg = self._config

        # Update HWM
        if equity > self._equity_hwm:
            self._equity_hwm = equity

        # Calculate drawdown
        if self._equity_hwm > 0:
            self._current_dd_pct = (self._equity_hwm - equity) / self._equity_hwm * 100.0
        else:
            self._current_dd_pct = 0.0

        # Record for velocity check
        self._equity_history.append((now, equity))

        # ── Velocity check: rapid drop detection ──
        if self._check_velocity(equity, now):
            self._trigger_kill(
                reason=f"velocity_breach: >{cfg.velocity_pct:.1f}% drop in {cfg.velocity_window_sec:.0f}s",
                now_ts=now,
            )
            return self._state

        # ── Threshold checks (escalating) ──
        if self._current_dd_pct >= cfg.kill_pct:
            if self._state != "killed":
                self._trigger_kill(
                    reason=f"drawdown {self._current_dd_pct:.1f}% >= kill threshold {cfg.kill_pct:.1f}%",
                    now_ts=now,
                )
        elif self._current_dd_pct >= cfg.reduce_pct:
            if self._state not in ("reduce_only", "killed"):
                self._trigger_reduce_only(
                    reason=f"drawdown {self._current_dd_pct:.1f}% >= reduce threshold {cfg.reduce_pct:.1f}%",
                    now_ts=now,
                )
        elif self._current_dd_pct >= cfg.warning_pct:
            if self._state == "normal":
                self._state = "warning"
                if now - self._last_warning_ts > 300:  # Don't spam warnings
                    self._last_warning_ts = now
                    logger.warning(
                        "DrawdownBreaker WARNING: dd=%.1f%% >= %.1f%% (hwm=%.2f, current=%.2f)",
                        self._current_dd_pct, cfg.warning_pct, self._equity_hwm, equity,
                    )
        else:
            # Drawdown recovered below warning threshold
            if self._state == "warning":
                self._state = "normal"
                logger.info(
                    "DrawdownBreaker: drawdown recovered to %.1f%%, state → normal",
                    self._current_dd_pct,
                )

        return self._state

    def _check_velocity(self, current_equity: float, now: float) -> bool:
        """Check if equity dropped too fast (velocity breach)."""
        cfg = self._config
        cutoff = now - cfg.velocity_window_sec

        # Find the equity at the start of the velocity window
        for ts, eq in self._equity_history:
            if ts >= cutoff:
                if eq > 0:
                    drop_pct = (eq - current_equity) / eq * 100.0
                    if drop_pct >= cfg.velocity_pct:
                        return True
                break

        return False

    def _trigger_reduce_only(self, reason: str, now_ts: float) -> None:
        """Enter reduce-only mode via KillSwitch."""
        self._state = "reduce_only"
        self._kill_switch.trigger(
            scope=KillScope.GLOBAL,
            key="*",
            mode=KillMode.REDUCE_ONLY,
            reason=f"drawdown_breaker: {reason}",
            source="drawdown_breaker",
            ttl_seconds=self._config.reduce_ttl_sec,
            now_ts=now_ts,
        )
        logger.error(
            "DrawdownBreaker REDUCE_ONLY: %s (hwm=%.2f, dd=%.1f%%)",
            reason, self._equity_hwm, self._current_dd_pct,
        )
        self._send_alert(f"REDUCE_ONLY: {reason}", severity="error")

    def _trigger_kill(self, reason: str, now_ts: float) -> None:
        """Enter hard-kill mode via KillSwitch."""
        self._state = "killed"
        self._kill_switch.trigger(
            scope=KillScope.GLOBAL,
            key="*",
            mode=KillMode.HARD_KILL,
            reason=f"drawdown_breaker: {reason}",
            source="drawdown_breaker",
            now_ts=now_ts,
        )
        logger.critical(
            "DrawdownBreaker HARD_KILL: %s (hwm=%.2f, dd=%.1f%%)",
            reason, self._equity_hwm, self._current_dd_pct,
        )
        self._send_alert(f"HARD_KILL: {reason}", severity="critical")

    def _send_alert(self, message: str, severity: str = "error") -> None:
        """Send alert via alert manager if available."""
        if self._alert_manager is not None:
            try:
                from monitoring.alerts.base import Alert, Severity
                sev = Severity.CRITICAL if severity == "critical" else Severity.ERROR
                self._alert_manager.fire(Alert(
                    rule_name="drawdown_breaker",
                    severity=sev,
                    message=f"DrawdownBreaker: {message}",
                ))
            except Exception:
                logger.warning("DrawdownBreaker alert send failed", exc_info=True)

    def reset(self, new_hwm: float | None = None) -> None:
        """Reset the breaker state. Use after manual intervention."""
        self._state = "normal"
        self._current_dd_pct = 0.0
        if new_hwm is not None:
            self._equity_hwm = new_hwm
        self._equity_history.clear()
        # Clear any kill switch records from drawdown_breaker
        self._kill_switch.clear(scope=KillScope.GLOBAL, key="*")
        logger.info("DrawdownBreaker reset (hwm=%.2f)", self._equity_hwm)

    def get_status(self) -> dict:
        """Return current status for health endpoint."""
        return {
            "state": self._state,
            "drawdown_pct": round(self._current_dd_pct, 2),
            "equity_hwm": round(self._equity_hwm, 2),
            "thresholds": {
                "warning_pct": self._config.warning_pct,
                "reduce_pct": self._config.reduce_pct,
                "kill_pct": self._config.kill_pct,
                "velocity_pct": self._config.velocity_pct,
                "velocity_window_sec": self._config.velocity_window_sec,
            },
        }
