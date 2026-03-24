"""Continuous drawdown circuit breaker — Rust-delegated computation.

Monitors equity on every pipeline tick. Triggers reduce-only or hard-kill
on the KillSwitch when drawdown thresholds are breached.

Velocity detection: if equity drops >5% in 15 minutes, triggers immediate
hard-kill to protect against flash crashes or fat-finger errors.

Core computation delegated to RustDrawdownBreaker; Python layer bridges
Rust actions to KillSwitch calls and logging.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from _quant_hotpath import RustDrawdownBreaker as _RustDrawdownBreaker

from risk.kill_switch import KillMode, KillScope, KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class DrawdownBreakerConfig:
    """Thresholds for the drawdown circuit breaker."""
    warning_pct: float = 10.0       # 10% drawdown -> warning log
    reduce_pct: float = 15.0        # 15% drawdown -> reduce-only mode
    kill_pct: float = 20.0          # 20% drawdown -> hard kill
    velocity_pct: float = 5.0       # 5% drop in velocity_window -> hard kill
    velocity_window_sec: float = 900.0  # 15 minutes
    reduce_ttl_sec: int = 3600      # reduce-only mode lasts 1 hour by default


class DrawdownCircuitBreaker:
    """Continuous equity drawdown monitor — Rust-delegated.

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
        self._last_warning_ts: float = 0.0
        self._inner = _RustDrawdownBreaker(
            warning_pct=self._config.warning_pct,
            reduce_pct=self._config.reduce_pct,
            kill_pct=self._config.kill_pct,
            velocity_pct=self._config.velocity_pct,
            velocity_window_sec=self._config.velocity_window_sec,
            reduce_ttl_sec=self._config.reduce_ttl_sec,
        )

    @property
    def state(self) -> str:
        return self._inner.state

    @property
    def current_drawdown_pct(self) -> float:
        return self._inner.current_drawdown_pct

    @property
    def equity_hwm(self) -> float:
        return self._inner.equity_hwm

    def on_equity_update(self, equity: float, now_ts: float | None = None) -> str:
        """Process an equity update. Returns the current state."""
        now = now_ts if now_ts is not None else time.time()
        prev_state = self._inner.state
        new_state, action = self._inner.on_equity_update(equity, now)
        self._handle_action(new_state, action, equity, now, prev_state)
        return new_state

    def _handle_action(
        self, new_state: str, action: Any, equity: float, now: float, prev_state: str = "normal"
    ) -> None:
        """Bridge Rust action tuples to KillSwitch calls."""
        if action is None:
            if new_state == "warning":
                if now - self._last_warning_ts > 300:
                    self._last_warning_ts = now
                    cfg = self._config
                    logger.warning(
                        "DrawdownBreaker WARNING: dd=%.1f%% >= %.1f%% (hwm=%.2f, current=%.2f)",
                        self._inner.current_drawdown_pct, cfg.warning_pct, self._inner.equity_hwm, equity,
                    )
            elif new_state == "normal" and prev_state == "warning":
                logger.info(
                    "DrawdownBreaker: drawdown recovered to %.1f%%, state -> normal",
                    self._inner.current_drawdown_pct,
                )
            return
        mode, reason = action
        if mode == "reduce_only":
            self._kill_switch.trigger(
                scope=KillScope.GLOBAL, key="*", mode=KillMode.REDUCE_ONLY,
                reason=f"drawdown_breaker: {reason}", source="drawdown_breaker",
                ttl_seconds=self._config.reduce_ttl_sec, now_ts=now,
            )
            logger.error(
                "DrawdownBreaker REDUCE_ONLY: %s (hwm=%.2f, dd=%.1f%%)",
                reason, self._inner.equity_hwm, self._inner.current_drawdown_pct,
            )
            self._send_alert(f"REDUCE_ONLY: {reason}", severity="error")
        elif mode == "hard_kill":
            self._kill_switch.trigger(
                scope=KillScope.GLOBAL, key="*", mode=KillMode.HARD_KILL,
                reason=f"drawdown_breaker: {reason}", source="drawdown_breaker", now_ts=now,
            )
            logger.critical(
                "DrawdownBreaker HARD_KILL: %s (hwm=%.2f, dd=%.1f%%)",
                reason, self._inner.equity_hwm, self._inner.current_drawdown_pct,
            )
            self._send_alert(f"HARD_KILL: {reason}", severity="critical")
        elif mode == "clear":
            self._kill_switch.clear(scope=KillScope.GLOBAL, key="*")

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

    def checkpoint(self) -> dict:
        """Return state for persistence."""
        return self._inner.checkpoint()

    def restore_checkpoint(self, data: dict) -> None:
        """Restore state from checkpoint."""
        self._inner.restore_checkpoint(data)

    def reset(self, new_hwm: float | None = None) -> None:
        """Reset the breaker state. Use after manual intervention."""
        _, action = self._inner.reset(new_hwm)
        if action:
            mode, _ = action
            if mode == "clear":
                self._kill_switch.clear(scope=KillScope.GLOBAL, key="*")
        logger.info("DrawdownBreaker reset (hwm=%.2f)", self._inner.equity_hwm)

    def get_status(self) -> dict:
        """Return current status for health endpoint."""
        return self._inner.get_status()
