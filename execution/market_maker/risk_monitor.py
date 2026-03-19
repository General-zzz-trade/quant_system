"""Risk monitoring and circuit breaker for market maker."""

from __future__ import annotations

import logging
import time

from .config import MarketMakerConfig

log = logging.getLogger(__name__)


class RiskMonitor:
    """Enforces PnL limits, circuit breaker, and stale order cleanup.

    States:
        running  – normal quoting
        paused   – circuit breaker active, resume after cooldown
        killed   – daily loss limit hit, no recovery until manual reset
    """

    def __init__(self, cfg: MarketMakerConfig) -> None:
        self._cfg = cfg
        self._state: str = "running"
        self._pause_until: float = 0.0
        self._kill_reason: str = ""

    @property
    def state(self) -> str:
        if self._state == "paused" and time.monotonic() >= self._pause_until:
            log.info("Circuit breaker cooldown expired, resuming")
            self._state = "running"
        return self._state

    @property
    def can_quote(self) -> bool:
        return self.state == "running"

    @property
    def is_killed(self) -> bool:
        return self._state == "killed"

    def check(self, daily_pnl: float, consecutive_losses: int) -> str:
        """Evaluate risk state. Returns current state after checks."""
        # Daily loss limit → hard kill
        if daily_pnl <= -self._cfg.daily_loss_limit:
            if self._state != "killed":
                self._state = "killed"
                self._kill_reason = f"daily_loss={daily_pnl:.2f}"
                log.error("KILL SWITCH: %s", self._kill_reason)
            return self._state

        # Circuit breaker: consecutive losses → temporary pause
        if consecutive_losses >= self._cfg.circuit_breaker_losses:
            if self._state == "running":
                self._state = "paused"
                self._pause_until = time.monotonic() + self._cfg.circuit_breaker_pause_s
                log.warning(
                    "Circuit breaker: %d consecutive losses, pausing %.0fs",
                    consecutive_losses,
                    self._cfg.circuit_breaker_pause_s,
                )
            return self._state

        # If we were paused and losses reset, resume
        if self._state == "paused" and consecutive_losses < self._cfg.circuit_breaker_losses:
            self._state = "running"

        return self.state

    def force_kill(self, reason: str) -> None:
        """External kill (e.g. SIGINT handler)."""
        self._state = "killed"
        self._kill_reason = reason
        log.error("FORCE KILL: %s", reason)

    def reset(self) -> None:
        """Manual reset (e.g. new trading day)."""
        self._state = "running"
        self._pause_until = 0.0
        self._kill_reason = ""
        log.info("Risk monitor reset")
