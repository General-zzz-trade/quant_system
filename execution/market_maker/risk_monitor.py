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
        self._pause_trigger_losses: int | None = None

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

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def check(self, daily_pnl: float, consecutive_losses: int) -> str:
        """Evaluate risk state. Returns current state after checks."""
        if self._state == "killed":
            return self._state

        # Daily loss limit → hard kill
        if daily_pnl <= -self._cfg.daily_loss_limit:
            if self._state != "killed":
                self._state = "killed"
                self._kill_reason = f"daily_loss={daily_pnl:.2f}"
                log.error("KILL SWITCH: %s", self._kill_reason)
            return self._state

        current_state = self.state

        # If loss streak cleared, fully re-arm the breaker.
        if consecutive_losses < self._cfg.circuit_breaker_losses:
            self._pause_trigger_losses = None
            if current_state == "paused":
                self._state = "running"
            return self.state

        # Circuit breaker: consecutive losses → temporary pause.
        # After cooldown expires, allow quoting to resume until a *new* loss
        # changes the streak again; otherwise the runner would stay paused forever.
        if current_state == "running":
            if (
                self._pause_trigger_losses is None
                or consecutive_losses != self._pause_trigger_losses
            ):
                self._state = "paused"
                self._pause_until = time.monotonic() + self._cfg.circuit_breaker_pause_s
                self._pause_trigger_losses = consecutive_losses
                log.warning(
                    "Circuit breaker: %d consecutive losses, pausing %.0fs",
                    consecutive_losses,
                    self._cfg.circuit_breaker_pause_s,
                )
                return self._state

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
        self._pause_trigger_losses = None
        log.info("Risk monitor reset")
