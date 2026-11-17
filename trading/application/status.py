"""The status side of a cycle (ADR-0013 point 4, S3c): one snapshot of the runtime for people and the watchdog.
Every field read from a store or the exchange is "unreadable (...)" instead of an exception: a lost state
database must not stop the cycle that reports it. Moved from runtime.py without changes in behaviour."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trading.application.equity import EquitySource
from trading.application.ports import Clock, DrawdownGuard, KillSwitch, Notifier
from trading.application.session import TargetSession
from trading.domain.bars import HOUR


class StatusReporter:
    def __init__(self, sessions: dict[str, TargetSession], *, mode: str, clock: Clock, spec_passed: object,
                 notifier: Notifier | None, kill: KillSwitch | None, drawdown: DrawdownGuard | None,
                 equity: EquitySource,
                 market_failures: Callable[[], int], pause_status: Callable[[str], str | None],
                 config_digest: str | None = None) -> None:
        self.sessions, self.mode, self.clock, self.spec_passed, self.notifier = sessions, mode, clock, spec_passed, \
            notifier
        self.kill: Any = kill
        self.drawdown: Any = drawdown
        self.equity = equity
        self._market_failures, self._pause_status = market_failures, pause_status
        self.config_digest = config_digest          # ADR-0017: which configuration this run uses

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "at": self.clock(), "spec_passed_gate": self.spec_passed,
                               "alerts": getattr(self.notifier, "channel", None),
                               "market_failures": self._market_failures(), "config_digest": self.config_digest,
                               "symbols": {}}
        if self.kill is not None:
            out["kill"] = readable(lambda: self.kill.state().__dict__)
            out["drawdown_latched"] = readable(self.drawdown.latched)
            out["equity"] = str(self.equity.value)
        for symbol, s in self.sessions.items():
            stops = s.stops
            out["symbols"][symbol] = {
                "last_bar_open": s.cursor, "target": s.last_target[0], "target_ready": s.last_target[1],
                "entry_ready": s.entry_ready, "entry_reason": s.entry_reason,
                "funding_interval_hours": s.funding_interval // HOUR if s.funding_interval else None,
                "state": s.state.value,
                "position": readable(lambda: str(s.position_qty())),
                "sleeve_equity": str(s.sleeve_equity()),
                "protective_stop": readable(stops.has_active_stop) if stops else None,
                "flatten_failures": s.flattener.failures if s.flattener else None,
                "entry_pause": readable(lambda: self._pause_status(symbol))}
        return out


def readable(read: Callable[[], Any]) -> Any:
    """A status field from a store: "unreadable" instead of an exception. A lost state database must not stop
    the cycle that reports it; reconciliation already blocks entries and raises the issue (ADR-0009)."""
    try:
        return read()
    except Exception as exc:
        return f"unreadable ({type(exc).__name__})"


__all__ = ["StatusReporter", "readable"]
