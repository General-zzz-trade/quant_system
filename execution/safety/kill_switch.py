# execution/safety/kill_switch.py
"""Execution-layer kill switch — gates order submission at the execution boundary.

Manual block delegated to RustKillSwitch for lock-free fast path.
Scope-based kill logic remains in risk.kill_switch.KillSwitch (Python).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from risk.kill_switch import KillRecord, KillScope, KillSwitch as RiskKillSwitch

from _quant_hotpath import RustKillSwitch as _RustKillSwitch


class ExecutionKillSwitch:
    """
    Rust-accelerated execution gate.

    Manual block uses RustKillSwitch (lock-free); scope-based kills delegate
    to risk.kill_switch.KillSwitch.
    """

    def __init__(self, risk_ks: Optional[RiskKillSwitch] = None) -> None:
        self._ks = risk_ks or RiskKillSwitch()
        self._rust = _RustKillSwitch()
        self._manual_reason: str = ""

    def gate_order(
        self,
        *,
        symbol: str,
        strategy_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Return (allowed, reason)."""
        # Fast path: Rust manual block check
        if self._rust.is_armed():
            return False, f"manual_block: {self._manual_reason}"

        allowed, rec = self._ks.allow_order(
            symbol=symbol, strategy_id=strategy_id, reduce_only=reduce_only,
        )
        if not allowed and rec is not None:
            return False, f"kill_switch: scope={rec.scope.value} mode={rec.mode.value} reason={rec.reason}"
        return True, None

    def block_all(self, reason: str = "manual") -> None:
        self._manual_reason = reason
        self._rust.arm("GLOBAL", "*", "hard_kill", reason)

    def unblock(self) -> None:
        self._manual_reason = ""
        self._rust.disarm("GLOBAL", "*")

    @property
    def is_blocked(self) -> bool:
        return bool(self._rust.is_armed())

    def trigger(self, *, scope: KillScope, key: str, **kwargs: Any) -> KillRecord:
        return self._ks.trigger(scope=scope, key=key, **kwargs)

    def clear(self, *, scope: KillScope, key: str) -> bool:
        return self._ks.clear(scope=scope, key=key)
