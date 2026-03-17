# execution/safety/kill_switch.py
"""Execution-layer kill switch — gates order submission at the execution boundary."""
from __future__ import annotations

from threading import RLock
from typing import Any, Optional, Tuple

from risk.kill_switch import KillRecord, KillScope, KillSwitch as RiskKillSwitch


class ExecutionKillSwitch:
    """
    执行层闸门 — 在 bridge 发送订单前做最后一道检查。
    复用 risk.kill_switch.KillSwitch 底层实现。
    """

    def __init__(self, risk_ks: Optional[RiskKillSwitch] = None) -> None:
        self._ks = risk_ks or RiskKillSwitch()
        self._lock = RLock()
        self._manual_block: bool = False
        self._manual_reason: str = ""

    def gate_order(
        self,
        *,
        symbol: str,
        strategy_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """返回 (allowed, reason)。"""
        with self._lock:
            if self._manual_block:
                return False, f"manual_block: {self._manual_reason}"

        allowed, rec = self._ks.allow_order(
            symbol=symbol, strategy_id=strategy_id, reduce_only=reduce_only,
        )
        if not allowed and rec is not None:
            return False, f"kill_switch: scope={rec.scope.value} mode={rec.mode.value} reason={rec.reason}"
        return True, None

    def block_all(self, reason: str = "manual") -> None:
        with self._lock:
            self._manual_block = True
            self._manual_reason = reason

    def unblock(self) -> None:
        with self._lock:
            self._manual_block = False
            self._manual_reason = ""

    @property
    def is_blocked(self) -> bool:
        with self._lock:
            return self._manual_block

    def trigger(self, *, scope: KillScope, key: str, **kwargs: Any) -> KillRecord:
        return self._ks.trigger(scope=scope, key=key, **kwargs)

    def clear(self, *, scope: KillScope, key: str) -> bool:
        return self._ks.clear(scope=scope, key=key)
