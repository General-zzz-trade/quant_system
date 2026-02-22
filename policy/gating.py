# policy/gating.py
"""Trading policy gating — pre-trade checks and circuit breakers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class GateResult(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    THROTTLE = "throttle"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """门控决策。"""
    result: GateResult
    reasons: tuple[str, ...] = ()

    @staticmethod
    def allow() -> GateDecision:
        return GateDecision(result=GateResult.ALLOW)

    @staticmethod
    def block(*reasons: str) -> GateDecision:
        return GateDecision(result=GateResult.BLOCK, reasons=reasons)


class TradingGate:
    """交易门控 — 组合多个检查条件。"""

    def __init__(self) -> None:
        self._halted: bool = False
        self._block_reasons: list[str] = []

    def halt(self, reason: str = "manual halt") -> None:
        self._halted = True
        self._block_reasons.append(reason)

    def resume(self) -> None:
        self._halted = False
        self._block_reasons.clear()

    def check(self) -> GateDecision:
        if self._halted:
            return GateDecision.block(*self._block_reasons)
        return GateDecision.allow()

    @property
    def is_halted(self) -> bool:
        return self._halted
