"""Replay verifier — validates replay results for margin, balance, and determinism."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class Violation:
    timestamp: float
    message: str
    severity: str = "error"


@dataclass
class VerificationResult:
    passed: bool
    violations: List[Violation] = field(default_factory=list)


class ReplayVerifier:
    """Validates replay results for correctness and safety."""

    def verify_margin_never_exceeded(
        self,
        snapshots: Sequence[Dict[str, Any]],
        max_leverage: float = 10.0,
    ) -> VerificationResult:
        """Check that no snapshot exceeds max leverage."""
        violations = []
        for snap in snapshots:
            balance = snap.get("balance", 0)
            notional = abs(snap.get("position_notional", 0))
            if balance > 0 and notional / balance > max_leverage:
                violations.append(Violation(
                    timestamp=snap.get("ts", 0),
                    message=f"Leverage {notional/balance:.1f}x exceeds {max_leverage}x",
                ))
        return VerificationResult(passed=len(violations) == 0, violations=violations)

    def verify_fills_respect_balance(
        self,
        fills: Sequence[Dict[str, Any]],
        snapshots: Sequence[Dict[str, Any]],
    ) -> VerificationResult:
        """Check fills don't occur when balance is negative."""
        violations = []
        for fill in fills:
            ts = fill.get("ts", 0)
            # Find closest preceding snapshot
            prev_balance = None
            for snap in snapshots:
                if snap.get("ts", 0) <= ts:
                    prev_balance = snap.get("balance", 0)
            if prev_balance is not None and prev_balance < 0:
                violations.append(Violation(
                    timestamp=ts,
                    message=f"Fill when balance negative: {prev_balance:.2f}",
                ))
        return VerificationResult(passed=len(violations) == 0, violations=violations)

    def verify_deterministic(
        self,
        result_a: Any,
        result_b: Any,
    ) -> bool:
        """Check two replay results are identical."""
        if result_a.events_processed != result_b.events_processed:
            return False
        if len(result_a.order_log) != len(result_b.order_log):
            return False
        for a, b in zip(result_a.order_log, result_b.order_log):
            if a.get("symbol") != b.get("symbol"):
                return False
            if a.get("side") != b.get("side"):
                return False
            if a.get("qty") != b.get("qty"):
                return False
        return True
