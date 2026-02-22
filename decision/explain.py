# decision/explain.py
"""Decision explanation — human-readable decision trace."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DecisionExplanation:
    """决策解释 — 人类可读的决策痕迹。"""
    symbol: str
    action: str                    # buy / sell / hold / flat
    reason_codes: tuple[str, ...]  # compact reason tags
    signal_details: Mapping[str, Any] = field(default_factory=dict)
    risk_check_passed: bool = True
    risk_details: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        reasons = ", ".join(self.reason_codes) if self.reason_codes else "none"
        risk_status = "passed" if self.risk_check_passed else "BLOCKED"
        return f"{self.symbol}: {self.action} (reasons: {reasons}, risk: {risk_status})"


def explain_decision(
    symbol: str,
    action: str,
    signal_results: Mapping[str, Any] | None = None,
    risk_results: Mapping[str, Any] | None = None,
    reason_codes: Sequence[str] = (),
) -> DecisionExplanation:
    """从决策组件的输出构建解释。"""
    risk_passed = True
    if risk_results:
        risk_passed = risk_results.get("allowed", True)
    return DecisionExplanation(
        symbol=symbol,
        action=action,
        reason_codes=tuple(reason_codes),
        signal_details=dict(signal_results or {}),
        risk_check_passed=risk_passed,
        risk_details=dict(risk_results or {}),
    )
