"""Compatibility shell: the entry gate lives in trading.application.risk.entry_gate (migration step 4)."""
from __future__ import annotations

from trading.application.risk.entry_gate import CompositeRiskGate, GateResult

__all__ = ["CompositeRiskGate", "GateResult"]
