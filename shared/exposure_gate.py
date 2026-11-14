"""Compatibility shell: the cap is trading.domain.risk.exposure, the check trading.application.risk.exposure_gate."""
from __future__ import annotations

from trading.application.risk.exposure_gate import AccountExposureGate
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE

__all__ = ["AccountExposureGate", "MAX_TOTAL_LEVERAGE"]
