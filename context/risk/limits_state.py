# context/risk/limits_state.py
"""Risk limits state — track limit usage."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class LimitUsage:
    """单个限制的使用情况。"""
    limit_name: str
    current: Decimal
    maximum: Decimal
    utilization_pct: Decimal

    @property
    def breached(self) -> bool:
        return self.current > self.maximum


class LimitsState:
    """风险限额追踪。"""

    def __init__(self) -> None:
        self._limits: Dict[str, dict] = {}

    def set_limit(self, name: str, maximum: Decimal) -> None:
        if name not in self._limits:
            self._limits[name] = {"current": Decimal("0"), "max": maximum}
        else:
            self._limits[name]["max"] = maximum

    def update_usage(self, name: str, current: Decimal) -> None:
        if name not in self._limits:
            self._limits[name] = {"current": current, "max": Decimal("0")}
        else:
            self._limits[name]["current"] = current

    def get(self, name: str) -> Optional[LimitUsage]:
        entry = self._limits.get(name)
        if entry is None:
            return None
        mx = entry["max"]
        cur = entry["current"]
        pct = cur / mx if mx > 0 else Decimal("0")
        return LimitUsage(limit_name=name, current=cur, maximum=mx, utilization_pct=pct)

    @property
    def breached_limits(self) -> list[str]:
        return [
            name for name, e in self._limits.items()
            if e["max"] > 0 and e["current"] > e["max"]
        ]
