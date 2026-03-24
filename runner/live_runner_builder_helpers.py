"""Sub-builder helpers for LiveRunner.build().

Extracted from live_runner.py to keep it under 500 lines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _find_module_attr(decision_bridge: Any, attr: str) -> Any:
    """Walk DecisionBridge.modules to find first module carrying `attr`."""
    if decision_bridge is None:
        return None
    for mod in getattr(decision_bridge, 'modules', []):
        val = getattr(mod, attr, None)
        if val is not None:
            return val
        inner = getattr(mod, 'inner', None)
        if inner is not None:
            val = getattr(inner, attr, None)
            if val is not None:
                return val
    return None


@dataclass
class _SubsystemReport:
    """Structured startup logging: track which subsystems succeeded/failed."""
    succeeded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    def record(self, name: str, ok: bool, error: str = "") -> None:
        if ok:
            self.succeeded.append(name)
        else:
            self.failed[name] = error

    def log_summary(self) -> None:
        if self.succeeded:
            logger.info(
                "Subsystems OK (%d): %s", len(self.succeeded), ", ".join(self.succeeded),
            )
        if self.failed:
            logger.warning(
                "Subsystems FAILED (%d): %s",
                len(self.failed),
                "; ".join(f"{k}: {v}" for k, v in self.failed.items()),
            )
