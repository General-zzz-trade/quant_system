"""Compatibility shell: the stop manager is trading.application.execution.stops (migration step 9b).

The application class takes its clock by injection; this subclass keeps ``clock=time.time`` as the
default for the legacy callers (``runner.binance_target``, ``runner.binance_main``, tests and probes).
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from trading.application.execution.stops import ACTIVE, MAX_STOP_PCT, MIN_STOP_PCT, TRIGGERED
from trading.application.execution.stops import ProtectiveStopManager as _ProtectiveStopManager


class ProtectiveStopManager(_ProtectiveStopManager):
    def __init__(self, adapter: Any, executor: Any, symbol: str, *, stop_pct: Decimal,
                 clock: Any = time.time) -> None:
        super().__init__(adapter, executor, symbol, stop_pct=stop_pct, clock=clock)


__all__ = ["ACTIVE", "MAX_STOP_PCT", "MIN_STOP_PCT", "ProtectiveStopManager", "TRIGGERED"]
