from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple

from state._util import ensure_utc


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Risk limits for Route B.

    These are deliberately conservative defaults for a single-symbol system.
    Tune per strategy/venue later.
    """

    max_leverage: Decimal = Decimal("5")
    max_position_notional: Optional[Decimal] = None  # e.g. Decimal("5000")
    max_drawdown_pct: Decimal = Decimal("0.30")  # 30% peak-to-trough
    block_on_equity_le_zero: bool = True


@dataclass(frozen=True, slots=True)
class RiskState:
    """Risk facts/flags (Route B).

    - blocked: hard stop for execution
    - halted: manual halt via CONTROL event
    - level/message: latest RISK event summary
    - flags: set of active limit flags
    - equity_peak/drawdown_pct: basic equity risk tracking
    """

    blocked: bool = False
    halted: bool = False

    level: Optional[str] = None
    message: Optional[str] = None
    flags: Tuple[str, ...] = ()

    equity_peak: Decimal = Decimal("0")
    drawdown_pct: Decimal = Decimal("0")

    last_ts: Optional[datetime] = None

    def with_update(
        self,
        *,
        blocked: bool,
        halted: bool,
        level: Optional[str],
        message: Optional[str],
        flags: Tuple[str, ...],
        equity_peak: Decimal,
        drawdown_pct: Decimal,
        ts: Optional[datetime],
    ) -> "RiskState":
        return RiskState(
            blocked=blocked,
            halted=halted,
            level=level,
            message=message,
            flags=flags,
            equity_peak=equity_peak,
            drawdown_pct=drawdown_pct,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )
