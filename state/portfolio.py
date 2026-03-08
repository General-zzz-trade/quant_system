from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Portfolio aggregation — derived state from account + positions + market.

    Computation handled by RustPortfolioReducer (state_reducers.rs).
    This dataclass is the Python-side container for the result.
    """

    total_equity: Decimal
    cash_balance: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal

    gross_exposure: Decimal
    net_exposure: Decimal
    leverage: Optional[Decimal]

    margin_used: Decimal
    margin_available: Decimal
    margin_ratio: Optional[Decimal]

    symbols: Tuple[str, ...] = ()
    last_ts: Optional[datetime] = None
