# context/market/account/account_snapshot.py
"""Account snapshot within market context — read-only view."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Optional


@dataclass(frozen=True, slots=True)
class MarketAccountSnapshot:
    """市场层级的账户快照（区别于 context/account_state.py 的顶级快照）。"""
    venue: str
    balance: Decimal
    equity: Decimal
    margin_used: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
