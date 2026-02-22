# context/market/account/account_state.py
"""Market-level account state — per-venue account tracking."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from context.market.account.account_snapshot import MarketAccountSnapshot


class MarketAccountState:
    """交易所级别的账户状态追踪。"""

    def __init__(self, *, venue: str, initial_balance: Decimal = Decimal("0")) -> None:
        self._venue = venue
        self._balance = initial_balance
        self._equity = initial_balance
        self._margin_used = Decimal("0")
        self._unrealized_pnl = Decimal("0")

    def update_balance(self, balance: Decimal) -> None:
        self._balance = balance
        self._equity = balance + self._unrealized_pnl

    def update_margin(self, used: Decimal) -> None:
        self._margin_used = used

    def update_unrealized_pnl(self, pnl: Decimal) -> None:
        self._unrealized_pnl = pnl
        self._equity = self._balance + pnl

    def snapshot(self) -> MarketAccountSnapshot:
        return MarketAccountSnapshot(
            venue=self._venue,
            balance=self._balance,
            equity=self._equity,
            margin_used=self._margin_used,
            margin_available=self._equity - self._margin_used,
            unrealized_pnl=self._unrealized_pnl,
        )
