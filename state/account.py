from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from state._util import ensure_utc


@dataclass(frozen=True, slots=True)
class AccountState:
    """Account facts (SSOT).

    v1 minimal accounting:
    - balance (cash/equity base)
    - margin used / available (if provided by events)
    - realized/unrealized pnl (unrealized may be updated later in Portfolio layer)
    - fees paid
    """

    currency: str
    balance: Decimal

    margin_used: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("0")

    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")

    last_ts: Optional[datetime] = None

    @classmethod
    def initial(cls, *, currency: str, balance: Decimal) -> "AccountState":
        return cls(
            currency=currency,
            balance=balance,
            margin_used=Decimal("0"),
            margin_available=balance,
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            last_ts=None,
        )

    def with_update(
        self,
        *,
        balance: Decimal,
        margin_used: Decimal,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        fees_paid: Decimal,
        ts: Optional[datetime],
    ) -> "AccountState":
        margin_available = balance - margin_used
        return AccountState(
            currency=self.currency,
            balance=balance,
            margin_used=margin_used,
            margin_available=margin_available,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            fees_paid=fees_paid,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )
