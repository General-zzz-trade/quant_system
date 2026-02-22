from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Optional, Tuple

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state._util import ensure_utc


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Portfolio aggregation (Route B).

    This is *derived* state: it summarizes account + positions + market into
    portfolio-level metrics used by risk/strategy.

    Design goals:
    - Deterministic: same inputs -> same outputs.
    - Auditable: fields are plain facts/derived facts.
    - Conservative types: Decimal + UTC timestamps.
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

    @staticmethod
    def compute(
        *,
        account: AccountState,
        positions: Mapping[str, PositionState],
        market: MarketState,
        ts: Optional[datetime],
    ) -> "PortfolioState":
        # Determine MTM per symbol (v1 is single-symbol, but keep generic)
        gross = Decimal("0")
        net = Decimal("0")
        unreal = Decimal("0")
        syms = sorted(positions.keys())

        for sym in syms:
            pos = positions[sym]
            qty = pos.qty
            if qty == 0:
                continue

            # mark price preference: market for primary symbol, otherwise pos.last_price
            mark = None
            if sym == market.symbol and market.last_price is not None:
                mark = market.last_price
            elif pos.last_price is not None:
                mark = pos.last_price

            if mark is None:
                # If we can't MTM, treat as zero exposure in derived metrics.
                continue

            notional = abs(qty) * mark
            gross += notional
            net += qty * mark

            if pos.avg_price is not None:
                unreal += (mark - pos.avg_price) * qty

        total_equity = account.balance + unreal
        lev: Optional[Decimal] = None
        if total_equity > 0:
            lev = (gross / total_equity) if gross != 0 else Decimal("0")

        mr: Optional[Decimal] = None
        if account.margin_used > 0:
            mr = total_equity / account.margin_used

        return PortfolioState(
            total_equity=total_equity,
            cash_balance=account.balance,
            realized_pnl=account.realized_pnl,
            unrealized_pnl=unreal,
            fees_paid=account.fees_paid,
            gross_exposure=gross,
            net_exposure=net,
            leverage=lev,
            margin_used=account.margin_used,
            margin_available=account.margin_available,
            margin_ratio=mr,
            symbols=tuple(syms),
            last_ts=ensure_utc(ts) if ts is not None else None,
        )
