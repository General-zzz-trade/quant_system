"""Account gross exposure check against the hard cap in trading.domain.risk.exposure.

Every check reads a fresh account snapshot. Gross exposure is the sum of |position qty| x mark over all
symbols, plus the notional of unresolved entry intents in the order journal, plus the proposed order.
Any error, stale or non-positive equity blocks the entry; reductions never reach this check.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from trading.application.ports import ExchangeAccount, OrderJournal
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE, exposure_breach, validated_leverage


class AccountExposureGate:
    def __init__(self, adapter: ExchangeAccount, journal: OrderJournal | None = None, *,
                 max_leverage: Decimal = MAX_TOTAL_LEVERAGE,
                 prices: Callable[[str], Decimal] | None = None) -> None:
        self.adapter, self.journal = adapter, journal
        self.max_leverage = validated_leverage(max_leverage)
        self.prices = prices or (lambda symbol: Decimal(str(adapter.get_ticker(symbol)["lastPrice"])))

    def exposure(self) -> tuple[Decimal, Decimal]:
        """(equity, current gross notional including pending entry intents)."""
        equity = Decimal(str(self.adapter.get_account_equity()))
        if not equity.is_finite() or equity <= 0:
            raise ValueError("account equity unavailable or non-positive")
        gross = Decimal(0)
        for position in self.adapter.get_positions(strict=True):
            mark = Decimal(str(position.mark_price or 0)) or self.prices(position.symbol)
            gross += abs(Decimal(str(position.qty))) * mark
        if self.journal is not None:
            for row in self.journal.orders(unresolved=True):
                if not row.get("reducing") and row.get("qty"):
                    gross += abs(Decimal(row["qty"])) * self.prices(row["symbol"])
        return equity, gross

    def check_would_exceed(self, *, symbol: str, venue: str, side: str,
                           proposed_notional_usd: float) -> tuple[bool, str]:
        try:
            proposed = Decimal(str(proposed_notional_usd))
            if not proposed.is_finite() or proposed <= 0:
                return False, "invalid proposed notional"
            equity, gross = self.exposure()
            reason = exposure_breach(equity, gross, proposed, self.max_leverage)
            return not reason, reason
        except Exception as exc:
            return False, f"exposure check failed: {type(exc).__name__}"
