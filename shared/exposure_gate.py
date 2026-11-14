"""Account-level gross exposure cap: total notional <= MAX_TOTAL_LEVERAGE x equity.

The cap is a code constant. Configuration may LOWER it (``max_leverage``) but
never raise it. Every check reads a fresh exchange snapshot; any error, stale
or non-positive equity blocks the entry (reductions never reach this gate).

Gross exposure = sum(|position qty| x mark) over all symbols
               + notional of unresolved entry intents in the order journal
               + the proposed order.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading.application.ports import ExchangeAccount, OrderJournal

MAX_TOTAL_LEVERAGE = Decimal("3")


class AccountExposureGate:
    def __init__(self, adapter: ExchangeAccount, journal: OrderJournal | None = None, *,
                 max_leverage: Decimal = MAX_TOTAL_LEVERAGE, prices: Any = None) -> None:
        max_leverage = Decimal(str(max_leverage))
        if not max_leverage.is_finite() or not 0 < max_leverage <= MAX_TOTAL_LEVERAGE:
            raise ValueError(f"max_leverage must be in (0, {MAX_TOTAL_LEVERAGE}]")
        self.adapter, self.journal, self.max_leverage = adapter, journal, max_leverage
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
            limit = equity * self.max_leverage
            if gross + proposed > limit:
                return False, (f"gross exposure {gross + proposed:.2f} would exceed "
                               f"{self.max_leverage}x equity ({limit:.2f})")
            return True, ""
        except Exception as exc:
            return False, f"exposure check failed: {type(exc).__name__}"
