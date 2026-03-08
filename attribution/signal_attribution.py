"""Signal-level P&L attribution.

Traces every fill back to the originating signal (IntentEvent.origin)
via the intent→order→fill chain, then computes per-signal P&L.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from _quant_hotpath import rust_attribute_by_signal


@dataclass(frozen=True)
class SignalPnL:
    """P&L breakdown for a single signal origin."""

    origin: str
    realized_pnl: float
    unrealized_pnl: float
    fee_cost: float
    trade_count: int
    win_rate: float


@dataclass(frozen=True)
class SignalAttributionReport:
    """Full signal attribution report."""

    by_signal: Mapping[str, SignalPnL]
    total_pnl: float
    unattributed_pnl: float


def attribute_by_signal(
    intents: Sequence[Mapping[str, object]],
    orders: Sequence[Mapping[str, object]],
    fills: Sequence[Mapping[str, object]],
    current_prices: Optional[Mapping[str, float]] = None,
) -> SignalAttributionReport:
    """Attribute fills to signal origins via intent→order→fill chain."""
    r = rust_attribute_by_signal(
        list(intents),
        list(orders),
        list(fills),
        dict(current_prices) if current_prices else None,
    )

    by_signal = {}
    for origin, sig_dict in r["by_signal"].items():
        by_signal[origin] = SignalPnL(
            origin=sig_dict["origin"],
            realized_pnl=sig_dict["realized_pnl"],
            unrealized_pnl=sig_dict["unrealized_pnl"],
            fee_cost=sig_dict["fee_cost"],
            trade_count=sig_dict["trade_count"],
            win_rate=sig_dict["win_rate"],
        )

    return SignalAttributionReport(
        by_signal=by_signal,
        total_pnl=r["total_pnl"],
        unattributed_pnl=r["unattributed_pnl"],
    )
