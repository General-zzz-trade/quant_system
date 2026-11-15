"""SimulatedExchange: the exchange ports over one research bar broker per symbol (ADR-0013 decision 1, S10).

Paper mode is to send its orders through the same gates, order lifecycle, fill consumer and stop manager as
execution (S10 part 3); this adapter is the venue they then talk to. Its fills come from the research
``BarBroker``, so paper and research fills stay identical by construction.

A protective stop (``STOP_MARKET closePosition``) is modelled by a ``StopModel`` (decision L): it triggers when a
closed bar's low (a sell stop) or high (a buy stop) reaches the trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading.adapters.simulated.bar_broker import ReplayConfig
from trading.domain.bars import Bar


class StopModel(Protocol):
    """Decision L: whether a close-all stop on ``side`` at ``trigger`` fills within ``bar``, and at what price."""

    def triggered(self, side: str, trigger: Decimal, bar: Bar) -> Decimal | None: ...


@dataclass(frozen=True)
class BarStopModel:
    """Reached by the bar's low (a sell stop) or high (a buy stop); fills at the trigger, or at the open when the
    open has already gapped through it, worse by ``slippage_bps``: the research broker's isolated-mode stop rule
    and its stop slippage."""

    slippage_bps: Decimal = Decimal(ReplayConfig.stop_slippage_bps)

    def __post_init__(self) -> None:
        if not self.slippage_bps.is_finite() or not 0 <= self.slippage_bps < 10000:
            raise ValueError("stop slippage must be in [0, 10000) bps")

    def triggered(self, side: str, trigger: Decimal, bar: Bar) -> Decimal | None:
        slip = self.slippage_bps / 10000
        if side == "sell" and bar.low <= trigger:
            return min(bar.open, trigger) * (1 - slip)
        if side == "buy" and bar.high >= trigger:
            return max(bar.open, trigger) * (1 + slip)
        return None


__all__ = ["BarStopModel", "StopModel"]
