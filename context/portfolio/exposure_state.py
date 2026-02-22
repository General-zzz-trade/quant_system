# context/portfolio/exposure_state.py
"""Exposure state — track and limit portfolio exposure."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    """暴露快照。"""
    symbol: str
    notional: Decimal
    pct_of_equity: Decimal
    side: str             # "long" / "short" / "flat"


class ExposureState:
    """暴露追踪 — 按品种追踪暴露度。"""

    def __init__(self) -> None:
        self._exposures: Dict[str, dict] = {}

    def update(self, symbol: str, *, qty: Decimal, price: Decimal, equity: Decimal) -> None:
        notional = abs(qty) * price
        pct = notional / equity if equity > 0 else Decimal("0")
        side = "long" if qty > 0 else ("short" if qty < 0 else "flat")
        self._exposures[symbol] = {
            "notional": notional, "pct": pct, "side": side,
        }

    def get(self, symbol: str) -> Optional[ExposureSnapshot]:
        e = self._exposures.get(symbol)
        if e is None:
            return None
        return ExposureSnapshot(
            symbol=symbol, notional=e["notional"],
            pct_of_equity=e["pct"], side=e["side"],
        )

    @property
    def total_gross(self) -> Decimal:
        return sum((e["notional"] for e in self._exposures.values()), Decimal("0"))
