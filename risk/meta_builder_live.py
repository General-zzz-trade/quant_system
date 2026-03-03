# risk/meta_builder_live.py
"""Build RiskEvalMetaBuilder for LiveRunner from coordinator state."""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from risk.aggregator import RiskEvalMetaBuilder

logger = logging.getLogger(__name__)


def build_live_meta_builder(
    coordinator: Any,
    equity_source: Callable[[], float],
) -> RiskEvalMetaBuilder:
    """Construct a RiskEvalMetaBuilder that extracts portfolio facts from coordinator state."""

    def _build_meta(symbol: str = "") -> Mapping[str, Any]:
        view = coordinator.get_state_view()
        positions = view.get("positions", {})
        equity = equity_source()

        gross_notional = 0.0
        net_notional = 0.0
        positions_notional: dict[str, float] = {}

        for sym, pos in positions.items():
            qty = float(getattr(pos, "qty", 0))
            mark = float(getattr(pos, "mark_price", 0) or getattr(pos, "entry_price", 0) or 0)
            notional = qty * mark
            positions_notional[sym] = notional
            gross_notional += abs(notional)
            net_notional += notional

        symbol_notional = positions_notional.get(symbol, 0.0)
        symbol_weight = abs(symbol_notional) / gross_notional if gross_notional > 0 else 0.0

        return {
            "equity": equity,
            "gross_notional": gross_notional,
            "net_notional": net_notional,
            "positions_notional": positions_notional,
            "symbol_weight": symbol_weight,
        }

    def _build_for_intent(intent: Any) -> Mapping[str, Any]:
        symbol = getattr(intent, "symbol", "")
        return _build_meta(symbol)

    def _build_for_order(order: Any) -> Mapping[str, Any]:
        symbol = getattr(order, "symbol", "")
        meta = dict(_build_meta(symbol))
        price = getattr(order, "price", None)
        if price is not None:
            meta["market_price"] = float(price)
        return meta

    return RiskEvalMetaBuilder(
        build_for_intent=_build_for_intent,
        build_for_order=_build_for_order,
    )
