# execution/adapters/binance/mapper_instrument.py
"""Map Binance exchangeInfo to canonical InstrumentInfo."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from execution.models.instruments import InstrumentInfo


def _find_filter(filters: Sequence[Mapping[str, Any]], filter_type: str) -> Mapping[str, Any]:
    for f in filters:
        if f.get("filterType") == filter_type:
            return f
    return {}


def map_instrument(raw: Mapping[str, Any]) -> InstrumentInfo:
    """从 exchangeInfo.symbols[] 映射单个品种。"""
    symbol = str(raw.get("symbol", ""))
    filters = raw.get("filters", [])

    price_f = _find_filter(filters, "PRICE_FILTER")
    lot_f = _find_filter(filters, "LOT_SIZE")
    notional_f = _find_filter(filters, "MIN_NOTIONAL")

    return InstrumentInfo(
        symbol=symbol,
        base_asset=str(raw.get("baseAsset", "")),
        quote_asset=str(raw.get("quoteAsset", "USDT")),
        price_precision=int(raw.get("pricePrecision", 2)),
        qty_precision=int(raw.get("quantityPrecision", 3)),
        tick_size=Decimal(str(price_f.get("tickSize", "0.01"))),
        lot_size=Decimal(str(lot_f.get("stepSize", "0.001"))),
        min_qty=Decimal(str(lot_f.get("minQty", "0.001"))),
        max_qty=Decimal(str(lot_f.get("maxQty", "999999"))),
        min_notional=Decimal(str(notional_f.get("notional", "5"))),
    )


def map_instruments(raws: Sequence[Mapping[str, Any]]) -> list[InstrumentInfo]:
    """批量映射品种信息。"""
    return [map_instrument(r) for r in raws]
