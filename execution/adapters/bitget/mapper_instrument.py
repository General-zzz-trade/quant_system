# execution/adapters/bitget/mapper_instrument.py
"""Map Bitget contracts to canonical InstrumentInfo."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Sequence

from execution.models.instruments import InstrumentInfo


def map_instrument(raw: Mapping[str, Any]) -> InstrumentInfo:
    """Map a single Bitget contract entry to InstrumentInfo.

    Bitget contracts fields:
    - symbol, baseCoin, quoteCoin
    - pricePrecision, quantityPrecision (or volumePlace)
    - priceEndStep → tick_size
    - sizeMultiplier → lot_size
    - minTradeNum → min_qty
    - maxSymbolOpenNum → max_qty (optional)
    """
    symbol = str(raw.get("symbol", "")).upper()
    price_prec = int(raw.get("pricePrecision", raw.get("pricePlace", 2)))
    qty_prec = int(raw.get("quantityPrecision", raw.get("volumePlace", 3)))

    tick_size = Decimal(str(raw.get("priceEndStep", 10 ** -price_prec)))
    lot_size = Decimal(str(raw.get("sizeMultiplier", 10 ** -qty_prec)))

    min_qty_raw = raw.get("minTradeNum", raw.get("minTradeUSDT", "0"))
    min_qty = Decimal(str(min_qty_raw)) if min_qty_raw else Decimal("0")

    max_qty_raw = raw.get("maxSymbolOpenNum")
    max_qty = Decimal(str(max_qty_raw)) if max_qty_raw else None

    min_notional = Decimal(str(raw.get("minTradeUSDT", "5")))

    return InstrumentInfo(
        venue="bitget",
        symbol=symbol,
        base_asset=str(raw.get("baseCoin", "")).upper(),
        quote_asset=str(raw.get("quoteCoin", "USDT")).upper(),
        price_precision=price_prec,
        qty_precision=qty_prec,
        tick_size=tick_size,
        lot_size=lot_size,
        min_qty=min_qty,
        max_qty=max_qty,
        min_notional=min_notional,
        contract_type="perpetual",
        margin_asset="USDT",
    )


def map_instruments(raws: Sequence[Mapping[str, Any]]) -> list[InstrumentInfo]:
    """Batch map contracts."""
    return [map_instrument(r) for r in raws]
