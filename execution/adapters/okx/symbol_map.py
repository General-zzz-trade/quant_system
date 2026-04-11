"""OKX symbol + quantity conversion.

Two axes of conversion:

1. **Symbol name**: the rest of the codebase uses Binance/Bybit style
   (`BTCUSDT`) while OKX uses dash-separated (`BTC-USDT-SWAP`).

2. **Quantity unit**: the rest of the codebase thinks in **coin units**
   (e.g. 0.05 BTC) while OKX SWAP orders are priced in **contracts**.
   For BTC-USDT-SWAP one contract represents 0.01 BTC (ctVal=0.01), for
   ETH-USDT-SWAP one contract represents 0.1 ETH.  Conversion:

       contracts = coin_qty / ctVal
       coin_qty  = contracts * ctVal

   The result is rounded **down** to the nearest `lotSz` to avoid
   OKX rejecting the order as "size not multiple of lot".

Critical safety: getting this wrong means either placing an order 100x
too large (missing /ctVal) or 100x too small (double /ctVal).  Unit tests
in tests/unit/execution/adapters/okx/ cover every round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Mapping


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    """Static per-instrument metadata pulled from `/public/instruments`."""

    inst_id: str       # OKX format, e.g. "BTC-USDT-SWAP"
    ct_val: Decimal    # coin amount per 1 contract (e.g. 0.01 BTC)
    ct_val_ccy: str    # base currency, e.g. "BTC"
    lot_sz: Decimal    # contract size step
    min_sz: Decimal    # minimum contracts per order
    tick_sz: Decimal   # price tick
    max_lever: int     # maximum leverage

    @classmethod
    def from_api(cls, data: Mapping[str, str]) -> "InstrumentMeta":
        return cls(
            inst_id=data["instId"],
            ct_val=Decimal(str(data["ctVal"])),
            ct_val_ccy=data["ctValCcy"],
            lot_sz=Decimal(str(data["lotSz"])),
            min_sz=Decimal(str(data["minSz"])),
            tick_sz=Decimal(str(data["tickSz"])),
            max_lever=int(float(data.get("lever") or 0)),
        )


# Hard-coded mapping internal ↔ OKX.  We only support linear USDT perps.
_SYMBOL_TO_OKX: dict[str, str] = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
}
_OKX_TO_SYMBOL: dict[str, str] = {v: k for k, v in _SYMBOL_TO_OKX.items()}


def to_okx_symbol(internal: str) -> str:
    """BTCUSDT → BTC-USDT-SWAP.  Raises KeyError for unmapped symbols."""
    key = internal.upper()
    # Strip _4h / _1h suffix — OKX instrument is the same across timeframes
    for suffix in ("_4H", "_4h", "_1H", "_1h", "_15M", "_15m"):
        if key.endswith(suffix.upper()):
            key = key[: -len(suffix)]
            break
    return _SYMBOL_TO_OKX[key]


def from_okx_symbol(okx_id: str) -> str:
    """BTC-USDT-SWAP → BTCUSDT.  Raises KeyError for unmapped instruments."""
    return _OKX_TO_SYMBOL[okx_id]


def coin_to_contracts(coin_qty: float, meta: InstrumentMeta) -> Decimal:
    """Convert a coin quantity to a contract count, rounded down to lotSz.

    Returns Decimal('0') if rounding brings it below `min_sz`.
    """
    if coin_qty <= 0:
        return Decimal("0")
    raw = Decimal(str(coin_qty)) / meta.ct_val
    # Round DOWN to the nearest lotSz step (never over-order)
    steps = (raw / meta.lot_sz).to_integral_value(rounding=ROUND_DOWN)
    size = steps * meta.lot_sz
    if size < meta.min_sz:
        return Decimal("0")
    return size


def contracts_to_coin(contracts: Decimal | float, meta: InstrumentMeta) -> Decimal:
    """Inverse of `coin_to_contracts` — what coin amount does this represent."""
    return Decimal(str(contracts)) * meta.ct_val


def round_price_to_tick(price: float, meta: InstrumentMeta) -> Decimal:
    """Round a price DOWN to the nearest tick (never overpay a taker buy)."""
    if price <= 0:
        return Decimal("0")
    raw = Decimal(str(price))
    steps = (raw / meta.tick_sz).to_integral_value(rounding=ROUND_DOWN)
    return steps * meta.tick_sz
