"""Polymarket JSON -> CanonicalFill / CanonicalOrder mapper."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional

from execution.models.fills import CanonicalFill
from execution.models.orders import CanonicalOrder, ingress_order_dedup_identity
from types import SimpleNamespace


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _dec(x: Any, field: str) -> Decimal:
    if x is None:
        raise ValueError(f"missing field: {field}")
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{field}: cannot convert to Decimal: {x!r}") from e


def _dec_or_zero(x: Any) -> Decimal:
    if x is None or x == "" or x == "0" or x == 0:
        return Decimal("0")
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")


def _int_ms(x: Any, field: str) -> int:
    if x is None:
        return 0
    try:
        return int(x)
    except Exception:
        return 0


def _norm_side(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: side")
    v = str(s).strip().lower()
    if v in ("buy", "b"):
        return "buy"
    if v in ("sell", "s"):
        return "sell"
    raise ValueError(f"unsupported side: {s!r}")


_STATUS_MAP = {
    "LIVE": "new",
    "ACTIVE": "new",
    "OPEN": "new",
    "MATCHED": "filled",
    "FILLED": "filled",
    "CANCELED": "canceled",
    "CANCELLED": "canceled",
    "EXPIRED": "expired",
    "REJECTED": "rejected",
}


def _norm_status(s: Any) -> str:
    if s is None:
        return "new"
    v = str(s).strip().upper()
    return _STATUS_MAP.get(v, v.lower())


def _norm_tif(order_type_raw: Any) -> Optional[str]:
    """Map Polymarket order type string to TIF."""
    if order_type_raw is None:
        return "gtc"
    v = str(order_type_raw).strip().upper()
    mapping = {
        "GTC": "gtc",
        "GTD": "gtc",
        "FOK": "fok",
        "IOC": "ioc",
    }
    return mapping.get(v, "gtc")


def _make_fill_id(venue: str, symbol: str, trade_id: str) -> str:
    return f"{venue}:{symbol}:fill:{trade_id}"


def _stable_hash(obj: Mapping[str, Any]) -> str:
    from execution.models.digest import stable_hash
    return stable_hash(obj)


# ------------------------------------------------------------------ #
# Fill mapper
# ------------------------------------------------------------------ #

def fill_from_polymarket(raw: Mapping[str, Any], venue: str = "polymarket") -> CanonicalFill:
    """Map a Polymarket trade/fill JSON dict to CanonicalFill."""
    trade_id = str(raw.get("id", ""))
    order_id = str(raw.get("order_id", raw.get("orderId", "")))
    if not trade_id:
        raise ValueError("missing field: id (trade_id)")

    market = raw.get("market", raw.get("slug", ""))
    asset_id = raw.get("asset_id", raw.get("token_id", ""))
    symbol = f"POLY:{market}:{asset_id}" if market else str(asset_id)

    side = _norm_side(raw.get("side"))
    qty = _dec(raw.get("size", raw.get("amount")), "size")
    price = _dec(raw.get("price"), "price")

    fee = _dec_or_zero(raw.get("fee", raw.get("commission")))
    fee_asset: Optional[str] = raw.get("fee_asset") or None

    ts_ms = _int_ms(raw.get("timestamp", raw.get("time")), "timestamp")

    fill_id = _make_fill_id(venue, symbol, trade_id)
    digest = _stable_hash({
        "symbol": symbol, "order_id": order_id, "trade_id": trade_id,
        "side": side, "qty": qty, "price": price, "fee": fee,
        "fee_asset": fee_asset or "", "ts_ms": ts_ms,
    })

    return CanonicalFill(
        venue=venue,
        symbol=symbol,
        order_id=order_id,
        trade_id=trade_id,
        fill_id=fill_id,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        fee_asset=fee_asset,
        liquidity=None,
        ts_ms=ts_ms,
        payload_digest=digest,
        raw=raw,
    )


# ------------------------------------------------------------------ #
# Order mapper
# ------------------------------------------------------------------ #

def order_from_polymarket(raw: Mapping[str, Any], venue: str = "polymarket") -> CanonicalOrder:
    """Map a Polymarket order JSON dict to CanonicalOrder."""
    order_id = str(raw.get("id", ""))
    if not order_id:
        raise ValueError("missing field: id (order_id)")

    market = raw.get("market", raw.get("slug", ""))
    asset_id = raw.get("asset_id", raw.get("token_id", ""))
    symbol = f"POLY:{market}:{asset_id}" if market else str(asset_id)

    side = _norm_side(raw.get("side"))
    status = _norm_status(raw.get("status"))

    order_type_raw = raw.get("type", raw.get("order_type", "GTC"))
    tif = _norm_tif(order_type_raw)
    order_type = "limit"  # Polymarket CLOB only supports limit orders

    qty = _dec(raw.get("original_size", raw.get("size", raw.get("amount"))), "original_size")
    price_raw = raw.get("price")
    price = _dec(price_raw, "price") if price_raw is not None else None

    filled_qty = _dec_or_zero(raw.get("size_matched", raw.get("filled_size", "0")))
    avg_price_raw = raw.get("avg_price", raw.get("average_price"))
    avg_price = _dec(avg_price_raw, "avg_price") if avg_price_raw is not None else None

    ts_ms = _int_ms(raw.get("timestamp", raw.get("time", raw.get("created_at"))), "timestamp")

    client_order_id = raw.get("client_order_id") or None
    coi = str(client_order_id) if client_order_id else None

    order_key, digest = ingress_order_dedup_identity(
        SimpleNamespace(
            venue=venue,
            symbol=symbol,
            order_id=order_id,
            client_order_id=coi,
            status=status,
            side=side,
            order_type=order_type,
            tif=tif,
            qty=qty,
            price=price,
            filled_qty=filled_qty,
            avg_price=avg_price,
            ts_ms=ts_ms,
        )
    )

    return CanonicalOrder(
        venue=venue,
        symbol=symbol,
        order_id=order_id,
        client_order_id=coi,
        status=status,
        side=side,
        order_type=order_type,
        tif=tif,
        qty=qty,
        price=price,
        filled_qty=filled_qty,
        avg_price=avg_price,
        ts_ms=ts_ms,
        order_key=order_key,
        payload_digest=digest,
        raw=raw,
    )
