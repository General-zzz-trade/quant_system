# execution/adapters/bitget/mapper_order.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Dict

from execution.models.orders import CanonicalOrder
from execution.adapters.bitget.dedup_order_keys import make_order_key, payload_digest_for_order


def _dec(x: Any, field: str) -> Decimal:
    if x is None:
        raise ValueError(f"missing field: {field}")
    if isinstance(x, bool):
        raise TypeError(f"{field}: bool is not allowed")
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{field}: cannot convert to Decimal: {x!r}") from e


def _dec_opt(x: Any) -> Optional[Decimal]:
    if x is None or x == "" or x == 0 or x == "0":
        return None
    try:
        return Decimal(str(x))
    except Exception:
        return None


def _int_ms(x: Any, field: str) -> int:
    if x is None:
        raise ValueError(f"missing field: {field}")
    if isinstance(x, bool):
        raise TypeError(f"{field}: bool is not allowed")
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"{field}: cannot convert to int: {x!r}") from e


def _norm_symbol(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: symbol")
    sym = str(s).strip().upper()
    if not sym:
        raise ValueError("symbol is empty")
    return sym


def _norm_side(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: side")
    v = str(s).strip().lower()
    if v in ("buy", "b"):
        return "buy"
    if v in ("sell", "s"):
        return "sell"
    raise ValueError(f"unsupported side: {s!r}")


def _norm_status(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: status")
    v = str(s).strip().lower()
    # Bitget statuses: new, partially_filled, filled, cancelled
    m = {
        "new": "new",
        "init": "new",
        "live": "new",
        "partially_filled": "partially_filled",
        "partial_fill": "partially_filled",
        "filled": "filled",
        "full_fill": "filled",
        "cancelled": "canceled",
        "canceled": "canceled",
        "rejected": "rejected",
        "expired": "expired",
    }
    if v in m:
        return m[v]
    return v


def _norm_order_type(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: order_type")
    return str(s).strip().lower()


def _norm_tif(s: Any) -> Optional[str]:
    if s is None:
        return None
    v = str(s).strip().lower()
    return v if v else None


def _extract_from_ws_order(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract from Bitget WebSocket order push (private channel)."""
    # WS push uses same field names as REST in V2 API
    if "instId" in raw or "ordId" in raw:
        # V2 WS format
        return {
            "symbol": raw.get("instId") or raw.get("symbol"),
            "side": raw.get("side"),
            "order_id": raw.get("ordId") or raw.get("orderId"),
            "client_order_id": raw.get("clOrdId") or raw.get("clientOid"),
            "status": raw.get("status") or raw.get("state"),
            "order_type": raw.get("ordType") or raw.get("orderType"),
            "tif": raw.get("force"),
            "qty": raw.get("sz") or raw.get("size"),
            "price": raw.get("px") or raw.get("price"),
            "filled_qty": raw.get("accBaseVolume") or raw.get("baseVolume") or "0",
            "avg_price": raw.get("avgPx") or raw.get("priceAvg"),
            "ts_ms": raw.get("uTime") or raw.get("cTime"),
        }
    return None


def _extract_from_rest_order(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract from Bitget REST order response."""
    if "orderId" not in raw and "ordId" not in raw:
        return None

    order_id = raw.get("orderId") or raw.get("ordId")
    return {
        "symbol": raw.get("symbol") or raw.get("instId"),
        "side": raw.get("side"),
        "order_id": order_id,
        "client_order_id": raw.get("clientOid") or raw.get("clOrdId"),
        "status": raw.get("status") or raw.get("state"),
        "order_type": raw.get("orderType") or raw.get("ordType"),
        "tif": raw.get("force"),
        "qty": raw.get("size") or raw.get("sz"),
        "price": raw.get("price") or raw.get("px"),
        "filled_qty": raw.get("baseVolume") or raw.get("accBaseVolume") or "0",
        "avg_price": raw.get("priceAvg") or raw.get("avgPx"),
        "ts_ms": raw.get("cTime") or raw.get("uTime"),
    }


@dataclass(frozen=True, slots=True)
class BitgetOrderMapper:
    venue: str = "bitget"

    def map_order(self, raw: Mapping[str, Any]) -> CanonicalOrder:
        extracted = (
            _extract_from_ws_order(raw)
            or _extract_from_rest_order(raw)
        )
        if extracted is None:
            raise ValueError("unsupported bitget order payload")

        symbol = _norm_symbol(extracted.get("symbol"))
        side = _norm_side(extracted.get("side"))

        order_id = extracted.get("order_id")
        if order_id is None:
            raise ValueError("missing field: order_id")
        order_id_s = str(order_id)

        client_order_id = extracted.get("client_order_id")
        coi = str(client_order_id) if client_order_id is not None and str(client_order_id).strip() else None

        status = _norm_status(extracted.get("status"))
        order_type = _norm_order_type(extracted.get("order_type"))
        tif = _norm_tif(extracted.get("tif"))

        qty = _dec(extracted.get("qty"), "qty")
        if qty <= 0:
            raise ValueError(f"qty must be >0, got {qty}")

        price = _dec_opt(extracted.get("price"))
        if price is not None and price <= 0:
            raise ValueError(f"price must be >0 when present, got {price}")

        filled_qty = _dec(extracted.get("filled_qty", "0"), "filled_qty")
        if filled_qty < 0:
            raise ValueError(f"filled_qty must be >=0, got {filled_qty}")

        avg_price = _dec_opt(extracted.get("avg_price"))
        if avg_price is not None and avg_price <= 0:
            avg_price = None

        ts_ms = _int_ms(extracted.get("ts_ms"), "ts_ms")

        order_key = make_order_key(venue=self.venue, symbol=symbol, order_id=order_id_s)
        digest = payload_digest_for_order(
            symbol=symbol,
            order_id=order_id_s,
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

        return CanonicalOrder(
            venue=self.venue,
            symbol=symbol,
            order_id=order_id_s,
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
