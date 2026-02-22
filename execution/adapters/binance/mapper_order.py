# execution/adapters/binance/mapper_order.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Dict

from execution.models.orders import CanonicalOrder
from execution.adapters.binance.dedup_order_keys import make_order_key, payload_digest_for_order


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
        d = Decimal(str(x))
        return d
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
    # Binance 常见 BUY/SELL
    if v == "buy":
        return "buy"
    if v == "sell":
        return "sell"
    raise ValueError(f"unsupported side: {s!r}")


def _norm_status(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: status")
    v = str(s).strip().upper()
    # Binance Spot/Futures: NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED
    m = {
        "NEW": "new",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "CANCELLED": "canceled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
        "PENDING_CANCEL": "pending_cancel",
    }
    if v in m:
        return m[v]
    return v.lower()


def _norm_order_type(s: Any) -> str:
    if s is None:
        raise ValueError("missing field: order_type")
    return str(s).strip().lower()


def _norm_tif(s: Any) -> Optional[str]:
    if s is None:
        return None
    v = str(s).strip().lower()
    return v if v else None


def _extract_from_spot_execution_report(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if str(raw.get("e", "")).lower() != "executionreport":
        return None
    return {
        "symbol": raw.get("s"),
        "side": raw.get("S"),
        "order_id": raw.get("i"),
        "client_order_id": raw.get("c"),
        "status": raw.get("X"),         # current order status
        "order_type": raw.get("o"),
        "tif": raw.get("f"),
        "qty": raw.get("q"),            # order qty
        "price": raw.get("p"),
        "filled_qty": raw.get("z"),     # cumulative filled qty
        "avg_price": raw.get("ap"),     # spot sometimes missing
        "ts_ms": raw.get("T") or raw.get("E"),
    }


def _extract_from_futures_order_trade_update(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if str(raw.get("e", "")).upper() != "ORDER_TRADE_UPDATE":
        return None
    o = raw.get("o")
    if not isinstance(o, Mapping):
        raise ValueError("ORDER_TRADE_UPDATE missing 'o'")
    return {
        "symbol": o.get("s"),
        "side": o.get("S"),
        "order_id": o.get("i"),
        "client_order_id": o.get("c"),
        "status": o.get("X"),           # order status
        "order_type": o.get("o"),
        "tif": o.get("f"),
        "qty": o.get("q"),              # orig qty
        "price": o.get("p"),
        "filled_qty": o.get("z"),       # cumulative filled qty
        "avg_price": o.get("ap"),       # avg price (futures)
        "ts_ms": o.get("T") or raw.get("E"),
    }


def _extract_from_rest_order(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    # REST: openOrders / allOrders style (spot/futures differ slightly)
    if ("symbol" not in raw) or ("orderId" not in raw and "order_id" not in raw and "i" not in raw):
        return None
    order_id = raw.get("orderId", raw.get("order_id", raw.get("i")))
    return {
        "symbol": raw.get("symbol"),
        "side": raw.get("side") or raw.get("S"),
        "order_id": order_id,
        "client_order_id": raw.get("clientOrderId") or raw.get("clientOrderID") or raw.get("c"),
        "status": raw.get("status") or raw.get("X"),
        "order_type": raw.get("type") or raw.get("orderType") or raw.get("o"),
        "tif": raw.get("timeInForce") or raw.get("f"),
        "qty": raw.get("origQty") or raw.get("q"),
        "price": raw.get("price") or raw.get("p"),
        "filled_qty": raw.get("executedQty") or raw.get("z"),
        "avg_price": raw.get("avgPrice") or raw.get("ap"),
        "ts_ms": raw.get("updateTime") or raw.get("time") or raw.get("transactTime") or raw.get("T"),
    }


@dataclass(frozen=True, slots=True)
class BinanceOrderMapper:
    venue: str = "binance"

    def map_order(self, raw: Mapping[str, Any]) -> CanonicalOrder:
        extracted = (
            _extract_from_spot_execution_report(raw)
            or _extract_from_futures_order_trade_update(raw)
            or _extract_from_rest_order(raw)
        )
        if extracted is None:
            raise ValueError("unsupported binance order payload")

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
