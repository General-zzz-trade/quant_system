# execution/adapters/binance/mapper_fill.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Dict

from execution.models.fills import CanonicalFill
from execution.adapters.binance.dedup_keys import make_fill_id, payload_digest_for_fill


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
    side = str(s).strip().lower()
    if side in ("buy", "b"):
        return "buy"
    if side in ("sell", "s"):
        return "sell"
    # Binance 常见为 "BUY"/"SELL"
    if side == "buy":
        return "buy"
    if side == "sell":
        return "sell"
    raise ValueError(f"unsupported side: {s!r}")


def _extract_from_ws_futures(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    # Futures user data: e=ORDER_TRADE_UPDATE, payload under o
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
        "trade_id": o.get("t"),
        "qty": o.get("l"),        # last filled qty
        "price": o.get("L"),      # last filled price
        "fee": o.get("n"),
        "fee_asset": o.get("N"),
        "ts_ms": o.get("T") or raw.get("E"),
        "is_maker": o.get("m"),
    }


def _extract_from_ws_spot(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    # Spot user data: e=executionReport
    if str(raw.get("e", "")).lower() != "executionreport":
        return None
    return {
        "symbol": raw.get("s"),
        "side": raw.get("S"),
        "order_id": raw.get("i"),
        "client_order_id": raw.get("c"),
        "trade_id": raw.get("t"),
        "qty": raw.get("l"),
        "price": raw.get("L"),
        "fee": raw.get("n"),
        "fee_asset": raw.get("N"),
        "ts_ms": raw.get("T") or raw.get("E"),
        "is_maker": raw.get("m"),
    }


def _extract_from_rest_trade(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    # REST trades: futures userTrades / spot myTrades
    if "symbol" not in raw or ("id" not in raw and "tradeId" not in raw):
        return None
    trade_id = raw.get("id", None)
    if trade_id is None:
        trade_id = raw.get("tradeId")
    is_maker = raw.get("maker", None)
    if is_maker is None:
        is_maker = raw.get("isMaker")
    return {
        "symbol": raw.get("symbol"),
        "side": raw.get("side") or raw.get("S"),
        "order_id": raw.get("orderId") or raw.get("i"),
        "client_order_id": raw.get("clientOrderId") or raw.get("clientOrderID") or raw.get("c"),
        "trade_id": trade_id,
        "qty": raw.get("qty") or raw.get("quantity") or raw.get("l"),
        "price": raw.get("price") or raw.get("L"),
        "fee": raw.get("commission") or raw.get("fee") or raw.get("n"),
        "fee_asset": raw.get("commissionAsset") or raw.get("feeAsset") or raw.get("N"),
        "ts_ms": raw.get("time") or raw.get("T") or raw.get("transactTime"),
        "is_maker": is_maker,
    }


@dataclass(frozen=True, slots=True)
class BinanceFillMapper:
    venue: str = "binance"

    def map_fill(self, raw: Mapping[str, Any]) -> CanonicalFill:
        # detect
        extracted = (
            _extract_from_ws_futures(raw)
            or _extract_from_ws_spot(raw)
            or _extract_from_rest_trade(raw)
        )
        if extracted is None:
            raise ValueError("unsupported binance fill payload")

        symbol = _norm_symbol(extracted.get("symbol"))
        side = _norm_side(extracted.get("side"))

        order_id = str(extracted.get("order_id")) if extracted.get("order_id") is not None else ""
        if not order_id:
            raise ValueError("missing field: order_id")

        trade_id = str(extracted.get("trade_id")) if extracted.get("trade_id") is not None else ""
        if not trade_id:
            raise ValueError("missing field: trade_id")

        qty = _dec(extracted.get("qty"), "qty")
        price = _dec(extracted.get("price"), "price")

        if qty <= 0:
            raise ValueError(f"qty must be >0, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be >0, got {price}")

        fee_raw = extracted.get("fee", None)
        fee = Decimal("0") if fee_raw in (None, "", "0", 0) else _dec(fee_raw, "fee")
        if fee < 0:
            raise ValueError(f"fee must be >=0, got {fee}")

        fee_asset = extracted.get("fee_asset")
        fee_asset_s = str(fee_asset).strip().upper() if fee_asset is not None and str(fee_asset).strip() else None

        ts_ms = _int_ms(extracted.get("ts_ms"), "ts_ms")

        is_maker = extracted.get("is_maker", None)
        liquidity: Optional[str] = None
        if isinstance(is_maker, bool):
            liquidity = "maker" if is_maker else "taker"

        fill_id = make_fill_id(venue=self.venue, symbol=symbol, trade_id=trade_id)
        digest = payload_digest_for_fill(
            symbol=symbol,
            order_id=order_id,
            trade_id=trade_id,
            side=side,
            qty=qty,
            price=price,
            fee=fee,
            fee_asset=fee_asset_s,
            ts_ms=ts_ms,
        )

        return CanonicalFill(
            venue=self.venue,
            symbol=symbol,
            order_id=order_id,
            trade_id=trade_id,
            fill_id=fill_id,
            side=side,
            qty=qty,
            price=price,
            fee=fee,
            fee_asset=fee_asset_s,
            liquidity=liquidity,
            ts_ms=ts_ms,
            payload_digest=digest,
            raw=raw,
        )
