# execution/adapters/bitget/mapper_fill.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Dict, Sequence

from execution.models.fills import CanonicalFill
from execution.adapters.bitget.dedup_keys import make_fill_id, payload_digest_for_fill


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
    raise ValueError(f"unsupported side: {s!r}")


def _extract_fee(raw: Mapping[str, Any]) -> tuple[Decimal, Optional[str]]:
    """Extract fee from Bitget fill — supports both feeDetail list and flat fields."""
    # Try feeDetail list first
    fee_detail = raw.get("feeDetail")
    if isinstance(fee_detail, (list, tuple)) and fee_detail:
        entry = fee_detail[0] if isinstance(fee_detail[0], dict) else {}
        total_fee = entry.get("totalFee") or entry.get("fee") or "0"
        fee_coin = entry.get("feeCoin") or entry.get("coin")
        fee = abs(Decimal(str(total_fee)))
        return fee, str(fee_coin).upper() if fee_coin else None

    # Flat fields
    fee_raw = raw.get("fee") or raw.get("n") or "0"
    fee = abs(Decimal(str(fee_raw)))
    fee_coin = raw.get("feeCoin") or raw.get("N")
    return fee, str(fee_coin).upper() if fee_coin else None


def _extract_from_rest_fill(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract from Bitget REST fills response."""
    trade_id = raw.get("tradeId") or raw.get("fillId")
    if trade_id is None:
        return None

    fee, fee_asset = _extract_fee(raw)

    return {
        "symbol": raw.get("symbol"),
        "side": raw.get("side"),
        "order_id": raw.get("orderId"),
        "trade_id": trade_id,
        "qty": raw.get("baseVolume") or raw.get("size") or raw.get("fillSz"),
        "price": raw.get("price") or raw.get("fillPx"),
        "fee": fee,
        "fee_asset": fee_asset,
        "ts_ms": raw.get("cTime") or raw.get("uTime"),
        "is_maker": raw.get("tradeScope"),  # "maker" / "taker" string
    }


def _extract_from_ws_fill(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract from Bitget WS fill push."""
    # WS fill pushes have similar structure
    trade_id = raw.get("tradeId") or raw.get("fillId")
    if trade_id is None or "orderId" not in raw:
        return None

    fee, fee_asset = _extract_fee(raw)

    return {
        "symbol": raw.get("symbol") or raw.get("instId"),
        "side": raw.get("side"),
        "order_id": raw.get("orderId") or raw.get("ordId"),
        "trade_id": trade_id,
        "qty": raw.get("baseVolume") or raw.get("fillSz") or raw.get("size"),
        "price": raw.get("price") or raw.get("fillPx"),
        "fee": fee,
        "fee_asset": fee_asset,
        "ts_ms": raw.get("cTime") or raw.get("uTime"),
        "is_maker": raw.get("tradeScope"),
    }


@dataclass(frozen=True, slots=True)
class BitgetFillMapper:
    venue: str = "bitget"

    def map_fill(self, raw: Mapping[str, Any]) -> CanonicalFill:
        extracted = (
            _extract_from_rest_fill(raw)
            or _extract_from_ws_fill(raw)
        )
        if extracted is None:
            raise ValueError("unsupported bitget fill payload")

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

        fee = extracted.get("fee", Decimal("0"))
        if not isinstance(fee, Decimal):
            fee = Decimal("0")
        fee_asset = extracted.get("fee_asset")

        ts_ms = _int_ms(extracted.get("ts_ms"), "ts_ms")

        # Determine liquidity from tradeScope
        trade_scope = extracted.get("is_maker")
        liquidity: Optional[str] = None
        if isinstance(trade_scope, str):
            if trade_scope.lower() == "maker":
                liquidity = "maker"
            elif trade_scope.lower() == "taker":
                liquidity = "taker"

        fill_id = make_fill_id(venue=self.venue, symbol=symbol, trade_id=trade_id)
        digest = payload_digest_for_fill(
            symbol=symbol,
            order_id=order_id,
            trade_id=trade_id,
            side=side,
            qty=qty,
            price=price,
            fee=fee,
            fee_asset=fee_asset,
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
            fee_asset=fee_asset,
            liquidity=liquidity,
            ts_ms=ts_ms,
            payload_digest=digest,
            raw=raw,
        )
