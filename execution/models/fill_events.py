from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from execution.safety.message_integrity import compute_payload_digest
from event.header import EventHeader
from event.types import FillEvent


@dataclass(frozen=True, slots=True)
class CanonicalFillIngressEvent:
    """Ingress event shape emitted into the engine for fill-driven state updates."""

    header: Any
    event_type: str

    symbol: str
    qty: float
    side: Optional[str]
    price: float

    fee: float = 0.0
    realized_pnl: float = 0.0
    margin_change: float = 0.0
    cash_delta: float = 0.0

    venue: Optional[str] = None
    order_id: Optional[str] = None
    fill_id: Optional[str] = None
    trade_id: Optional[str] = None
    payload_digest: Optional[str] = None

    @property
    def quantity(self) -> float:
        return self.qty


def canonical_fill_to_public_event(fill: Any, *, source: str = "execution") -> FillEvent:
    """Convert CanonicalFill-like objects into the public minimal FillEvent contract."""
    header = EventHeader.new_root(
        event_type=FillEvent.event_type,
        version=FillEvent.VERSION,
        source=source,
    )
    return FillEvent(
        header=header,
        fill_id=str(getattr(fill, "fill_id")),
        order_id=str(getattr(fill, "order_id")),
        symbol=str(getattr(fill, "symbol")),
        qty=Decimal(str(getattr(fill, "qty"))),
        price=Decimal(str(getattr(fill, "price"))),
    )


def canonical_fill_to_ingress_event(fill: Any) -> CanonicalFillIngressEvent:
    """Convert CanonicalFill-like objects into the richer ingress event contract."""
    return build_ingress_fill_event(
        symbol=getattr(fill, "symbol", ""),
        qty=getattr(fill, "qty", 0),
        side=getattr(fill, "side", None),
        price=getattr(fill, "price", 0),
        fee=getattr(fill, "fee", 0),
        realized_pnl=getattr(fill, "realized_pnl", 0),
        margin_change=getattr(fill, "margin_change", 0),
        cash_delta=getattr(fill, "cash_delta", 0),
        venue=getattr(fill, "venue", None),
        order_id=getattr(fill, "order_id", None),
        fill_id=getattr(fill, "fill_id", None),
        trade_id=getattr(fill, "trade_id", None),
        payload_digest=getattr(fill, "payload_digest", None),
        ts=getattr(fill, "ts", None),
        ts_ms=getattr(fill, "ts_ms", None),
    )


def build_ingress_fill_event(
    *,
    symbol: Any,
    qty: Any,
    side: Any,
    price: Any,
    fee: Any = 0,
    realized_pnl: Any = 0,
    margin_change: Any = 0,
    cash_delta: Any = 0,
    venue: Any = None,
    order_id: Any = None,
    fill_id: Any = None,
    trade_id: Any = None,
    payload_digest: Any = None,
    ts: Any = None,
    ts_ms: Any = None,
) -> CanonicalFillIngressEvent:
    """Build the ingress-side richer fill fact used by pipeline/state updates."""
    header = SimpleNamespace(
        ts=_coerce_ts(ts, ts_ms=ts_ms),
        event_type="FILL",
        event_id=None,
    )
    return CanonicalFillIngressEvent(
        header=header,
        event_type="FILL",
        symbol=str(symbol),
        qty=float(Decimal(str(qty))),
        side=_normalize_optional_side(side),
        price=float(Decimal(str(price))),
        fee=float(Decimal(str(fee))),
        realized_pnl=float(Decimal(str(realized_pnl))),
        margin_change=float(Decimal(str(margin_change))),
        cash_delta=float(Decimal(str(cash_delta))),
        venue=_normalize_optional_str(venue),
        order_id=_normalize_optional_str(order_id),
        fill_id=_normalize_optional_str(fill_id),
        trade_id=_normalize_optional_str(trade_id),
        payload_digest=_normalize_optional_str(payload_digest),
    )


def ingress_fill_dedup_identity(event: CanonicalFillIngressEvent) -> Tuple[Tuple[str, str, str], str]:
    """Return the stable dedup key and digest for ingress fill processing."""
    venue = event.venue or "unknown"
    symbol = event.symbol or ""

    fill_key = event.fill_id or event.trade_id
    if fill_key is None:
        fill_key = _stable_hash(
            {
                "order_id": event.order_id,
                "ts": event.header.ts.isoformat() if getattr(event.header, "ts", None) else None,
                "price": event.price,
                "qty": event.qty,
                "side": event.side,
            }
        )

    digest = event.payload_digest
    if digest is None:
        digest = _stable_hash(
            {
                "venue": event.venue,
                "symbol": event.symbol,
                "order_id": event.order_id,
                "fill_id": event.fill_id,
                "trade_id": event.trade_id,
                "ts": event.header.ts.isoformat() if getattr(event.header, "ts", None) else None,
                "price": event.price,
                "qty": event.qty,
                "side": event.side,
                "fee": event.fee,
            }
        )

    return (str(venue), str(symbol), str(fill_key)), str(digest)


def build_synthetic_ingress_fill_event(
    *,
    source: str,
    symbol: Any,
    side: Any,
    qty: Any,
    price: Any,
    fee: Any = 0,
    venue: Any = None,
    order_id: Any = None,
    identity_seed: Any = None,
    fill_seq: int = 1,
    ts: Any = None,
    ts_ms: Any = None,
) -> CanonicalFillIngressEvent:
    """Build a synthetic ingress fill with stable fill_id + payload_digest."""
    normalized_venue = _normalize_optional_str(venue) or "synthetic"
    normalized_symbol = str(symbol)
    normalized_order_id = _normalize_optional_str(order_id) or ""
    normalized_side = _normalize_optional_side(side) or ""
    normalized_qty = float(Decimal(str(qty)))
    normalized_price = float(Decimal(str(price)))
    normalized_fee = float(Decimal(str(fee)))
    normalized_seed = _normalize_optional_str(identity_seed) or ""

    fill_key = _stable_hash(
        {
            "source": source,
            "venue": normalized_venue,
            "symbol": normalized_symbol,
            "order_id": normalized_order_id,
            "seed": normalized_seed,
            "fill_seq": int(fill_seq),
        }
    )[:24]
    fill_id = f"{source}-fill-{fill_key}"

    payload_digest = compute_payload_digest(
        {
            "source": source,
            "venue": normalized_venue,
            "symbol": normalized_symbol,
            "order_id": normalized_order_id,
            "fill_id": fill_id,
            "fill_seq": int(fill_seq),
            "side": normalized_side,
            "qty": normalized_qty,
            "price": normalized_price,
            "fee": normalized_fee,
        }
    )

    return build_ingress_fill_event(
        symbol=normalized_symbol,
        qty=normalized_qty,
        side=normalized_side,
        price=normalized_price,
        fee=normalized_fee,
        venue=normalized_venue,
        order_id=normalized_order_id,
        fill_id=fill_id,
        payload_digest=payload_digest,
        ts=ts,
        ts_ms=ts_ms,
    )


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _normalize_optional_side(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).lower()


def _coerce_ts(ts: Any, *, ts_ms: Any = None) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)

    if ts_ms is not None:
        try:
            return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
        except Exception:
            pass

    if ts is not None:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            pass

    return datetime.now(tz=timezone.utc)


def _stable_hash(payload: Any) -> str:
    from execution.models.digest import stable_hash
    return stable_hash(payload)
