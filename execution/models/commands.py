from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional, Literal, Any

from execution.bridge.request_ids import RequestIdFactory, make_idempotency_key


Side = Literal["buy", "sell"]
CommandType = Literal["submit", "cancel"]


def _dec(x: Any, field_name: str) -> Decimal:
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{field_name}: cannot convert to Decimal: {x!r}") from e


def _norm_side(s: Any) -> Side:
    v = str(s).strip().lower()
    if v in ("buy", "b", "long"):
        return "buy"
    if v in ("sell", "s", "short"):
        return "sell"
    raise ValueError(f"unsupported side: {s!r}")


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class BaseCommand:
    # 关键：不进入 __init__，避免 default/non-default 顺序冲突
    command_type: CommandType = field(init=False)

    command_id: str
    created_ts: datetime
    actor: str

    venue: str
    symbol: str

    idempotency_key: str


@dataclass(frozen=True, slots=True)
class SubmitOrderCommand(BaseCommand):
    command_type: Literal["submit"] = field(init=False, default="submit")

    client_order_id: str = ""

    side: Side = "buy"
    order_type: str = "limit"              # lower
    tif: Optional[str] = None              # lower or None

    qty: Decimal = Decimal("0")
    price: Optional[Decimal] = None        # limit 时应提供

    reduce_only: bool = False


@dataclass(frozen=True, slots=True)
class CancelOrderCommand(BaseCommand):
    command_type: Literal["cancel"] = field(init=False, default="cancel")

    order_id: Optional[str] = None
    client_order_id: Optional[str] = None

    reason: Optional[str] = None


def make_submit_order_command(
    *,
    rid: RequestIdFactory,
    actor: str,
    venue: str,
    symbol: str,
    strategy: str,
    logical_id: str,
    side: Any,
    order_type: str,
    qty: Any,
    price: Any = None,
    tif: Optional[str] = None,
    reduce_only: bool = False,
    command_id: Optional[str] = None,
    created_ts: Optional[datetime] = None,
) -> SubmitOrderCommand:
    sym = str(symbol).strip().upper()
    v = str(venue).strip().lower()
    s = _norm_side(side)

    ot = str(order_type).strip().lower()
    tif_n = str(tif).strip().lower() if tif is not None else None

    q = _dec(qty, "qty")
    if q <= 0:
        raise ValueError(f"qty must be > 0, got {q}")

    p = None
    if price is not None and str(price) != "":
        p = _dec(price, "price")
        if p <= 0:
            raise ValueError(f"price must be > 0, got {p}")

    if ot == "limit" and p is None:
        raise ValueError("limit order requires price")

    coid = rid.client_order_id(strategy=strategy, symbol=sym, logical_id=logical_id)
    idem = make_idempotency_key(venue=v, action="submit", key=coid)

    return SubmitOrderCommand(
        command_id=command_id or f"cmd-submit-{coid}",
        created_ts=created_ts or _now_utc(),
        actor=actor,
        venue=v,
        symbol=sym,
        idempotency_key=idem,
        client_order_id=coid,
        side=s,
        order_type=ot,
        tif=tif_n,
        qty=q,
        price=p,
        reduce_only=bool(reduce_only),
    )


def make_cancel_order_command(
    *,
    actor: str,
    venue: str,
    symbol: str,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    reason: Optional[str] = None,
    command_id: Optional[str] = None,
    created_ts: Optional[datetime] = None,
) -> CancelOrderCommand:
    if not order_id and not client_order_id:
        raise ValueError("cancel requires order_id or client_order_id")

    v = str(venue).strip().lower()
    sym = str(symbol).strip().upper()

    key = client_order_id or order_id or ""
    idem = make_idempotency_key(venue=v, action="cancel", key=str(key))

    cid = command_id or f"cmd-cancel-{key}"
    return CancelOrderCommand(
        command_id=cid,
        created_ts=created_ts or _now_utc(),
        actor=actor,
        venue=v,
        symbol=sym,
        idempotency_key=idem,
        order_id=str(order_id) if order_id is not None else None,
        client_order_id=str(client_order_id) if client_order_id is not None else None,
        reason=reason,
    )
