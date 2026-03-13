# execution/ingress/router.py
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from _quant_hotpath import RustPayloadDedupGuard as _RustPayloadDedupGuard
from execution.models.fill_events import (
    CanonicalFillIngressEvent as IngressFillEvent,
    canonical_fill_to_ingress_event,
    ingress_fill_dedup_identity,
)


# ============================================================
# Event emitted into engine (must be routable as FILL -> PIPELINE)
# ============================================================

# ============================================================
# Execution safety: idempotency + payload mismatch detection
# ============================================================

class FillDeduplicator:
    """
    Idempotency contract:
    - Same (venue, symbol, fill_key) with same digest: DROP (idempotent)
    - Same (venue, symbol, fill_key) with different digest: FAIL FAST (data corruption)
    """
    def __init__(self) -> None:
        self._rust = _RustPayloadDedupGuard()

    def accept_or_raise(self, *, key: Tuple[str, str, str], digest: str) -> bool:
        packed = "\x1f".join(str(part) for part in key)
        return bool(self._rust.check_and_insert(packed, digest))


# ============================================================
# Ingress Router
# ============================================================

class FillIngressRouter:
    """
    Convert CanonicalFill -> IngressFillEvent -> coordinator.emit()

    coordinator contract:
    - has .emit(event, actor=...)
    """
    def __init__(self, *, coordinator: Any, default_actor: str = "venue:binance") -> None:
        self._coord = coordinator
        self._default_actor = default_actor
        self._dedup = FillDeduplicator()

    # ---------- public ----------

    def ingest_canonical_fill(self, fill: Any, *, actor: Optional[str] = None) -> bool:
        """
        Returns:
        - True  if accepted and emitted into engine
        - False if dropped due to idempotent duplicate
        Raises:
        - ValueError if duplicate key but payload mismatch
        """
        ev = self._to_fill_event(fill)

        key, digest = self._dedup_key_and_digest(ev)
        if not self._dedup.accept_or_raise(key=key, digest=digest):
            return False

        self._coord.emit(ev, actor=actor or self._default_actor)
        return True

    # ---------- internal ----------

    def _to_fill_event(self, fill: Any) -> IngressFillEvent:
        if _looks_like_canonical_fill(fill):
            return canonical_fill_to_ingress_event(fill)

        # Legacy / non-canonical fallback path.
        venue = _get_first(fill, "venue", "exchange", default=None)
        symbol = _get_first(fill, "symbol", default="")
        side = _get_first(fill, "side", default=None)
        qty = float(_get_first(fill, "qty", "quantity", default=0.0))
        price = float(_get_first(fill, "price", default=0.0))
        fee = float(_get_first(fill, "fee", default=0.0))
        realized_pnl = float(_get_first(fill, "realized_pnl", "realized", default=0.0))
        margin_change = float(_get_first(fill, "margin_change", default=0.0))
        cash_delta = float(_get_first(fill, "cash_delta", default=0.0))
        fill_id = _get_first(fill, "fill_id", default=None)
        trade_id = _get_first(fill, "trade_id", "tradeId", default=None)
        order_id = _get_first(fill, "order_id", "orderId", default=None)
        payload_digest = _get_first(fill, "payload_digest", "digest", default=None)
        ts = _coerce_ts(
            _get_first(fill, "ts", "timestamp", "time", default=None),
            ts_ms=_get_first(fill, "ts_ms", "timestamp_ms", "event_time_ms", default=None),
        )
        return IngressFillEvent(
            header=SimpleNamespace(ts=ts, event_type="FILL", event_id=None),
            event_type="FILL",
            symbol=str(symbol),
            qty=float(qty),
            side=str(side).lower() if side is not None else None,
            price=float(price),
            fee=float(fee),
            realized_pnl=float(realized_pnl),
            margin_change=float(margin_change),
            cash_delta=float(cash_delta),
            venue=str(venue) if venue is not None else None,
            order_id=str(order_id) if order_id is not None else None,
            fill_id=str(fill_id) if fill_id is not None else None,
            trade_id=str(trade_id) if trade_id is not None else None,
            payload_digest=str(payload_digest) if payload_digest is not None else None,
        )

    def _dedup_key_and_digest(self, ev: IngressFillEvent) -> Tuple[Tuple[str, str, str], str]:
        return ingress_fill_dedup_identity(ev)


# ============================================================
# helpers
# ============================================================

def _get_first(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
        if isinstance(obj, dict) and n in obj and obj[n] is not None:
            return obj[n]
    return default


def _looks_like_canonical_fill(fill: Any) -> bool:
    required = ("venue", "symbol", "order_id", "trade_id", "fill_id", "side", "qty", "price")
    return all(getattr(fill, name, None) is not None for name in required)


def _coerce_ts(ts: Any, *, ts_ms: Any = None) -> datetime:
    # priority: explicit datetime
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)

    # milliseconds timestamp
    if ts_ms is not None:
        try:
            ms = int(ts_ms)
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except Exception:
            pass

    # seconds timestamp
    if ts is not None:
        try:
            sec = float(ts)
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        except Exception:
            pass

    # fallback: now(UTC)
    return datetime.now(tz=timezone.utc)
