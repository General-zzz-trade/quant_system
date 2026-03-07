# execution/ingress/order_router.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional
import hashlib
import json

from engine.coordinator import EngineCoordinator

from _quant_hotpath import RustPayloadDedupGuard as _RustPayloadDedupGuard


def _stable_sha256(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


@dataclass(slots=True)
class OrderIngressRouter:
    """
    订单回报（ORDER_UPDATE）入口：

    - 幂等：同 order_key + 同 payload_digest => drop（返回 False）
    - 数据损坏：同 order_key + 不同 payload_digest => fail fast（抛 ValueError）
    - 发入 engine：event_type="ORDER_UPDATE"（应走 PIPELINE）
    """
    coordinator: EngineCoordinator
    default_actor: str = "venue:unknown"

    # slots=True 时，内部状态必须声明成字段
    _dedup: Any = field(default_factory=_RustPayloadDedupGuard, init=False, repr=False)

    def ingest_canonical_order(self, order: Any, *, actor: Optional[str] = None) -> bool:
        venue = getattr(order, "venue", None) or "unknown"
        symbol = getattr(order, "symbol", None) or getattr(order, "s", None)

        order_id = getattr(order, "order_id", None) or getattr(order, "id", None)
        client_order_id = getattr(order, "client_order_id", None) or getattr(order, "client_id", None)

        status = getattr(order, "status", None)
        side = getattr(order, "side", None)
        order_type = getattr(order, "order_type", None) or getattr(order, "type", None)
        tif = getattr(order, "tif", None)

        qty = getattr(order, "qty", None)
        price = getattr(order, "price", None)
        filled_qty = getattr(order, "filled_qty", None)
        avg_price = getattr(order, "avg_price", None)

        ts_ms = getattr(order, "ts_ms", None)
        if ts_ms is None:
            ts = datetime.now(tz=timezone.utc)
        else:
            ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)

        order_key = getattr(order, "order_key", None)
        if not order_key:
            order_key = f"{venue}:{symbol}:order:{order_id or client_order_id}"

        payload_digest = getattr(order, "payload_digest", None)
        if not payload_digest:
            payload_digest = _stable_sha256(
                {
                    "venue": venue,
                    "symbol": symbol,
                    "order_id": order_id,
                    "client_order_id": client_order_id,
                    "status": status,
                    "side": side,
                    "order_type": order_type,
                    "tif": tif,
                    "qty": qty,
                    "price": price,
                    "filled_qty": filled_qty,
                    "avg_price": avg_price,
                    "ts": ts.isoformat(),
                }
            )

        if not self._dedup.check_and_insert(order_key, payload_digest):
            return False

        ev_id = f"order_update:{order_key}:{payload_digest[:12]}"
        header = SimpleNamespace(ts=ts, event_id=ev_id)

        ev = SimpleNamespace(
            event_type="ORDER_UPDATE",
            header=header,
            venue=venue,
            symbol=symbol,
            order_id=order_id,
            client_order_id=client_order_id,
            status=status,
            side=side,
            order_type=order_type,
            tif=tif,
            qty=qty,
            price=price,
            filled_qty=filled_qty,
            avg_price=avg_price,
            order_key=order_key,
            payload_digest=payload_digest,
        )

        self.coordinator.emit(ev, actor=actor or self.default_actor)
        return True
