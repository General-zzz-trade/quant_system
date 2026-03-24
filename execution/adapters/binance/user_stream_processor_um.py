from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, Tuple


from _quant_hotpath import (  # type: ignore[import-untyped]
    rust_demux_user_stream,
    rust_parse_agg_trade,
)

logger = logging.getLogger(__name__)

# Rust-accelerated user stream demux — classifies event type without
# full JSON parsing.  Used in the binary hot path.
_demux_user_stream = rust_demux_user_stream

# Rust aggregate trade parser — extracts price/qty/side from raw JSON
# in a single FFI call.
_parse_agg_trade = rust_parse_agg_trade


class SupportsIngestFill(Protocol):
    def ingest_canonical_fill(self, fill: Any, *, actor: Optional[str] = None) -> bool: ...


class SupportsIngestOrder(Protocol):
    def ingest_canonical_order(self, order: Any, *, actor: Optional[str] = None) -> bool: ...


# -----------------------------
# mapper compatibility helpers
# -----------------------------
def _call_mapper(
    mapper: Any,
    payload: Mapping[str, Any],
    method_names: Tuple[str, ...],
) -> Any:
    """
    兼容不同 mapper 方法命名：
    1) 优先按 method_names 尝试
    2) 其次尝试 mapper.map(payload)
    3) 最后自动发现：扫描 mapper 上的 map*/from_* 方法，按关键字打分选最匹配的
    同时支持 payload root / payload["o"] 两种入参形态。
    """
    if mapper is None:
        return None

    # mapper 也可能就是一个函数
    if callable(mapper) and not hasattr(mapper, "__dict__"):
        return mapper(payload)

    alts: list[Mapping[str, Any]] = [payload]
    o = payload.get("o")
    if isinstance(o, Mapping):
        alts.append(o)

    # 1) 固定候选方法名
    for name in method_names:
        fn = getattr(mapper, name, None)
        if callable(fn):
            return _try_payloads(fn, alts)

    # 2) map(payload) 兜底
    fn = getattr(mapper, "map", None)
    if callable(fn):
        return _try_payloads(fn, alts)

    # 3) 自动发现
    fn2 = _auto_pick_mapper_fn(mapper, method_names=method_names, payload=payload)
    if fn2 is not None:
        return _try_payloads(fn2, alts)

    raise AttributeError(
        f"mapper has no supported method among: {method_names} or map(), and auto-discovery found none"
    )


def _try_payloads(fn: Any, payloads: Iterable[Mapping[str, Any]]) -> Any:
    first_exc: Optional[BaseException] = None
    for p in payloads:
        try:
            return fn(p)
        except (KeyError, TypeError, ValueError) as e:
            if first_exc is None:
                first_exc = e
            continue
    assert first_exc is not None
    raise first_exc


def _auto_pick_mapper_fn(
    mapper: Any,
    *,
    method_names: Tuple[str, ...],
    payload: Mapping[str, Any],
) -> Optional[Any]:
    tokens = set()
    for n in method_names:
        for t in n.lower().split("_"):
            if t:
                tokens.add(t)

    ev = payload.get("e")
    if isinstance(ev, str):
        for t in ev.lower().split("_"):
            tokens.add(t)

    cands: list[tuple[int, str, Any]] = []
    for name in dir(mapper):
        if name.startswith("_"):
            continue
        fn = getattr(mapper, name, None)
        if not callable(fn):
            continue

        lname = name.lower()
        if not (lname.startswith("map") or lname.startswith("from_") or "map" in lname):
            continue

        # 过滤明显不可能的方法（签名不对）
        try:
            sig = inspect.signature(fn)
            # bound method 通常只剩 1 个参数（payload）
            if len(sig.parameters) != 1:
                continue
        except Exception as e:
            logger.debug("Failed to inspect signature of %s: %s", name, e)

        score = 0
        for t in tokens:
            if t and t in lname:
                score += 2
        if "order" in lname:
            score += 1
        if "trade" in lname:
            score += 1
        if "update" in lname:
            score += 1
        if "user" in lname or "stream" in lname or "ws" in lname:
            score += 1

        if score > 0:
            cands.append((score, name, fn))

    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][2]


# -----------------------------
# processor
# -----------------------------
@dataclass(slots=True)
class BinanceUmUserStreamProcessor:
    """
    Binance UM user stream processor:
    - JSON -> demux -> mapper -> ingress
    幂等/坏重复 fail-fast 由 ingress/router 负责。
    """
    order_router: SupportsIngestOrder
    fill_router: SupportsIngestFill
    order_mapper: Any
    fill_mapper: Any
    default_actor: str = "venue:binance"

    def process_raw(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except Exception as e:
            raise ValueError(f"invalid json: {e}") from e
        if not isinstance(payload, Mapping):
            raise ValueError("user stream payload must be a JSON object")
        self.process_event(payload)

    def process_event(self, payload: Mapping[str, Any]) -> None:
        et = str(payload.get("e", "")).strip()
        if et == "ORDER_TRADE_UPDATE":
            self._handle_order_trade_update(payload)
            return
        # 其他事件先忽略（ACCOUNT_UPDATE 之后再接）
        return

    def _handle_order_trade_update(self, payload: Mapping[str, Any]) -> None:
        o = payload.get("o")
        if not isinstance(o, Mapping):
            raise ValueError("ORDER_TRADE_UPDATE missing 'o' object")

        actor = self.default_actor

        # 1) order update
        order_model = _call_mapper(
            self.order_mapper,
            payload,
            method_names=(
                "map_um_user_stream_order_trade_update",
                "map_um_order_trade_update",
                "map_order_trade_update",
                "from_um_order_trade_update",
                "map_user_stream",
                "map_ws_user_stream",
            ),
        )
        if order_model is not None:
            self.order_router.ingest_canonical_order(order_model, actor=actor)

        # 2) trade fill（只在 TRADE 且 l>0 时生成 fill）
        exec_type = str(o.get("x", "")).strip().upper()
        last_fill_qty = o.get("l", None)
        is_trade = exec_type == "TRADE" and last_fill_qty is not None and str(last_fill_qty) not in ("0", "0.0", "")

        if not is_trade:
            return

        fill_model = _call_mapper(
            self.fill_mapper,
            payload,
            method_names=(
                "map_um_user_stream_order_trade_update",
                "map_um_order_trade_update_fill",
                "map_order_trade_update_fill",
                "from_um_order_trade_update",
                "map_user_stream",
                "map_ws_user_stream",
            ),
        )
        if fill_model is not None:
            self.fill_router.ingest_canonical_fill(fill_model, actor=actor)
