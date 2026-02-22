# execution/models/serialization.py
"""JSON-friendly serialization helpers for execution models."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any, Dict, Mapping


def _convert_value(v: Any) -> Any:
    """递归转换值为 JSON 兼容类型。"""
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dict):
        return {k: _convert_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_convert_value(item) for item in v]
    if hasattr(v, 'value'):  # Enum
        return v.value
    return v


def model_to_dict(obj: Any) -> Dict[str, Any]:
    """
    将 frozen dataclass 转为 JSON 可序列化 dict。

    - Decimal → str
    - Enum → value
    - 嵌套 dataclass 递归处理
    """
    d = asdict(obj)
    return {k: _convert_value(v) for k, v in d.items()}


def _parse_decimal(v: Any) -> Decimal:
    if v is None:
        return Decimal("0")
    return Decimal(str(v))


def _parse_optional_decimal(v: Any) -> Decimal | None:
    if v is None:
        return None
    return Decimal(str(v))


def dict_to_canonical_order(d: Mapping[str, Any]) -> Any:
    """从 dict 恢复 CanonicalOrder。延迟导入避免循环。"""
    from execution.models.orders import CanonicalOrder
    return CanonicalOrder(
        venue=str(d["venue"]),
        symbol=str(d["symbol"]),
        order_id=str(d["order_id"]),
        client_order_id=d.get("client_order_id"),
        status=str(d["status"]),
        side=str(d["side"]),
        order_type=str(d["order_type"]),
        tif=d.get("tif"),
        qty=_parse_decimal(d["qty"]),
        price=_parse_optional_decimal(d.get("price")),
        filled_qty=_parse_decimal(d.get("filled_qty", "0")),
        avg_price=_parse_optional_decimal(d.get("avg_price")),
        ts_ms=int(d.get("ts_ms", 0)),
        order_key=str(d.get("order_key", "")),
        payload_digest=str(d.get("payload_digest", "")),
        raw=d.get("raw"),
    )


def dict_to_canonical_fill(d: Mapping[str, Any]) -> Any:
    """从 dict 恢复 CanonicalFill。延迟导入避免循环。"""
    from execution.models.fills import CanonicalFill
    return CanonicalFill(
        venue=str(d["venue"]),
        symbol=str(d["symbol"]),
        order_id=str(d["order_id"]),
        trade_id=str(d["trade_id"]),
        fill_id=str(d["fill_id"]),
        side=str(d["side"]),
        qty=_parse_decimal(d["qty"]),
        price=_parse_decimal(d["price"]),
        fee=_parse_decimal(d.get("fee", "0")),
        fee_asset=d.get("fee_asset"),
        liquidity=d.get("liquidity"),
        ts_ms=int(d.get("ts_ms", 0)),
        payload_digest=str(d.get("payload_digest", "")),
        raw=d.get("raw"),
    )
