# execution/config/mapping_config.py
"""Field mapping configuration for venue-to-canonical normalization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """单个字段的映射规则。"""
    source_key: str             # 交易所原始字段名
    target_key: str             # 标准化字段名
    transform: Optional[str] = None  # "lower" / "upper" / "decimal" / "int" / "bool" / None
    default: Optional[str] = None


@dataclass(frozen=True, slots=True)
class StatusMapping:
    """交易所状态码到标准 OrderStatus 的映射。"""
    venue_status: str
    canonical_status: str


@dataclass(frozen=True, slots=True)
class SideMapping:
    """交易所买卖方向到标准 side 的映射。"""
    venue_side: str
    canonical_side: str     # "buy" / "sell"


@dataclass(frozen=True, slots=True)
class MappingConfig:
    """
    一个交易所的完整字段映射配置。

    用于将交易所的原始 WS/REST 数据转换为 CanonicalOrder / CanonicalFill。
    """
    venue: str

    # 订单字段映射
    order_fields: tuple[FieldMapping, ...] = ()

    # 成交字段映射
    fill_fields: tuple[FieldMapping, ...] = ()

    # 状态映射
    status_map: tuple[StatusMapping, ...] = ()

    # 方向映射
    side_map: tuple[SideMapping, ...] = ()

    # symbol 转换规则
    symbol_transform: Optional[str] = None   # "upper" / None

    def resolve_status(self, venue_status: str) -> Optional[str]:
        vs = venue_status.upper()
        for sm in self.status_map:
            if sm.venue_status.upper() == vs:
                return sm.canonical_status
        return None

    def resolve_side(self, venue_side: str) -> Optional[str]:
        vs = venue_side.upper()
        for sm in self.side_map:
            if sm.venue_side.upper() == vs:
                return sm.canonical_side
        return None

    def resolve_field(
        self, fields: tuple[FieldMapping, ...], raw: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for fm in fields:
            val = raw.get(fm.source_key, fm.default)
            if val is not None and fm.transform:
                val = _apply_transform(val, fm.transform)
            result[fm.target_key] = val
        return result


def _apply_transform(value: Any, transform: str) -> Any:
    if transform == "lower":
        return str(value).lower()
    if transform == "upper":
        return str(value).upper()
    if transform == "decimal":
        from decimal import Decimal
        return Decimal(str(value))
    if transform == "int":
        return int(value)
    if transform == "bool":
        return bool(value)
    return value
