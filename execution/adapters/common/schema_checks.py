# execution/adapters/common/schema_checks.py
"""Lightweight schema validation for raw venue messages."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """交易所消息格式不符合预期。"""


def require_keys(
    data: Mapping[str, Any],
    keys: Sequence[str],
    *,
    context: str = "",
) -> None:
    """检查 dict 中必须包含指定 key。"""
    missing = [k for k in keys if k not in data]
    if missing:
        ctx = f" ({context})" if context else ""
        raise SchemaError(f"missing required keys{ctx}: {missing}")


def require_non_empty(
    data: Mapping[str, Any],
    keys: Sequence[str],
    *,
    context: str = "",
) -> None:
    """检查指定 key 存在且值非空。"""
    for k in keys:
        v = data.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            ctx = f" ({context})" if context else ""
            raise SchemaError(f"key {k!r} is empty or missing{ctx}")


def safe_get(
    data: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """从 dict 中按优先级获取值 — 返回第一个非 None 的。"""
    for k in keys:
        v = data.get(k)
        if v is not None:
            return v
    return default
