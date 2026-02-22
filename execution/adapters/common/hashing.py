# execution/adapters/common/hashing.py
"""Hashing utilities for payload digest and deduplication."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def payload_digest(data: Mapping[str, Any], *, length: int = 16) -> str:
    """
    计算 payload 的 SHA256 摘要（截断到 length 个字符）。

    用途：
    1. 幂等检查 — 同 key 不同 payload => 数据损坏
    2. 变更检测 — 同 key 同 digest => 已处理
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:length]


def stable_hash(text: str, *, length: int = 16) -> str:
    """对纯文本做稳定 hash。"""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:length]


def fill_key(*, venue: str, symbol: str, fill_id: str) -> str:
    """成交去重键。"""
    return f"{venue}|{symbol}|{fill_id}"


def order_key(*, venue: str, symbol: str, order_id: str) -> str:
    """订单去重键。"""
    return f"{venue}|{symbol}|{order_id}"
