# execution/adapters/common/time.py
"""Time utilities for adapter normalization."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


def now_ms() -> int:
    """当前 UTC 毫秒时间戳。"""
    return int(time.time() * 1000)


def now_utc() -> datetime:
    """当前 UTC datetime。"""
    return datetime.now(tz=timezone.utc)


def ms_to_datetime(ms: int) -> datetime:
    """毫秒时间戳转 UTC datetime。"""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """UTC datetime 转毫秒时间戳。"""
    return int(dt.timestamp() * 1000)


def coerce_ts_ms(value: Any) -> int:
    """
    灵活的时间戳转换 — 将各种格式统一为毫秒时间戳。

    支持：int (ms/s), float (s), datetime, str(数字)
    """
    if value is None:
        return now_ms()
    if isinstance(value, datetime):
        return datetime_to_ms(value)
    if isinstance(value, (int, float)):
        v = int(value)
        if v < 2_000_000_000:
            return v * 1000
        return v
    try:
        v = int(str(value))
        if v < 2_000_000_000:
            return v * 1000
        return v
    except (ValueError, TypeError):
        return now_ms()
