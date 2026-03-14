# execution/ingress/quarantine.py
"""Quarantine for suspicious or malformed execution messages."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class QuarantineReason(str, Enum):
    """隔离原因。"""
    SCHEMA_ERROR = "schema_error"
    DIGEST_MISMATCH = "digest_mismatch"
    SEQUENCE_GAP = "sequence_gap"
    STALE_TIMESTAMP = "stale_timestamp"
    UNKNOWN_SYMBOL = "unknown_symbol"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QuarantinedMessage:
    """被隔离的消息。"""
    message_id: str
    reason: QuarantineReason
    detail: str
    payload: Mapping[str, Any]
    quarantined_at: float
    venue: str = ""
    symbol: str = ""


class QuarantineStore:
    """
    消息隔离区。

    可疑消息先进入隔离区，经人工确认或自动超时后处理。
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._messages: Dict[str, QuarantinedMessage] = {}

    def quarantine(
        self,
        *,
        message_id: str,
        reason: QuarantineReason,
        detail: str,
        payload: Mapping[str, Any],
        venue: str = "",
        symbol: str = "",
    ) -> QuarantinedMessage:
        """将消息放入隔离区。"""
        msg = QuarantinedMessage(
            message_id=message_id,
            reason=reason,
            detail=detail,
            payload=payload,
            quarantined_at=time.monotonic(),
            venue=venue,
            symbol=symbol,
        )
        self._messages[message_id] = msg
        # 防止无限增长
        if len(self._messages) > self._max_size:
            oldest_key = next(iter(self._messages))
            del self._messages[oldest_key]
        return msg

    def release(self, message_id: str) -> Optional[QuarantinedMessage]:
        """释放隔离消息。"""
        return self._messages.pop(message_id, None)

    def get(self, message_id: str) -> Optional[QuarantinedMessage]:
        return self._messages.get(message_id)

    @property
    def pending(self) -> Sequence[QuarantinedMessage]:
        return list(self._messages.values())

    @property
    def count(self) -> int:
        return len(self._messages)

    def clear(self) -> int:
        n = len(self._messages)
        self._messages.clear()
        return n
