from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Mapping, Type

from event.types import BaseEvent, EventType
from event.header import EventHeader


# ============================================================
# 常量（制度级）
# ============================================================

K_PROTO_VERSION = "proto"
K_EVENT_TYPE = "event_type"
K_HEADER = "header"
K_BODY = "body"

# ============================================================
# JSON default encoder（制度级）
# - Decimal -> str（避免精度损失）
# - datetime -> ISO8601(UTC, Z)，必须 tz-aware
# - Enum -> value
# ============================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            raise EventCodecError("datetime must be tz-aware")
        return obj.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


PROTO_VERSION = 1


# ============================================================
# 异常
# ============================================================

class EventCodecError(RuntimeError):
    pass


# ============================================================
# Registry —— EventType -> EventClass
# ============================================================

class EventCodecRegistry:
    _registry: Dict[EventType, Type[BaseEvent]] = {}

    @classmethod
    def register(cls, event_cls: Type[BaseEvent]) -> None:
        """
        注册事件类型（制度级）

        规则：
        - event_cls.event_type 必须是 EventType
        - registry key 以 EventType 为唯一真理源
        - 不允许重复注册
        """
        event_type = getattr(event_cls, "event_type", None)

        if not isinstance(event_type, EventType):
            raise EventCodecError(
                f"{event_cls.__name__}.event_type 必须是 EventType"
            )

        if event_type in cls._registry:
            raise EventCodecError(
                f"event_type 已注册: {event_type}"
            )

        cls._registry[event_type] = event_cls

    @classmethod
    def get(cls, event_type: EventType) -> Type[BaseEvent]:
        try:
            return cls._registry[event_type]
        except KeyError:
            raise EventCodecError(
                f"未注册的 event_type: {event_type}"
            )

    @classmethod
    def has(cls, event_type: EventType) -> bool:
        return event_type in cls._registry

    @classmethod
    def assert_ready(cls) -> None:
        if not cls._registry:
            raise EventCodecError("EventCodecRegistry 为空，未注册任何事件类型")


# ============================================================
# 编码
# ============================================================

def encode_event(event: BaseEvent) -> Dict[str, Any]:
    """
    将事件编码为 dict（制度级、可回放）
    """
    EventCodecRegistry.assert_ready()

    event_type = event.event_type
    if not isinstance(event_type, EventType):
        raise EventCodecError("event.event_type 必须是 EventType")

    if not EventCodecRegistry.has(event_type):
        raise EventCodecError(f"未注册的 event_type: {event_type}")

    header = event.header
    if not isinstance(header, EventHeader):
        raise EventCodecError("event.header 必须是 EventHeader")

    header.validate()

    return {
        K_PROTO_VERSION: PROTO_VERSION,
        K_EVENT_TYPE: event_type.value,  # 仅在 payload 中使用 str
        K_HEADER: header.to_dict(),
        K_BODY: event.to_dict(),
    }


def encode_event_json(event: BaseEvent) -> str:
    """
    JSON 编码（跨进程 / 存储）

    约束：
    - 必须可稳定序列化（datetime/Decimal 等）
    - 失败应抛出 EventCodecError（fail-fast）
    """
    payload = encode_event(event)
    try:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError) as e:
        raise EventCodecError(f"JSON 编码失败: {e}") from e


# ============================================================
# 解码
# ============================================================

def decode_event(payload: Mapping[str, Any]) -> BaseEvent:
    """
    从 dict 解码事件（制度级、fail-fast）
    """
    EventCodecRegistry.assert_ready()

    # -------- proto --------
    proto = payload.get(K_PROTO_VERSION)
    if proto != PROTO_VERSION:
        raise EventCodecError(
            f"不支持的 proto 版本: {proto}"
        )

    # -------- event_type --------
    raw_type = payload.get(K_EVENT_TYPE)
    if not isinstance(raw_type, str):
        raise EventCodecError("payload 缺少合法 event_type")

    try:
        event_type = EventType(raw_type)
    except ValueError:
        raise EventCodecError(f"未知的 event_type: {raw_type}")

    event_cls = EventCodecRegistry.get(event_type)

    # -------- header --------
    header_raw = payload.get(K_HEADER)
    if not isinstance(header_raw, Mapping):
        raise EventCodecError("payload 缺少合法 header")

    header = EventHeader.from_dict(dict(header_raw))

    # -------- body --------
    body = payload.get(K_BODY)
    if not isinstance(body, Mapping):
        raise EventCodecError("payload 缺少合法 body")

    event = event_cls.from_dict(
        header=header,
        body=body,
    )

    # -------- 一致性校验 --------
    if event.event_type is not event_type:
        raise EventCodecError(
            f"event_type 不一致: payload={event_type}, event={event.event_type}"
        )

    return event


def decode_event_json(raw: str) -> BaseEvent:
    """
    JSON 解码
    """
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise EventCodecError(f"JSON 解析失败: {e}")

    if not isinstance(payload, Mapping):
        raise EventCodecError("JSON payload 非 dict")

    return decode_event(payload)
