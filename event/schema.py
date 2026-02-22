# event/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from event.types import BaseEvent, EventType
from event.errors import SchemaNotFoundError, EventValidationError


# ============================================================
# Helpers
# ============================================================


def _to_decimal(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, int):
        return Decimal(v)
    if isinstance(v, float):
        # float -> str -> Decimal: still not ideal, but avoids binary float repr where possible
        return Decimal(str(v))
    if isinstance(v, str):
        return Decimal(v)
    raise TypeError(f"cannot convert to Decimal: {type(v).__name__}")


def _type_name(t: type) -> str:
    try:
        return t.__name__
    except Exception:
        return str(t)


# ============================================================
# Schema Field Spec
# ============================================================


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """字段规格定义（制度级）

    设计原则：
    - schema 负责“结构 + 类型 + 值域”三层约束
    - 不修改 event，只做校验
    """

    name: str
    required: bool = True

    # 类型约束：若为 None，则只做 required 检查
    types: Optional[Tuple[type, ...]] = None

    # 值允许为 None（即使 required=True，也允许显式 None）
    allow_none: bool = False

    # 枚举集合
    enum: Optional[Tuple[Any, ...]] = None

    # 数值约束（适用于 int/Decimal/float/str 可转 Decimal）
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None

    # 长度约束（适用于 str / Sequence）
    min_len: Optional[int] = None
    max_len: Optional[int] = None

    # 字符串正则
    pattern: Optional[str] = None

    # datetime 是否必须 tz-aware（仅当 types 包含 datetime 或值为 datetime 时启用）
    tz_aware: Optional[bool] = None

    # 自定义校验：签名 validator(value) -> None
    validator: Optional[Callable[[Any], None]] = None

    def validate(self, value: Any) -> None:
        # ---- None 处理 ----
        if value is None:
            if self.allow_none:
                return
            raise EventValidationError(f"field '{self.name}' is None")

        # ---- 类型检查 ----
        if self.types is not None:
            if not isinstance(value, self.types):
                allowed = ",".join(_type_name(t) for t in self.types)
                raise EventValidationError(
                    f"field '{self.name}' type mismatch: {type(value).__name__} not in ({allowed})"
                )

        # ---- datetime tz-aware ----
        if isinstance(value, datetime) and self.tz_aware is not None:
            if self.tz_aware and value.tzinfo is None:
                raise EventValidationError(f"field '{self.name}' must be tz-aware datetime")
            if (not self.tz_aware) and value.tzinfo is not None:
                raise EventValidationError(f"field '{self.name}' must be naive datetime")

        # ---- enum ----
        if self.enum is not None:
            if value not in self.enum:
                raise EventValidationError(
                    f"field '{self.name}' not in enum: {value} not in {list(self.enum)}"
                )

        # ---- length constraints ----
        if self.min_len is not None or self.max_len is not None:
            if isinstance(value, (str, bytes, Sequence)):
                try:
                    ln = len(value)
                except Exception:
                    ln = None
                if ln is None:
                    raise EventValidationError(f"field '{self.name}' has no length")

                if self.min_len is not None and ln < self.min_len:
                    raise EventValidationError(f"field '{self.name}' length {ln} < {self.min_len}")
                if self.max_len is not None and ln > self.max_len:
                    raise EventValidationError(f"field '{self.name}' length {ln} > {self.max_len}")

        # ---- string pattern ----
        if self.pattern is not None:
            if not isinstance(value, str):
                raise EventValidationError(f"field '{self.name}' pattern requires str")
            if re.fullmatch(self.pattern, value) is None:
                raise EventValidationError(f"field '{self.name}' does not match pattern")

        # ---- numeric range ----
        if self.min_value is not None or self.max_value is not None:
            # 仅当 value 是数值/可转 Decimal 时启用
            try:
                dv = _to_decimal(value)
            except Exception as e:
                raise EventValidationError(
                    f"field '{self.name}' range check requires numeric: {type(value).__name__}"
                ) from e

            if self.min_value is not None:
                dmin = _to_decimal(self.min_value)
                if dv < dmin:
                    raise EventValidationError(f"field '{self.name}' {dv} < min {dmin}")

            if self.max_value is not None:
                dmax = _to_decimal(self.max_value)
                if dv > dmax:
                    raise EventValidationError(f"field '{self.name}' {dv} > max {dmax}")

        # ---- custom validator ----
        if self.validator is not None:
            try:
                self.validator(value)
            except EventValidationError:
                raise
            except Exception as e:
                raise EventValidationError(f"field '{self.name}' custom validator failed") from e


# ============================================================
# Event Schema
# ============================================================


PostValidator = Callable[[Mapping[str, Any], BaseEvent], None]


@dataclass(frozen=True)
class EventSchema:
    """EventSchema —— BaseEvent 的结构合同

    校验对象：
    - event.event_type
    - event.version
    - event.to_dict()（body）

    顶级机构常用做法：
    - allow_extra_fields=False（尽早发现事件漂移）
    - 字段类型/值域/时区强约束
    """

    event_type: EventType
    version: int
    fields: Tuple[FieldSpec, ...]

    allow_extra_fields: bool = True
    post_validators: Tuple[PostValidator, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate FieldSpec names in schema: {names}")

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def validate_event(self, event: BaseEvent) -> None:
        # -------- 1) 类型与版本 --------
        if event.event_type != self.event_type:
            raise EventValidationError(
                f"event_type mismatch: {event.event_type} != {self.event_type}"
            )

        if event.version != self.version:
            raise EventValidationError(
                f"version mismatch: {event.version} != {self.version}"
            )

        # -------- 2) body 结构 --------
        body_any = event.to_dict()
        if not isinstance(body_any, Mapping):
            raise EventValidationError("event.to_dict() must return a Mapping")

        body: Dict[str, Any] = dict(body_any)

        # required + field-level validate
        for spec in self.fields:
            if spec.name not in body:
                if spec.required:
                    raise EventValidationError(f"missing required field: {spec.name}")
                continue
            spec.validate(body.get(spec.name))

        # extra field drift detection
        if not self.allow_extra_fields:
            allowed = set(self.field_names)
            extra = [k for k in body.keys() if k not in allowed]
            if extra:
                raise EventValidationError(f"extra fields not allowed: {extra}")

        # -------- 3) post validators --------
        for pv in self.post_validators:
            try:
                pv(body, event)
            except EventValidationError:
                raise
            except Exception as e:
                raise EventValidationError("schema post-validator failed") from e

    # runtime 统一入口
    def validate(self, event: BaseEvent) -> None:
        self.validate_event(event)


# ============================================================
# Schema Registry
# ============================================================


class SchemaRegistry:
    """SchemaRegistry —— (event_type, version) → EventSchema"""

    def __init__(self) -> None:
        self._schemas: Dict[Tuple[EventType, int], EventSchema] = {}

    def register(self, schema: EventSchema) -> None:
        key = (schema.event_type, schema.version)
        self._schemas[key] = schema

    def get(self, event_type: EventType, version: int) -> EventSchema:
        key = (event_type, version)
        try:
            return self._schemas[key]
        except KeyError:
            raise SchemaNotFoundError(f"schema not found: {event_type} v{version}")

    def all(self) -> Iterable[EventSchema]:
        return self._schemas.values()
