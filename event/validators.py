# event/validators.py
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from event.errors import EventValidationError
from event.header import EventHeader
from event.lifecycle import EventLifecycle, LifecycleState


# ============================================================
# 基础抽象（冻结）
# ============================================================

class BaseValidator:
    """校验器基类：只负责校验，不做副作用（冻结）"""

    name: str = "validator"

    def validate(self, event: Any) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CompositeValidator(BaseValidator):
    """
    组合校验器：按顺序执行多个校验器（冻结）
    - 任意一个失败即抛异常
    """
    validators: Tuple[BaseValidator, ...] = field(default_factory=tuple)
    name: str = "composite"

    def validate(self, event: Any) -> None:
        for v in self.validators:
            v.validate(event)


# ============================================================
# 工具函数（冻结）
# ============================================================

def _get_header(event: Any) -> EventHeader:
    header = getattr(event, "header", None)
    if not isinstance(header, EventHeader):
        raise EventValidationError("事件缺少合法 EventHeader")
    return header


def _get_meta(header: EventHeader) -> Mapping[str, Any]:
    meta = getattr(header, "meta", None)
    if isinstance(meta, Mapping):
        return meta
    return {}


def _get_event_time(header: EventHeader) -> datetime:
    et = getattr(header, "event_time", None)
    if not isinstance(et, datetime):
        raise EventValidationError("EventHeader.event_time 非法或缺失")
    if et.tzinfo is None:
        raise EventValidationError("EventHeader.event_time 必须为 tz-aware datetime")
    return et


def _default_stream_key(event: Any) -> Tuple[str, str, str, str]:
    """
    默认“流键”：
    - event_type：区分事件类别（核心）
    - symbol/timeframe：区分行情流（如果有）
    - source：区分数据源（如果 header 有）
    目标：避免多品种/多流交错导致“全局单调”误杀
    """
    h = _get_header(event)
    event_type = str(getattr(h, "event_type", "")) or getattr(event, "EVENT_TYPE", "") or event.__class__.__name__
    symbol = str(getattr(event, "symbol", "")) if getattr(event, "symbol", None) is not None else ""
    timeframe = str(getattr(event, "timeframe", "")) if getattr(event, "timeframe", None) is not None else ""
    source = str(getattr(h, "source", "")) if getattr(h, "source", None) is not None else ""
    return (event_type, symbol, timeframe, source)


# ============================================================
# 具体校验器（冻结）
# ============================================================

@dataclass(slots=True)
class RequiredHeaderFieldsValidator(BaseValidator):
    """
    Header 字段必备性校验（冻结）
    - 只检查“必须存在且类型正确”的最小集合
    """
    name: str = "required-header"

    def validate(self, event: Any) -> None:
        h = _get_header(event)
        _ = _get_event_time(h)

        # event_id / trace_id 是否存在取决于你的 header 宪法
        # 这里不强制 event_id（由 header.validate() 负责），只做轻量兜底
        eid = getattr(h, "event_id", None)
        if eid is not None and not isinstance(eid, str):
            raise EventValidationError("EventHeader.event_id 类型必须为 str")


@dataclass(slots=True)
class LifecycleTerminalValidator(BaseValidator):
    """
    终态阻断校验（冻结）
    - 防止终态事件被重复注入/重复处理
    """
    lifecycle: EventLifecycle
    block_terminal: bool = True
    name: str = "lifecycle-terminal"

    def validate(self, event: Any) -> None:
        if not self.block_terminal:
            return
        state = self.lifecycle.state_of(event)
        if state in (LifecycleState.DISPATCHED, LifecycleState.FAILED, LifecycleState.DROPPED):
            raise EventValidationError(f"事件已处于终态，禁止再次注入: state={state.value}")


@dataclass(slots=True)
class StreamMonotonicTimeValidator(BaseValidator):
    """
    按“流键”做时间单调校验（冻结）
    - 解决多品种/多流交错的误杀问题
    - 默认线程安全（可关闭锁，但不建议）
    """
    stream_key_fn: Callable[[Any], Tuple[str, ...]] = _default_stream_key
    allow_equal: bool = False
    thread_safe: bool = True
    name: str = "time-monotonic-by-stream"

    # 内部状态：每个 stream_key 对应 last_ts
    _last_ts: Dict[Tuple[str, ...], datetime] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def validate(self, event: Any) -> None:
        h = _get_header(event)
        ts = _get_event_time(h)
        key = self.stream_key_fn(event)

        if self.thread_safe:
            with self._lock:
                self._check_and_update(key, ts)
        else:
            self._check_and_update(key, ts)

    def _check_and_update(self, key: Tuple[str, ...], ts: datetime) -> None:
        last = self._last_ts.get(key)
        if last is None:
            self._last_ts[key] = ts
            return

        if self.allow_equal:
            if ts < last:
                raise EventValidationError(f"事件时间倒流: key={key}, ts={ts}, last={last}")
        else:
            if ts <= last:
                raise EventValidationError(f"事件时间非递增: key={key}, ts={ts}, last={last}")

        self._last_ts[key] = ts


@dataclass(frozen=True, slots=True)
class ReplayModeValidator(BaseValidator):
    """
    回放/实盘模式限制（冻结）
    - 允许通过 header.mode 或 header.meta['mode'] 标记
    - 若没有标记，则默认放行（安全兜底）
    """
    replay_only: bool = False
    live_only: bool = False
    name: str = "replay-mode"

    def validate(self, event: Any) -> None:
        if not self.replay_only and not self.live_only:
            return

        h = _get_header(event)
        meta = _get_meta(h)

        mode = getattr(h, "mode", None)
        if mode is None:
            mode = meta.get("mode")

        # 未提供 mode：不阻断
        if mode is None:
            return

        mode_str = str(mode).lower()
        is_replay = mode_str in ("replay", "backtest", "paper_replay")
        is_live = mode_str in ("live", "paper_live", "production")

        if self.replay_only and not is_replay:
            raise EventValidationError(f"该事件仅允许回放模式: mode={mode_str}")

        if self.live_only and not is_live:
            raise EventValidationError(f"该事件仅允许实盘模式: mode={mode_str}")


@dataclass(frozen=True, slots=True)
class TraceValidator(BaseValidator):
    """
    Trace 一致性校验（冻结）
    - 兼容 trace 字段落在 header.* 或 header.meta
    - 不强制必须存在 trace；但一旦存在则必须自洽
    """
    name: str = "trace"

    def validate(self, event: Any) -> None:
        h = _get_header(event)
        meta = _get_meta(h)

        trace_id = getattr(h, "trace_id", None) or meta.get("trace_id")
        span_id = getattr(h, "span_id", None) or meta.get("span_id")
        parent_span_id = getattr(h, "parent_span_id", None) or meta.get("parent_span_id")
        depth = getattr(h, "trace_depth", None) or meta.get("trace_depth")

        # 完全没有 trace 信息：放行
        if trace_id is None and span_id is None and parent_span_id is None and depth is None:
            return

        # trace_id 一旦参与，必须为 str 且非空
        if not isinstance(trace_id, str) or not trace_id:
            raise EventValidationError("trace_id 非法（存在 trace 字段时必须为非空 str）")

        # span_id 可选，但若提供则必须为非空 str
        if span_id is not None and (not isinstance(span_id, str) or not span_id):
            raise EventValidationError("span_id 非法（必须为非空 str）")

        # parent_span_id 可选，但若提供则必须为非空 str
        if parent_span_id is not None and (not isinstance(parent_span_id, str) or not parent_span_id):
            raise EventValidationError("parent_span_id 非法（必须为非空 str）")

        # depth 可选，但若提供则必须为非负 int
        if depth is not None:
            try:
                d = int(depth)
            except Exception as exc:
                raise EventValidationError("trace_depth 必须可转为 int") from exc
            if d < 0:
                raise EventValidationError("trace_depth 不能为负数")


# ============================================================
# 默认构建（冻结）
# ============================================================

def build_default_validators(
    *,
    lifecycle: Optional[EventLifecycle] = None,
    enable_time_monotonic: bool = True,
    time_allow_equal: bool = False,
) -> EventValidators:
    """
    构建默认校验链（冻结）

    设计原则：
    - 默认链必须“可用于多品种/多流”
    - 时间单调采用“按流键单调”，避免误杀
    - lifecycle 校验仅在传入 lifecycle 时启用
    """
    vs: list[BaseValidator] = [
        RequiredHeaderFieldsValidator(),
        TraceValidator(),
    ]

    if enable_time_monotonic:
        vs.append(StreamMonotonicTimeValidator(allow_equal=time_allow_equal, thread_safe=True))

    if lifecycle is not None:
        vs.append(LifecycleTerminalValidator(lifecycle=lifecycle, block_terminal=True))

    return EventValidators(validator=CompositeValidator(validators=tuple(vs)))


# ============================================================
# EventValidators 外观（冻结）
# ============================================================

@dataclass(frozen=True, slots=True)
class EventValidators:
    """
    Runtime 使用的统一入口（冻结）
    """
    validator: BaseValidator

    def validate(self, event: Any) -> None:
        self.validator.validate(event)


# --- Rust acceleration ---
try:
    from _quant_hotpath import RustEventValidator  # noqa: F401
    _RUST_EVENT_VALIDATOR_AVAILABLE = True
except ImportError:
    _RUST_EVENT_VALIDATOR_AVAILABLE = False
