# engine/replay.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Protocol

# engine 侧
from engine.dispatcher import EventDispatcher, Route

# event 侧（仅作类型契约，不做实现依赖）
class EventSource(Protocol):
    """
    事件来源契约：
    - 只负责“按顺序提供事件”
    - 不关心 engine / state / reducer
    """
    def __iter__(self) -> Iterator[Any]: ...


class EventSink(Protocol):
    """
    可选：重放过程的观测器（例如进度、断点、校验）
    """
    def on_event(self, event: Any, index: int) -> None: ...
    def on_start(self, total: Optional[int]) -> None: ...
    def on_finish(self, processed: int) -> None: ...


# ============================================================
# Errors
# ============================================================

class ReplayError(RuntimeError):
    pass


class ReplayInterrupted(ReplayError):
    """外部请求中断 replay（例如校验失败、用户停止）"""


# ============================================================
# Replay Config
# ============================================================

@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """
    replay 的制度性配置（不含任何业务参数）
    """
    strict_order: bool = True           # 是否要求输入事件天然有序
    actor: str = "replay"               # 注入 dispatcher 的 actor
    stop_on_error: bool = True          # handler 抛异常时是否停止
    allow_drop: bool = True             # DROP 事件是否允许存在
    progress_interval: int = 0          # >0 时每 N 个事件回调一次进度


# ============================================================
# Replay Driver (frozen v1.0)
# ============================================================

class EventReplay:
    """
    EventReplay —— 统一的“时间回放驱动器”（冻结版 v1.0）

    设计铁律：
    1) replay 不直接调用 core / pipeline
    2) replay 只通过 dispatcher 注入事件
    3) replay 与 live 走完全相同的因果路径
    4) replay 不理解 state，只负责“把历史再发生一次”
    """

    def __init__(
        self,
        *,
        dispatcher: EventDispatcher,
        source: EventSource,
        sink: Optional[EventSink] = None,
        config: Optional[ReplayConfig] = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._source = source
        self._sink = sink
        self._cfg = config or ReplayConfig()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self) -> int:
        """
        执行 replay。

        返回：
        - 实际处理的事件数量（不含 DROP）
        """
        processed = 0

        total = None
        if hasattr(self._source, "__len__"):
            try:
                total = len(self._source)  # type: ignore[arg-type]
            except Exception:
                total = None

        if self._sink is not None:
            try:
                self._sink.on_start(total)
            except Exception:
                pass

        # strict_order 的比较键必须在相邻事件上可比。
        # 规则：优先使用两边都具备的 ts，其次 event_index，最后使用枚举 idx。
        last_ts: Optional[int] = None
        last_eidx: Optional[int] = None
        last_fb: int = 0

        for idx, event in enumerate(self._source, start=1):
            # 可选：顺序校验（严格但不过度误判）
            if self._cfg.strict_order:
                cur_ts, cur_eidx, cur_fb = self._order_keys(event, fallback=idx)

                if last_ts is not None and cur_ts is not None:
                    if cur_ts < last_ts:
                        raise ReplayError("事件顺序不合法（逆序）")
                elif last_eidx is not None and cur_eidx is not None:
                    if cur_eidx < last_eidx:
                        raise ReplayError("事件顺序不合法（逆序）")
                else:
                    if cur_fb < last_fb:
                        raise ReplayError("事件顺序不合法（逆序）")

                # 更新可用键，便于后续事件选择共同键比较
                if cur_ts is not None:
                    last_ts = cur_ts
                if cur_eidx is not None:
                    last_eidx = cur_eidx
                last_fb = cur_fb

            # DROP 预判（是否允许）
            route = self._preview_route(event)
            if route is Route.DROP and not self._cfg.allow_drop:
                raise ReplayError("存在 DROP 事件且 allow_drop=False")

            # 注入 dispatcher（与 live 完全一致）
            try:
                self._dispatcher.dispatch(event=event, actor=self._cfg.actor)
            except Exception as e:
                if self._cfg.stop_on_error:
                    raise
                # 否则：吞掉异常继续
            else:
                if route is not Route.DROP:
                    processed += 1

            # 观测回调
            if self._sink is not None:
                try:
                    self._sink.on_event(event, processed)
                except ReplayInterrupted:
                    raise
                except Exception:
                    pass

            # 进度回调
            if self._cfg.progress_interval > 0 and processed % self._cfg.progress_interval == 0:
                if self._sink is not None:
                    try:
                        self._sink.on_event(event, processed)
                    except Exception:
                        pass

        if self._sink is not None:
            try:
                self._sink.on_finish(processed)
            except Exception:
                pass

        return processed

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _order_keys(event: Any, fallback: int) -> tuple[Optional[int], Optional[int], int]:
        """Return (ts_key, event_index_key, fallback).

        The replay driver uses strict_order to reject obviously out-of-order inputs, but must avoid
        false positives when some events have ts and others only have event_index.
        """
        # ts can be either on the event itself or inside header.
        ts0 = getattr(event, "ts", None)
        if isinstance(ts0, (int, float)):
            ts_key: Optional[int] = int(ts0)
        else:
            header = getattr(event, "header", None)
            ts1 = getattr(header, "ts", None)
            ts_key = int(ts1) if isinstance(ts1, (int, float)) else None

        header = getattr(event, "header", None)
        eidx = getattr(header, "event_index", None)
        eidx_key: Optional[int] = int(eidx) if isinstance(eidx, int) else None

        return ts_key, eidx_key, int(fallback)

    def _preview_route(self, event: Any) -> Route:
        """
        在不 dispatch 的情况下，预判路由（用于校验）
        """
        # 复用 dispatcher 的私有规则（不调用 handlers）
        # 这里直接调用受保护方法是“制度允许”的：
        # replay 属于 engine 内部设施
        return self._dispatcher._route_for(event)  # type: ignore[attr-defined]
