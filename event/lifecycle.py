from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================
# LifecycleState —— 运行期状态（制度级、可审计）
# ============================================================

class LifecycleState(str, Enum):
    CREATED = "created"
    ENQUEUED = "enqueued"
    DISPATCH_START = "dispatch_start"
    HANDLED = "handled"
    DISPATCHED = "dispatched"
    RETRY = "retry"
    DROPPED = "dropped"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: Dict[LifecycleState, Set[LifecycleState]] = {
    LifecycleState.CREATED: {LifecycleState.ENQUEUED},
    LifecycleState.ENQUEUED: {LifecycleState.DISPATCH_START},
    LifecycleState.DISPATCH_START: {
        LifecycleState.HANDLED,
        LifecycleState.RETRY,
        LifecycleState.DROPPED,
        LifecycleState.FAILED,
    },
    LifecycleState.HANDLED: {LifecycleState.DISPATCHED},
    LifecycleState.DISPATCHED: set(),
    LifecycleState.RETRY: {LifecycleState.ENQUEUED, LifecycleState.DROPPED, LifecycleState.FAILED},
    LifecycleState.DROPPED: set(),
    LifecycleState.FAILED: set(),
}


# ============================================================
# 异常（只携带 event_id，不携带 event 本体）
# ============================================================

@dataclass(frozen=True, slots=True)
class EventLifecycleError(RuntimeError):
    message: str
    event_id: str

    def __str__(self) -> str:  # 便于日志与定位
        return f"{self.message} (event_id={self.event_id})"


# ============================================================
# EventLifecycle —— 事件运行态管理器（不进入 codec / replay）
# ============================================================

class EventLifecycle:
    def __init__(self) -> None:
        self._lock = Lock()
        self._states: Dict[str, LifecycleState] = {}

    def _event_key(self, event: Any) -> str:
        """
        运行期弱依赖：仅尝试从 event.header.event_id 获取主键。
        不引入对 EventHeader/BaseEvent 的强依赖，避免循环依赖。
        """
        try:
            header = getattr(event, "header", None)
            event_id = getattr(header, "event_id", None)
            if isinstance(event_id, str) and event_id:
                return event_id
        except Exception as e:
            logger.debug("Failed to extract event_id from event: %s", e)
        # 兜底：仅用于开发期/测试期，生产应确保 event_id 存在
        return str(id(event))

    def state_of(self, event: Any) -> Optional[LifecycleState]:
        key = self._event_key(event)
        with self._lock:
            return self._states.get(key)

    def ensure_created(self, event: Any) -> None:
        """
        保证该 event 至少存在 CREATED 状态（幂等）
        """
        key = self._event_key(event)
        with self._lock:
            self._states.setdefault(key, LifecycleState.CREATED)

    def transition(self, event: Any, new_state: LifecycleState) -> LifecycleState:
        """
        推进生命周期状态（严格状态机，非法转移直接失败）
        """
        key = self._event_key(event)
        with self._lock:
            cur = self._states.get(key, LifecycleState.CREATED)

            allowed = _ALLOWED_TRANSITIONS.get(cur)
            if allowed is None:
                raise EventLifecycleError(
                    message=f"未知生命周期状态: {cur}",
                    event_id=key,
                )

            if new_state not in allowed:
                raise EventLifecycleError(
                    message=f"非法生命周期转移: {cur} -> {new_state}",
                    event_id=key,
                )

            self._states[key] = new_state
            return new_state

    def reset(self, event: Any) -> None:
        """
        清理该事件的生命周期（用于测试或短期内存释放）
        """
        key = self._event_key(event)
        with self._lock:
            self._states.pop(key, None)

    def snapshot(self) -> Dict[str, str]:
        """
        返回当前所有事件生命周期的只读快照（用于监控/调试）
        """
        with self._lock:
            return {k: v.value for k, v in self._states.items()}
