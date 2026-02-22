# core.py
from __future__ import annotations

from typing import Any, Mapping, Optional

from engine.coordinator import EngineCoordinator


class CoreFacade:
    """
    Core Facade（冻结版 v1.0）

    职责：
    - 作为 legacy / strategy / 外部代码的统一入口
    - 所有事件转发给 EngineCoordinator
    - 不做任何 state 更新、不做时间、不做调度
    """

    def __init__(self, coordinator: EngineCoordinator) -> None:
        self._coord = coordinator

    # -------------------------
    # Event entry
    # -------------------------

    def emit(self, event: Any, *, actor: str = "core") -> None:
        """
        唯一合法入口：转发给 engine
        """
        self._coord.emit(event, actor=actor)

    # -------------------------
    # Legacy helpers（可选）
    # -------------------------

    def on_market_event(self, event: Any) -> None:
        self.emit(event, actor="market")

    def on_timer_event(self, event: Any) -> None:
        self.emit(event, actor="timer")

    # -------------------------
    # Read-only state
    # -------------------------

    def get_state(self) -> Mapping[str, Any]:
        """
        只读视图（debug / monitor / legacy）
        """
        return self._coord.get_state_view()

    @property
    def phase(self) -> str:
        return self._coord.phase.value
