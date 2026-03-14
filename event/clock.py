# event/clock.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional



@dataclass
class EventClock:
    """
    EventClock —— 系统时间源（Replay / Live 统一）
    """
    current_time: datetime | None = None
    bar_index: int = 0

    def update_from_event_time(
        self,
        *,
        event_time: datetime,
        bar_index: int,
    ) -> None:
        self.current_time = event_time
        self.bar_index = bar_index
# ============================================================
# Errors
# ============================================================

class ClockError(RuntimeError):
    pass


# ============================================================
# Clock Mode
# ============================================================

class ClockMode(str, Enum):
    REPLAY = "replay"
    LIVE = "live"


# ============================================================
# Snapshot
# ============================================================

@dataclass(frozen=True, slots=True)
class ClockSnapshot:
    """
    Clock 的不可变快照（用于 checkpoint / debug / metrics）
    """
    mode: ClockMode
    ts: datetime
    bar_index: int

    last_event_time: Optional[datetime]
    last_wall_time: Optional[datetime]

    drift_ms: Optional[int]


# ============================================================
# Clock
# ============================================================

class Clock:
    """
    事件驱动时钟（机构级）

    约定：
    - ts 永远代表“系统时间真相”（回放 = 事件时间）
    - LIVE 模式下 wall_time 只用于延迟观测，不参与状态推进
    """

    def __init__(
        self,
        *,
        mode: ClockMode,
        strict_monotonic: bool = True,
        allow_equal_timestamps: bool = False,
    ) -> None:
        self._mode = mode
        self._strict_monotonic = strict_monotonic
        self._allow_equal = allow_equal_timestamps

        self._ts: Optional[datetime] = None
        self._bar_index: int = 0

        self._last_event_time: Optional[datetime] = None
        self._last_wall_time: Optional[datetime] = None

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ClockError("Clock 禁止使用 naive datetime")
        return ts.astimezone(timezone.utc)

    # --------------------------------------------------------
    # 基本访问
    # --------------------------------------------------------

    @property
    def mode(self) -> ClockMode:
        return self._mode

    @property
    def ts(self) -> datetime:
        if self._ts is None:
            raise ClockError("Clock 尚未初始化")
        return self._ts

    @property
    def bar_index(self) -> int:
        return self._bar_index

    # --------------------------------------------------------
    # 核心更新逻辑
    # --------------------------------------------------------

    def update_from_event_time(
        self,
        *,
        event_time: datetime,
        wall_time: Optional[datetime] = None,
        bar_index: Optional[int] = None,
    ) -> None:
        """
        由事件时间推进 Clock（唯一合法推进方式）
        """
        event_time = self._ensure_utc(event_time)

        if wall_time is not None:
            wall_time = self._ensure_utc(wall_time)

        # 单调性校验
        if self._last_event_time is not None:
            if event_time < self._last_event_time:
                if self._strict_monotonic:
                    raise ClockError("事件时间回退")
            elif event_time == self._last_event_time:
                if not self._allow_equal:
                    raise ClockError("事件时间重复")

        # 推进
        self._ts = event_time
        self._last_event_time = event_time

        if bar_index is not None:
            self._bar_index = int(bar_index)
        else:
            self._bar_index += 1

        if self._mode == ClockMode.LIVE:
            self._last_wall_time = wall_time

    # --------------------------------------------------------
    # Reset（受限操作）
    # --------------------------------------------------------

    def reset(
        self,
        *,
        ts: datetime,
        bar_index: int = 0,
        force: bool = False,
    ) -> None:
        """
        重置 Clock

        规则：
        - REPLAY 模式：允许
        - LIVE 模式：必须 force=True
        """
        if self._mode == ClockMode.LIVE and not force:
            raise ClockError("LIVE 模式禁止 reset Clock（需 force=True）")

        ts = self._ensure_utc(ts)

        self._ts = ts
        self._bar_index = int(bar_index)
        self._last_event_time = ts
        self._last_wall_time = None

    # --------------------------------------------------------
    # 比较工具
    # --------------------------------------------------------

    def is_after(self, ts: datetime) -> bool:
        ts = self._ensure_utc(ts)
        return self.ts > ts

    def is_before(self, ts: datetime) -> bool:
        ts = self._ensure_utc(ts)
        return self.ts < ts

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    def snapshot(self) -> ClockSnapshot:
        """
        生成不可变快照
        """
        drift_ms: Optional[int] = None

        if (
            self._mode == ClockMode.LIVE
            and self._last_event_time is not None
            and self._last_wall_time is not None
        ):
            drift = self._last_wall_time - self._last_event_time
            drift_ms = int(drift.total_seconds() * 1000)

        return ClockSnapshot(
            mode=self._mode,
            ts=self.ts,
            bar_index=self._bar_index,
            last_event_time=self._last_event_time,
            last_wall_time=self._last_wall_time,
            drift_ms=drift_ms,
        )
