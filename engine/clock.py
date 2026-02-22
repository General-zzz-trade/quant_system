# engine/clock.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable
import time


class ClockMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"


@runtime_checkable
class Clock(Protocol):
    """
    Engine 时间制度的唯一来源。

    机构级原则：
    - 所有“当前时间”只能从 clock 获取
    - replay 必须可控、单调、确定性
    - live 必须只读（不可推进）
    """

    @property
    def mode(self) -> ClockMode: ...

    def now(self) -> Any: ...
    def set(self, ts: Any) -> None: ...
    def advance_to(self, ts: Any) -> None: ...
    def advance_by(self, seconds: float) -> None: ...


class ClockError(RuntimeError):
    pass


class ClockMonotonicError(ClockError):
    pass


class ClockImmutableError(ClockError):
    pass


def _to_float_seconds(ts: Any) -> Optional[float]:
    """
    尝试把 ts 转成 float 秒（用于比较/推进）
    支持：
    - int/float
    - 具有 timestamp() 方法的对象（datetime 等）
    - 具有 .ts 或 .timestamp 字段（你后续可按需扩展）
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)

    # datetime-like
    if hasattr(ts, "timestamp") and callable(getattr(ts, "timestamp")):
        try:
            return float(ts.timestamp())
        except Exception:
            return None

    # common attribute names
    for attr in ("ts", "timestamp"):
        if hasattr(ts, attr):
            v = getattr(ts, attr)
            if isinstance(v, (int, float)):
                return float(v)
            if hasattr(v, "timestamp") and callable(getattr(v, "timestamp")):
                try:
                    return float(v.timestamp())
                except Exception:
                    return None
    return None


@dataclass(slots=True)
class ReplayClock:
    """
    Replay 模式时钟：可控、可推进、单调递增。

    设计要求：
    - 初始 ts 可为空：由第一条 event set/advance_to 决定
    - 所有推进必须单调（防止回放产生时间倒流导致状态不确定）
    """
    _ts: Any = None

    @property
    def mode(self) -> ClockMode:
        return ClockMode.REPLAY

    def now(self) -> Any:
        return self._ts

    def set(self, ts: Any) -> None:
        # set 也必须单调（制度强制）
        self.advance_to(ts)

    def advance_to(self, ts: Any) -> None:
        if ts is None:
            return

        if self._ts is None:
            self._ts = ts
            return

        old = _to_float_seconds(self._ts)
        new = _to_float_seconds(ts)

        # 如果无法比较（自定义时间类型），退化为“引用一致性”策略：只允许同对象或字符串化不降序
        if old is None or new is None:
            # 保守：只允许不倒退的“字符串比较”（仍可能不完美，但至少确定性）
            if str(ts) < str(self._ts):
                raise ClockMonotonicError(f"ReplayClock time moved backwards: {self._ts} -> {ts}")
            self._ts = ts
            return

        if new < old:
            raise ClockMonotonicError(f"ReplayClock time moved backwards: {self._ts} -> {ts}")

        self._ts = ts

    def advance_by(self, seconds: float) -> None:
        if seconds < 0:
            raise ClockMonotonicError(f"ReplayClock cannot advance by negative seconds: {seconds}")
        if self._ts is None:
            # 没有基准时间则以 0 为基准（由使用者决定是否允许）
            self._ts = float(seconds)
            return

        old = _to_float_seconds(self._ts)
        if old is None:
            # 无法数值推进，退化：把推进值附加到字符串（确定性但不用于真实交易）
            self._ts = f"{self._ts}+{seconds}s"
            return

        self._ts = old + float(seconds)


@dataclass(slots=True)
class LiveClock:
    """
    Live 模式时钟：只读，不允许推进（防止业务代码改写时间语义）。

    默认：
    - now() 返回 time.time()（wall-clock 秒）
    - 如你未来要用“交易所时间”或 “bar 时间”，应由 Event/Header 提供，
      并在 scheduler/dispatcher 层显式选择使用 event.ts 还是 clock.now()
    """
    time_fn: Any = time.time  # 可注入（测试/模拟）

    @property
    def mode(self) -> ClockMode:
        return ClockMode.LIVE

    def now(self) -> float:
        return float(self.time_fn())

    def set(self, ts: Any) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")

    def advance_to(self, ts: Any) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")

    def advance_by(self, seconds: float) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")
