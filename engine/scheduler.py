# engine/scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple
import time

from engine.clock import Clock, ClockMode, ClockError


# ============================================================
# Events (scheduler 只产生“时间类事件”，不碰业务，不碰 state)
# ============================================================

@dataclass(frozen=True, slots=True)
class TimerEvent:
    """固定间隔触发（heartbeat / risk tick / reconcile tick 等）"""
    ts: Any
    name: str
    interval_s: float
    seq: int


@dataclass(frozen=True, slots=True)
class BarCloseEvent:
    """bar 收盘触发（1m/5m/1h 等）"""
    ts: Any
    symbol: str
    timeframe_s: int
    bar_index: int


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """交易时段边界（可用于 reset / roll / reconcile）"""
    ts: Any
    symbol: str
    kind: str  # "open" | "close"


# ============================================================
# Contracts
# ============================================================

class EmitFn(Protocol):
    def __call__(self, event: Any, *, actor: str) -> None: ...


class Scheduler(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def poll(self) -> int: ...
    def step(self, *, max_events: int = 1) -> int: ...


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True, slots=True)
class TimerSpec:
    name: str
    interval_s: float


@dataclass(frozen=True, slots=True)
class BarSpec:
    symbol: str
    timeframe_s: int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """
    v1.0 冻结边界：
    - scheduler 只负责“时间 -> 事件”
    - 事件统一通过 emit(actor="scheduler") 注入 engine
    - 不做线程化；由外部主循环调用 poll()/step()
    """
    timers: Sequence[TimerSpec] = ()
    bars: Sequence[BarSpec] = ()
    emit_actor: str = "scheduler"

    # Live 模式下最小 sleep（避免 busy loop）
    live_min_sleep_s: float = 0.02

    # Replay 模式下：是否允许补齐缺失 tick（一般不需要）
    # v1.0：保持 False，保证确定性（只在明确触发点发事件）
    replay_fill_ticks: bool = False


# ============================================================
# Utilities
# ============================================================

def _to_float(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp") and callable(getattr(ts, "timestamp")):
        try:
            return float(ts.timestamp())
        except Exception:
            return None
    return None


def _floor_div(a: float, b: float) -> int:
    return int(a // b)


# ============================================================
# Base Implementation
# ============================================================

class BaseScheduler:
    def __init__(self, *, cfg: SchedulerConfig, clock: Clock, emit: EmitFn) -> None:
        self._cfg = cfg
        self._clock = clock
        self._emit = emit

        self._running: bool = False

        # timers bookkeeping
        self._timer_next: Dict[str, float] = {}
        self._timer_seq: Dict[str, int] = {}

        # bars bookkeeping
        self._bar_last_index: Dict[Tuple[str, int], int] = {}  # (symbol, timeframe) -> last_bar_index

        # init schedules
        now = _to_float(self._clock.now())
        if now is None:
            now = 0.0

        for t in cfg.timers:
            self._timer_seq[t.name] = 0
            self._timer_next[t.name] = now + float(t.interval_s)

        for b in cfg.bars:
            # 初始化为 “当前时间所在 bar 的 index - 1”，确保下一次 close 会发出
            tf = int(b.timeframe_s)
            idx = _floor_div(now, tf) - 1
            self._bar_last_index[(b.symbol, tf)] = idx

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def poll(self) -> int:
        """
        Live 模式主循环：外部 while True 调用 poll()
        - 该函数自己会做一个很小 sleep，避免 busy loop
        """
        if not self._running:
            return 0
        n = self._drain(max_events=10_000)  # 上限足够大，避免积压
        # 仅在 live 下 sleep
        if getattr(self._clock, "mode", None) == ClockMode.LIVE:
            time.sleep(self._cfg.live_min_sleep_s)
        return n

    def step(self, *, max_events: int = 1) -> int:
        """
        Replay 模式主循环：外部按 step 驱动（可做单步/快进）
        """
        if not self._running:
            return 0
        return self._drain(max_events=max_events)

    def _drain(self, *, max_events: int) -> int:
        raise NotImplementedError


# ============================================================
# LiveScheduler
# ============================================================

class LiveScheduler(BaseScheduler):
    """
    LiveScheduler：基于 LiveClock.now() 触发 timer/bar close 事件

    注意：
    - 这里的 now() 是 wall clock（默认）
    - 如果你要改成“交易所时间”，应在上层把 clock.now() 换成 exchange clock
    """
    def _drain(self, *, max_events: int) -> int:
        fired = 0
        now_ts = self._clock.now()
        now = _to_float(now_ts)
        if now is None:
            return 0

        # 1) timers
        for t in self._cfg.timers:
            if fired >= max_events:
                break
            name = t.name
            nxt = self._timer_next.get(name)
            if nxt is None:
                continue
            if now >= nxt:
                seq = self._timer_seq.get(name, 0) + 1
                self._timer_seq[name] = seq
                # 递进到下一个触发点（允许跳过多次，避免 drift）
                interval = float(t.interval_s)
                # 保障下一次一定在未来
                k = max(1, int((now - nxt) // interval) + 1)
                self._timer_next[name] = nxt + k * interval

                self._emit(
                    TimerEvent(ts=now_ts, name=name, interval_s=interval, seq=seq),
                    actor=self._cfg.emit_actor,
                )
                fired += 1

        # 2) bars
        for b in self._cfg.bars:
            if fired >= max_events:
                break
            tf = int(b.timeframe_s)
            key = (b.symbol, tf)
            last_idx = self._bar_last_index.get(key, -1)
            cur_idx = _floor_div(now, tf)

            # bar close：当 cur_idx > last_idx 时，意味着至少有一个 bar 已经结束
            if cur_idx > last_idx:
                # 只发“紧邻的下一根 close”（v1.0：避免一次补太多导致 flood）
                next_idx = last_idx + 1
                close_ts_f = (next_idx + 1) * tf  # 该 bar 的 close 时间（秒）
                # close_ts 使用 float 秒，保持一致性（后续可换 datetime）
                close_ts = close_ts_f

                self._emit(
                    BarCloseEvent(ts=close_ts, symbol=b.symbol, timeframe_s=tf, bar_index=next_idx),
                    actor=self._cfg.emit_actor,
                )
                self._bar_last_index[key] = next_idx
                fired += 1

        return fired


# ============================================================
# ReplayScheduler
# ============================================================

class ReplayScheduler(BaseScheduler):
    """
    ReplayScheduler：不 sleep，不依赖系统时间。
    外部应在 replay loop 中：
      - clock.advance_to(event.ts)
      - coordinator.emit(event)
      - scheduler.step(...) 产生定时事件（以 replay clock 的 now 为准）
    """

    def _drain(self, *, max_events: int) -> int:
        fired = 0
        now_ts = self._clock.now()
        now = _to_float(now_ts)
        if now is None:
            return 0

        # timers：严格基于 next trigger，不补齐缺失 tick（保证确定性）
        for t in self._cfg.timers:
            if fired >= max_events:
                break
            name = t.name
            nxt = self._timer_next.get(name)
            if nxt is None:
                continue
            if now >= nxt:
                seq = self._timer_seq.get(name, 0) + 1
                self._timer_seq[name] = seq
                interval = float(t.interval_s)
                # replay 下只递进 1 次（不补齐缺口，保证确定性）
                self._timer_next[name] = nxt + interval

                self._emit(
                    TimerEvent(ts=now_ts, name=name, interval_s=interval, seq=seq),
                    actor=self._cfg.emit_actor,
                )
                fired += 1

        # bars：严格按 bar close 触发（同样只发下一根）
        for b in self._cfg.bars:
            if fired >= max_events:
                break
            tf = int(b.timeframe_s)
            key = (b.symbol, tf)
            last_idx = self._bar_last_index.get(key, -1)
            cur_idx = _floor_div(now, tf)
            if cur_idx > last_idx:
                next_idx = last_idx + 1
                close_ts = (next_idx + 1) * tf
                self._emit(
                    BarCloseEvent(ts=close_ts, symbol=b.symbol, timeframe_s=tf, bar_index=next_idx),
                    actor=self._cfg.emit_actor,
                )
                self._bar_last_index[key] = next_idx
                fired += 1

        return fired


# ============================================================
# Factory
# ============================================================

def build_scheduler(*, cfg: SchedulerConfig, clock: Clock, emit: EmitFn) -> BaseScheduler:
    """
    根据 clock.mode 构建对应 scheduler
    """
    if clock.mode == ClockMode.LIVE:
        return LiveScheduler(cfg=cfg, clock=clock, emit=emit)
    if clock.mode == ClockMode.REPLAY:
        return ReplayScheduler(cfg=cfg, clock=clock, emit=emit)
    raise ClockError(f"Unknown clock mode: {clock.mode}")
