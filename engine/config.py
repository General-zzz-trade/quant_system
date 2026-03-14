# engine/config.py
from __future__ import annotations

from dataclasses import dataclass

from engine.clock import ClockMode
from engine.scheduler import SchedulerConfig, TimerSpec, BarSpec
from engine.guards import GuardConfig


# ============================================================
# Clock
# ============================================================

@dataclass(frozen=True, slots=True)
class ClockConfig:
    mode: ClockMode = ClockMode.LIVE


# ============================================================
# Observability
# ============================================================

@dataclass(frozen=True, slots=True)
class MetricsConfig:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TracingConfig:
    enabled: bool = True


# ============================================================
# Engine (top-level)
# ============================================================

@dataclass(frozen=True, slots=True)
class EngineConfig:
    """
    Engine 的唯一配置入口（声明式、可版本化）

    原则：
    - 不包含策略参数
    - 不包含交易所/adapter 细节
    """
    # identity
    symbol_default: str
    currency: str = "USDT"
    starting_balance: float = 0.0

    # subsystems
    clock: ClockConfig = ClockConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    guards: GuardConfig = GuardConfig()
    metrics: MetricsConfig = MetricsConfig()
    tracing: TracingConfig = TracingConfig()

    # runtime
    attach_runtime: bool = True  # 是否在 bootstrap 时自动 attach runtime


# ============================================================
# Helpers (safe builders)
# ============================================================

def default_scheduler(symbol: str, timeframe_s: int = 60) -> SchedulerConfig:
    return SchedulerConfig(
        timers=[TimerSpec(name="heartbeat", interval_s=1.0)],
        bars=[BarSpec(symbol=symbol, timeframe_s=timeframe_s)],
    )
