# engine/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from engine.config import EngineConfig
from engine.coordinator import EngineCoordinator, CoordinatorConfig
from engine.clock import LiveClock, ReplayClock, Clock
from engine.scheduler import build_scheduler, BaseScheduler
from engine.guards import build_basic_guard
from engine.metrics import MetricsRegistry
from engine.tracing import Tracer
from engine.loop import EngineLoop, LoopConfig


@dataclass(slots=True)
class EngineBundle:
    """
    启动后的 engine 组合体（v1.1：加入 loop/guard）

    - loop：生产级事件循环（inbox + guard 执行点）
    - coordinator：内核（dispatcher/pipeline/bridges）
    - scheduler：时间→事件（通过 loop.submit 注入）
    """
    loop: EngineLoop
    coordinator: EngineCoordinator
    scheduler: Optional[BaseScheduler]
    clock: Clock
    metrics: Optional[MetricsRegistry]
    tracer: Optional[Tracer]


def _build_clock(cfg) -> Clock:
    if cfg.mode.name == "LIVE":
        return LiveClock()
    if cfg.mode.name == "REPLAY":
        return ReplayClock()
    raise ValueError(f"Unknown ClockMode: {cfg.mode}")


def bootstrap_engine(
    *,
    cfg: EngineConfig,
    runtime: Optional[Any] = None,
    decision_bridge: Optional[Any] = None,
    execution_bridge: Optional[Any] = None,
    loop_cfg: Optional[LoopConfig] = None,
) -> EngineBundle:
    """
    Engine 的唯一装配入口（v1.1：生产骨架）

    1) clock
    2) observability
    3) coordinator（不直接 attach runtime）
    4) guard + loop（执行点在 loop）
    5) scheduler（emit -> loop.submit）
    """
    # 1) clock
    clock = _build_clock(cfg.clock)

    # 2) observability
    metrics = MetricsRegistry() if cfg.metrics.enabled else None
    tracer = Tracer() if cfg.tracing.enabled else None

    # 3) coordinator（注意：这里不再传 runtime，避免回调线程直通 dispatch）
    coord_cfg = CoordinatorConfig(
        symbol_default=cfg.symbol_default,
        currency=cfg.currency,
        starting_balance=cfg.starting_balance,
    )
    coordinator = EngineCoordinator(
        cfg=coord_cfg,
        decision_bridge=decision_bridge,
        execution_bridge=execution_bridge,
        runtime=None,
    )

    # 4) guard + loop
    guard = build_basic_guard(cfg.guards)
    loop = EngineLoop(coordinator=coordinator, guard=guard, cfg=loop_cfg)

    # runtime attach（IO -> inbox）
    if runtime is not None and cfg.attach_runtime:
        loop.attach_runtime(runtime)

    # 5) scheduler（time -> event -> inbox）
    scheduler = None
    if cfg.scheduler is not None:
        scheduler = build_scheduler(
            cfg=cfg.scheduler,
            clock=clock,
            emit=lambda ev, actor: loop.submit(ev, actor=actor),
        )
        scheduler.start()

    # start engine
    coordinator.start()

    return EngineBundle(
        loop=loop,
        coordinator=coordinator,
        scheduler=scheduler,
        clock=clock,
        metrics=metrics,
        tracer=tracer,
    )
