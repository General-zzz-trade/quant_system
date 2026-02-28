"""System bootstrap — wires ConfigService to all modules.

This module provides a single ``bootstrap()`` function that reads
configuration, constructs the InterceptorChain, configures the EventBus,
and returns a ready-to-use ``SystemContext`` containing all wired components.

Configuration keys (with defaults)::

    # Bus
    bus.capacity              = 10000
    bus.high_watermark        = 0.8
    bus.overflow_policy       = "drop_lowest"   # or "reject"

    # Pipeline
    pipeline.snapshot_on_change_only = true
    pipeline.fail_on_missing_symbol  = false

    # Risk
    risk.fail_safe_action     = "reject"   # or "kill"
    risk.reject_on_reduce     = false
    risk.kill_switch.halt_pipeline = true

    # Saga
    saga.max_completed        = 10000

    # Observability
    observability.max_spans   = 10000
    observability.log_continue = false

Usage::

    from core.bootstrap import bootstrap

    ctx = bootstrap(
        config_file="config.json",
        defaults={"bus.capacity": 5000},
    )
    # ctx.bus, ctx.chain, ctx.saga_manager, etc. are ready
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from core.bus import BoundedEventBus, BusConfig, OverflowPolicy
from core.config import ConfigService
from core.effects import Effects, live_effects, test_effects
from core.interceptors import InterceptorChain, PipelineInterceptor
from core.observability import LoggingInterceptor, MetricsInterceptor, TracingInterceptor
from core.plugins import PluginRegistry, get_registry
from engine.saga import SagaManager


@dataclass
class SystemContext:
    """All wired system components, ready to use.

    This is the single point of truth for dependency injection —
    no module should construct its own dependencies.
    """
    config: ConfigService
    effects: Effects
    bus: BoundedEventBus
    chain: InterceptorChain
    saga_manager: SagaManager

    # Observability interceptors (accessible for inspection/testing)
    tracing: TracingInterceptor
    logging_interceptor: LoggingInterceptor
    metrics_interceptor: MetricsInterceptor

    # Plugin registries by category
    plugins: Dict[str, PluginRegistry] = field(default_factory=dict)

    # Optional production components (wired by LiveRunner when available)
    event_store: Optional[Any] = None
    state_store: Optional[Any] = None
    latency_tracker: Optional[Any] = None
    alert_manager: Optional[Any] = None
    tracer: Optional[Any] = None  # infra.tracing.otel.Tracer


def bootstrap(
    *,
    defaults: Optional[Dict[str, Any]] = None,
    config_file: Optional[str] = None,
    env_prefix: str = "QS_",
    effects: Optional[Effects] = None,
    extra_interceptors: Sequence[PipelineInterceptor] = (),
) -> SystemContext:
    """Wire all system components from configuration.

    Parameters
    ----------
    defaults : dict, optional
        Default configuration values (lowest priority).
    config_file : str, optional
        Path to JSON/YAML config file.
    env_prefix : str
        Environment variable prefix for config lookup.
    effects : Effects, optional
        Pre-built effects container.  If None, ``live_effects()`` is used.
    extra_interceptors : sequence of PipelineInterceptor
        Additional interceptors to insert into the chain (e.g., RiskInterceptor).
    """
    # ── 1. Configuration ─────────────────────────────────
    all_defaults = _DEFAULT_CONFIG.copy()
    if defaults:
        all_defaults.update(defaults)

    config = ConfigService(
        defaults=all_defaults,
        file_path=config_file,
        env_prefix=env_prefix,
    )

    # ── 2. Effects ───────────────────────────────────────
    fx = effects or live_effects()

    # ── 3. Event Bus ─────────────────────────────────────
    overflow_str = config.get_or("bus.overflow_policy", "drop_lowest", str)
    overflow = (
        OverflowPolicy.REJECT
        if overflow_str.lower() == "reject"
        else OverflowPolicy.DROP_LOWEST
    )

    bus = BoundedEventBus(BusConfig(
        capacity=config.get_or("bus.capacity", 10_000, int),
        high_watermark=config.get_or("bus.high_watermark", 0.8, float),
        overflow_policy=overflow,
    ))

    # ── 4. Observability Interceptors ────────────────────
    otel_tracer = None
    try:
        from infra.tracing.otel import Tracer as OtelTracer
        otel_tracer = OtelTracer(service_name="quant_system")
    except Exception:
        pass  # OTel tracer optional

    obs_max_spans = config.get_or("observability.max_spans", 10_000, int)
    tracing = TracingInterceptor(
        max_spans=obs_max_spans,
        tracer=otel_tracer,
    )

    logging_ic = LoggingInterceptor(
        log_fn=_make_log_fn(fx),
        max_entries=config.get_or("observability.max_log_entries", 5_000, int),
        log_continue=config.get_or("observability.log_continue", False, bool),
    )

    metrics_ic = MetricsInterceptor(metrics=fx.metrics)

    # ── 5. Interceptor Chain ─────────────────────────────
    # Order: tracing → metrics → [extra (risk, etc.)] → logging
    interceptors: list[PipelineInterceptor] = [
        tracing,
        metrics_ic,
        *extra_interceptors,
    ]
    chain = InterceptorChain(interceptors)

    # ── 6. Saga Manager ─────────────────────────────────
    saga_manager = SagaManager(
        max_completed=config.get_or("saga.max_completed", 10_000, int),
    )

    # ── 7. Plugin Registries ─────────────────────────────
    plugins = {
        "strategy": get_registry("strategy"),
        "alpha": get_registry("alpha"),
        "venue": get_registry("venue"),
        "indicator": get_registry("indicator"),
    }

    return SystemContext(
        config=config,
        effects=fx,
        bus=bus,
        chain=chain,
        saga_manager=saga_manager,
        tracing=tracing,
        logging_interceptor=logging_ic,
        metrics_interceptor=metrics_ic,
        plugins=plugins,
        tracer=otel_tracer,
    )


def bootstrap_test(
    *,
    defaults: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    extra_interceptors: Sequence[PipelineInterceptor] = (),
) -> SystemContext:
    """Bootstrap for tests — deterministic effects, small capacities."""
    test_defaults = {
        "bus.capacity": 1_000,
        "observability.max_spans": 100,
        "observability.max_log_entries": 100,
        "saga.max_completed": 100,
    }
    if defaults:
        test_defaults.update(defaults)

    return bootstrap(
        defaults=test_defaults,
        effects=test_effects(seed=seed),
        extra_interceptors=extra_interceptors,
    )


# ── Internals ────────────────────────────────────────────

_DEFAULT_CONFIG: Dict[str, Any] = {
    "bus.capacity": 10_000,
    "bus.high_watermark": 0.8,
    "bus.overflow_policy": "drop_lowest",
    "pipeline.snapshot_on_change_only": True,
    "pipeline.fail_on_missing_symbol": False,
    "risk.fail_safe_action": "reject",
    "risk.reject_on_reduce": False,
    "risk.kill_switch.halt_pipeline": True,
    "saga.max_completed": 10_000,
    "observability.max_spans": 10_000,
    "observability.max_log_entries": 5_000,
    "observability.log_continue": False,
}


def _make_log_fn(fx: Effects) -> Any:
    """Create a log function from effects."""
    def log_fn(level: str, message: str, **kwargs: Any) -> None:
        dispatch = {
            "debug": fx.log.debug,
            "info": fx.log.info,
            "warning": fx.log.warning,
            "error": fx.log.error,
        }
        dispatch.get(level, fx.log.info)(message)
    return log_fn
