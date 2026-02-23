"""Tests for core.bootstrap — system wiring and configuration."""
from __future__ import annotations

from core.bootstrap import SystemContext, bootstrap, bootstrap_test
from core.bus import BoundedEventBus
from core.clock import SimulatedClock
from core.config import ConfigService
from core.effects import Effects, InMemoryMetrics
from core.interceptors import InterceptorChain, PassthroughInterceptor
from core.observability import LoggingInterceptor, MetricsInterceptor, TracingInterceptor
from engine.saga import SagaManager


class TestBootstrap:
    def test_returns_system_context(self) -> None:
        ctx = bootstrap_test()
        assert isinstance(ctx, SystemContext)
        assert isinstance(ctx.config, ConfigService)
        assert isinstance(ctx.effects, Effects)
        assert isinstance(ctx.bus, BoundedEventBus)
        assert isinstance(ctx.chain, InterceptorChain)
        assert isinstance(ctx.saga_manager, SagaManager)
        assert isinstance(ctx.tracing, TracingInterceptor)
        assert isinstance(ctx.logging_interceptor, LoggingInterceptor)
        assert isinstance(ctx.metrics_interceptor, MetricsInterceptor)

    def test_test_bootstrap_uses_simulated_clock(self) -> None:
        ctx = bootstrap_test()
        assert isinstance(ctx.effects.clock, SimulatedClock)

    def test_test_bootstrap_uses_in_memory_metrics(self) -> None:
        ctx = bootstrap_test()
        assert isinstance(ctx.effects.metrics, InMemoryMetrics)

    def test_custom_defaults(self) -> None:
        ctx = bootstrap_test(defaults={"bus.capacity": 500})
        # We can verify via config
        cap = ctx.config.get("bus.capacity", int)
        assert cap == 500

    def test_extra_interceptors(self) -> None:
        extra = PassthroughInterceptor()
        ctx = bootstrap_test(extra_interceptors=[extra])
        # Chain should contain: tracing, metrics, passthrough
        assert len(ctx.chain.interceptors) == 3
        assert ctx.chain.interceptors[2] is extra

    def test_live_bootstrap(self) -> None:
        ctx = bootstrap(defaults={"bus.capacity": 100})
        assert isinstance(ctx, SystemContext)
        # Live effects use SystemClock, not SimulatedClock
        from core.clock import SystemClock
        assert isinstance(ctx.effects.clock, SystemClock)

    def test_config_override_via_defaults(self) -> None:
        ctx = bootstrap_test(defaults={
            "observability.max_spans": 42,
            "saga.max_completed": 77,
        })
        assert ctx.config.get("observability.max_spans", int) == 42
        assert ctx.config.get("saga.max_completed", int) == 77

    def test_overflow_policy_reject(self) -> None:
        ctx = bootstrap_test(defaults={"bus.overflow_policy": "reject"})
        from core.bus import OverflowPolicy
        assert ctx.bus._cfg.overflow_policy == OverflowPolicy.REJECT

    def test_default_config_values(self) -> None:
        ctx = bootstrap_test()
        assert ctx.config.get("bus.capacity", int) == 1000  # test default override
        assert ctx.config.get("risk.reject_on_reduce", bool) is False
