# engine/coordinator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import logging as _logging
import threading

from engine.dispatcher import EventDispatcher, Route
from engine.pipeline import (
    PipelineConfig, PipelineOutput, StatePipeline,
)
from engine.coordinator_handlers import (
    handle_pipeline_event,
    handle_decision_event,
    handle_execution_event,
)
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from event.events import BaseEvent
from _quant_hotpath import (  # type: ignore[import-untyped]
    RustStateStore,
    RustEventValidator,
    RustInMemoryEventStore,
    RustInterceptorChain,
    RustTradingGate,
    rust_validate_side,
    rust_validate_signal_side,
    rust_validate_venue,
    rust_validate_order_type,
    rust_validate_tif,
    rust_validate_numeric_range,
)

_SCALE = 100_000_000
_logger = _logging.getLogger(__name__)


# ============================================================
# Contracts (避免对 event.runtime 强绑定)
# ============================================================

class RuntimeLike(Protocol):
    """
    EventRuntime 的最小契约（兼容不同实现）：

    - subscribe(handler) or on(handler): 注册事件回调
    - unsubscribe(handler) or off(handler): 取消注册（可选）
    """
    def subscribe(self, handler: Callable[[Any], None]) -> None: ...
    def unsubscribe(self, handler: Callable[[Any], None]) -> None: ...


# ============================================================
# Engine Coordinator
# ============================================================

class EnginePhase(str, Enum):
    INIT = "init"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    """
    Coordinator 的制度性配置（不含任何策略参数）

    约束：
    - coordinator 不做策略
    - coordinator 不做 IO（execution adapter 由 bridge 内部完成）
    - coordinator 不实现时间制度（clock/scheduler 后续模块接入）
    """
    symbol_default: str
    symbols: Tuple[str, ...] = ()  # 空=退化为单品种 (symbol_default,)
    currency: str = "USDT"
    starting_balance: float = 0.0

    # pipeline 制度
    pipeline_config: PipelineConfig = PipelineConfig()

    # 观测钩子（可选）
    on_pipeline_output: Optional[Callable[[PipelineOutput], None]] = None
    on_snapshot: Optional[Callable[[Any], None]] = None
    feature_hook: Optional[Any] = None

    # 是否在 pipeline 未推进时也回调 on_pipeline_output
    #（默认 False：只在事实事件推进时触发，保持因果清晰）
    emit_on_non_advanced: bool = False

    # RustTickProcessor — full hot-path (features + predict + state update + export)
    # When set, bypasses feature_hook + pipeline for MARKET events.
    tick_processor: Optional[Any] = None


class EngineCoordinator:
    """
    EngineCoordinator —— engine 的“运行时总控”（冻结版 v1.0）

    目标：把 dispatcher / pipeline / decision / execution 串成单一因果链，
    并提供一个机构级“唯一入口”：

        runtime(or replay) -> dispatcher -> pipeline -> decision -> execution -> dispatcher ...

    冻结铁律：
    1) state 的唯一写通道：只能通过 StatePipeline.apply()
    2) coordinator 不写策略，不写 IO（只负责调度与因果串联）
    3) live 与 replay 走同一条入口：EventDispatcher.dispatch()
    4) handler 内不吞异常：异常上抛由更高层（guards/errors）裁决（后续补齐）
    """

    def __init__(
        self,
        *,
        cfg: CoordinatorConfig,
        dispatcher: Optional[EventDispatcher] = None,
        pipeline: Optional[StatePipeline] = None,
        decision_bridge: Optional[DecisionBridge] = None,
        execution_bridge: Optional[ExecutionBridge] = None,
        runtime: Optional[Any] = None,
        store: Optional[Any] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._cfg = cfg
        self._phase: EnginePhase = EnginePhase.INIT

        # core modules
        self._dispatcher = dispatcher or EventDispatcher()

        # Tick processor fast path (replaces feature_hook + pipeline for MARKET events)
        # Can be a single RustTickProcessor or a dict of {symbol: RustTickProcessor}
        self._tick_processor_raw = cfg.tick_processor
        self._tick_processors: Optional[dict[str, Any]] = None
        if isinstance(cfg.tick_processor, dict):
            self._tick_processors = cfg.tick_processor
        elif cfg.tick_processor is not None:
            self._tick_processors = {cfg.symbol_default: cfg.tick_processor}

        self._feature_hook = cfg.feature_hook

        # Auto-create RustStateStore when no explicit store/pipeline given.
        # When tick_processor is set, it owns the state — no separate store needed.
        if self._tick_processors is not None:
            # Use first tick_processor as the store for get_state_view compat
            self._store = next(iter(self._tick_processors.values()))
            self._pipeline = None
        elif store is not None:
            self._store = store
            if pipeline is not None:
                self._pipeline = pipeline
            else:
                self._pipeline = StatePipeline(store=self._store, config=cfg.pipeline_config)
        elif pipeline is not None:
            self._store = None
            self._pipeline = pipeline
        else:
            effective_symbols = cfg.symbols or (cfg.symbol_default,)
            self._store = RustStateStore(
                list(effective_symbols),
                cfg.currency,
                int(cfg.starting_balance * _SCALE),
            )
            self._pipeline = StatePipeline(store=self._store, config=cfg.pipeline_config)

        # bridges（允许外部注入实现）
        self._decision_bridge = decision_bridge
        self._execution_bridge = execution_bridge

        # runtime（可选：Live 模式使用）
        self._runtime = runtime
        self._runtime_handler: Optional[Callable[[Any], None]] = None

        self._last_snapshot: Optional[Any] = None
        self._cached_view: Optional[Mapping[str, Any]] = None

        # Rust event validation + audit store
        self._event_validator = RustEventValidator(max_seen=10000)
        self._event_store = RustInMemoryEventStore()

        # Rust interceptor chain: pre/post event processing hooks
        self._interceptor_chain = RustInterceptorChain()

        # Rust trading gate: halt/resume mechanism
        self._trading_gate = RustTradingGate()

        # ---- register handlers (single chain) ----
        self._dispatcher.register(route=Route.PIPELINE, handler=self._handle_pipeline_event)
        self._dispatcher.register(route=Route.DECISION, handler=self._handle_decision_event)
        self._dispatcher.register(route=Route.EXECUTION, handler=self._handle_execution_event)
        # DROP route：不注册 handler（明确无副作用）

        # 如果传了 runtime，则默认 start 时 attach
        if self._runtime is not None:
            self.attach_runtime(self._runtime)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    @property
    def phase(self) -> EnginePhase:
        return self._phase

    @property
    def dispatcher(self) -> EventDispatcher:
        return self._dispatcher

    @property
    def pipeline(self) -> StatePipeline:
        assert self._pipeline is not None, "pipeline not initialized"
        return self._pipeline

    def get_state_view(self) -> Mapping[str, Any]:
        """对外只暴露只读视图（便于 debug/监控；不承诺稳定字段）

        Cached per event cycle — invalidated on each emit().
        """
        with self._lock:
            if self._cached_view is not None:
                return self._cached_view
            bundle = dict(self._store.export_state())
            markets = dict(bundle["markets"])
            view = {
                "phase": self._phase.value,
                "symbol_default": self._cfg.symbol_default,
                "event_index": int(bundle["event_index"]),
                "last_event_id": bundle.get("last_event_id"),
                "last_ts": bundle.get("last_ts"),
                "markets": markets,
                "market": markets.get(self._cfg.symbol_default, next(iter(markets.values()))),
                "account": bundle["account"],
                "positions": dict(bundle["positions"]),
                "portfolio": bundle["portfolio"],
                "risk": bundle["risk"],
                "last_snapshot": self._last_snapshot,
            }
            self._cached_view = view
            return view

    def start(self) -> None:
        with self._lock:
            if self._phase == EnginePhase.RUNNING:
                return
            self._phase = EnginePhase.RUNNING

    def stop(self) -> None:
        with self._lock:
            if self._phase == EnginePhase.STOPPED:
                return
            self._phase = EnginePhase.STOPPED
            # 尽量做安全解绑（若 runtime 支持）
            if self._runtime is not None and self._runtime_handler is not None:
                self.detach_runtime()

    def halt_trading(self, reason: str = "") -> None:
        """Halt trading via RustTradingGate. All subsequent events are dropped until resumed."""
        self._trading_gate.halt()
        _logger.warning("Trading halted%s", f": {reason}" if reason else "")

    def resume_trading(self) -> None:
        """Resume trading after a halt."""
        self._trading_gate.resume()
        _logger.info("Trading resumed")

    @property
    def is_trading_halted(self) -> bool:
        return self._trading_gate.is_halted

    @property
    def interceptor_chain(self) -> RustInterceptorChain:
        """Expose interceptor chain for pre/post event hooks."""
        return self._interceptor_chain

    def emit(self, event: Any, *, actor: str = "live") -> None:
        """
        engine 的统一入口（手动注入 / tests / live 注入 / scheduler 注入）

        - live：actor="live"
        - replay：actor="replay"
        """
        with self._lock:
            if self._phase == EnginePhase.INIT:
                # 允许 INIT 阶段注入（用于 warmup / tests），但建议显式 start
                pass
            if self._phase == EnginePhase.STOPPED:
                raise RuntimeError("EngineCoordinator is stopped")

        # Trading gate: drop events when halted (safety mechanism)
        if self._trading_gate.is_halted:
            _logger.warning("Trading halted — dropping %s event", type(event).__name__)
            return

        # Interceptor chain: run pre-event hooks (warn-only, never crash)
        try:
            self._interceptor_chain.run(event)
        except Exception:
            _logger.debug("Interceptor chain error (non-fatal)", exc_info=True)

        # Invalidate cached view on state change
        self._cached_view = None

        # Type safety: log non-BaseEvent emissions at debug level
        if __debug__ and not isinstance(event, BaseEvent):
            _logger.debug(
                "Non-BaseEvent emitted: %s", type(event).__name__,
            )

        # ── Rust event validation (warn-only, never crash) ──
        try:
            side = getattr(event, "side", None)
            if side is not None:
                if not rust_validate_side(str(side)):
                    _logger.warning("Invalid side '%s' on %s", side, type(event).__name__)

            signal_side = getattr(event, "signal_side", None)
            if signal_side is not None:
                if not rust_validate_signal_side(str(signal_side)):
                    _logger.warning("Invalid signal_side '%s' on %s", signal_side, type(event).__name__)

            venue = getattr(event, "venue", None)
            if venue is not None:
                if not rust_validate_venue(str(venue)):
                    _logger.warning("Invalid venue '%s' on %s", venue, type(event).__name__)

            order_type = getattr(event, "order_type", None)
            if order_type is not None:
                if not rust_validate_order_type(str(order_type)):
                    _logger.warning("Invalid order_type '%s' on %s", order_type, type(event).__name__)

            tif = getattr(event, "tif", None)
            if tif is not None:
                if not rust_validate_tif(str(tif)):
                    _logger.warning("Invalid tif '%s' on %s", tif, type(event).__name__)

            # Validate price is in sane range if present
            price = getattr(event, "price", None)
            if price is not None:
                try:
                    price_f = float(price)
                    if not rust_validate_numeric_range(price_f, 0.0, 1e12):
                        _logger.warning("Price out of range: %s on %s", price, type(event).__name__)
                except (TypeError, ValueError):
                    pass

            # Store event for audit trail
            self._event_store.append(event)
        except Exception:
            _logger.debug("Event validation skipped due to unexpected error", exc_info=True)

        # 不要在 lock 内 dispatch（防止 handler 再次 emit 导致死锁/长锁）
        self._dispatcher.dispatch(event=event, actor=actor)

    # --------------------------------------------------------
    # Runtime attach/detach (Live mode)
    # --------------------------------------------------------

    def attach_runtime(self, runtime: Any) -> None:
        """
        兼容不同 runtime 的订阅接口：
        - runtime.subscribe(handler)
        - runtime.on(handler)
        """
        with self._lock:
            self._runtime = runtime

            def _handler(ev: Any) -> None:
                # runtime 回调一律走 emit（统一入口）
                self.emit(ev, actor="live")

            self._runtime_handler = _handler

            if hasattr(runtime, "subscribe") and callable(getattr(runtime, "subscribe")):
                runtime.subscribe(_handler)
            elif hasattr(runtime, "on") and callable(getattr(runtime, "on")):
                runtime.on(_handler)
            else:
                # runtime 不支持订阅：保持可用（用户可手动 emit）
                self._runtime_handler = None


    def attach_decision_bridge(self, bridge: DecisionBridge) -> None:
        """Attach a DecisionBridge to enable decision modules after pipeline output."""
        self._decision_bridge = bridge

    def attach_execution_bridge(self, bridge: ExecutionBridge) -> None:
        """Attach an ExecutionBridge to enable live execution handling for order events."""
        self._execution_bridge = bridge

    def restore_from_snapshot(self, snapshot: Any) -> None:
        """Restore engine state from a StateSnapshot. Only valid during INIT phase.
        Accepts snapshots with either Python or Rust state types."""
        with self._lock:
            if self._phase != EnginePhase.INIT:
                raise RuntimeError("Can only restore during INIT phase")
            self._store.load_exported(
                dict(snapshot.markets),
                dict(snapshot.positions),
                snapshot.account,
                event_index=snapshot.bar_index,
                last_event_id=snapshot.event_id,
                last_ts=(snapshot.ts.isoformat() if getattr(snapshot, "ts", None) is not None else None),
                portfolio=getattr(snapshot, "portfolio", None),
                risk=getattr(snapshot, "risk", None),
            )

    def detach_runtime(self) -> None:
        with self._lock:
            runtime = self._runtime
            handler = self._runtime_handler
            self._runtime = None
            self._runtime_handler = None

        if runtime is None or handler is None:
            return

        if hasattr(runtime, "unsubscribe") and callable(getattr(runtime, "unsubscribe")):
            runtime.unsubscribe(handler)
        elif hasattr(runtime, "off") and callable(getattr(runtime, "off")):
            runtime.off(handler)
        # 否则无能为力：忽略

    # --------------------------------------------------------
    # Dispatcher handlers (delegated to coordinator_handlers.py)
    # --------------------------------------------------------

    def _handle_pipeline_event(self, event: Any) -> None:
        handle_pipeline_event(self, event)

    def _handle_decision_event(self, event: Any) -> None:
        handle_decision_event(self, event)

    def _handle_execution_event(self, event: Any) -> None:
        handle_execution_event(self, event)
