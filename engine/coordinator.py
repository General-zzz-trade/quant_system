# engine/coordinator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import threading

from engine.dispatcher import EventDispatcher, Route
from engine.pipeline import PipelineConfig, PipelineInput, PipelineOutput, StatePipeline
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from state.rust_adapters import (
    account_from_rust,
    account_to_rust,
    market_from_rust,
    market_to_rust,
    portfolio_from_rust,
    portfolio_to_rust,
    position_from_rust,
    position_to_rust,
    risk_from_rust,
    risk_to_rust,
)

from _quant_hotpath import (
    RustAccountState,
    RustMarketState,
    RustPositionState,
    RustStateStore,
)

_SCALE = 100_000_000


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

        # Auto-create RustStateStore when no explicit store/pipeline given.
        # This makes the store path (state on Rust heap) the default,
        # eliminating per-event Python↔Rust conversions.
        if store is not None:
            self._store = store
        elif pipeline is not None:
            self._store = None
        else:
            effective_symbols = cfg.symbols or (cfg.symbol_default,)
            self._store = RustStateStore(
                list(effective_symbols),
                cfg.currency,
                int(cfg.starting_balance * _SCALE),
            )

        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = StatePipeline(store=self._store, config=cfg.pipeline_config)

        # bridges（允许外部注入实现）
        self._decision_bridge = decision_bridge
        self._execution_bridge = execution_bridge

        # runtime（可选：Live 模式使用）
        self._runtime = runtime
        self._runtime_handler: Optional[Callable[[Any], None]] = None

        self._last_snapshot: Optional[Any] = None

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
        return self._pipeline

    def get_state_view(self) -> Mapping[str, Any]:
        """对外只暴露只读视图（便于 debug/监控；不承诺稳定字段）"""
        with self._lock:
            bundle = dict(self._store.export_state())
            markets = {
                sym: market_from_rust(mkt) if isinstance(mkt, RustMarketState) else mkt
                for sym, mkt in dict(bundle["markets"]).items()
            }
            account = (
                account_from_rust(bundle["account"])
                if isinstance(bundle["account"], RustAccountState)
                else bundle["account"]
            )
            positions = {
                sym: position_from_rust(pos) if isinstance(pos, RustPositionState) else pos
                for sym, pos in dict(bundle["positions"]).items()
            }
            return {
                "phase": self._phase.value,
                "symbol_default": self._cfg.symbol_default,
                "event_index": int(bundle["event_index"]),
                "last_event_id": bundle.get("last_event_id"),
                "last_ts": bundle.get("last_ts"),
                "markets": markets,
                "market": markets.get(self._cfg.symbol_default, next(iter(markets.values()))),
                "account": account,
                "positions": positions,
                "portfolio": portfolio_from_rust(bundle["portfolio"]),
                "risk": risk_from_rust(bundle["risk"]),
                "last_snapshot": self._last_snapshot,
            }

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
            rust_markets = {
                sym: mkt if isinstance(mkt, RustMarketState) else market_to_rust(mkt)
                for sym, mkt in snapshot.markets.items()
            }
            rust_account = (
                snapshot.account
                if isinstance(snapshot.account, RustAccountState)
                else account_to_rust(snapshot.account)
            )
            rust_positions = {
                sym: pos if isinstance(pos, RustPositionState) else position_to_rust(pos)
                for sym, pos in dict(snapshot.positions).items()
            }
            rust_portfolio = None
            if getattr(snapshot, "portfolio", None) is not None:
                rust_portfolio = portfolio_to_rust(snapshot.portfolio)
            rust_risk = None
            if getattr(snapshot, "risk", None) is not None:
                rust_risk = risk_to_rust(snapshot.risk)

            self._store.load_exported(
                rust_markets,
                rust_positions,
                rust_account,
                event_index=snapshot.bar_index,
                last_event_id=snapshot.event_id,
                last_ts=(snapshot.ts.isoformat() if getattr(snapshot, "ts", None) is not None else None),
                portfolio=rust_portfolio,
                risk=rust_risk,
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
    # Dispatcher handlers
    # --------------------------------------------------------

    def _handle_pipeline_event(self, event: Any) -> None:
        """PIPELINE handler：事实事件推进 state 的唯一入口"""
        features = None
        if self._cfg.feature_hook is not None:
            features = self._cfg.feature_hook.on_event(event)

        inp = PipelineInput(
            event=event,
            event_index=0,
            symbol_default=self._cfg.symbol_default,
            markets={},
            account=None,
            positions={},
            features=features,
        )

        out = self._pipeline.apply(inp)

        with self._lock:
            if out.snapshot is not None:
                self._last_snapshot = out.snapshot

        # 观测钩子（不进锁，不阻塞主链路）
        if self._cfg.on_pipeline_output is not None:
            if out.advanced or self._cfg.emit_on_non_advanced:
                self._cfg.on_pipeline_output(out)

        if self._cfg.on_snapshot is not None and out.snapshot is not None:
            self._cfg.on_snapshot(out.snapshot)

        # 决策触发制度：仅在 MARKET 事件推进时触发
        if self._decision_bridge is not None and out.advanced and out.snapshot is not None:
            from engine.pipeline import _detect_kind
            if _detect_kind(event) == "MARKET":
                self._decision_bridge.on_pipeline_output(out)

    def _handle_decision_event(self, event: Any) -> None:
        """
        DECISION handler：冻结版 v1.0 默认不做任何副作用。

        说明：
        - decision_bridge 由 pipeline 输出驱动（out.snapshot）产生“意见事件”
        - dispatcher 会把意见事件再次路由：
            - OrderEvent -> EXECUTION
            - 其他 -> DECISION / DROP（视 event 类型而定）

        后续在 risk/guards 完善后，可以在这里接入：
        - risk gate / kill-switch
        - decision event audit
        """
        # v1.0：明确 no-op（不吞异常，不产生副作用）
        return

    def _handle_execution_event(self, event: Any) -> None:
        """
        EXECUTION handler：所有下单只允许走 ExecutionBridge
        """
        if self._execution_bridge is None:
            raise RuntimeError("ExecutionBridge is not attached")
        self._execution_bridge.handle_event(event)
