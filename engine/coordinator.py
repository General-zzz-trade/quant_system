# engine/coordinator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import threading
import time as _time_mod
from datetime import datetime as _datetime_type

from engine.dispatcher import EventDispatcher, Route
from engine.pipeline import (
    PipelineConfig, PipelineInput, PipelineOutput, StatePipeline,
    _detect_kind, _LazyConvertMapping, _build_snapshot,
)
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
_time_mod_time = lambda: int(_time_mod.time())


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
        self._tick_processors: Optional[dict] = None
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
        """对外只暴露只读视图（便于 debug/监控；不承诺稳定字段）

        Cached per event cycle — invalidated on each emit().
        """
        with self._lock:
            if self._cached_view is not None:
                return self._cached_view
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
            view = {
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
            self._cached_view = view
            return view

    def get_positions_raw(self) -> Mapping[str, Any]:
        """Fast path: export only positions from Rust (no Decimal conversion)."""
        with self._lock:
            return dict(self._store.get_positions())

    def get_account_raw(self) -> Any:
        """Fast path: export only account from Rust (no Decimal conversion)."""
        with self._lock:
            return self._store.get_account()

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

        # Invalidate cached view on state change
        self._cached_view = None
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
        if self._tick_processors is not None:
            kind = _detect_kind(event)
            symbol = getattr(event, "symbol", self._cfg.symbol_default)
            tp = self._tick_processors.get(symbol)
            if tp is None:
                tp = next(iter(self._tick_processors.values()))
            if kind == "MARKET":
                self._handle_market_tick_fast(event, tp)
                return
            if kind == "FILL":
                tp.process_fill(event)
                return
            if kind == "FUNDING":
                tp.process_funding(event)
                return
            return

        # ── Original slow path ──
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
            if _detect_kind(event) == "MARKET":
                self._decision_bridge.on_pipeline_output(out)

    def _handle_market_tick_fast(self, event: Any, tp: Any) -> None:
        """Fast path: single Rust call for features + predict + state + features dict."""
        symbol = getattr(event, "symbol", self._cfg.symbol_default)
        close_f = float(getattr(event, "close", 0))
        volume = float(getattr(event, "volume", 0) or 0)
        high = float(getattr(event, "high", 0) or 0)
        low = float(getattr(event, "low", 0) or 0)
        open_ = float(getattr(event, "open", 0) or 0)

        ts = getattr(event, "ts", None)
        if isinstance(ts, _datetime_type):
            hour_key = int(ts.timestamp()) // 3600
            ts_str = ts.isoformat()
        elif isinstance(ts, str):
            ts_str = ts
            hour_key = _time_mod_time() // 3600
        else:
            ts_str = None
            hour_key = _time_mod_time() // 3600

        # Push external data (delegated to feature_hook sources via tick_processor)
        fh = self._feature_hook
        if fh is not None:
            ext_count = fh._ext_push_count.get(symbol, 0)
            if ext_count % 5 == 0:
                src = fh._resolve_bar_sources(symbol, event)
                fh._ext_cache[symbol] = src
                tp.push_external_data(symbol, **src)
            else:
                cached = fh._ext_cache.get(symbol)
                if cached is not None:
                    trades = float(getattr(event, "trades", 0) or 0)
                    taker_buy_volume = float(getattr(event, "taker_buy_volume", 0) or 0)
                    quote_volume = float(getattr(event, "quote_volume", 0) or 0)
                    taker_buy_quote_volume = float(getattr(event, "taker_buy_quote_volume", 0) or 0)
                    ts_ev = ts
                    cached["hour"] = ts_ev.hour if isinstance(ts_ev, _datetime_type) else -1
                    cached["dow"] = ts_ev.weekday() if isinstance(ts_ev, _datetime_type) else -1
                    cached["trades"] = trades
                    cached["taker_buy_volume"] = taker_buy_volume
                    cached["quote_volume"] = quote_volume
                    cached["taker_buy_quote_volume"] = taker_buy_quote_volume
                    tp.push_external_data(symbol, **cached)
                else:
                    src = fh._resolve_bar_sources(symbol, event)
                    fh._ext_cache[symbol] = src
                    tp.push_external_data(symbol, **src)
            fh._ext_push_count[symbol] = ext_count + 1

        # Determine warmup status
        warmup_done = True
        if fh is not None:
            bar_count = fh._bar_count.get(symbol, 0) + 1
            fh._bar_count[symbol] = bar_count
            warmup_done = bar_count >= fh._warmup_bars

        # Single Rust call: features + predict + state + pre-built features dict
        result = tp.process_tick_full(
            symbol, close_f, volume, high, low, open_, hour_key,
            warmup_done=warmup_done, ts=ts_str,
        )

        # Use pre-built features dict from Rust (eliminates ~35μs Python dict ops)
        features = result.features_dict

        # Cross-asset features (if enabled)
        if fh is not None and fh._cross_asset is not None:
            funding_rate = features.get("funding_rate")
            fh._cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate,
                                   high=high, low=low)
            cross_feats = fh._cross_asset.get_features(symbol)
            features.update(cross_feats)

        # Tag features with symbol for downstream hooks (alpha health, etc.)
        features["_symbol"] = symbol

        # Store last features in feature_hook for non-market event lookups
        if fh is not None:
            fh._last_features[symbol] = features

        # Skip Decimal conversion: pass Rust objects directly to snapshot/output.
        snapshot = _build_snapshot(
            raw_event=event,
            event_index=result.event_index,
            markets=result.markets,
            account=result.account,
            positions=result.positions,
            portfolio=result.portfolio,
            risk=result.risk,
            features=features,
            skip_convert=True,
        )

        self._last_snapshot = snapshot

        # Build PipelineOutput for hooks + decision bridge
        out = PipelineOutput(
            markets=result.markets,
            account=result.account,
            positions=result.positions,
            portfolio=result.portfolio,
            risk=result.risk,
            features=features,
            event_index=result.event_index,
            last_event_id=result.last_event_id,
            last_ts=result.last_ts,
            snapshot=snapshot,
            advanced=True,
        )

        if self._cfg.on_pipeline_output is not None:
            self._cfg.on_pipeline_output(out)

        if self._cfg.on_snapshot is not None:
            self._cfg.on_snapshot(snapshot)

        if self._decision_bridge is not None:
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
