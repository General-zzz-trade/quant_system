# LiveRunner Decomposition Design

**Date**: 2026-03-15
**Status**: Reviewed
**Goal**: Decompose the 2,017-line LiveRunner God Object into small, independently testable modules with a transparent ~100-line assembly script.

## Problem

`runner/live_runner.py` is a God Object:
- 2,017 LOC, 53 attributes (most `Optional[Any]`)
- `build()` has 30 parameters, 12 `_build_*` sub-methods
- `LiveRunnerConfig` has 102 fields
- Mixes construction, runtime, and lifecycle concerns
- Cannot test any subsystem without constructing the entire runner
- Adding features means adding more attributes and build phases

## Scope

### Remove (not needed for current single-symbol BTC trading)

| Subsystem | LOC (est) | Reason |
|-----------|-----------|--------|
| AdaptiveBtcConfig | ~60 | Experimental, unvalidated |
| EnsembleCombiner (multi-TF) | ~40 | Not enabled |
| PortfolioAllocator | ~50 | Single-symbol, no cross-asset |
| CorrelationComputer + Gate | ~80 | Single-symbol, no correlation |
| RegimeSizer | ~40 | Unvalidated |
| LiveSignalTracker | ~50 | Debug only, not core |
| DataScheduler + FreshnessMonitor | ~60 | Over-engineered |
| ModuleReloader | ~30 | Rarely used |

**Estimated removal**: ~410 LOC from LiveRunner, ~15 attributes

### Keep (core trading path)

| Module | Responsibility |
|--------|---------------|
| TradingEngine | Features → Inference → Signal + SIGHUP model reload |
| RiskManager | KillSwitch + RiskGate + gate chain + margin check |
| OrderManager | OrderStateMachine + TimeoutTracker + dedup |
| BinanceExecutor | WS/REST order submission + user stream + KillSwitchBridge |
| RecoveryManager | 8-component checkpoint/restore + startup reconciliation |
| LifecycleManager | start sequence + stop ordering + SIGTERM/SIGINT + systemd watchdog |
| RunnerLoop | 1-second poll loop + timeout checks + bar dispatch |

## Target Architecture

### Module Decomposition

```
runner/
├── run_trading.py          # ~120 LOC: assemble & start (replaces LiveRunner.build)
├── trading_engine.py       # ~200 LOC: features + inference + signal + model reload
├── risk_manager.py         # ~120 LOC: kill switch + risk gate + gate chain + margin
├── order_manager.py        # ~120 LOC: OSM + timeout + dedup
├── binance_executor.py     # ~180 LOC: WS/REST orders + user stream + fill recording
├── recovery_manager.py     # ~180 LOC: 8-component checkpoint + restore + reconcile
├── lifecycle_manager.py    # ~150 LOC: start/stop sequencing + signals + watchdog
├── runner_loop.py          # ~120 LOC: main event loop (poll + dispatch)
├── config.py               # ~80 LOC: ~50-field config (remove truly unused only)
├── emit_handler.py         # existing (169 LOC, keep as-is)
└── graceful_shutdown.py    # existing (keep as-is)
```

### Module Interfaces

Each module is a plain class with explicit `__init__` parameters (no `Any`).

#### TradingEngine

```python
class TradingEngine:
    """Features → ML prediction → trading signal + model hot-reload."""

    def __init__(self, feature_hook: FeatureComputeHook,
                 inference_bridge: RustInferenceBridge,
                 symbols: list[str],
                 model_dir: str = "models_v8"):
        ...

    def on_bar(self, symbol: str, bar: dict) -> Signal | None:
        """Process one bar, return signal or None."""
        ...

    def reload_models(self) -> dict[str, str]:
        """Hot-reload models from model_dir. Called on SIGHUP.
        Returns {symbol: status} for each reloaded model.
        Handles both ModelRegistry and direct .pkl file paths.
        """
        ...

    def checkpoint(self) -> dict:
        """Serialize feature_hook + inference_bridge state."""
        ...

    def restore(self, state: dict) -> None:
        """Restore from checkpoint."""
        ...
```

#### RiskManager

```python
class RiskManager:
    """Pre-trade risk: kill switch + gate chain (risk gate, margin)."""

    def __init__(self, kill_switch: KillSwitch,
                 risk_gate: RiskGate,
                 max_position: float, max_notional: float,
                 max_open_orders: int,
                 fetch_margin: Callable[[], float] | None = None):
        ...

    def check(self, signal: Signal, osm_open_count: int) -> tuple[bool, str]:
        """Full gate chain: risk gate + margin + kill switch."""
        ...

    def kill(self, reason: str) -> None: ...

    def checkpoint(self) -> dict:
        """Serialize kill switch state."""
        ...
```

#### OrderManager

```python
class OrderManager:
    """Track order lifecycle: submit → ack → fill/cancel/timeout."""

    def __init__(self, timeout_sec: float = 30.0):
        ...

    def submit(self, order: OrderEvent) -> str: ...
    def on_ack(self, order_id: str, venue_id: str) -> None: ...
    def on_fill(self, fill: FillEvent) -> None: ...
    def check_timeouts(self) -> list[str]:
        """Called each poll cycle. Returns timed-out order IDs."""
        ...
    @property
    def open_count(self) -> int: ...
```

#### BinanceExecutor

```python
class BinanceExecutor:
    """Send orders to Binance via WS-API or REST, receive fills via user stream.

    Wraps venue_client in KillSwitchBridge. Optionally wraps in
    ShadowExecutionAdapter for paper trading (shadow_mode).
    """

    def __init__(self, venue_client, kill_switch: KillSwitch,
                 use_ws: bool = True, shadow_mode: bool = False):
        ...

    def send(self, order: OrderEvent) -> dict: ...
    def cancel(self, order_id: str) -> dict: ...
    def start_user_stream(self, on_fill: Callable) -> None: ...
    def stop_user_stream(self) -> None: ...
    def get_positions(self) -> list: ...
    def get_balances(self) -> dict: ...
```

#### RecoveryManager

```python
class RecoveryManager:
    """8-component checkpoint/restore + startup reconciliation.

    Components: kill_switch, inference_bridge, feature_hook, correlation,
    timeout_tracker, exit_manager, regime_gate, drawdown_breaker.
    """

    def __init__(self, state_dir: str,
                 engine: TradingEngine,
                 risk: RiskManager,
                 orders: OrderManager,
                 coordinator: EngineCoordinator,
                 interval_sec: float = 300.0):
        ...

    def save(self) -> None:
        """Save all 8 components + state store atomically."""
        ...

    def restore(self) -> bool:
        """Restore all components from last checkpoint. Returns True if found."""
        ...

    def reconcile_startup(self, executor: BinanceExecutor) -> list[str]:
        """Compare coordinator state vs venue positions, heal differences."""
        ...
```

#### LifecycleManager

```python
class LifecycleManager:
    """Manages startup sequence, shutdown ordering, signals, and watchdog.

    Shutdown order (15 subsystems, specific sequence):
    1. User stream
    2. Checkpointer
    3. Health/alerts/margin/reconcile
    4. Runtime/loop/coordinator
    5. WS order gateway
    """

    def __init__(self, engine: TradingEngine, executor: BinanceExecutor,
                 recovery: RecoveryManager, loop: RunnerLoop,
                 enable_watchdog: bool = False):
        ...

    def install_signal_handlers(self) -> None:
        """SIGTERM → graceful stop, SIGHUP → engine.reload_models()."""
        ...

    def start(self) -> None:
        """Perf tuning → recovery.restore → executor.start_user_stream →
        loop.start. Blocks until stop."""
        ...

    def stop(self) -> None:
        """Execute shutdown sequence in correct order."""
        ...
```

#### RunnerLoop

```python
class RunnerLoop:
    """1-second poll loop: dispatch bars, check timeouts, periodic tasks."""

    def __init__(self, engine: TradingEngine, risk: RiskManager,
                 orders: OrderManager, executor: BinanceExecutor,
                 emit_handler: LiveEmitHandler):
        ...

    def on_bar(self, symbol: str, bar: dict) -> None:
        """Engine → risk check → emit order if signal."""
        ...

    def poll(self) -> None:
        """Called every 1s: check timeouts, health."""
        ...

    def start(self, ws_runtime) -> None:
        """Enter WS event loop. Blocks."""
        ...

    def stop(self) -> None: ...
```

### Assembly Script (`run_trading.py`)

```python
#!/usr/bin/env python3
"""Start live trading. Every step visible."""

import signal
from runner.config import TradingConfig
from runner.trading_engine import TradingEngine
from runner.risk_manager import RiskManager
from runner.order_manager import OrderManager
from runner.binance_executor import BinanceExecutor
from runner.recovery_manager import RecoveryManager
from runner.lifecycle_manager import LifecycleManager
from runner.runner_loop import RunnerLoop
from runner.emit_handler import LiveEmitHandler

def main():
    config = TradingConfig.from_yaml("config/production.yaml")
    apply_perf_tuning()  # CPU affinity, nice, GC

    # 1. Feature + ML engine (owns model reload)
    engine = TradingEngine.from_config(config)

    # 2. Risk (kill switch + gate chain)
    risk = RiskManager(
        kill_switch=KillSwitch(),
        risk_gate=RiskGate.from_config(config),
        max_position=config.max_position,
        max_notional=config.max_notional,
        max_open_orders=config.max_open_orders,
    )

    # 3. Orders
    orders = OrderManager(timeout_sec=config.order_timeout)

    # 4. Execution (wraps venue client in KillSwitchBridge)
    venue_client = create_binance_client(config)
    executor = BinanceExecutor(
        venue_client=venue_client,
        kill_switch=risk.kill_switch,
        use_ws=config.use_ws_orders,
        shadow_mode=config.shadow_mode,
    )

    # 5. Coordinator + pipeline (existing engine code, untouched)
    coordinator = build_coordinator(config, engine.feature_hook)

    # 6. Emit handler (gate chain → executor)
    emit = LiveEmitHandler(executor, risk, orders)

    # 7. Recovery (8-component bundle)
    recovery = RecoveryManager(
        state_dir=config.state_dir,
        engine=engine,
        risk=risk,
        orders=orders,
        coordinator=coordinator,
    )

    # 8. Event loop
    loop = RunnerLoop(engine, risk, orders, executor, emit)

    # 9. Lifecycle (startup/shutdown sequencing + signals)
    lifecycle = LifecycleManager(engine, executor, recovery, loop)
    lifecycle.start()  # blocks

if __name__ == "__main__":
    main()
```

### Config Simplification

Current `LiveRunnerConfig`: 102 fields. Target `TradingConfig`: ~50 fields.

```python
@dataclass(frozen=True)
class TradingConfig:
    # Identity
    symbols: tuple[str, ...]
    venue: str = "binance"
    testnet: bool = True
    shadow_mode: bool = True

    # Feature/ML
    model_dir: str = "models_v8"
    kline_interval: str = "1m"

    # Signal/strategy
    deadzone: float = 0.5
    min_hold_bars: int = 24
    trend_follow: bool = False
    max_hold: int = 0
    monthly_gate: bool = False

    # Risk
    max_position: float = 0.1
    max_notional: float = 500.0
    max_open_orders: int = 3
    max_daily_loss: float = 100.0

    # Execution
    use_ws_orders: bool = True
    order_timeout: float = 30.0

    # Recovery
    state_dir: str = "state"
    data_dir: str = "data"
    checkpoint_interval: float = 300.0
    enable_persistent_stores: bool = True
    enable_reconcile: bool = True
    reconcile_on_startup: bool = True
    reconcile_interval_sec: float = 300.0
    enable_preflight: bool = True

    # Monitoring (optional)
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_health_check: bool = False
    health_port: int = 8080
    log_level: str = "INFO"
```

**Removed**: adaptive_btc_*, ensemble_*, portfolio_allocator_*, correlation_*, regime_sizer_*, signal_tracker_*, data_scheduler_*, freshness_*, module_reloader_* (~50 fields cut).

## Migration Strategy

1. **Phase 1**: Create new modules alongside existing LiveRunner (no breaking changes)
2. **Phase 2**: Write `run_trading.py` using new modules, verify with `scripts/testnet_smoke` (18/20 pass) and `scripts/run_paper_trading --testnet` (shadow mode 24h)
3. **Phase 3**: Add tests for each new module independently (target: each module has ≥5 unit tests)
4. **Phase 4**: Run old and new runners in parallel for 2 weeks, compare fill counts, latency, reconciliation
5. **Phase 5**: Deprecate `LiveRunner.build()`, update CLAUDE.md

Old code is NOT deleted until new code matches old on all metrics for 2 weeks.

## What Stays Unchanged

- `engine/coordinator.py` — untouched
- `engine/pipeline.py` — untouched
- `engine/feature_hook.py` — untouched
- `rust/` — untouched
- `execution/adapters/binance/` — untouched
- `decision/` — untouched
- All existing tests — untouched

## Helpers That Need a Home

| Current location | New home | Notes |
|-----------------|----------|-------|
| `_FillRecordingAdapter` (live_runner.py:1948) | `binance_executor.py` | Wraps executor to record fills |
| `_reconcile_startup` (live_runner.py:1914) | `recovery_manager.py` | Position/balance comparison |
| `OperatorControlMixin` | Drop for now | Re-add as optional HTTP control plane later |
| `OperatorObservabilityMixin` | Drop for now | Re-add as optional metrics export later |
| Event recorder `_event_recorder_ref` patching | `run_trading.py` | Inject at construction, no post-hoc patching |

## Success Criteria

- [ ] No file > 200 LOC in new runner/
- [ ] Each module testable without constructing others
- [ ] `run_trading.py` < 150 LOC
- [ ] `TradingConfig` < 55 fields
- [ ] Zero `Optional[Any]` — all types explicit
- [ ] SIGHUP model reload works via `engine.reload_models()`
- [ ] Shutdown ordering matches current 15-subsystem sequence
- [ ] Testnet smoke test passes (18/20+)
- [ ] Paper trading shadow mode runs 24h without crash
- [ ] Old LiveRunner still works (parallel running for 2 weeks)
