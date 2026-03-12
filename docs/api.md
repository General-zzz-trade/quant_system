# API Documentation

Key interfaces, protocols, and event types for the quant trading system.

Scope note:

- This document describes the current Python-default production runtime unless stated otherwise.
- For runtime ownership and path selection, see [`runtime_truth.md`](/quant_system/docs/runtime_truth.md).
- For execution semantics, see [`execution_contracts.md`](/quant_system/docs/execution_contracts.md).
- For model loading and promotion semantics, see [`model_governance.md`](/quant_system/docs/model_governance.md).

## Core Protocols

### AlphaModel

Defined in `alpha/base.py`. All alpha models must satisfy this protocol.

```python
class AlphaModel(Protocol):
    name: str

    def predict(
        self, *, symbol: str, ts: datetime, features: Dict[str, Any]
    ) -> Optional[Signal]:
        ...
```

Returns a `Signal` dataclass:

```python
@dataclass(frozen=True)
class Signal:
    symbol: str
    ts: datetime
    side: str        # "long" | "short" | "flat"
    strength: float  # 0.0 to 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
```

Production models: `LgbmAlphaModel`, `XgbAlphaModel`. Experimental: `LstmAlphaModel`, `TransformerAlphaModel` (marked experimental, require OOD detection wrapper).

### DecisionModule

Defined in `engine/decision_bridge.py`. Strategy, risk overlay, and portfolio modules implement this.

```python
class DecisionModule(Protocol):
    def decide(self, snapshot: Any) -> Iterable[Any]:
        """
        Input: StateSnapshot
        Output: opinion events (IntentEvent, OrderEvent, etc.)

        Rules:
        - Read-only: never modify state
        - Return opinions, not actions
        - Deterministic for a given snapshot
        """
        ...
```

The `DecisionBridge` collects outputs from all modules and feeds them back through the dispatcher for routing to execution or risk layers.

### VenueAdapter

Defined in `execution/adapters/base.py`. Exchange connectivity interface.

```python
class VenueAdapter(Protocol):
    venue: str

    def list_instruments(self) -> Tuple[InstrumentInfo, ...]: ...
    def get_balances(self) -> BalanceSnapshot: ...
    def get_positions(self) -> Tuple[VenuePosition, ...]: ...
    def get_open_orders(self, *, symbol: Optional[str] = None) -> Tuple[CanonicalOrder, ...]: ...
    def get_recent_fills(self, *, symbol: Optional[str] = None, since_ms: int = 0) -> Tuple[CanonicalFill, ...]: ...
```

Implementations: `BinanceRestClient`.

### ExecutionAdapter

Defined in `engine/execution_bridge.py`. Lower-level order routing interface used by `ExecutionBridge`.

```python
class ExecutionAdapter(Protocol):
    def send_order(self, order_event: Any) -> Iterable[Any]:
        """
        Input: OrderEvent
        Output: 0-N events (FillEvent, RejectEvent)

        Rules:
        - Does not modify engine state
        - Returns fill results as events
        """
        ...
```

The `KillSwitchBridge` wraps any adapter to enforce kill switch rules before forwarding to the venue.

## Engine Components

### EngineCoordinator

Central orchestrator. Defined in `engine/coordinator.py`.

```python
coordinator = EngineCoordinator(cfg=CoordinatorConfig(
    symbol_default="BTCUSDT",
    symbols=("BTCUSDT", "ETHUSDT"),
    currency="USDT",
    starting_balance=10000.0,
    feature_hook=feature_hook,         # Optional: FeatureComputeHook
    on_pipeline_output=monitoring_hook, # Optional: callback on state change
    on_snapshot=correlation_update,     # Optional: callback on snapshot
))

# Lifecycle
coordinator.start()
coordinator.emit(event, actor="live")  # Unified entry point
coordinator.stop()

# Read-only state access
view = coordinator.get_state_view()
# Returns: phase, markets, account, positions, event_index, last_snapshot

# Attach subsystems
coordinator.attach_decision_bridge(decision_bridge)
coordinator.attach_execution_bridge(execution_bridge)
coordinator.attach_runtime(ws_runtime)
coordinator.restore_from_snapshot(snapshot)  # Only during INIT phase
```

**Event dispatch chain**: `emit()` -> `EventDispatcher` -> route to `PIPELINE` / `DECISION` / `EXECUTION` / `DROP`.

State mutation rule: only `StatePipeline.apply()` can write state. Everything else is read-only.

### CoordinatorConfig

```python
@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    symbol_default: str                    # Primary symbol
    symbols: Tuple[str, ...] = ()          # All symbols (empty = single)
    currency: str = "USDT"
    starting_balance: float = 0.0
    pipeline_config: PipelineConfig = PipelineConfig()
    on_pipeline_output: Optional[Callable] = None  # Monitoring hook
    on_snapshot: Optional[Callable] = None          # Snapshot callback
    feature_hook: Optional[Any] = None              # Feature computation
    emit_on_non_advanced: bool = False              # Emit on non-advancing events
```

### LiveRunner

Current default production entry point. Defined in [`runner/live_runner.py`](/quant_system/runner/live_runner.py).

```python
# Build from config object
runner = LiveRunner.build(
    config=LiveRunnerConfig(
        symbols=("BTCUSDT", "ETHUSDT"),
        venue="binance",
        shadow_mode=False,
        health_port=8080,
    ),
    venue_clients={"binance": binance_client},
    decision_modules=[my_strategy],
    feature_computer=feature_computer,
    alpha_models=[lgbm_model],
    fetch_venue_state=fetch_fn,
    fetch_margin=margin_fn,
)

# Build from YAML config file
runner = LiveRunner.from_config(
    Path("config/local.yaml"),
    venue_clients={"binance": binance_client},
)

# Run (blocks until signal or stop)
runner.start()

# Access state
runner.fills          # List of fill records
runner.event_index    # Current event count
runner.kill_switch    # Direct kill switch access
runner.coordinator    # Engine coordinator access
```

`LiveRunner.build()` assembles the current default production stack:
- EngineCoordinator + EngineLoop
- KillSwitchBridge (wraps venue client)
- CorrelationComputer + CorrelationGate
- RiskGate (pre-execution size/notional checks)
- MarginMonitor + ReconcileScheduler
- SystemHealthMonitor + AlertManager
- LatencyTracker + AttributionTracker
- GracefulShutdown (SIGTERM/SIGINT handling)
- Optional: FeatureComputeHook + LiveInferenceBridge
- Optional: persistent SQLite stores
- Optional: HTTP health endpoint

Note:

- The repository also contains a standalone Rust trader under [`ext/rust/src/bin/main.rs`](/quant_system/ext/rust/src/bin/main.rs).
- That binary is an important runtime path, but it is not the default production truth source today.

## Risk Components

### RiskGate

Pre-execution check. Defined in `execution/safety/risk_gate.py`.

```python
gate = RiskGate(
    config=RiskGateConfig(
        max_position_notional=100_000,
        max_order_notional=50_000,
        max_open_orders=20,
        max_portfolio_notional=500_000,
    ),
    get_positions=lambda: coordinator.get_state_view()["positions"],
    is_killed=lambda: kill_switch.is_killed() is not None,
)

result = gate.check(order_event)  # Returns RiskCheckResult(allowed, reason)
```

### CorrelationGate

Blocks new positions when portfolio correlation is too concentrated. Defined in `risk/correlation_gate.py`.

```python
gate = CorrelationGate(
    computer=correlation_computer,
    config=CorrelationGateConfig(
        max_avg_correlation=0.7,
        max_position_correlation=0.85,
        min_data_points=20,
    ),
)

decision = gate.should_allow("ETHUSDT", existing_symbols=["BTCUSDT"])
# Returns RiskDecision(ok, action, violations)
```

### KillSwitch

Circuit breaker with scoped shutdown. Defined in `risk/kill_switch.py`.

```python
kill_switch = KillSwitch()

# Trigger
kill_switch.trigger(
    scope=KillScope.GLOBAL,     # GLOBAL | ACCOUNT | PORTFOLIO | STRATEGY | SYMBOL
    key="*",                     # Scope key (e.g. symbol name)
    mode=KillMode.HARD_KILL,    # HARD_KILL | REDUCE_ONLY
    reason="drawdown_exceeded",
    source="margin_monitor",
)

# Query
record = kill_switch.is_killed()       # Returns KillRecord or None
record = kill_switch.is_killed(        # Check specific scope
    scope=KillScope.SYMBOL, key="BTCUSDT"
)

# Reset
kill_switch.reset(scope=KillScope.GLOBAL, key="*")
```

## Event Types

All events inherit from `BaseEvent` (defined in `event/types.py`). Events are frozen dataclasses with `to_dict()` and `from_dict()` for serialization.

### MarketEvent

```python
MarketEvent(
    header=header,
    ts=datetime(2026, 1, 15, tzinfo=timezone.utc),
    symbol="BTCUSDT",
    open=Decimal("42000"),
    high=Decimal("42500"),
    low=Decimal("41800"),
    close=Decimal("42200"),
    volume=Decimal("1500"),
)
```

### IntentEvent

```python
IntentEvent(
    header=header,
    intent_id="intent-abc123",
    symbol="BTCUSDT",
    side="buy",              # "buy" | "sell"
    target_qty=Decimal("0.01"),
    reason_code="signal",    # signal | rebalance | risk | manual
    origin="strategy_v1",
)
```

### OrderEvent

```python
OrderEvent(
    header=header,
    order_id="order-xyz789",
    intent_id="intent-abc123",
    symbol="BTCUSDT",
    side="buy",
    qty=Decimal("0.01"),
    price=Decimal("42200"),  # None for market orders
)
```

### FillEvent

```python
FillEvent(
    header=header,
    fill_id="fill-001",
    order_id="order-xyz789",
    symbol="BTCUSDT",
    qty=Decimal("0.01"),
    price=Decimal("42195"),
)
```

### Other Events

| Type | Key Fields | Purpose |
|------|------------|---------|
| `SignalEvent` | `signal_id`, `symbol`, `side`, `strength` | Strategy signal output |
| `RiskEvent` | `rule_id`, `level` (info/warn/block), `message` | Risk system ruling |
| `ControlEvent` | `command` (halt/resume/flush/shutdown), `reason` | System control |
| `FundingEvent` | `ts`, `symbol`, `funding_rate`, `mark_price` | Perpetual funding settlement |

### EventType Enum

```python
class EventType(Enum):
    MARKET = "market"
    SIGNAL = "signal"
    INTENT = "intent"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    CONTROL = "control"
    FUNDING = "funding"
```

## State Types

### StateSnapshot

Immutable snapshot at an event boundary. Defined in `state/snapshot.py`.

```python
@dataclass(frozen=True, slots=True)
class StateSnapshot:
    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    event_type: str
    bar_index: int
    markets: Mapping[str, MarketState]
    positions: Mapping[str, PositionState]
    account: AccountState
    portfolio: Optional[PortfolioState] = None
    risk: Optional[RiskState] = None
    features: Optional[Mapping[str, Any]] = None
```

`snapshot.market` property returns `MarketState` for the primary symbol.

### Domain Types

Rich domain wrappers defined in `event/types.py`:

```python
Side.BUY / Side.SELL           # Trading direction
Symbol(value="BTCUSDT")        # Normalized symbol
Venue.BINANCE / Venue.SIM      # Supported venues
Qty.of(0.01)                   # Quantity (Decimal)
Price.of(42000)                # Price (Decimal)
Money.of(10000, "USDT")       # Monetary amount
OrderType.MARKET / .LIMIT      # Order type
TimeInForce.GTC / .IOC / .FOK  # Time in force
```

## Portfolio Optimization

### Objectives

Defined in `portfolio/optimizer/objectives.py`:

| Objective | Description |
|-----------|-------------|
| `MaxSharpe` | Maximize Sharpe ratio |
| `MinVariance` | Minimize portfolio variance |
| `RiskParity` | Equalize risk contribution |
| `MaxReturn` | Maximize expected return |

### Black-Litterman

```python
from portfolio.optimizer.black_litterman import BlackLittermanOptimizer

optimizer = BlackLittermanOptimizer(
    risk_aversion=2.5,
    tau=0.05,
)
weights = optimizer.optimize(
    market_caps={"BTCUSDT": 1e12, "ETHUSDT": 3e11},
    cov_matrix=cov,
    views=views,
    view_confidences=confidences,
)
```

### Kelly Allocator

```python
from portfolio.allocator_kelly import KellyAllocator

allocator = KellyAllocator(fraction=0.25)  # Quarter Kelly
weights = allocator.allocate(
    expected_returns={"BTCUSDT": 0.05, "ETHUSDT": 0.03},
    cov_matrix=cov,
)
```
