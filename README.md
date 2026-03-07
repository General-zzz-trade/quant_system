# Quant System

Production-grade quantitative trading system for crypto perpetual futures. ML-driven alpha generation, multi-exchange execution, institutional risk management, and **Rust-accelerated kernel** (PyO3).

## Architecture

```
                          +------------------+
                          |  Market Data WS  |  (Binance)
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  EngineCoordinator|
                          |  (Python shell)   |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
     +--------v--------+  +-------v--------+  +-------v--------+
     |  StatePipeline  |  |  FeatureHook   |  |  ML Inference  |
     |  (Rust kernel)  |  |  (Rust engine) |  |  (LightGBM C++)|
     +--------+--------+  +-------+--------+  +-------+--------+
              |                    |                    |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          |  DecisionBridge  |
                          |  (Rust sizing)   |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
     +--------v--------+  +-------v--------+  +-------v--------+
     |  RiskEvaluator  |  | CorrelationGate|  |  KillSwitch    |
     |  (Rust, 6 rules)|  | (concentration)|  |  (circuit break)|
     +--------+--------+  +-------+--------+  +-------+--------+
              |                    |                    |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          | ExecutionBridge  |
                          | (order routing)  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  Exchange API    |
                          |  (REST + WS)     |
                          +------------------+
```

**Event flow**: MarketEvent -> StatePipeline (Rust) -> FeatureEngine (Rust) -> ML Inference -> DecisionBridge (Rust sizing) -> RiskEvaluator (Rust) -> ExecutionBridge -> Exchange

All state mutations go through `rust_pipeline_apply()`. Decision modules are read-only -- they return "opinion events" (IntentEvent, OrderEvent), never modify state directly.

## Rust Kernel

The system's hot path runs on a unified Rust crate (`ext/rust/` -> `_quant_hotpath`), built via PyO3 + maturin.

### Performance

| Component | Speedup vs Python | Description |
|-----------|------------------|-------------|
| Pipeline (state reduction) | **5.67x** | Single `rust_pipeline_apply()` call per event |
| Event processing | **23.9x** | `RustMarketEvent` / `RustFillEvent` zero-FFI fields |
| Feature computation | **5-10x** | `RustFeatureEngine` 105 features incremental |
| Sizing math | **3-5x** | `rust_fixed_fraction_qty()` f64 vs Decimal |

### What Rust Owns

| Layer | Rust Module | Exports |
|-------|-------------|---------|
| State types | `state_types.rs` | `RustMarketState`, `RustAccountState`, `RustPositionState` (Fd8 i64 fixed-point) |
| State reduction | `state_reducers.rs` | `RustMarketReducer`, `RustAccountReducer`, `RustPositionReducer` |
| Pipeline | `pipeline.rs` | `rust_pipeline_apply()`, `rust_normalize_to_facts()`, `rust_detect_event_kind()` |
| State store | `state_store.rs` | `RustStateStore` -- state on Rust heap, export on demand |
| Events | `rust_events.rs` | `RustMarketEvent`, `RustFillEvent`, `RustFundingEvent` |
| Risk engine | `risk_engine.rs` | `RustRiskEvaluator` (max_position, leverage, drawdown, exposure, concentration) |
| Feature engine | `feature_engine.rs` | `RustFeatureEngine` -- 105 features, incremental `push_bar()` |
| Decision math | `decision_math.rs` | `rust_fixed_fraction_qty()`, `rust_volatility_adjusted_qty()`, `rust_apply_allocation_constraints()` |
| Fixed-point | `fixed_decimal.rs` | `Fd8` type (i64 x 10^8), eliminates Decimal parsing |

**Crate stats**: 44 Rust modules, ~16,500 LOC, 125 exports (43 classes + 82 functions), 2,594 tests passing.

### What Python Owns

| Layer | Reason |
|-------|--------|
| ML inference | LightGBM C++ binding, pickle models |
| Exchange adapters | IO-bound (WebSocket/REST), not CPU bottleneck |
| Strategy research | Rapid iteration, notebooks, walk-forward |
| Config / logging / monitoring | Infrastructure glue |

## Quick Start

### Prerequisites

- Python 3.12+
- Rust toolchain (rustc 1.75+, maturin)

### Installation

```bash
# Build Rust kernel (required)
make rust

# Core only (stdlib, zero dependencies)
pip install -e .

# Full production stack
pip install -e ".[live,data,ml,config,monitoring]"

# Development
pip install -e ".[live,data,ml,config,monitoring,dev,test]"
```

### Configuration

```bash
# Copy example config
cp infra/config/examples/live.yaml config/local.yaml

# Set credentials via environment (never in config files)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Run Backtest

```bash
python -m runner.backtest_runner \
  --csv data_files/btcusdt_1h.csv \
  --symbol BTCUSDT \
  --window 20 \
  --qty 0.01
```

### Run Live

```bash
# Production
python -m runner.live_runner --config config/local.yaml

# Paper trading (simulated orders, real market data)
python -m runner.live_paper_runner --config config/paper.yaml

# Shadow mode
python -m runner.live_runner --config config/local.yaml --shadow
```

## Module Overview

| Module | Description |
|--------|-------------|
| `engine/` | EngineCoordinator event loop, StatePipeline, dispatcher, feature hook |
| `ext/rust/` | Rust PyO3 kernel -- state, pipeline, features, risk, events, sizing |
| `alpha/` | ML models (LightGBM), inference bridge, drift detection |
| `risk/` | KillSwitch, CorrelationGate, risk aggregation |
| `execution/` | Exchange adapters (Binance), rate limiting, reconciliation |
| `portfolio/` | Black-Litterman, Kelly allocator, MVO, risk parity |
| `decision/` | DecisionEngine, ML decision, regime gating, sizing |
| `features/` | Feature computation, enriched computer (105 features) |
| `strategies/` | Multi-timeframe ensemble, factor strategies |
| `research/` | Walk-forward validation, overfit detection |
| `state/` | State types, Rust adapters, snapshot management |
| `event/` | Event types (Market, Signal, Intent, Order, Fill, Risk, Control) |
| `runner/` | LiveRunner, PaperRunner, BacktestRunner |
| `infra/` | Config loader, structured logging, secret resolution |
| `monitoring/` | SLO/SLI tracking, Prometheus metrics, alerts |

## Event Types

| Event | Purpose |
|-------|---------|
| `MarketEvent` | OHLCV kline data from exchange |
| `SignalEvent` | Strategy signal (long/short/flat + strength) |
| `IntentEvent` | Trading intent (buy/sell + target qty + reason) |
| `OrderEvent` | Execution instruction (order ID, price, qty) |
| `FillEvent` | Confirmed fill from exchange |
| `RiskEvent` | Risk system ruling (info/warn/block) |
| `ControlEvent` | System control (halt/resume/shutdown) |
| `FundingEvent` | Perpetual funding rate settlement |

## Walk-Forward Results

| Asset | WF Pass | Sharpe | Return | Strategy |
|-------|---------|--------|--------|----------|
| BTC | 18/21 | 2.39 | +262% | Strategy F (stable_icir) |
| ETH | 15/21 | 1.19 | +189% | Strategy F |
| SOL | 13/17 | 1.80 | +301% | Greedy selector |

## Testing

```bash
# Unit tests (fast, no external deps)
python -m pytest tests/ -x -q

# With Rust parity checks
python -m pytest tests/ -x -q -k "rust"

# All tests with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Build

```bash
# Build Rust kernel
make rust

# Or manually:
cd ext/rust && maturin build --release
pip install target/wheels/*.whl --force-reinstall
cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/
```

## Project Structure

```
quant_system/
  engine/            # Core event loop and coordination
  ext/rust/           # Rust PyO3 kernel (44 modules, ~16,500 LOC)
  alpha/              # ML alpha models and inference
  risk/               # Risk management and kill switch
  execution/          # Exchange connectivity
  portfolio/          # Portfolio optimization
  decision/           # Trading decision logic
  features/           # Feature engineering
  state/              # State management + Rust adapters
  event/              # Event type definitions
  runner/             # Entry points (live, paper, backtest)
  _quant_hotpath/     # Built Rust shared library
  research/           # Research and validation tools
  monitoring/         # Observability stack
  infra/              # Config, logging, secrets
  tests/              # Unit + integration tests (2,594 passing)
```

## License

Proprietary
