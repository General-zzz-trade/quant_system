# Quant System

Production-grade quantitative trading system for crypto perpetual futures. ML-driven alpha generation, institutional risk management, and a **Rust-accelerated kernel** embedded in a Python runtime.

## Documentation Map

Use these documents as the current source of truth:

- Runtime truth: [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md)
- Runtime contracts: [`docs/runtime_contracts.md`](/quant_system/docs/runtime_contracts.md)
- Production recovery/runbook: [`docs/production_runbook.md`](/quant_system/docs/production_runbook.md)
- Execution contracts: [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md)
- Model governance: [`docs/model_governance.md`](/quant_system/docs/model_governance.md)
- Full codebase assessment: [`research.md`](/quant_system/research.md)
- Refactor / convergence status: [`tasks/refactor_master_plan.md`](/quant_system/tasks/refactor_master_plan.md)

Historical plans under [`tasks/`](/quant_system/tasks) and forward-looking rewrite notes under [`docs/`](/quant_system/docs) should be read as dated planning documents unless they explicitly state they are current truth.

## Architecture

```
                          +------------------+
                          |  Market Data WS  |  (Binance)
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  EngineCoordinator|
                          |  (event loop)     |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
     +--------v--------+  +-------v--------+  +-------v--------+
     |  StatePipeline  |  |  FeatureHook   |  |  ML Inference  |
     |  (Rust kernel)  |  |  (Rust engine) |  |  (LightGBM)    |
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

**Current production path**: `runner/live_runner.py` assembles the Python runtime, while Rust owns the hot path for state, features, and several execution/risk primitives.

**Event flow**: MarketEvent -> StatePipeline (Rust-backed store) -> FeatureEngine (Rust) -> ML Inference -> DecisionBridge -> Risk Gates -> ExecutionBridge -> Exchange

All state mutations go through the Rust-backed state pipeline. Decision modules are read-only: they return opinion events (`IntentEvent`, `OrderEvent`) and do not mutate state directly.

## Rust Kernel

The system's hot path runs on a unified Rust crate (`ext/rust/` -> `_quant_hotpath`), built via PyO3 + maturin.

Rust owns the hot path, but the default production orchestrator is still Python (`runner/live_runner.py`). The repository also contains a standalone Rust trader (`ext/rust/src/bin/main.rs`) as an evolving runtime path rather than the default entrypoint.

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
| Cross-asset | `cross_asset.rs` | `RustCrossAssetComputer` -- rolling beta, correlation, EMA |
| Decision math | `decision_math.rs` | `rust_fixed_fraction_qty()`, `rust_volatility_adjusted_qty()` |
| Decision signals | `decision_signals.rs` | `rust_rolling_sharpe`, `rust_max_drawdown`, `rust_strategy_weights` |
| Portfolio | `portfolio_allocator.rs` | `rust_allocate_portfolio()` -- leverage/turnover/notional caps |
| Factor signals | `factor_signals.rs` | `rust_momentum_score`, `rust_volatility_score`, `rust_adx`, `rust_carry_score` |
| Microstructure | `microstructure.rs` | `RustVPINCalculator`, `RustStreamingMicrostructure` |
| Attribution | `attribution.rs` | `rust_compute_pnl`, `rust_compute_cost_attribution`, `rust_attribute_by_signal`, `rust_flush_orderbook_bar` |
| Ensemble | `ensemble_calibrate.rs` | `rust_adaptive_ensemble_calibrate` (IC/inverse-vol/ridge) |
| Inference | `inference_bridge.rs` | `RustInferenceBridge` (z-score, min-hold, monthly gate) |
| Tree predict | `tree_predict.rs` | `RustTreePredictor` (native LightGBM/XGBoost inference) |
| Regime | `regime_buffer.rs` | `RustRegimeBuffer` -- regime detection buffer |
| Fixed-point | `fixed_decimal.rs` | `Fd8` type (i64 x 10^8), eliminates Decimal parsing |

**Crate stats**: 67 exported Rust modules in `_quant_hotpath`, 82 Rust source files under `ext/`, standalone `quant_trader` binary included.

### What Python Owns

| Layer | Reason |
|-------|--------|
| ML inference | LightGBM C++ binding, model files |
| Exchange adapters | IO-bound (WebSocket/REST), not CPU bottleneck |
| Strategy research | Rapid iteration, walk-forward validation |
| Risk aggregation | Pure rule orchestration, zero math |
| Config / logging / monitoring | Infrastructure glue |

## Walk-Forward Results

| Asset | WF Pass | Sharpe | Return | Strategy |
|-------|---------|--------|--------|----------|
| BTC | 18/21 (86%) | 2.39 | +262% | Strategy F (stable_icir) |
| ETH | 15/21 (71%) | 1.19 | +189% | Strategy F |
| SOL | 13/17 (76%) | 1.80 | +301% | Greedy selector |

## Quick Start

### Prerequisites

- Python 3.12+
- Rust toolchain (rustc 1.75+, maturin)

### Installation

```bash
# Build Rust kernel (required)
make rust

# Full production stack
pip install -e ".[live,data,ml,config,monitoring]" --break-system-packages

# Development
pip install -e ".[live,data,ml,config,monitoring,dev,test]" --break-system-packages
```

### Configuration

```bash
# Copy example env
cp .env.example .env

# Set credentials (never in config files)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Run Backtest

```bash
python3 scripts/backtest_alpha_v8.py \
  --symbol BTCUSDT \
  --strategy-f \
  --selector stable_icir
```

### Run Walk-Forward Validation

```bash
python3 scripts/walkforward_validate.py \
  --symbol BTCUSDT \
  --strategy-f \
  --selector stable_icir
```

### Run Paper Trading

```bash
python3 -m runner.live_paper_runner --config infra/config/examples/testnet_v8_gate_v2.yaml
```

### Run Live

```bash
python3 -m runner.live_runner --config config/local.yaml
```

## Module Overview

| Module | Files | Description |
|--------|-------|-------------|
| `engine/` | 19 | EngineCoordinator event loop, StatePipeline, dispatcher, feature hook |
| `ext/rust/` | 82 | Rust crate, PyO3 kernel, and standalone `quant_trader` binary |
| `features/` | 27 | Feature computation, enriched computer, cross-asset |
| `decision/` | 88 | DecisionEngine, signals, sizing, execution policy, regime/rebalance modules |
| `alpha/` | 30 | ML models, inference bridge, drift/OOD monitoring |
| `risk/` | 19 | KillSwitch, CorrelationGate, risk aggregation, margin monitoring |
| `execution/` | 195 | Exchange adapters, order state machine, dedup, reconciliation, algos |
| `portfolio/` | 78 | Black-Litterman, Kelly, risk parity, risk model, optimizers |
| `state/` | 12 | State types, Rust adapters, snapshot management |
| `event/` | 22 | Event types, header, store, checkpoint, security |
| `regime/` | 4 | Regime detection (volatility, trend) |
| `strategies/` | 19 | HFT strategies, imbalance scalper |
| `runner/` | 14 | LiveRunner, PaperRunner, BacktestRunner, graceful shutdown |
| `monitoring/` | 27 | Prometheus metrics, Grafana dashboards, Telegram alerts, health server |
| `infra/` | 25 | Config loader, structured JSON logging, model signing, threading utils |
| `scripts/` | 113 | Training, walk-forward validation, alpha research, data download |
| `research/` | 27 | Experiment tracking, Monte Carlo, sensitivity analysis, model registry |
| `tests/` | 249 | Unit + integration + replay + regression + performance tests |

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

## Testing

```bash
# All tests
python3 -m pytest tests/ -x -q
cd ext/rust && cargo test

# Rust parity checks
python3 -m pytest tests/ -x -q -k "rust"

# Benchmarks
python3 -m pytest tests/ -m benchmark
```

## Build

```bash
# Build Rust kernel
make rust

# Or manually:
cd ext/rust && maturin build --release
pip install target/wheels/*.whl --force-reinstall --break-system-packages
cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/
```

## Deployment

```bash
# Docker build (multi-stage, includes Rust compilation)
docker build -t quant-system .

# Docker Compose (paper trading + monitoring)
docker compose --profile monitoring up -d

# Kubernetes
kubectl apply -f deploy/k8s/
```

## Project Structure

```
quant_system/
  engine/            # Core event loop and coordination (19 files)
  ext/rust/          # Rust crate + PyO3 kernel + standalone binary (82 Rust files)
  features/          # Feature engineering (27 files)
  decision/          # Trading decision logic (88 files)
  alpha/             # ML alpha models and inference (30 files)
  risk/              # Risk management and kill switch (19 files)
  execution/         # Exchange connectivity and execution platform (195 files)
  portfolio/         # Portfolio optimization (78 files)
  state/             # State management + Rust adapters (12 files)
  event/             # Event type definitions (22 files)
  regime/            # Market regime detection (4 files)
  strategies/        # HFT strategies (19 files)
  runner/            # Entry points: live, paper, backtest (14 files)
  monitoring/        # Prometheus, Grafana, Telegram (27 files)
  infra/             # Config, logging, secrets (25 files)
  scripts/           # Training, research, data download (113 files)
  research/          # Experiment tracking, Monte Carlo (27 files)
  _quant_hotpath/    # Built Rust shared library
  tests/             # Unit + integration + replay + regression + performance (249 files)
  deploy/            # Docker, K8s, Prometheus, Grafana configs
```

## License

Proprietary
