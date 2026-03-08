# Quant System

Production-grade quantitative trading system for crypto perpetual futures. ML-driven alpha generation, multi-exchange execution, institutional risk management, and **Rust-accelerated kernel** (PyO3).

## Architecture

```
                          +------------------+
                          |  Market Data WS  |  (Binance / Bitget)
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

**Event flow**: MarketEvent -> StatePipeline (Rust) -> FeatureEngine (Rust, 105 features) -> ML Inference -> DecisionBridge (Rust sizing) -> RiskEvaluator (Rust) -> ExecutionBridge -> Exchange

All state mutations go through `rust_pipeline_apply()`. Decision modules are read-only -- they return "opinion events" (IntentEvent, OrderEvent), never modify state directly.

## Rust Kernel

The system's hot path runs on a unified Rust crate (`ext/rust/` -> `_quant_hotpath`), built via PyO3 + maturin.

**No Python fallbacks remain** -- all computation kernels require the Rust crate.

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
| Regime | `regime_buffer.rs` | `RustRegimeBuffer` -- regime detection buffer |
| Fixed-point | `fixed_decimal.rs` | `Fd8` type (i64 x 10^8), eliminates Decimal parsing |

**Crate stats**: 55 Rust modules, ~16,700 LOC, 154 exports, 52 Rust tests + 2,650 Python tests passing.

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
| `engine/` | 21 | EngineCoordinator event loop, StatePipeline, dispatcher, feature hook |
| `ext/rust/` | 55 | Rust PyO3 kernel -- state, pipeline, features, risk, events, sizing |
| `features/` | 31 | Feature computation, enriched computer (105 features), cross-asset |
| `decision/` | 82 | DecisionEngine, signals (technical + factor + ML), sizing, execution policy |
| `alpha/` | 31 | ML models (LightGBM, XGBoost, LSTM, Transformer), inference bridge, drift detection |
| `risk/` | 24 | KillSwitch, CorrelationGate, risk aggregation, 6 risk rules |
| `execution/` | 192 | Exchange adapters (Binance, Bitget), order state machine, dedup, reconciliation |
| `portfolio/` | 80 | Black-Litterman, Kelly, risk parity, risk model (vol, correlation, covariance, tail) |
| `state/` | 14 | State types, Rust adapters, snapshot management |
| `event/` | 26 | Event types, header, store, checkpoint, replay, security |
| `regime/` | 7 | Regime detection (volatility, trend, HMM) |
| `strategies/` | 19 | HFT strategies, imbalance scalper |
| `runner/` | 14 | LiveRunner, PaperRunner, BacktestRunner, graceful shutdown |
| `monitoring/` | 27 | Prometheus metrics, Grafana dashboards, Telegram alerts, health server |
| `infra/` | 25 | Config loader, structured JSON logging, model signing, threading utils |
| `scripts/` | 55 | Training, walk-forward validation, alpha research, data download |
| `research/` | 30 | Experiment tracking, Monte Carlo, sensitivity analysis, model registry |
| `tests/` | 212 | Unit + integration + regression tests (2,650 passing) |

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
# All tests (2,650 Python + 52 Rust)
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
  engine/            # Core event loop and coordination (21 files)
  ext/rust/          # Rust PyO3 kernel (55 modules, ~16,700 LOC)
  features/          # Feature engineering (31 files, 105 features)
  decision/          # Trading decision logic (82 files)
  alpha/             # ML alpha models and inference (31 files)
  risk/              # Risk management and kill switch (24 files)
  execution/         # Exchange connectivity (192 files)
  portfolio/         # Portfolio optimization (80 files)
  state/             # State management + Rust adapters (14 files)
  event/             # Event type definitions (26 files)
  regime/            # Market regime detection (7 files)
  strategies/        # HFT strategies (19 files)
  runner/            # Entry points: live, paper, backtest (14 files)
  monitoring/        # Prometheus, Grafana, Telegram (27 files)
  infra/             # Config, logging, secrets (25 files)
  scripts/           # Training, research, data download (55 files)
  research/          # Experiment tracking, Monte Carlo (30 files)
  _quant_hotpath/    # Built Rust shared library
  tests/             # Unit + integration + regression (212 files, 2,650 tests)
  deploy/            # Docker, K8s, Prometheus, Grafana configs
```

## License

Proprietary
