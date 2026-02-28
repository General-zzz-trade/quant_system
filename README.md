# Quant System

Production-grade quantitative trading system for crypto perpetual futures. ML-driven alpha generation, multi-exchange execution, institutional risk management, and C++ accelerated compute.

## Architecture

```
                          +------------------+
                          |  Market Data WS  |  (Binance, Bitget)
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
     |  (state updates)|  |  (C++ accel)   |  |  (LightGBM/XGB)|
     +--------+--------+  +-------+--------+  +-------+--------+
              |                    |                    |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          |  DecisionBridge  |
                          |  (strategy logic)|
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
     +--------v--------+  +-------v--------+  +-------v--------+
     |  RiskGate       |  | CorrelationGate|  |  KillSwitch    |
     |  (size/notional)|  | (concentration)|  |  (circuit break)|
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

**Event flow**: MarketEvent -> StatePipeline -> DecisionBridge -> RiskGates -> ExecutionBridge -> Exchange

All state mutations go through `StatePipeline.apply()`. Decision modules are read-only — they return "opinion events" (IntentEvent, OrderEvent), never modify state directly.

## Quick Start

### Prerequisites

- Python 3.12+
- C++ compiler with C++17 support (optional, for acceleration)
- `pybind11` (optional, for C++ bindings)

### Installation

```bash
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

# Shadow mode (simulated orders, real market data)
python -m runner.live_runner --config config/local.yaml --shadow
```

## Module Overview

| Module | Description |
|--------|-------------|
| `engine/` | EngineCoordinator event loop, StatePipeline, dispatcher, guards |
| `alpha/` | ML models (LightGBM, XGBoost), inference bridge, drift detection |
| `risk/` | KillSwitch, CorrelationGate, CorrelationComputer, risk aggregation |
| `execution/` | Exchange adapters (Binance, Bitget), rate limiting, reconciliation |
| `portfolio/` | Black-Litterman, Kelly allocator, MVO, risk parity, rebalancing |
| `decision/` | DecisionEngine, regime-aware gating, intent validation, sizing |
| `strategies/` | Multi-timeframe ensemble, factor strategies, stat-arb, HFT |
| `research/` | Walk-forward validation, overfit detection, combinatorial CV |
| `features/` | Feature computation, C++ accelerated rolling windows |
| `data/` | Quality monitoring, lineage tracking, backup management |
| `monitoring/` | SLO/SLI tracking, Prometheus metrics, Grafana dashboards, alerts |
| `event/` | Event types (Market, Signal, Intent, Order, Fill, Risk, Control) |
| `state/` | MarketState, AccountState, PositionState, StateSnapshot |
| `runner/` | LiveRunner, BacktestRunner, preflight checks, graceful shutdown |
| `ext/` | C++ pybind11 extensions (rolling, cross-sectional, portfolio math) |
| `infra/` | Config loader, structured logging, secret resolution |
| `deploy/` | K8s manifests, Dockerfile, CI/CD workflows |

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
# Unit tests (fast, no external deps)
python -m pytest tests_unit/ -x -q

# Integration tests
python -m pytest tests/ -x -q --tb=short

# All tests with coverage
python -m pytest tests/ tests_unit/ --cov=. --cov-report=term-missing

# Performance benchmarks
python -m pytest tests/ -m benchmark -v
```

## C++ Acceleration

Six pybind11 modules provide 5x-90x speedups over pure Python:

| Module | Functions | Speedup |
|--------|-----------|---------|
| `rolling_window` | SMA, EMA, RSI, MACD, Bollinger, ATR, VWAP | 8-15x |
| `cross_sectional` | momentum_rank, rolling_beta, relative_strength | 8-76x |
| `portfolio_math` | sample_cov, ewma_cov, rolling_corr | 39-90x |
| `factor_math` | factor exposures, factor_model_cov, specific_risk | 21-45x |
| `feature_selection` | correlation_select, mutual_info_select | 5-26x |
| `linalg` | Black-Litterman posterior, matrix inverse | 34x |

```bash
make rolling    # Build all C++ extensions
make clean      # Remove compiled .so files
```

Automatic fallback to Python implementations if C++ is not built.

## Deployment

```bash
# Docker build (multi-stage)
docker build --target live -t quant-system:latest .

# Kubernetes
kubectl apply -f deploy/k8s/
```

See [docs/operations.md](docs/operations.md) for the full production operations manual.

## Project Structure

```
quant_system/
  engine/          # Core event loop and coordination
  alpha/           # ML alpha models and inference
  risk/            # Risk management and kill switch
  execution/       # Exchange connectivity
  portfolio/       # Portfolio optimization
  decision/        # Trading decision logic
  strategies/      # Strategy implementations
  research/        # Research and validation tools
  features/        # Feature engineering
  data/            # Data management
  monitoring/      # Observability stack
  event/           # Event type definitions
  state/           # State management
  runner/          # Entry points (live, backtest)
  ext/rolling/     # C++ pybind11 source
  infra/           # Config, logging, secrets
  deploy/          # K8s, Docker, CI/CD
  tests/           # Integration tests
  tests_unit/      # Unit tests
```

## License

Proprietary
