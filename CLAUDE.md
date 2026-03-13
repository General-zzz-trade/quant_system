## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
pytest tests/ -x -q          # Fast tests (~27s, excludes slow+benchmark)
pytest tests/ -x -q -m ""   # ALL tests including slow (~35s)
pytest tests/unit/ -x -q     # Unit tests only
pytest -m slow               # Slow tests only (parity, NN, XGB)
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test    # Rust unit tests (57 tests)
```

**CRITICAL after Rust build**: copy .so to local package (shadows system install):
```bash
cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/
```

**Binary build** (standalone Rust trader, requires Python linkage):
```bash
cd ext/rust && RUSTFLAGS="-C link-arg=-L/usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" cargo build --release --bin quant_trader
./target/release/quant_trader --config config.testnet.yaml [--dry-run]
```

**Docker deployment**:
```bash
docker compose up -d trader-rust      # Start Rust trader (testnet)
docker compose logs -f trader-rust    # Follow logs
curl localhost:9090/metrics           # Prometheus metrics
curl -X POST localhost:9090/kill      # Emergency kill switch
```

## Architecture

```
engine/          Pipeline + coordinator (event -> state transitions)
features/        Feature computation (EnrichedFeatureComputer, 105 features)
decision/        Trading signals, ensemble, regime detection, rebalancing
alpha/           ML models + inference bridge
execution/       Order routing, state machine, dedup
state/           State types + Rust adapters
attribution/     P&L + cost + signal attribution (thin Rust wrappers)
event/           Event types + runtime protocol
strategies/      HFT + multi-factor strategy implementations
ext/rust/        Unified Rust crate -> _quant_hotpath (67 modules, ~25K LOC)
ext/rust/src/bin/ Standalone trading binary (main.rs + config.rs, ~2.6K LOC)
runner/          Live/paper/backtest entry points
regime/          Regime detection (volatility, trend)
risk/            Risk limits + kill switch
portfolio/       Allocator, rebalance, optimizer
monitoring/      Alerts, health checks, metrics, Prometheus, Grafana
infra/           Logging (structured JSON), networking
scripts/         Training, walk-forward validation, alpha research
```

**Data flow (Python)**: Market event -> FeatureComputeHook (RustFeatureEngine) -> Pipeline
  (RustStateStore) -> DecisionModule -> ExecutionPolicy -> OrderRouter
**Fast path (Python)**: Market event -> RustTickProcessor.process_tick_full() (features+predict+state+features_dict in one Rust call) -> DecisionModule
**Binary path**: Binance WS → parse_kline/aggTrade → RustTickProcessor.process_tick_native() → risk gates → WS-API order
**Order path**: DecisionModule -> BinanceWsOrderGateway (WS-API, ~4ms) or REST fallback (~30ms)

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 67 .rs modules, ~25,000 LOC
- Exports: 64 classes + 106 functions
- Binary: `quant_trader` standalone trading binary (no Python runtime)
- Naming: `cpp_*` = C++ migration functions, `rust_*` = new kernel modules
- State types use i64 fixed-point (Fd8, x10^8); `_SCALE = 100_000_000`
- feature_hook.py always uses Rust (no Python fallback)
- `RustStateStore` keeps state on Rust heap, Python gets snapshots via `get_*()`

Key exports:
- State: `RustStateStore`, `RustMarketState`, `RustPositionState`, `RustAccountState`
- Events: `RustMarketEvent`, `RustFillEvent`, `RustFundingEvent`
- Features: `RustFeatureEngine` (105 features), `RustCrossAssetComputer`
- Risk: `RustRiskEvaluator`, `RustKillSwitch`, `RustCircuitBreaker`
- Decision: `rust_rolling_sharpe`, `rust_max_drawdown`, `rust_strategy_weights`
- Portfolio: `rust_allocate_portfolio`, `rust_fixed_fraction_qty`
- Pipeline: `rust_pipeline_apply`, `RustProcessResult`
- Factors: `rust_momentum_score`, `rust_volatility_score`, `rust_adx`, `rust_carry_score`
- Inference: `RustInferenceBridge` (z-score, min-hold, monthly gate, short signal)
- Attribution: `rust_compute_pnl`, `rust_compute_cost_attribution`, `rust_attribute_by_signal`
- Orderbook: `rust_flush_orderbook_bar`, `rust_extract_orderbook_features`
- Ensemble: `rust_adaptive_ensemble_calibrate`
- Unified: `RustUnifiedPredictor` (zero-copy feature→predict→signal pipeline, 2.9x faster)
- TickProcessor: `RustTickProcessor` (full hot path: features+predict+state in single call), `RustTickResult`
- WebSocket: `RustWsClient` (tokio-tungstenite, GIL-free recv+send), `rust_parse_agg_trade`
- OrderGateway: `RustWsOrderGateway` (WS-API order submission with Rust HMAC signing, ~4ms vs ~30ms REST)
- MicroAlpha: `MicroAlpha` (aggTrade-driven trade flow/volume/large trade signals)
- Transport: `RustWsTransport` in `execution/adapters/binance/ws_transport_rust.py`

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 105 enriched feature definitions
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `ext/rust/src/bin/main.rs` — Standalone Rust trading binary (WS + ML + orders)
- `ext/rust/src/bin/config.rs` — Binary config (YAML + model config.json overrides)
- `runner/live_runner.py` — Production entry point (Python)

## Live Integration Subsystems

- **Alpha Health Monitor**: `AlphaHealthMonitor` in `monitoring/alpha_health.py` — tracks per-symbol IC, gates orders via `position_scale()` (0.0/0.5/1.0). Wired in `EngineMonitoringHook` and `_emit()` gate in live_runner.
- **WS-API Orders**: `WsOrderAdapter` in `execution/adapters/binance/ws_order_adapter.py` — ~4ms WS-API with REST fallback. Enable via `LiveRunnerConfig(use_ws_orders=True)`.
- **Adaptive BTC Config**: `AdaptiveConfigSelector.select_robust()` runs periodically (24h) for BTCUSDT only. Updates `LiveInferenceBridge.update_params()` on `confidence="high"`. Enable via `LiveRunnerConfig(adaptive_btc_enabled=True)`.
- **Auto-Retrain**: `scripts/auto_retrain.py --notify-runner --alert` sends SIGHUP to runner + Telegram/webhook alerts. Systemd timer in `infra/systemd/auto-retrain.timer` (Sunday 2am UTC).
- **SIGHUP Model Reload**: Works with both ModelRegistry (registry-based) and direct file reload (models_v8/*.pkl). Runner always installs SIGHUP handler.

## Gotchas

- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `RustFeatureEngine` uses its own window sizes (not LiveFeatureComputer params)
- Tests require `_quant_hotpath` built; `pytest.importorskip("_quant_hotpath")` guards Rust tests
- Production models in `models_v8/` (LightGBM)
- CrossAssetComputer: must push benchmark (BTCUSDT) **before** altcoins each bar
- Fd8 conversion: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- `features/dynamic_selector.py` keeps `_rankdata`/`_spearman_ic` for scripts (not fallback)
- `pip install` requires `--break-system-packages` flag (no venv, system Python 3.12)
- No Python fallbacks remain: rolling.py, multi_timeframe.py, factor signals all require Rust
- `features/_rolling_py.py` only has `rolling_apply` (RollingWindow class deleted)
- Binary build requires `-lpython3.12` link flag (PyO3 symbols)
- Binary config priority: model `config.json` > YAML `per_symbol` > YAML `strategy` defaults
- Binance minimum notional: $100 per order (error -4164), fraction≥0.05 for testnet
