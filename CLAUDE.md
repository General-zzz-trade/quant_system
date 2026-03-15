## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
make test                    # ALL gates (py + exec + rust + lint, matches CI)
pytest tests/unit/ -x -q     # Unit tests only (~18s)
pytest tests/ -x -q -m ""   # ALL tests including slow (~35s)
pytest execution/tests/ -x -q  # Execution subsystem tests (67 tests)
pytest tests/unit/ib/ -x -q   # IB adapter tests (37 tests)
pytest tests/unit/runner/ -x -q    # Runner tests (298 tests)
pytest tests/unit/runner_v2/ -x -q  # Decomposed runner tests (42 tests)
pytest tests/unit/bybit/ -x -q   # Bybit adapter tests (14 tests)
pytest -m slow               # Slow tests only (parity, NN, XGB)
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test    # Rust unit tests (82 tests)
ruff check --select E,W,F . # Lint (matches CI gate)
```

**Validation & paper trading**:
```bash
python3 -m scripts.testnet_smoke --public-only                     # Testnet connectivity (no API key)
python3 -m scripts.testnet_smoke                                   # Full smoke test (18 checks)
python3 -m scripts.walkforward_validate --symbol BTCUSDT --no-hpo  # Walk-forward OOS (~20s)
python3 -m scripts.walkforward_validate --symbol BTCUSDT           # Walk-forward + HPO (~30min)
python3 -m scripts.run_paper_trading --symbols BTCUSDT --testnet   # Paper trading (shadow mode)
```

**CRITICAL after Rust build**: copy .so then verify:
```bash
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "import _quant_hotpath; print(_quant_hotpath.rust_version())"  # verify import works
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
core/            Bootstrap, config, bus, clock, effects, observability
engine/          Pipeline + coordinator (event -> state transitions)
features/        Feature computation (EnrichedFeatureComputer, 105 features)
decision/        Trading signals, ensemble, regime detection, rebalancing
alpha/           ML models + inference bridge
execution/       Order routing, state machine, dedup
  adapters/binance/    Binance USDT-M futures (47 files, ~3.7K LOC)
  adapters/bybit/      Bybit V5 linear perpetuals (demo/testnet/live)
  adapters/ib/         Interactive Brokers multi-asset (stocks/forex/futures/options/crypto)
  adapters/polymarket/ Polymarket prediction market CLOB adapter
  adapters/generic/    CCXT-based unified 100+ exchange adapter
state/           State types + Rust adapters
attribution/     P&L + cost + signal attribution (thin Rust wrappers)
event/           Event types + runtime protocol
strategies/      HFT + multi-factor strategy implementations
ext/rust/        Unified Rust crate -> _quant_hotpath (66 .rs files, ~24K LOC)
ext/rust/src/bin/ Standalone trading binary (main.rs + config.rs, ~2.6K LOC)
runner/          Live/paper/backtest entry points
regime/          Regime detection (volatility, trend)
risk/            Risk limits + kill switch
portfolio/       Allocator, rebalance, optimizer
monitoring/      Alerts, health checks, metrics, Prometheus, Grafana
infra/           Logging (structured JSON), networking
polymarket/      Polymarket 5m BTC Up/Down — collector, features, signals, runner
models_v8/       Production LightGBM models (BTCUSDT_gate_v2, ETHUSDT_gate_v2, SOLUSDT)
research/        Alpha research, factor backtests, hyperopt, Monte Carlo
scripts/         Training, walk-forward, research, data, ops (7 subdirs + symlinks for compat)
```

**Data flow (Python)**: Market event -> FeatureComputeHook (RustFeatureEngine) -> Pipeline
  (RustStateStore) -> DecisionModule -> ExecutionPolicy -> OrderRouter
**Fast path (Python, candidate optimization)**: Market event -> RustTickProcessor.process_tick_full() (features+predict+state+features_dict in one Rust call) -> DecisionModule
**Binary path**: Binance WS → parse_kline/aggTrade → RustTickProcessor.process_tick_native() → risk gates → WS-API order
**Order path**: DecisionModule -> BinanceWsOrderGateway (WS-API, ~4ms) or REST fallback (~30ms)

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 66 .rs modules, ~24K LOC
- Exports: ~27 PyO3 classes + ~100 functions (see `lib.rs`)
- Binary: `quant_trader` standalone trading binary (no Python runtime)
- Naming: `cpp_*` = C++ migration functions, `rust_*` = new kernel modules
- State types use i64 fixed-point (Fd8, x10^8); `_SCALE = 100_000_000`
- feature_hook.py always uses Rust (no Python fallback)
- `RustStateStore` keeps state on Rust heap, Python gets snapshots via `get_*()`

Key export categories (see `lib.rs` for full list):
- **State**: `RustStateStore`, `RustMarketState`, `RustPositionState`, `RustAccountState`
- **Features**: `RustFeatureEngine` (105 features), `RustCrossAssetComputer`
- **Risk**: `RustRiskEvaluator`, `RustKillSwitch`, `RustCircuitBreaker`
- **Pipeline**: `rust_pipeline_apply`, `RustProcessResult`, `RustUnifiedPredictor`
- **TickProcessor**: `RustTickProcessor` (candidate optimization: features+predict+state in single call; not default production path)
- **Networking**: `RustWsClient`, `RustWsOrderGateway` (WS-API, ~4ms), `MicroAlpha`
- **Inference**: `RustInferenceBridge` (z-score, min-hold, monthly gate, short signal)

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 105 enriched feature definitions
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `ext/rust/src/bin/main.rs` — Standalone Rust trading binary (WS + ML + orders)
- `ext/rust/src/bin/config.rs` — Binary config (YAML + model config.json overrides)
- `runner/live_runner.py` — Production entry point (Python)
- `runner/emit_handler.py` — LiveEmitHandler (ORDER gate chain + FILL tracking)
- `runner/recovery.py` — Crash recovery: 8-component atomic bundle (kill_switch, inference_bridge, feature_hook, correlation, timeout, exit_manager, regime_gate, drawdown_breaker)
- `runner/config.py` — LiveRunnerConfig (93 fields); factory methods: `.lite()`, `.paper()`, `.testnet_full()`, `.prod()`
- `features/feature_catalog.py` — PRODUCTION_FEATURES frozenset (122 features: 105 enriched + 17 cross-asset); `validate_model_features()` for schema checks
- `execution/adapters/ib/adapter.py` — IB multi-asset adapter (stocks, forex, futures, options, crypto via IB Gateway)
- `execution/adapters/ib/mapper.py` — IB Contract/Fill/Position → Canonical types; `make_contract()` builder
- `polymarket/collector.py` — 5m BTC Up/Down real-time CLOB orderbook collector + BS fair value (V2: direct CLOB, not Gamma cache)

## Live Integration Subsystems

- **Alpha Health Monitor**: `AlphaHealthMonitor` in `monitoring/alpha_health.py` — tracks per-symbol IC, gates orders via `position_scale()` (0.0/0.5/1.0). Wired in `EngineMonitoringHook` and `LiveEmitHandler` gate chain.
- **WS-API Orders**: `WsOrderAdapter` in `execution/adapters/binance/ws_order_adapter.py` — ~4ms WS-API with REST fallback. Enable via `LiveRunnerConfig(use_ws_orders=True)`.
- **Adaptive BTC Config**: `AdaptiveConfigSelector.select_robust()` runs periodically (24h) for BTCUSDT only. Updates `LiveInferenceBridge.update_params()` on `confidence="high"`. Enable via `LiveRunnerConfig(adaptive_btc_enabled=True)`.
- **Auto-Retrain**: `scripts/auto_retrain.py --notify-runner --alert` sends SIGHUP to runner + Telegram/webhook alerts. Systemd timer in `infra/systemd/auto-retrain.timer` (Sunday 2am UTC).
- **SIGHUP Model Reload**: Works with both ModelRegistry (registry-based) and direct file reload (models_v8/*.pkl). Runner always installs SIGHUP handler.

## Venue Adapters

All adapters implement `VenueAdapter` protocol (`execution/adapters/base.py`); registered via `AdapterRegistry.register(venue, adapter)`.

| Venue | Protocol | Notes |
|-------|----------|-------|
| Binance | REST + WS-API (~4ms) | Primary. USDT-M futures. 47 files, ~3.7K LOC |
| IB | `ib_insync` → IB Gateway | Port 4002=paper, 4001=live. All asset classes via `make_contract(symbol, sec_type)` |
| Polymarket | CLOB REST + Gamma discovery | L2 auth (HMAC-SHA256). Collector V2 uses CLOB orderbook directly, stores SQLite |
| Bybit | REST V5 + WS | Demo/testnet/live. USDT perpetuals. HMAC-SHA256 auth |
| Generic | CCXT | 100+ exchanges, unified interface |

## Environment

```bash
export BINANCE_TESTNET_API_KEY=...    # from testnet.binancefuture.com (GitHub login)
export BINANCE_TESTNET_API_SECRET=... # required for authenticated endpoints + paper trading
```

**Quick start → paper trading**:
1. Set env vars above
2. `python3 -m scripts.testnet_smoke --public-only` — verify connectivity
3. `python3 -m scripts.walkforward_validate --symbol BTCUSDT --no-hpo` — verify model OOS performance
4. `python3 -m scripts.run_paper_trading --symbols BTCUSDT --testnet` — start shadow trading

**Walk-forward baseline** (2025-03-15, BTCUSDT 1h, 20 folds):
- No HPO: Sharpe 0.26, +17% total, 12/20 positive (recent 4 folds: Sharpe 0.98)
- HPO: Sharpe 0.38, **+104% total**, 12/20 positive (recent 4 folds: **Sharpe 2.27**)

## Gotchas

- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `RustFeatureEngine` uses its own window sizes (not LiveFeatureComputer params)
- Tests require `_quant_hotpath` built; `pytest.importorskip("_quant_hotpath")` guards Rust tests
- Production models in `models_v8/` (LightGBM)
- CrossAssetComputer: must push benchmark (BTCUSDT) **before** altcoins each bar
- Fd8 conversion: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- `features/dynamic_selector.py` keeps `_rankdata`/`_spearman_ic` for scripts (not fallback)
- `pip install` requires `--break-system-packages` flag (no venv, system Python 3.12)
- Live hot-path has no Python fallbacks (rolling.py, multi_timeframe.py require Rust); research scripts still have Python-only paths with known divergences (see `signal_postprocess.py`)
- Binary build requires `-lpython3.12` link flag (PyO3 symbols)
- Binary config priority: model `config.json` > YAML `per_symbol` > YAML `strategy` defaults
- Binance minimum notional: $100 per order (error -4164), fraction≥0.05 for testnet
- `RustFeatureEngine.checkpoint()` / `restore_checkpoint()` persist rolling windows as bar history JSON — crash recovery replays bars to rebuild EMA/RSI/ATR state
- Event recorder captures risk/control events in addition to market/fill/signal/order
- State ownership: `RustStateStore` is primary position truth (decisions read from it); `OrderStateMachine` is execution audit trail only (exception: `RiskGate.max_open_orders` reads OSM open-order count)
- Recovery bundle now includes 8 components (drawdown_breaker HWM added); `save_all_auxiliary_state()` / `restore_all_auxiliary_state()` are the primary entry points
- `GateChain._apply_scale()` preserves Decimal type (no float conversion); uses `dataclasses.replace()` on frozen OrderEvent to avoid mutation
- `LiveEmitHandler` attribution tracks only accepted orders (rejected orders excluded from IC/Sharpe); fill supports PARTIALLY_FILLED via `ev.is_partial` or `ev.status`
- Python fallback in `signal_postprocess.py` now uses `_enforce_hold_single_pass()` matching Rust `enforce_hold_step` exactly (parity verified 2026-03-15)
- IB Gateway requires Xvfb on headless Linux (`Xvfb :99 &; export DISPLAY=:99`); login uses virtual keyboard (manual VNC login required first time, then stays running)
- Polymarket Gamma API `outcomePrices`/`bestBid`/`bestAsk` are **cached/stale** — always use CLOB orderbook (`clob.polymarket.com/book?token_id=`) for real prices
- IB `VenueType` enum includes FOREX, CFD, OPTIONS in addition to SPOT/FUTURES/PERPETUAL
