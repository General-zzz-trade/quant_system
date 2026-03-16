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
pytest tests/unit/bitget/ -x -q  # Bitget adapter tests (14 tests)
pytest -m slow               # Slow tests only (parity, NN, XGB)
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test    # Rust unit tests (82 tests)
ruff check --select E,W,F . # Lint (matches CI gate)
```

**Walk-forward validation**:
```bash
python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo                          # Quick OOS (~20s)
python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo --realistic              # Realistic engine
python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo --realistic --adaptive-stop  # + ATR trailing stop
python3 -m scripts.walkforward_validate --symbol ETHUSDT                                    # Full HPO (~30min)
python3 -m scripts.walkforward_validate --symbol ETHUSDT --selector stable_greedy           # Stability-filtered features
```

**Alpha strategy (demo/live)**:
```bash
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws  # Full portfolio (production)
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT --ws --dry-run         # Signal only, no orders
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT --once --dry-run       # Single bar then exit
sudo systemctl restart bybit-alpha.service                                   # Restart service
sudo systemctl status bybit-alpha.service                                    # Check service
tail -f /quant_system/logs/bybit_alpha.log                                   # Follow live logs
```

**Data & model management**:
```bash
python3 -m scripts.data.download_15m_klines                                  # Update 15m kline data (incremental)
python3 -m scripts.data.download_funding_rates --symbols ETHUSDT SOLUSDT     # Update funding rate history
python3 -m scripts.auto_retrain --include-15m --force                        # Retrain 1h + 15m models
python3 -m scripts.auto_retrain --only-15m --force                           # Retrain 15m models only
python3 -m scripts.auto_retrain --dry-run                                    # Preview retrain without saving
```

**Monitoring & diagnostics**:
```bash
python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log  # Live vs backtest comparison
python3 -m scripts.testnet_smoke --public-only                                # Exchange connectivity check
python3 -m scripts.research.research_funding_alpha --symbol ETHUSDT           # Funding alpha IC analysis
python3 -m scripts.research.research_15m_alpha --symbol ETHUSDT               # 15m alpha research
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
features/        Feature computation (EnrichedFeatureComputer, 111 features incl. V12 cross-asset)
  dynamic_selector.py  Feature selection: greedy_ic, stable_icir, stability_filtered_greedy
  feature_catalog.py   PRODUCTION_FEATURES frozenset (122 = 105 enriched + 17 cross-asset)
decision/        Trading signals, ensemble, regime detection, rebalancing
  backtest_module.py   MLSignalDecisionModule (z-score, min-hold, trend-hold, regime gate)
  exit_manager.py      Trailing stop, z-reversal, deadzone fade exits
alpha/           ML models + inference bridge
  horizon_ensemble.py  Multi-horizon IC-weighted ensemble (12h + 24h)
  adaptive_config.py   Dynamic deadzone/min_hold selection (24h sweep)
execution/       Order routing, state machine, dedup
  adapters/binance/    Binance USDT-M futures (47 files, ~3.7K LOC, WS-API ~4ms)
  adapters/bybit/      Bybit V5 linear perpetuals (demo/testnet/live, 6 files)
  adapters/bitget/     Bitget V2 USDT-M futures (HMAC-SHA256+base64, passphrase, 6 files)
  adapters/ib/         Interactive Brokers multi-asset (stocks/forex/futures/options/crypto)
  adapters/polymarket/ Polymarket prediction market CLOB adapter
  adapters/generic/    CCXT-based unified 100+ exchange adapter
  sim/                 Backtest engines (realistic_backtest.py, limit_order_book.py, cost_constants.py)
state/           State types + Rust adapters
attribution/     P&L + cost + signal attribution (thin Rust wrappers)
event/           Event types + runtime protocol
strategies/      HFT + multi-factor strategy implementations
ext/rust/        Unified Rust crate -> _quant_hotpath (66 .rs files, ~24K LOC)
ext/rust/src/bin/ Standalone trading binary (main.rs + config.rs, ~2.6K LOC)
runner/          Live/paper/backtest entry points
  gate_chain.py        GateChain: 8-gate order pipeline (correlation → risk → portfolio → alpha health)
  emit_handler.py      LiveEmitHandler: ORDER/FILL routing with audit trail
regime/          Regime detection (volatility, trend)
risk/            Risk limits + kill switch
portfolio/       Allocator, rebalance, optimizer
monitoring/      Alerts, health checks, metrics, Prometheus, Grafana
infra/           Logging (structured JSON), networking, systemd units
polymarket/      Polymarket 5m BTC Up/Down — collector, features, signals, runner
models_v8/       Production models (Ridge primary 60% + LightGBM 40%)
  ETHUSDT_gate_v2/   1h model (v11, IC 0.075, 14 features, min_hold=18)
  ETHUSDT_15m/       15m model (v1, IC 0.075, 14 features, h=32 horizon)
  SUIUSDT/           1h model (v1, 15 features, 6/7 PASS Sharpe 1.63)
  AXSUSDT/           1h model (v1, 15 features, 13/17 PASS Sharpe 1.25)
  BTCUSDT_gate_v2/   1h model (FAIL — not deployed)
research/        Alpha research, factor backtests, hyperopt, Monte Carlo
scripts/         7 subdirs + symlinks for compat
  ops/               run_bybit_alpha.py, compare_live_backtest.py
  data/              download_15m_klines.py, download_funding_rates.py
  research/          backtest_funding_alpha.py, backtest_vol_squeeze.py, leverage_survival_sim.py
  walkforward/       walkforward_validate.py
  training/          train_v7_alpha.py, train_15m.py
data_files/      CSV data: {SYMBOL}_{1h,15m,1m}.csv, {SYMBOL}_funding.csv, fear_greed_index.csv
logs/            bybit_alpha.log, polymarket-collector.log, retrain_cron.log
tests/           unit/ (runner, bybit, bitget, decision, features), integration/ (constraint parity)
```

**Data flow (live alpha)**:
```
Bybit WS kline → EnrichedFeatureComputer.on_bar() → Ridge(60%)+LightGBM(40%) ensemble predict
  → z-score normalize → deadzone discretize → min-hold enforce
  → PortfolioCombiner (AGREE ONLY: both 1h+15m agree → trade)
  → ATR adaptive stop (initial → breakeven → trailing)
  → Bybit REST market order
```

**Data flow (Python engine)**: Market event → FeatureComputeHook (RustFeatureEngine) → Pipeline
  (RustStateStore) → DecisionModule → ExecutionPolicy → OrderRouter
**Fast path**: Market event → RustTickProcessor.process_tick_full() → DecisionModule
**Binary path**: Binance WS → RustTickProcessor.process_tick_native() → risk gates → WS-API order
**Order path**: DecisionModule → BinanceWsOrderGateway (WS-API, ~4ms) or REST fallback (~30ms)

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
- **Selection**: `cpp_greedy_ic_select_np`, `cpp_stable_icir_select`, `cpp_feature_icir_report`

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 105 enriched feature definitions
- `features/dynamic_selector.py` — Feature selection: greedy, stable_icir, stability_filtered_greedy
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `ext/rust/src/constraint_pipeline.rs` — Signal constraints: z-score, discretize, min-hold, trend-hold (batch + incremental)
- `ext/rust/src/bin/main.rs` — Standalone Rust trading binary (WS + ML + orders)
- `runner/live_runner.py` — Production entry point (Python)
- `runner/emit_handler.py` — LiveEmitHandler (ORDER gate chain + FILL tracking)
- `runner/gate_chain.py` — GateChain: 8 gates with `process_with_audit()` for structured ORDER_AUDIT logging
- `runner/recovery.py` — Crash recovery: 8-component atomic bundle
- `runner/config.py` — LiveRunnerConfig (93 fields); factory methods: `.lite()`, `.paper()`, `.testnet_full()`, `.prod()`
- `features/feature_catalog.py` — PRODUCTION_FEATURES frozenset (122 features); `validate_model_features()`
- `scripts/ops/run_bybit_alpha.py` — **Primary alpha runner**: dual 1h+15m AGREE ONLY combo, ATR adaptive stop, Kelly 1.4x leverage, PortfolioCombiner, WS + realtime stop
- `scripts/ops/compare_live_backtest.py` — Live vs backtest signal/PnL comparison tool
- `scripts/data/download_15m_klines.py` — Incremental 15m kline data download from Binance
- `scripts/shared/signal_postprocess.py` — Signal pipeline: z-score → discretize → min-hold → trend-hold (Rust + Python parity)
- `execution/sim/realistic_backtest.py` — Realistic backtest: intra-bar stop, margin model, Almgren-Chriss slippage, adaptive ATR stop
- `execution/sim/limit_order_book.py` — Simulated LOB: FIFO queue, partial fills, stop orders, TTL expiry
- `execution/adapters/ib/adapter.py` — IB multi-asset adapter via IB Gateway
- `polymarket/collector.py` — 5m BTC Up/Down real-time CLOB orderbook collector + BS fair value

## Live Integration Subsystems

- **Dual Alpha COMBO**: `run_bybit_alpha.py` runs 1h+15m alphas simultaneously via separate WS clients (kline.60 + kline.15). `PortfolioCombiner` AGREE ONLY mode: trades only when both alphas agree direction (Sharpe 5.48 vs weighted 3.18). Individual runners set to `dry_run=True`; combiner manages single exchange position. Per-symbol position cap: 30% of equity × leverage. Conviction scaling: both agree=100%, one only=50%.
- **Adaptive Stop-Loss**: ATR-based 3-phase stop in both `run_bybit_alpha.py` (realtime tick-level) and `realistic_backtest.py` (`--adaptive-stop`). Phase 1: initial ATR×2.0. Phase 2: breakeven after 1×ATR profit. Phase 3: trailing at peak - ATR×0.3 after 0.8×ATR profit. Hard limits: 0.3%-5%. Grid-search optimized: 18/20 folds, 75% trailing stop wins.
- **Alpha Health Monitor**: `AlphaHealthMonitor` in `monitoring/alpha_health.py` — tracks per-symbol IC, gates orders via `position_scale()` (0.0/0.5/1.0).
- **WS-API Orders**: `WsOrderAdapter` in `execution/adapters/binance/ws_order_adapter.py` — ~4ms WS-API with REST fallback.
- **Auto-Retrain**: `scripts/auto_retrain.py` — supports `--include-15m`, `--only-15m` for 15m models. Downloads fresh data → trains LightGBM+XGBoost → validates → saves if passes. Systemd timer: Sunday 2am UTC. SIGHUP to runner for hot reload.
- **Live Comparison**: `scripts/ops/compare_live_backtest.py` — parses live log, replays z-scores through signal logic, reports agreement rate + PnL divergence.

## Venue Adapters

All adapters implement `VenueAdapter` protocol (`execution/adapters/base.py`); registered via `AdapterRegistry.register(venue, adapter)`.

| Venue | Protocol | Min Order (ETH) | Notes |
|-------|----------|-----------------|-------|
| Binance | REST + WS-API (~4ms) | $20 | Primary. USDT-M futures. 47 files, ~3.7K LOC. Live API connected. |
| Bybit | REST V5 + WS | $5 | Demo trading active. HMAC-SHA256. Systemd service running. |
| Bitget | REST V2 + WS | $5 | USDT-M + stock tokens. HMAC-SHA256+base64, passphrase header. |
| IB | `ib_insync` → IB Gateway | N/A | Port 4002=paper, 4001=live. All asset classes. Requires Xvfb. |
| Polymarket | CLOB REST | N/A | L2 auth. Collector V2: direct CLOB orderbook, SQLite storage. |
| Generic | CCXT | varies | 100+ exchanges, unified interface. |

## Environment

```bash
# Bybit Demo (currently active)
# Keys hardcoded in scripts/ops/run_bybit_alpha.py create_adapter()

# Binance Live (API connected, account empty — needs USDT deposit)
# API key stored, connection verified 2026-03-15

# Binance Testnet (for paper trading)
export BINANCE_TESTNET_API_KEY=...    # from testnet.binancefuture.com
export BINANCE_TESTNET_API_SECRET=...
```

**Current deployment** (systemd):
```bash
# Service: bybit-alpha.service
# Command: python3 -m scripts.run_bybit_alpha --symbols ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws
# Status: active (running), Bybit Demo, AGREE ONLY mode (ETH), independent (SUI/AXS)
# Logs: /quant_system/logs/bybit_alpha.log
```

**Quick start → demo trading**:
1. Service auto-starts: `sudo systemctl status bybit-alpha.service`
2. Check signals: `tail -f logs/bybit_alpha.log`
3. Verify model: `python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo --realistic`
4. Compare live/backtest: `python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log`

**Walk-forward baselines** (2026-03-15):
- ETHUSDT 1h (min_hold=18, 20 folds): Sharpe **1.52**, **+397%**, **14/20 positive** → **PASS**
- ETHUSDT 15m h=32 (4 folds): Sharpe **1.04**, **+121%**, **3/4 positive** → **PASS**
- ETHUSDT adaptive stop (15/20): Sharpe **1.35**, **+286%** → **PASS**
- SUIUSDT 1h (7 folds): Sharpe **1.63**, **+150%**, **6/7 positive** → **PASS**
- AXSUSDT 1h (17 folds): Sharpe **1.25**, **+241%**, **13/17 positive** → **PASS**
- BTCUSDT 1h: Sharpe -0.21, 10/20 → FAIL. SOLUSDT 15m: 1/4 → FAIL. (Not traded)
- Kelly optimal leverage: **1.4x** (half-Kelly 0.7x; 3x+ → >50% bust rate, geometric mean turns negative)
- Dual alpha AGREE backtest: Sharpe **5.48**, +1,141%, 56% WR, signal correlation 0.077

## Signal Pipeline

```
Raw prediction (Ridge 60% + LightGBM 40% ensemble, walk-forward validated)
  → Rolling z-score (window=720, warmup=180)
  → Long-only clip (optional)
  → Discretize: z > deadzone → +1, z < -deadzone → -1, else 0
  → Min-hold enforce (18 bars for 1h, 16 for 15m)
  → Trend-hold extend (optional: extend when trend intact, symmetric long/short)
  → Monthly gate (optional: close <= SMA → flat)
  → Vol-adaptive scaling (optional: signal × target_vol/realized_vol)
```
Constraint pipeline implemented identically in Rust (`constraint_pipeline.rs`) and Python (`signal_postprocess.py`). Parity verified via `tests/integration/test_constraint_parity.py`.

## Gotchas

- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `RustFeatureEngine` uses its own window sizes (not LiveFeatureComputer params)
- Tests require `_quant_hotpath` built; `pytest.importorskip("_quant_hotpath")` guards Rust tests
- Production models in `models_v8/` (LightGBM + XGBoost pkl files)
- CrossAssetComputer: must push benchmark (BTCUSDT) **before** altcoins each bar
- Fd8 conversion: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- `features/dynamic_selector.py` keeps `_rankdata`/`_spearman_ic` for scripts (not fallback)
- `pip install` requires `--break-system-packages` flag (no venv, system Python 3.12)
- Live hot-path has no Python fallbacks (rolling.py, multi_timeframe.py require Rust)
- Binary build requires `-lpython3.12` link flag (PyO3 symbols)
- Binary config priority: model `config.json` > YAML `per_symbol` > YAML `strategy` defaults
- Binance min notional: ETHUSDT $20, BTCUSDT $100; Bybit/Bitget all $5
- `RustFeatureEngine.checkpoint()` / `restore_checkpoint()` persist rolling windows as bar history JSON
- State ownership: `RustStateStore` = position truth; `OrderStateMachine` = execution audit trail only
- Recovery bundle: 8 components; `save_all_auxiliary_state()` / `restore_all_auxiliary_state()`
- `GateChain._apply_scale()` preserves Decimal type; uses `dataclasses.replace()` on frozen OrderEvent
- `LiveEmitHandler` attribution tracks only accepted orders; fill supports PARTIALLY_FILLED
- Python fallback in `signal_postprocess.py` uses `_enforce_hold_single_pass()` matching Rust exactly (parity verified)
- IB Gateway requires Xvfb on headless Linux (`Xvfb :99 &; export DISPLAY=:99`)
- Polymarket Gamma API prices are **cached/stale** — always use CLOB orderbook for real prices
- Walk-forward `--realistic` uses `close ± 0.5%` as high/low unless real CSV data available; `--adaptive-stop` auto-injects real high/low
- Adaptive stop ATR params: `mult=2.0, trail_trigger=0.8, trail_step=0.3, breakeven=1.0` (grid-search optimized)
- `PortfolioCombiner` sets individual runners to `dry_run=True` — combiner manages single exchange position
- Bitget API: success code `"00000"` (string); signing = base64(HMAC-SHA256) + ACCESS-PASSPHRASE header
- Feature selection: greedy IC is optimal (stability-filtered and fixed-feature approaches both hurt performance)
- Funding/vol-squeeze/OI-divergence as standalone alphas all FAIL after costs — alpha is in ML multi-factor combination only
- `ETHUSDT_15m` in SYMBOL_CONFIG uses `"symbol": "ETHUSDT"` (real exchange symbol) + `"interval": "15"` (separate WS)
- `EnrichedFeatureComputer.on_bar(btc_close=...)` — V12 cross-asset features need BTC price; without it, 6 `btc_*` features return None (safe degradation)
- SYMBOL_CONFIG: `SUIUSDT` size=0.1, `AXSUSDT` size=1.0 (lot sizes match Bybit minimums)
- Ridge model uses its own feature list (`ridge_features` in horizon_models) which may differ from LGBM features
- `PortfolioCombiner` caps each symbol at 30% of equity × leverage to prevent single-symbol overexposure
- Ridge won 20-fold walk-forward (15/20 PASS, Sharpe 0.54) vs LGBM+XGB ensemble (11/20 FAIL)
