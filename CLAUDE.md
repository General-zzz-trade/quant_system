## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
make test                    # ALL gates (py + exec + rust + lint, matches CI)
pytest tests/unit/ -x -q     # Unit tests only (~18s)
pytest tests/ -x -q -m ""   # ALL tests including slow (~35s)
pytest execution/tests/ -x -q  # Execution subsystem tests (67 tests)
pytest tests/unit/runner/ -x -q    # Runner tests (303 tests)
pytest tests/unit/runner_v2/ -x -q  # Decomposed runner tests (47 tests)
pytest tests/unit/bybit/ -x -q   # Bybit adapter tests (60 tests)
pytest tests/unit/state/ -x -q   # State module tests (87 tests)
pytest tests/unit/event/ -x -q   # Event module tests (145 tests)
pytest tests/unit/strategies/ -x -q  # Strategy registry tests (10 tests)
pytest -m slow               # Slow tests only (parity, NN, XGB)
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test --no-default-features  # Rust unit tests (198 tests)
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
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws  # Full portfolio (LiveRunner, default)
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT --ws --dry-run         # Signal only, no orders
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT --once --dry-run       # Single bar then exit
python3 -m scripts.run_bybit_alpha --symbols ETHUSDT --ws --legacy          # Use AlphaRunner (deprecated fallback)
sudo systemctl restart bybit-alpha.service                                   # Restart service
sudo systemctl status bybit-alpha.service                                    # Check service
tail -f /quant_system/logs/bybit_alpha.log                                   # Follow live logs
```

**Data & model management**:
```bash
python3 -m scripts.data.download_15m_klines                                  # Update 15m kline data (incremental)
python3 -m scripts.data.download_5m_klines --symbols ETHUSDT                 # Download 5m kline data
python3 -m scripts.data.download_funding_rates --symbols ETHUSDT SOLUSDT     # Update funding rate history
python3 -m scripts.data.download_oi_data --symbols ETHUSDT BTCUSDT           # Download OI/LS/Taker data (Binance, ~28d max)
python3 -m scripts.data.download_multi_exchange_funding --symbols ETHUSDT    # Multi-exchange funding rates (Binance+Bybit)
python3 -m scripts.auto_retrain --include-15m --force                        # Retrain 1h + 15m models
python3 -m scripts.auto_retrain --only-15m --force                           # Retrain 15m models only
python3 -m scripts.auto_retrain --dry-run                                    # Preview retrain without saving
python3 -m scripts.training.train_all_production --dry-run                   # Validate all production models
python3 -m scripts.training.train_all_production --force                     # Force retrain all production models
```

**Polymarket collector**:
```bash
python3 -m polymarket.collector --mode intra --db data/polymarket/collector.db  # Start 5m+15m CLOB collector
kill $(cat data/polymarket/collector.pid)                                       # Stop collector
```

**Monitoring & diagnostics**:
```bash
python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log  # Live vs backtest comparison
python3 -m scripts.testnet_smoke --public-only                                # Exchange connectivity check
python3 -m scripts.ops.security_scan                                          # Security audit (secrets, notional, bare-except)
python3 -m scripts.ops.ops_dashboard                                          # Unified ops status dashboard
python3 -m scripts.ops.pre_live_checklist                                     # Automated pre-live readiness check
python3 -m scripts.ops.shadow_mode_check                                      # Shadow trading health report
python3 -m scripts.ops.demo_tracker                                           # Update track record from logs
python3 -m scripts.ops.weekly_report                                          # Generate weekly performance report
```

**CRITICAL after Rust build**: copy .so then verify:
```bash
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)), 'exports')"  # verify import works
```

**Binary build** (standalone Rust trader, requires Python linkage):
```bash
cd ext/rust && RUSTFLAGS="-C link-arg=-L/usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" cargo build --release --bin quant_trader
./target/release/quant_trader --config config.testnet.yaml [--dry-run]
```

## Architecture

```
core/            Bootstrap, config, bus, clock, effects, observability
engine/          Pipeline + coordinator (event -> state transitions)
features/        Feature computation (EnrichedFeatureComputer, 127 features incl. ADX + V12 cross-asset + V13 OI + V14 dominance)
  dynamic_selector.py  Feature selection: greedy_ic, stable_icir, stability_filtered_greedy
  feature_catalog.py   PRODUCTION_FEATURES frozenset (153 = 127 enriched + 17 cross-asset + 4 dominance + 3 funding spread + 2 onchain)
  dominance_computer.py  V14 BTC/ETH ratio features (Python + Rust dual path)
  funding_spread.py    Multi-exchange funding rate spread (3 features)
  onchain_flow.py      Exchange inflow z-score + MA ratio (2 features)
decision/        Trading signals, ensemble, regime detection (CompositeRegimeDetector + ParamRouter wired), rebalancing
alpha/           ML models + inference bridge (horizon_ensemble.py, adaptive_config.py)
execution/       Order routing, state machine, dedup
  adapters/binance/    Binance USDT-M futures (43 files, ~3.7K LOC, WS-API ~4ms)
  adapters/bybit/      Bybit V5 linear perpetuals (demo/testnet/live, 6 files)
  adapters/polymarket/ Polymarket prediction market CLOB adapter
  sim/                 Backtest engines (realistic_backtest.py, limit_order_book.py, cost_constants.py)
state/           State types + Rust adapters
attribution/     P&L + cost + signal attribution (thin Rust wrappers)
event/           Event types + runtime protocol
ext/rust/        Unified Rust crate -> _quant_hotpath (77 .rs files, ~30K LOC)
ext/rust/src/bin/ Standalone trading binary (main.rs + config.rs, ~2.6K LOC)
runner/          Live/paper/backtest (gate_chain.py, emit_handler.py, recovery.py, config.py)
  builders/            13 phase builders (live) + 5 legacy builders (backcompat); see __init__.py
    rust_components_builder.py  Phase 1.5: 9 Rust component initialization
    combo_builder.py            Dual-alpha AGREE mode (combine_signals)
  gates/               Custom order gates (Phase 2 convergence)
    adaptive_stop_gate.py       ATR 3-phase stop-loss gate
    equity_leverage_gate.py     Kelly-bracket leverage + z-score scaling
    consensus_scaling_gate.py   Cross-symbol consensus sizing
  live_runner.py       Primary production entry point (Python, converged from AlphaRunner)
regime/          Regime detection (volatility, trend, composite, param_router — all wired)
risk/            Risk limits + kill switch + StagedRiskManager (equity-based staging)
strategies/      Strategy registry + protocol (alpha_momentum, polymarket_rsi)
portfolio/       Allocator, rebalance, optimizer
monitoring/      Alerts, health checks, metrics, Prometheus, Grafana
infra/           Logging (structured JSON), networking, systemd units
polymarket/      Polymarket 5m BTC Up/Down — collector, features, signals, runner, maker (A-S)
models_v8/       Production models (Ridge 60% + LightGBM 40%): ETHUSDT_gate_v2, ETHUSDT_15m, SUIUSDT, AXSUSDT, BTCUSDT_gate_v2
research/        Alpha research, factor backtests, hyperopt, Monte Carlo
scripts/         7 subdirs + symlinks for compat
  ops/               Split into 13 modules (see below)
    config.py            SYMBOL_CONFIG, constants, MAX_ORDER_NOTIONAL
    data_fetcher.py      Binance OI/LS/taker data fetch
    model_loader.py      load_model(), create_adapter()
    alpha_runner.py      AlphaRunner class (signal + trade, 12 Rust components)
    portfolio_manager.py PortfolioManager class
    portfolio_combiner.py PortfolioCombiner (AGREE ONLY)
    hedge_runner.py      BTC+ALT hedge
    pnl_tracker.py       Unified PnL tracking
    run_bybit_alpha.py   main() entry + WS loop
    shadow_mode_check.py Shadow trading log analysis + health report
    ops_dashboard.py     Unified status dashboard (service, models, signals, data)
    pre_live_checklist.py Automated pre-live readiness checks
  data/              download_15m_klines.py, download_5m_klines.py, download_funding_rates.py, download_oi_data.py
  research/          backtest_funding_alpha.py, backtest_vol_squeeze.py, polymarket_binary_alpha.py
  walkforward/       walkforward_validate.py
  training/          train_v7_alpha.py, train_15m.py, train_all_production.py
data_files/      CSV data: {SYMBOL}_{1h,15m,5m}.csv, {SYMBOL}_funding.csv, {SYMBOL}_oi_1h.csv
logs/            bybit_alpha.log, retrain_cron.log
tests/           unit/ (runner, bybit, decision, features, state, event, monitoring, strategies, polymarket, scripts), integration/ (crash_recovery, fault_injection, constraint_parity, ...)
```

**Data flow (live alpha via AlphaRunner)**:
```
Bybit WS kline → RustFeatureEngine.push_bar(+OI/LS from Binance API)
  → _check_regime(close) — vol/trend/ranging filter + dynamic deadzone
  → Ridge(60%)+LightGBM(40%) ensemble predict
  → _check_regime(close, feat_dict) — CompositeRegime updates deadzone/min_hold (BTC only)
  → RustInferenceBridge.apply_constraints (z-score + deadzone + min-hold)
  → RustRiskEvaluator.check_drawdown + RustKillSwitch
  → RustOrderStateMachine.register + RustCircuitBreaker.allow_request
  → Bybit REST market order (with orderLinkId dedup)
  → RustStateStore.process_event (position truth)
```

**Python engine path**: Market event → FeatureComputeHook → Pipeline (RustStateStore) → DecisionModule → OrderRouter
**Binary path**: Binance WS → RustTickProcessor.process_tick_native() → risk gates → WS-API order (~4ms)

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 77 .rs modules, ~30K LOC
- Exports: ~38 PyO3 classes + ~100 functions (195 total, see `lib.rs`)
- Binary: `quant_trader` standalone trading binary (no Python runtime)
- Naming: `cpp_*` = C++ migration functions, `rust_*` = new kernel modules
- State types use i64 fixed-point (Fd8, x10^8); `_SCALE = 100_000_000`
- feature_hook.py always uses Rust (no Python fallback)
- `RustStateStore` keeps state on Rust heap, Python gets snapshots via `get_*()`
- `pnl_tracker.rs` — RustPnLTracker (record_close, win_rate, drawdown; Python fallback in scripts/ops/pnl_tracker.py)
- `drawdown_breaker.rs` — RustDrawdownBreaker (4-state machine, velocity detection; return-action pattern, Python bridges to KillSwitch)
- `regime_detector.rs` — RustCompositeRegimeDetector + RustRegimeParamRouter (vol percentile + ADX trend + param routing; Python fallback in regime/composite.py + param_router.py)
- `adaptive_stop.rs` — RustAdaptiveStopGate (3-phase ATR stop, per-symbol state; Python fallback in runner/gates/adaptive_stop_gate.py)
- `risk_rules.rs` + `risk_aggregator.rs` — RustRiskAggregator (7 rules: MaxPosition, LeverageCap, MaxDrawdown, PortfolioLimits, OrderFrequency, CorrelationLimit, VaR; Mutex-protected stats; NaN→reject)
- `saga_manager.rs` — RustSagaManager (order lifecycle state machine, match-exhaustive transitions, TTL auto-expire)
- `event_validation.rs` — RustEventValidator (bounded LRU dedup, monotonic time, type-specific validation)
- `correlation.rs` — RustCorrelationComputer (rolling Pearson pairwise correlation, log return tracking)
- `gate_chain.rs` — RustGateChain (9 gate types in single FFI call: equity_leverage, consensus, drawdown, correlation, alpha_health, regime, staged_risk, notional, min_qty)
- `incremental_trackers.rs` — Standalone EMA/RSI/ATR/ADX trackers (reusable outside feature engine)

Key exports (see `lib.rs`): State (RustStateStore, RustMarketState, RustPositionState, RustAccountState), Features (RustFeatureEngine, RustCrossAssetComputer), Risk (RustRiskEvaluator, RustKillSwitch, RustCircuitBreaker, RustDrawdownBreaker, RustRiskAggregator), Pipeline (rust_pipeline_apply, RustUnifiedPredictor), Networking (RustWsClient, RustWsOrderGateway), Inference (RustInferenceBridge), Selection (cpp_greedy_ic_select_np, cpp_stable_icir_select), Analytics (RustPnLTracker), Regime (RustCompositeRegimeDetector, RustRegimeParamRouter, RustRegimeResult, RustRegimeParams), Gates (RustAdaptiveStopGate, RustGateChain), Saga (RustSagaManager), Validation (RustEventValidator), Correlation (RustCorrelationComputer), Allocation (RustPortfolioAllocator).

## Rust Pipeline (12/12 components in production)

AlphaRunner uses all 12 Rust components: RustFeatureEngine (120 features), RustInferenceBridge (z-score+deadzone+min-hold+max-hold), RustRiskEvaluator (drawdown+leverage), RustKillSwitch (global emergency stop), RustOrderStateMachine (order lifecycle), RustCircuitBreaker (3-failure/120s backoff), RustStateStore (position truth), RustFillEvent+RustMarketEvent (zero-copy), rust_pipeline_apply (atomic reducer). RustUnifiedPredictor, RustTickProcessor, RustWsClient imported but not active (see alpha_runner.py).

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 127 enriched feature definitions (V1-V14 + ADX)
- `features/feature_catalog.py` — PRODUCTION_FEATURES frozenset (153 features); `validate_model_features()`
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `ext/rust/src/constraint_pipeline.rs` — Signal constraints (batch + incremental)
- `runner/live_runner.py` — Production entry point (Python)
- `runner/gate_chain.py` — GateChain: up to 13 gates with `process_with_audit()` (incl. StagedRiskGate, AdaptiveStopGate, EquityLeverageGate, ConsensusScalingGate)
- `runner/config.py` — LiveRunnerConfig (~85 fields); factory: `.lite()`, `.paper()`, `.prod()`
- `scripts/ops/config.py` — SYMBOL_CONFIG, constants, MAX_ORDER_NOTIONAL
- `scripts/ops/alpha_runner.py` — AlphaRunner: legacy signal + trade (deprecated, use `--legacy`)
- `scripts/ops/portfolio_combiner.py` — PortfolioCombiner (AGREE ONLY mode, enforces MAX_ORDER_NOTIONAL)
- `scripts/ops/run_bybit_alpha.py` — **Primary entry point**: defaults to LiveRunner, `--legacy` for AlphaRunner
- `scripts/shared/signal_postprocess.py` — Signal pipeline (Rust + Python parity)
- `execution/sim/realistic_backtest.py` — Realistic backtest: intra-bar stop, Almgren-Chriss slippage, adaptive ATR stop
- `polymarket/collector.py` — 5m+15m CLOB collector + BS fair value + RSI signal
- `polymarket/strategies/maker_5m.py` — Avellaneda-Stoikov market maker for binary outcomes
- `polymarket/strategies/inventory_manager.py` — Inventory tracking + expiry actions
- `strategies/registry.py` — StrategyRegistry: register/discover/instantiate strategies
- `strategies/base.py` — StrategyProtocol + Signal dataclass
- `decision/regime_bridge.py` — RegimeAwareDecisionModule (CompositeRegimeDetector + ParamRouter)
- `risk/staged_risk.py` — StagedRiskManager: 5-stage equity-based risk ladder
- `docs/wiring_truth.md` — Module integration status table (what's wired, what's not)

## Live Integration Subsystems

- **Dual Alpha COMBO**: 1h+15m alphas via separate WS (kline.60 + kline.15). `PortfolioCombiner` AGREE ONLY: both must agree direction (Sharpe 5.48). Per-symbol cap: 30% equity x leverage. Conviction: both=100%, one=50%.
- **Adaptive Stop-Loss**: ATR 3-phase: initial ATRx2.0 → breakeven after 1xATR profit → trailing at peak-ATRx0.3. Hard limits: 0.3%-5%.
- **Alpha Health Monitor**: `monitoring/alpha_health.py` — per-symbol IC tracking, gates via `position_scale()` (0.0/0.5/1.0).
- **Auto-Retrain**: `scripts/auto_retrain.py` — `--include-15m`, `--only-15m`. Systemd timer: Sunday 2am UTC. SIGHUP for hot reload.

## Venue Adapters

All adapters implement `VenueAdapter` protocol (`execution/adapters/base.py`); registered via `AdapterRegistry.register(venue, adapter)`.

| Venue | Protocol | Min Order | Notes |
|-------|----------|-----------|-------|
| Binance | REST + WS-API (~4ms) | $20 (ETH) | Primary. USDT-M futures. 43 files, ~3.7K LOC. Live API connected. |
| Bybit | REST V5 + WS | $5 | Demo trading active. HMAC-SHA256. Systemd service running. |
| Polymarket | CLOB REST | N/A | L2 auth. Collector V2: direct CLOB orderbook, SQLite storage. |

## Environment

```bash
# Required (set in .env or export):
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export BYBIT_BASE_URL=https://api-demo.bybit.com  # or https://api.bybit.com for live
# See .env.example for all optional vars (Binance, Polymarket)
```

**Current deployment** (systemd):
```bash
# Service: bybit-alpha.service
# Command: python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws
# Status: active (running), Bybit Demo, 5 symbols (BTC h=96, ETH 1h+15m AGREE, SUI/AXS independent)
# Logs: /quant_system/logs/bybit_alpha.log
# Deploy truth: docs/deploy_truth.md (systemd/compose/CI consistency)
```

**Quick start → demo trading**:
1. Service auto-starts: `sudo systemctl status bybit-alpha.service`
2. Check signals: `tail -f logs/bybit_alpha.log`
3. Verify model: `python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo --realistic`
4. Compare live/backtest: `python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log`

**Walk-forward baselines** (2026-03-15):
- ETHUSDT 1h (min_hold=18, 20 folds): Sharpe **1.52**, **+389%**, **14/20 positive** → **PASS**
- ETHUSDT 15m h=32 (4 folds): Sharpe **1.04**, **+121%**, **3/4 positive** → **PASS**
- ETHUSDT adaptive stop (15/20): Sharpe **1.35**, **+286%** → **PASS**
- SUIUSDT 1h (7 folds): Sharpe **1.63**, **+150%**, **6/7 positive** → **PASS**
- AXSUSDT 1h (17 folds): Sharpe **1.25**, **+241%**, **13/17 positive** → **PASS**
- **BTCUSDT V14** (h=96, dominance): Sharpe **2.03**, **+552%**, **16/20 positive** → **PASS** (BTC/ETH ratio is #1 feature)
- **BTCUSDT Regime-Adaptive**: Sharpe **5.52**, **20/20 positive** → **+120% over baseline** (composite regime → param routing)
- FAIL: SUIUSDT 15m (9/23), SOLUSDT 15m (1/4), AXSUSDT 15m (Sharpe 0.39), ETHUSDT 5m (IC near zero)
- Kelly optimal leverage: **1.4x** (half-Kelly 0.7x; 3x+ → >50% bust rate). Demo uses **10x** (all tiers)
- Dual alpha AGREE backtest: Sharpe **5.48**, +1,141%, 56% WR, signal correlation 0.077
- Polymarket RSI(5) 5m: accuracy **55.0%**, **23/23 folds PASS**, $25/day@$10/bet

## Signal Pipeline

```
Raw prediction (Ridge 60% + LightGBM 40% ensemble, walk-forward validated)
  → Rolling z-score (window=720, warmup=180)
  → Long-only clip (optional)
  → Discretize: z > deadzone → +1, z < -deadzone → -1, else 0
  → Min-hold enforce (18 bars for 1h, 16 for 15m)
  → Trend-hold extend (optional: extend when trend intact, symmetric long/short)
  → Monthly gate (optional: close <= SMA → flat)
  → Vol-adaptive scaling (optional: signal x target_vol/realized_vol)
```
Constraint pipeline implemented identically in Rust (`constraint_pipeline.rs`) and Python (`signal_postprocess.py`). Parity verified via `tests/integration/test_constraint_parity.py`.

## Gotchas

**Build & environment**:
- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `pip install` requires `--break-system-packages` (no venv, system Python 3.12)
- Binary build requires `-lpython3.12` link flag (PyO3 symbols)
- Live hot-path has no Python fallbacks (rolling.py, multi_timeframe.py require Rust)

**Rust/Python interface**:
- Fd8 conversion: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- `RustFeatureEngine` uses its own window sizes; `checkpoint()`/`restore_checkpoint()` persist as bar history JSON
- State ownership: `RustStateStore` = position truth; `OrderStateMachine` = execution audit trail only
- On restart, `_reconcile_position()` syncs StateStore with exchange positions via `_record_fill()`
- Feature hook source exceptions isolated via `_safe_call_source()` — NaN on failure, bar continues
- `SagaManager`: Python uses `RLock`, Rust (`RustSagaManager`) uses match-exhaustive state machine with mandatory TTL
- `RustRiskAggregator` replaces Python aggregator's Lock with Rust Mutex — stats never lost under concurrency
- `RustEventValidator` uses bounded LRU (default 100K) instead of unbounded HashSet — prevents dedup memory leak
- `RustGateChain` processes all gates in single FFI call — no per-gate Python↔Rust switching
- Binary config priority: model `config.json` > YAML `per_symbol` > YAML `strategy` defaults

**Trading & safety**:
- `MAX_ORDER_NOTIONAL = $5,000` hard limit in config.py — enforced in both AlphaRunner (clamp, not block) and PortfolioCombiner
- `_round_to_step()` applied in ALL code paths (adaptive sizing, base size, exception fallback) — prevents Bybit `Qty invalid` rejections
- Margin pre-flight check: AlphaRunner and PortfolioCombiner check `available` balance before sending orders to avoid `ab not enough` errors
- Binance min notional: ETHUSDT $20, BTCUSDT $100; Bybit all $5
- `PortfolioCombiner` sets individual runners to `dry_run=True`; caps each symbol at 30% of equity x leverage
- SYMBOL_CONFIG: `SUIUSDT` size=10, step=10 (Bybit qtyStep=10); `AXSUSDT` size=5.0, step=0.1
- `ETHUSDT_15m` in SYMBOL_CONFIG uses `"symbol": "ETHUSDT"` + `"interval": "15"` (separate WS)
- `_safe_val()` handles NaN/None→0.0 for model input; Rust engine returns NaN for unfed features
- `BinanceOICache`: OI data fetched in background thread (55s refresh), no longer blocks stop-loss
- `_NEUTRAL_DEFAULTS`: NaN features use neutral values (ls_ratio→1.0, rsi_14→50.0), not 0.0
- `order_utils.py`: `reliable_close_position()` replaces bare `close_position()` calls; `clamp_notional()` enforces $5,000 limit at all order sites
- `PortfolioCombiner` uses `PnLTracker` (no duplicate PnL tracking); records fills to `RustStateStore` on both open and close
- `PortfolioManager.record_position()` syncs COMBO fill positions without re-executing orders
- `RustStateStore` initialized with real exchange balance (Fd8); equity=0 bug fixed
- `HedgeRunner` uses 2% hysteresis band (ratio < MA×0.98 open, > MA×1.02 close) to prevent noise trading

**Features & models**:
- ADX(14): computed incrementally in `enriched_computer.py` via `_ADXTracker`; needs 2×14=28 bars warmup; used by `TrendRegimeDetector`
- CrossAssetComputer: must push benchmark (BTCUSDT) **before** altcoins each bar; call `begin_bar()` to reset per-bar tracking; warns if order violated
- `EnrichedFeatureComputer.on_bar(btc_close=...)` — V12 needs BTC price; `on_bar(eth_close=...)` — V14 needs ETH price; missing → None (safe)
- V13 OI features (5): IC validated, 28 days data only. V14 dominance (4): `btc_dom_*`, dual path (Python `dominance_computer.py` + Rust `push_dominance()`)
- Ridge model uses its own feature list (`ridge_features`) which may differ from LGBM features
- Feature selection: greedy IC is optimal (stability-filtered and fixed-feature approaches both hurt)
- BTC model h=96 uses `min_hold=48` (2 days), `max_hold=288` (12 days) — much slower than ETH's h=24/min_hold=18
- `batch_feature_engine.py` `_add_dominance_features()` requires ETHUSDT_1h.csv

**Alpha research conclusions**:
- Only 1h (all symbols) and 15m (ETH only) work; 5m/SUI-15m/AXS-15m/SOL-15m all FAIL
- Funding/vol-squeeze/OI-divergence/altcoin-rotation as standalone alphas all FAIL after costs
- Polymarket ML classifier loses to simple RSI rule — RSI's selectivity is the alpha

**Production path (converged)**:
- `LiveRunner` (`runner/live_runner.py`) is the **primary production path** — `run_bybit_alpha.py` defaults to it
- `AlphaRunner` (`scripts/ops/alpha_runner.py`) is **deprecated** — use `--legacy` flag to fall back; retained 3 months
- LiveRunner has all AlphaRunner capabilities: 12 Rust components, regime detection, adaptive stop, consensus scaling, combo AGREE mode
- `composite_regime_symbols` config controls per-symbol CompositeRegime (BTC only by default); ETH/SUI/AXS use fixed params
- Polymarket `AvellanedaStoikovMaker` is standalone; needs manual startup, not wired to runner

**Regime & risk wiring** (2026-03-17):
- `RegimeAwareDecisionModule` defaults to `CompositeRegimeDetector` (was Vol+Trend separately)
- `RegimeParamRouter` activated when `enable_regime_sizing=True` via `enable_param_routing` flag
- `RegimePolicy` blocks composite crisis labels (any label with "crisis" in value)
- `StagedRiskManager` wired as `StagedRiskGate` in gate_chain; equity stages: survival→growth→stable→safe→institutional
- `initial_equity` field added to `LiveRunnerConfig` (default $500); `RustStateStore` uses real exchange balance via Fd8 at startup

**Infrastructure**:
- `docs/deploy_truth.md` is deployment truth; `infra/systemd/` must sync with `/etc/systemd/system/`
- Burn-in gate: Phase A(Paper 3d) → B(Shadow 3d) → C(Testnet 3d) = 9 days before live; configs in `config/burnin.{paper,shadow,testnet}.yaml`
- Live config: `config/production.live.yaml` (ETHUSDT, Kelly 0.7x real / 10x demo, $5K order cap)
- Alert rules: `config/alerts.live.yaml` (balance $400 kill, $50 single-loss warn, 2h stale, IC decay)
- Dockerfile: multi-stage (`ci`/`paper`/`live`); docker-compose.yml with paper + `--profile live` profiles
- `scripts/ops/security_scan.py`: checks hardcoded secrets, .env gitignored, MAX_ORDER_NOTIONAL, bare-except blocks
- `regime/param_router.py`: only BTC uses regime-adaptive params; ETH keeps fixed (fixed outperforms adaptive for ETH)
- Binance OI history API only retains ~28 days; `download_oi_data.py` cron every 6h to accumulate
- Polymarket Gamma API prices are **cached/stale** — always use CLOB orderbook for real prices
- Polymarket collector: background process (not systemd); PID in `data/polymarket/collector.pid`
- Walk-forward `--realistic` uses `close +/- 0.5%` as high/low unless real CSV; `--adaptive-stop` auto-injects real high/low
