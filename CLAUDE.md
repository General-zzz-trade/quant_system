## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
make test                    # Core local gate; CI still adds security/model-check/compose smoke/framework integration
pytest tests/unit/ -x -q     # Unit tests only (~18s)
pytest tests/ -x -q -m ""   # ALL tests including slow (~35s)
pytest execution/tests/ -x -q  # Execution subsystem tests (71 tests)
pytest tests/unit/runner/ -x -q    # Runner tests (543 tests)
pytest tests/unit/runner_v2/ -x -q  # Decomposed runner tests (47 tests)
pytest tests/unit/bybit/ -x -q   # Bybit adapter tests (60 tests)
pytest tests/unit/state/ -x -q   # State module tests (87 tests)
pytest tests/unit/event/ -x -q   # Event module tests (145 tests)
pytest tests/unit/strategies/ -x -q  # Strategy registry tests (10 tests)
pytest -m slow               # Slow tests only (parity, NN, XGB)
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test  # Rust unit tests (default feature set for tests)
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

**Active trading services**:
```bash
# Strategy H: 4h primary + 1h scaler (4 runners, 2 WS — 15m disabled after WF FAIL):
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
sudo systemctl restart bybit-alpha.service

# Model hot-reload (no restart, <200ms):
sudo kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)

# Polymarket dry-run (currently active):
python3 -m scripts.run_polymarket_dryrun --bet-size 10 --rsi-low 30 --rsi-high 70
```

**Data & model management**:
```bash
python3 -m scripts.data.download_15m_klines                                  # Update 15m kline data (incremental)
python3 -m scripts.data.download_5m_klines --symbols ETHUSDT                 # Download 5m kline data
python3 -m scripts.data.download_funding_rates --symbols ETHUSDT SOLUSDT     # Update funding rate history
python3 -m scripts.data.download_oi_data --symbols ETHUSDT BTCUSDT           # Download OI/LS/Taker data (Binance, ~28d max)
python3 -m scripts.data.download_multi_exchange_funding --symbols ETHUSDT    # Multi-exchange funding rates (Binance+Bybit)
python3 -m scripts.auto_retrain --include-15m --force                        # Retrain 1h + 15m models
python3 -m scripts.auto_retrain --include-4h --force                         # Retrain 1h + 4h models
python3 -m scripts.auto_retrain --only-4h --force                            # Retrain 4h models only
python3 -m scripts.auto_retrain --daily --include-4h --sighup                # Daily lightweight retrain + hot-reload
python3 -m scripts.auto_retrain --dry-run                                    # Preview retrain without saving
python3 -m scripts.training.train_4h_daily --all                             # Train all 4h + daily models
python3 -m scripts.training.train_all_production --force                     # Force retrain all production models
```

**Polymarket collector**:
```bash
python3 -m polymarket.collector --mode intra --db data/polymarket/collector.db  # Start 5m+15m CLOB collector
kill $(cat data/polymarket/collector.pid)                                       # Stop collector
```

**Monitoring & diagnostics**:
```bash
python3 -m scripts.ops.health_watchdog                                        # Health check (services+data+account), auto-restarts stale
python3 -m scripts.ops.health_watchdog --json                                 # JSON output for automation
python3 -m scripts.ops.auto_bug_scan --severity warning                       # Static bug scan (30 patterns)
python3 -m scripts.ops.ops_dashboard                                          # Unified ops status dashboard
python3 -m scripts.ops.demo_tracker                                           # Update track record from logs
python3 -m scripts.ops.weekly_report                                          # Generate weekly performance report
python3 -m scripts.ops.security_scan                                          # Security audit (secrets, notional, bare-except)
python3 -m scripts.ops.pre_live_checklist                                     # Pre-live readiness check
python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log  # Live vs backtest comparison
python3 -m scripts.ops.signal_reconcile --hours 24                            # Live vs backtest signal consistency
python3 -m scripts.ops.signal_reconcile --hours 168 --alert                   # Weekly signal check + Telegram
python3 -m monitoring.ic_decay_monitor                                        # IC decay check (GREEN/YELLOW/RED)
python3 -m monitoring.ic_decay_monitor --alert                                # IC decay + Telegram alert
python3 -m scripts.ops.shadow_compare --model-a models_v8/BTCUSDT_gate_v2 --model-b models_v8/BTCUSDT_4h --symbol BTCUSDT --days 90  # A/B model comparison
python3 -m scripts.research.monte_carlo_risk                                  # Monte Carlo risk simulation (10K runs)
python3 -m monitoring.notify                                                  # Test Telegram notification
python3 -m scripts.ops.live_validation_dashboard                              # 30-day validation status
python3 -m scripts.ops.live_validation_dashboard --markdown                   # Markdown report
python3 -m scripts.ops.slippage_analyzer --hours 168                          # Weekly slippage analysis
```

**Automated operations** (systemd timers + cron):
```bash
sudo systemctl list-timers --all | grep -E "health|retrain|refresh|ic-decay"  # Check all timers
# health-watchdog.timer  — every 5 min (service health + data freshness + Telegram alerts, 5h tolerance for 4h bars)
# data-refresh.timer     — every 6 hours (kline + funding + OI sync)
# auto-retrain.timer     — Sunday 2am UTC (walk-forward retrain 1h models)
# daily-retrain.timer    — daily 2am UTC (lightweight 4h retrain + SIGHUP hot-reload)
# ic-decay-monitor.timer — daily 3am UTC (IC decay detection + Telegram alerts)
# cron: demo_tracker (hourly), weekly_report (Sun 3am), bug_scan (Sun 1am), OI download (6h), Deribit options (30min)
```

**CRITICAL after Rust build**: copy .so then verify:
```bash
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)), 'exports')"  # verify import works
```

Rust Python extension builds must explicitly enable `python`, for example:

```bash
cd ext/rust && maturin build --release --features python
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
features/        Feature computation, catalog, dominance / funding / onchain helpers
decision/        Trading signals, ensemble, regime detection, rebalancing
alpha/           ML models + inference bridge + model loader
execution/       Adapters, state machine, dedup, reconcile, ingress, observability
state/           State types + Rust adapters
attribution/     P&L + cost + signal attribution
event/           Event types + runtime protocol
ext/rust/        Unified Rust crate -> _quant_hotpath + standalone candidate binary
runner/          Framework runtime (LiveRunner), builders, recovery, control plane
monitoring/      Alerts, health checks, metrics, ops views
infra/           Logging, config, systemd units
research/        Registry, artifacts, experiment helpers
scripts/         Active alpha / MM entrypoints, ops tooling, research/data wrappers
    alpha_runner.py      AlphaRunner class (signal + trade, 12 Rust components)
    portfolio_manager.py PortfolioManager class
    portfolio_combiner.py PortfolioCombiner (AGREE ONLY)
    hedge_runner.py      BTC+ALT hedge
    pnl_tracker.py       Unified PnL tracking
    run_bybit_alpha.py   main() entry + WS loop
    shadow_mode_check.py Shadow trading log analysis + health report
    ops_dashboard.py     Unified status dashboard (service, models, signals, data)
    pre_live_checklist.py Automated pre-live readiness checks
  data/              download_15m_klines.py, download_5m_klines.py, download_funding_rates.py, download_oi_data.py, download_cross_market.py, download_stablecoin_supply.py, download_deribit_pcr.py, download_onchain.py
  research/          backtest_small_capital.py, backtest_strategy_h.py, optimize_tf_integration.py, monte_carlo_risk.py
  walkforward/       walkforward_validate.py
  training/          train_v7_alpha.py, train_15m.py, train_4h_daily.py, train_all_production.py
data_files/      CSV data: {SYMBOL}_{1h,15m,5m}.csv, {SYMBOL}_funding.csv, {SYMBOL}_oi_1h.csv, cross_market_daily.csv, etf_volume_daily.csv, stablecoin_daily.csv, fear_greed_index.csv, {SYMBOL}_dvol_1h.csv
models_v8/       Model dirs: {SYMBOL}_gate_v2 (1h), {SYMBOL}_15m, {SYMBOL}_4h
logs/            bybit_alpha.log, retrain_cron.log
tests/           unit/ (runner, bybit, decision, features, state, event, monitoring, strategies, polymarket, scripts), integration/ (crash_recovery, fault_injection, constraint_parity, ...)
```

**Data flow (HFT Signal — inactive, Sharpe -5)**:
```
Bybit WS (depth+trade per symbol) → SymbolEngine.on_depth/on_trade (thread-locked)
  → 5min bar aggregation → 8-layer vote:
    funding(0.5x) + momentum + liquidation + OB imbalance + on-chain + MR(1.5x) + BTC-lead(2.0x) + DVOL(block)
  → Trend filter (MA50 blocks counter-trend) → Correlation guard (max 1 position)
  → Kelly sizing → Limit order → TP/SL/Timeout exit (MR: 0.15%/0.1%/60s, BL: 0.1%/0.08%/30s, normal: 0.3%/0.5%/301s)
```

**Data flow (Strategy H — 4h primary + 1h scaler, T-1 corrected)**:
```
┌─ 4h Runners (PRIMARY, cap BTC 15% / ETH 10%) ──────────────┐
│ Bybit WS kline.240 → push_bar → vol-adaptive deadzone       │
│ → Ridge(60%)+LGBM(40%) ensemble → z-score → discretize     │
│ → 4h独立下单 (不受COMBO限制)                                  │
│ → signal → _consensus_signals["BTCUSDT_4h"]                 │
└──────────────────────────────────────────────────────────────┘
        │ 4h signal (gate input)
        ▼
┌─ 1h Runners (SCALER, cap BTC 8% / ETH 6%) ─────────────────┐
│ Bybit WS kline.60 → push_bar → regime filter                │
│ → ensemble predict → z-score → MultiTFConfluenceGate        │
│   (reads 4h signal: agree→1.3x, oppose→0.3x, neutral→0.7x) │
│ → BB Entry Scaler → dynamic leverage (DD-based)             │
│ → Maker limit order (PostOnly, 45s timeout) → Bybit REST    │
└──────────┬───────────────────────────────────────────────────┘
           │ 1h+15m COMBO (AGREE ONLY, secondary)
           ▼
┌─ 15m Runners (COMBO, cap 5%) ───────────────────────────────┐
│ Bybit WS kline.15 → same pipeline → PortfolioCombiner      │
└─────────────────────────────────────────────────────────────┘

Cross-market features (T-1 shift): SPY/treasury/ETF_volume/DVOL/FGI/stablecoin/on-chain
```

**Python engine path**: Market event → FeatureComputeHook → Pipeline (RustStateStore) → DecisionModule → OrderRouter
**Binary path**: Binance WS → RustTickProcessor.process_tick_native() → risk gates → WS-API order (~4ms)

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 77 .rs modules, ~30K LOC; see `lib.rs` for full export list
- Exports: ~38 PyO3 classes + ~100 functions (195 total)
- Binary: `quant_trader` standalone trading binary (no Python runtime)
- Naming: `cpp_*` = C++ migration functions, `rust_*` = new kernel modules
- State types use i64 fixed-point (Fd8, x10^8); `_SCALE = 100_000_000`
- feature_hook.py always uses Rust (no Python fallback)
- `RustStateStore` keeps state on Rust heap, Python gets snapshots via `get_*()`
- `RustGateChain` processes 9 gate types in single FFI call (no per-gate Python↔Rust switching)

## Rust Pipeline (12/12 components in production)

AlphaRunner uses all 12 Rust components: RustFeatureEngine (120 features), RustInferenceBridge (z-score+deadzone+min-hold+max-hold), RustRiskEvaluator (drawdown+leverage), RustKillSwitch (global emergency stop), RustOrderStateMachine (order lifecycle), RustCircuitBreaker (3-failure/120s backoff), RustStateStore (position truth), RustFillEvent+RustMarketEvent (zero-copy), rust_pipeline_apply (atomic reducer). RustUnifiedPredictor, RustTickProcessor, RustWsClient imported but not active (see alpha_runner.py).

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 157 enriched feature definitions (V1-V19: ADX, OI, dominance, DVOL)
- `features/feature_catalog.py` — PRODUCTION_FEATURES frozenset (183 features); `validate_model_features()`
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `ext/rust/src/constraint_pipeline.rs` — Signal constraints (batch + incremental)
- `runner/live_runner.py` — Framework live runtime entry point
- `runner/gate_chain.py` — GateChain: up to 16 gates with `process_with_audit()` (incl. StagedRiskGate, AdaptiveStopGate, MultiTFConfluence, LiquidationCascade, CarryCost)
- `runner/config.py` — LiveRunnerConfig (~85 fields); factory: `.lite()`, `.paper()`, `.prod()`
- `scripts/ops/config.py` — SYMBOL_CONFIG (BTC+ETH × 1h/15m/4h = 6 runners), constants, MAX_ORDER_NOTIONAL
- `scripts/ops/alpha_runner.py` — AlphaRunner: Strategy H runtime (4h primary + 1h scaler + gate chain + SIGHUP hot-reload + checkpoint per-runner)
- `scripts/ops/signal_reconcile.py` — Live vs backtest signal consistency validation
- `scripts/ops/daily_reconciliation.py` — Live vs backtest signal reconciliation + slippage analysis
- `scripts/ops/shadow_compare.py` — A/B model comparison (shadow testing without execution)
- `scripts/training/train_4h_daily.py` — 4h/daily model trainer (LGBM + T-1 cross-market)
- `alpha/online_ridge.py` — Online Ridge with RLS incremental weight updates (enabled for 4h runners)
- `features/options_flow.py` — Options flow features from Deribit (gamma, max pain, vega, PCR, IV features)
- `features/batch_feature_engine.py` — 185+ features (V1-V24: including IV, stablecoin, ETF volume)
- `monitoring/ic_decay_monitor.py` — IC decay detection (GREEN/YELLOW/RED + Telegram alert)
- `execution/adapters/venue_router.py` — Multi-venue routing (Bybit + Hyperliquid, fee-optimal)
- `execution/adapters/hyperliquid/` — Hyperliquid DEX adapter (0% maker, EIP-712 signing)
- `regime/eth_regime_proxy.py` — ETH parameter routing using BTC regime labels
- `core/validation.py` — NaN/Inf input validation for prices, quantities, signals
- `monitoring/pipeline_metrics.py` — Thread-safe pipeline counters (bars/signals/orders/errors)
- `scripts/ops/portfolio_combiner.py` — PortfolioCombiner (AGREE ONLY mode, enforces MAX_ORDER_NOTIONAL)
- `scripts/ops/run_bybit_alpha.py` — 当前活跃 directional alpha service 入口
- `scripts/shared/signal_postprocess.py` — Signal pipeline (Rust + Python parity)
- `execution/sim/realistic_backtest.py` — Realistic backtest: intra-bar stop, Almgren-Chriss slippage, adaptive ATR stop
- `polymarket/collector.py` — 5m+15m CLOB collector + BS fair value + RSI signal
- `polymarket/strategies/maker_5m.py` — Avellaneda-Stoikov market maker for binary outcomes
- `polymarket/strategies/inventory_manager.py` — Inventory tracking + expiry actions
- `strategies/registry.py` — StrategyRegistry: register/discover/instantiate strategies
- `strategies/base.py` — StrategyProtocol + Signal dataclass
- `scripts/run_hft_signal.py` — HFT Signal engine: 8-layer voting (funding, momentum, liq, OB, on-chain, MR, BTC-lead, DVOL)
- `scripts/run_polymarket_dryrun.py` — Polymarket RSI(30/70) taker dry-run validation
- `scripts/run_binary_signal.py` — Binary 5m signal trader (Bybit)
- `scripts/ops/health_watchdog.py` — Auto health check + Telegram alerts + service restart
- `scripts/ops/auto_bug_scan.py` — Static bug scanner (30 patterns: bare-except, unchecked-api, etc.)
- `monitoring/notify.py` — Unified Telegram + console notification dispatcher
- `execution/adapters/hyperliquid/` — Hyperliquid DEX adapter (229 perps, 0.035% taker)
- `decision/regime_bridge.py` — RegimeAwareDecisionModule (CompositeRegimeDetector + ParamRouter)
- `risk/staged_risk.py` — StagedRiskManager: 5-stage equity-based risk ladder
- `docs/wiring_truth.md` — Module integration status table (what's wired, what's not)

## Live Integration Subsystems

- **Dual Alpha COMBO**: 1h+15m alphas via separate WS (kline.60 + kline.15). `PortfolioCombiner` AGREE ONLY: both must agree direction (Sharpe 5.48). Per-symbol cap: 45% equity x leverage (BTC+ETH only). Conviction: both=100%, one=50%.
- **Alpha Expansion Gates**: 3 new gates in production — MultiTFConfluence (1h/4h alignment), LiquidationCascade (cascade protection), CarryCost (funding/basis adjustment). Wired in `alpha_runner.py._evaluate_gates()`.
- **Online Ridge**: `alpha/online_ridge.py` — RLS incremental weight updates (λ=0.997). Enabled for 4h runners via `enable_online_ridge()`. Weight drift monitoring with Telegram alerts.
- **Adaptive Stop-Loss**: ATR 3-phase: initial ATRx1.2 → breakeven after 0.5×ATR profit → trailing at 0.2×ATR step. Hard limits: 0.3%-5%.
- **4h Z-Score Stop**: 1h/15m runners exit immediately when 4h model signal flips against position (IC 0.29-0.43 reversal signal).
- **Dynamic Leverage**: DD≥10%→0.75x, DD≥20%→0.5x, DD≥35%→0.25x leverage reduction.
- **Entry Scaler (BB)**: BB position → scale entry size (oversold=1.2x, overbought=0.3x). MaxDD reduced ~50%.
- **Vol-Adaptive Deadzone**: deadzone × (realized_vol / vol_median), clamped [0.5x, 2.0x]. Low vol→lower dz.
- **SIGHUP Hot-Reload**: `kill -HUP` reloads all 6 models in <200ms without restart. Triggered by auto_retrain.
- **Per-Runner Checkpoint**: Each runner saves/restores independently (`BTCUSDT_4h.json` etc). Instant recovery on restart.
- **Alpha Health Monitor**: `monitoring/alpha_health.py` — per-symbol IC tracking, gates via `position_scale()` (0.0/0.5/1.0).
- **IC Decay Monitor**: `monitoring/ic_decay_monitor.py` — daily 03:00 UTC, GREEN/YELLOW/RED levels, Telegram alerts.
- **Auto-Retrain**: Weekly (Sunday 2am, 1h models) + Daily (2am, 4h models with --sighup). IC tolerance gate.
- **Signal Reconciliation**: `scripts/ops/signal_reconcile.py` — live vs backtest signal match rate (target >90%).
- **VenueRouter**: Dual-venue support (Bybit + Hyperliquid). Fee-optimal routing with automatic fallback.

## Venue Adapters

All adapters implement `VenueAdapter` protocol (`execution/adapters/base.py`); registered via `AdapterRegistry.register(venue, adapter)`.

| Venue | Protocol | Min Order | Notes |
|-------|----------|-----------|-------|
| Binance | REST + WS-API (~4ms) | $20 (ETH) | Primary. USDT-M futures. 43 files, ~3.7K LOC. Live API connected. |
| Bybit | REST V5 + WS | $5 | Demo trading active. HMAC-SHA256. |
| Hyperliquid | REST (info/exchange) | varies | 229 perps. 0% maker taker 0.035%. EIP-712 signing. No KYC. 39ms latency. |
| Polymarket | CLOB REST | N/A | 0% fee. Collector V2: direct CLOB orderbook, SQLite. RSI dry-run active. |

## Environment

```bash
# Required (set in .env or export):
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export BYBIT_BASE_URL=https://api-demo.bybit.com  # or https://api.bybit.com for live
# See .env.example for all optional vars (Binance, Polymarket)
```

**Current deployment** (2026-03-22):
```bash
# ACTIVE:
#   bybit-alpha.service → Strategy H: 4 runners (BTC+ETH × 1h/4h), 2 WS (kline.60/240) — 15m disabled after WF FAIL
#     Sharpe 2.25 (T-1 corrected), $500→$2.15M backtest, demo $35K equity
#   polymarket collector + dryrun
# ACTIVE timers:
#   health-watchdog.timer (5min, 5h tolerance for 4h), data-refresh.timer (6h),
#   auto-retrain.timer (Sun 2am), daily-retrain.timer (daily 2am + SIGHUP),
#   ic-decay-monitor.timer (daily 3am)
# INACTIVE (validated but not recommended):
#   hft-signal.service (Sharpe -5), bybit-mm.service (spread < fee)
```

**Quick start**:
1. Health check: `python3 -m scripts.ops.health_watchdog`
2. Strategy H status: `tail -f logs/bybit_alpha.log | grep HEARTBEAT`
3. Start/restart: `sudo systemctl restart bybit-alpha.service`
4. Hot-reload models: `sudo kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)`
5. Signal check: `python3 -m scripts.ops.signal_reconcile --hours 24`
6. IC health: `python3 -m monitoring.ic_decay_monitor`
7. Timers: `sudo systemctl list-timers --all | grep -E "health|retrain|refresh|ic-decay"`

**Walk-forward baselines** (2026-03-22, T-1 corrected): see `results/walkforward/` for full data.
- **4h PASS**: BTCUSDT (WF Sharpe 3.62, 20/22), ETHUSDT (WF Sharpe 4.57, 21/21)
- **1h PASS**: ETHUSDT (Sharpe 3.92, 18/21), BTCUSDT (Sharpe 2.43, 15/22)
- **15m FAIL**: BTCUSDT (WF Sharpe 0.27, 3/4), ETHUSDT (WF Sharpe -1.36, 1/4) — overfit, consider disabling
- FAIL: all 5m/1m, all altcoins
- Production focus: **BTC + ETH only, 4h primary** (Strategy H)
- Kelly optimal leverage: **14x full Kelly, 7x half-Kelly** (Monte Carlo 10K runs). Demo uses **10x**. Production recommended **3x** (MaxDD ~18%)

## Signal Pipeline

```
Raw prediction (Ridge 60% + LightGBM 40% ensemble, T-1 cross-market features)
  → Online Ridge update (4h runners: RLS incremental weight adaptation)
  → Rolling z-score (4h: window=180/warmup=45, 1h: 720/180)
  → Vol-adaptive deadzone: dz × (realized_vol / vol_median), clamped [0.5x, 2.0x]
  → Discretize: z > deadzone → +1, z < -deadzone → -1, else 0
  → Min-hold enforce (BTC 4h: 6 bars=24h, ETH 4h: 18 bars=72h, BTC 1h: 24, ETH 1h: 18)
  → Monthly gate (BTC only: close <= SMA(120@4h/480@1h) → skip longs)
  → Gate chain: LiquidationCascade → MultiTFConfluence(4h model signal) → CarryCost
  → BB Entry Scaler (oversold→1.2x, overbought→0.3x)
  → Dynamic leverage (DD≥10%→0.75x, ≥20%→0.5x, ≥35%→0.25x)
  → Maker limit order (PostOnly, 45s timeout, 1-tick spread→bid/ask)
```
Constraint pipeline implemented identically in Rust (`constraint_pipeline.rs`) and Python (`signal_postprocess.py`). Parity verified via `tests/integration/test_constraint_parity.py`.

**Gate chain (alpha expansion)**:
- `MultiTFConfluenceGate`: 1h vs 4h trend alignment → aligned 1.2x, opposed 0.5x (MaxDD -20-25%)
- `LiquidationCascadeGate`: zscore>3 block, >2 scale 0.3x, OI unwind 0.5x
- `CarryCostGate`: funding+basis carry >30%/yr → 0.4x, >10% → 0.7x, favorable >5% → 1.15x

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
- `PortfolioCombiner` sets individual runners to `dry_run=True`; caps each symbol at 45% of equity x leverage (BTC+ETH only)
- SYMBOL_CONFIG: BTC+ETH only (SUI/AXS removed 2026-03-21 due to poor liquidity)
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
- BTC model h=96 uses `deadzone=1.0`, `min_hold=24` (1 day), `max_hold=144` (6 days), `monthly_gate=SMA(480)` — optimized 2026-03-21 (was dz=0.8/mh=48/maxh=288)
- ETH model uses `deadzone=0.4`, `min_hold=18`, `max_hold=60` — fixed params (regime-adaptive hurts ETH)
- `batch_feature_engine.py` `_add_dominance_features()` requires ETHUSDT_1h.csv
- `OnlineRidge` (alpha/online_ridge.py): RLS incremental weight updates, forgetting_factor=0.997, drift>0.5 triggers warning
- `OptionsFlowComputer` (features/options_flow.py): 7 features from Deribit options DB (gamma, max_pain, vega, PCR, IV term slope)

**Alpha research conclusions** (2026-03-22, T-1 corrected — no look-ahead bias):
- **Strategy H (production)**: 4h primary + 1h scaler, Sharpe 2.25, $500→$2.15M/6.5yr @10x — PASS
- **CRITICAL**: Cross-market features IC dropped 90% after T-1 correction (GBTC IC 0.27→0.03). Old Sharpe 4.37 was inflated.
- 4h Alpha: BTC WF Sharpe 3.62 (20/22), ETH 4.57 (21/21) — strongest timeframe
- 1h Alpha (T-1): BTC Sharpe 2.43 (15/22), ETH 3.92 (18/21)
- 15m: only ETH marginal (Sharpe 1.04); BTC 15m Sharpe 0.91 FAIL
- 5m/1m HFT: Sharpe -5 to -25 — ALL FAIL
- Market making: NOT viable (Bybit spread<fee, Hyperliquid adverse selection>spread)
- Neural networks: Ridge > MLP > LGBM on 4h OOS (LGBM overfits low-frequency data)
- Strongest features (T-1, no bias): DVOL zscore IC=0.074, ETF volume IC=0.11, funding zscore IC=0.052
- Cross-exchange arb: NOT viable (spread 2-5bps < fees 9.5bps)
- TWAP: NOT needed ($5K orders = 0.015% of bar volume)
- FOMC day: IC=0.061 (p=0.003) but decaying and only 8/year

**Current path split**:
- `scripts/run_polymarket_dryrun.py` — **currently active** dry-run validation
- `scripts/ops/run_bybit_alpha.py` — BTC+ETH Alpha (RECOMMENDED, BTC 4.37 ETH 4.67)
- `scripts/run_hft_signal.py` — HFT 8-layer signal (stopped, Sharpe -5)
- `scripts/run_bybit_mm.py` — market maker (stopped, not viable)
- `runner/live_runner.py` — framework convergence target
- `composite_regime_symbols` config controls per-symbol CompositeRegime (BTC only); ETH fixed params
- `ETHRegimeProxy` (regime/eth_regime_proxy.py): uses BTC regime labels for ETH parameter routing (ETH-specific param table)

**Regime & risk wiring** (2026-03-17):
- `RegimeAwareDecisionModule` defaults to `CompositeRegimeDetector` (was Vol+Trend separately)
- `RegimeParamRouter` activated when `enable_regime_sizing=True` via `enable_param_routing` flag
- `RegimePolicy` blocks composite crisis labels (any label with "crisis" in value)
- `StagedRiskManager` wired as `StagedRiskGate` in gate_chain; equity stages: survival→growth→stable→safe→institutional
- `initial_equity` field added to `LiveRunnerConfig` (default $500); `RustStateStore` uses real exchange balance via Fd8 at startup

**Automated operations** (2026-03-21):
- `health-watchdog.timer`: every 5min — checks services, data freshness, account equity; auto-restarts stale; Telegram alerts on status change
- `data-refresh.timer`: every 6h — syncs klines, funding, OI from Binance
- `auto-retrain.timer`: Sunday 2am UTC — walk-forward retrain with IC/Sharpe gates
- Cron: `demo_tracker` (hourly), `weekly_report` (Sun 3am), `auto_bug_scan` (Sun 1am), `OI download` (6h)
- `monitoring/notify.py`: unified Telegram dispatch; needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in .env
- `scripts/ops/auto_bug_scan.py`: 30 static patterns (bare-except, unchecked-api, mutable-default, float-equality)
- Pre-commit hook: ruff lint + API key check + critical bug scan + core tests (~5s)
- Health status: `data/runtime/health_status.json`; alert history: `data/runtime/alert_history.jsonl`

**Infrastructure**:
- `docs/deploy_truth.md` is deployment truth; `infra/systemd/` must sync with `/etc/systemd/system/`
- Burn-in gate: Phase A(Paper 3d) → B(Shadow 3d) → C(Testnet 3d) = 9 days before live; configs in `config/burnin.{paper,shadow,testnet}.yaml`
- Live config: `config/production.live.yaml` (ETHUSDT, Kelly 0.7x real / 10x demo, $5K order cap)
- Alert rules: `config/alerts.live.yaml` (balance $400 kill, $50 single-loss warn, 2h stale, IC decay)
- Dockerfile: multi-stage (`ci`/`paper`/`live`); `docker-compose.yml` manages `quant-paper` / `quant-live` / `quant-framework`
- `scripts/ops/security_scan.py`: checks hardcoded secrets, .env gitignored, MAX_ORDER_NOTIONAL, bare-except blocks
- `regime/param_router.py`: only BTC uses regime-adaptive params; ETH keeps fixed (fixed outperforms adaptive for ETH)
- Binance OI history API only retains ~28 days; `download_oi_data.py` cron every 6h to accumulate
- Polymarket Gamma API prices are **cached/stale** — always use CLOB orderbook for real prices
- Polymarket collector: background process (not systemd); PID in `data/polymarket/collector.pid`
- Walk-forward `--realistic` uses `close +/- 0.5%` as high/low unless real CSV; `--adaptive-stop` auto-injects real high/low
