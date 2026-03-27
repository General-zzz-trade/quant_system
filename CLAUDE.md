## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
make test                    # Core local gate
pytest tests/unit/ -x -q     # Unit tests (~10s)
pytest tests/unit/runner/ -x -q    # Runner tests
pytest tests/unit/bybit/ -x -q    # Bybit adapter tests
pytest tests/unit/state/ -x -q    # State module tests
pytest tests/unit/event/ -x -q    # Event module tests
pytest tests/unit/decision/ -x -q # Decision module tests
cargo test                   # Rust unit tests
ruff check --select E,W,F . # Lint (matches CI gate)
```

**Active trading**:
```bash
# Strategy H: 4h primary + 1h scaler (framework-native, 4 runners, 2 WS):
python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
sudo systemctl restart bybit-alpha.service

# Model hot-reload (no restart, <200ms):
sudo kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)
```

**Data & model management**:
```bash
python3 -m data.downloads.data_refresh                            # Full data sync (klines + funding + OI)
python3 -m alpha.retrain.cli --include-4h --force --sighup        # Retrain 1h + 4h models + hot-reload
python3 -m alpha.retrain.cli --daily --include-4h --sighup        # Daily lightweight retrain + hot-reload
python3 -m alpha.retrain_15m                                      # 15m retrain (stricter thresholds, V14+regime features)
python3 -m alpha.retrain_15m --dry-run                            # 15m validation only
```

**Monitoring**:
```bash
python3 -m monitoring.watchdog                                     # Health check + auto-restart + Telegram
python3 -m monitoring.ic_decay_monitor --alert                     # IC decay + Telegram
python3 -m monitoring.data_quality_check                           # Data quality (OHLC + gaps)
python3 -m monitoring.data_quality_check --symbol BTCUSDT --json   # Single symbol JSON
python3 -m monitoring.rolling_sharpe                               # Per-symbol rolling Sharpe (GREEN/YELLOW/RED)
```

**CRITICAL after Rust build**: copy .so then verify:
```bash
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)), 'exports')"  # expect 202
```

Rust builds must enable `python` feature: `maturin build --release --features python`

## Architecture

```
decision/        DecisionModule protocol, AlphaDecisionModule, signals, sizing
  modules/alpha.py   Framework-native decision logic + audit logging
  signals/           EnsemblePredictor + SignalDiscretizer
  sizing/            AdaptivePositionSizer (equity-tier + IC + vol, Rust delegate)
  rust/              19 .rs (constraint pipeline, inference bridge, ML predict, sizer, exit mgr, micro alpha)

engine/          EngineCoordinator, Pipeline, Bridges, Dispatcher
  coordinator.py     Event orchestration hub
  pipeline.py        State transitions (→ RustStateStore)
  feature_hook.py    Bridges RustFeatureEngine + 13 data sources into pipeline
  rust/              9 .rs (tick_processor [disabled], pipeline, guards)

event/           Event types (Rust PyO3 driven, Python thin wrappers)
  events.py          8 event classes (MarketEvent, OrderEvent, FillEvent, etc.)
  rust/              9 .rs (EventHeader, event classes, validators)

state/           State management (Rust types, zero Python dataclass)
  snapshot.py        StateSnapshot container
  store.py           SQLite persistence + Rust to_dict/from_dict
  rust/              19 .rs (types, reducers, store)

execution/       Exchange adapters (Bybit, Hyperliquid, Binance)
  adapters/bybit/    Production adapter + execution_adapter (3x retry)
  safety/            CircuitBreaker, KillSwitch, OrderLimiter (Rust delegates)
  rust/              6 .rs (order state machine, WS client)

features/        Feature computation (141 Rust features + batch 192)
  enriched_computer  Incremental features (Rust PyO3 trackers)
  batch_feature_engine  Batch computation for training/backtest
  rust/              26 .rs (FeatureEngine, indicators, incremental trackers, cross-asset, microstructure)

risk/            Risk gates (StagedRisk, AdaptiveStop, GateChain)
  rust/              14 .rs (gate_chain, risk engine, adaptive stop, drawdown, aggregator)

runner/          Runtime (alpha_main entry, backtest, recovery)
  alpha_main.py      PRODUCTION entry (EngineCoordinator + WS + real-time monitor)
  limit_order_manager.py  Limit order pre-placement (z > 0.7×dz → limit order)
  (config in strategy/config.py)

alpha/           ML models (loader, online Ridge, auto-retrain)
  retrain/cli.py     Retrain CLI (correct entry point for systemd timers)
monitoring/      Ops (watchdog, IC decay, data quality, Telegram, rolling Sharpe, decision audit)
attribution/     PnL tracking (Rust-backed PnLTracker, integrated into alpha_main)
strategy/        Strategy config + regime detection + execution policy + gates
  config.py        SYMBOL_CONFIG, MAX_ORDER_NOTIONAL_PCT, LEVERAGE_LADDER
  regime/          CompositeRegime + ParamRouter + Rust detector
data/            Data downloads + quality checks
infra/           Logging, config, systemd, errors
research/        Research scripts + Rust tools
```

**Data flow (Strategy H)**:
```
Bybit WS kline → MarketEvent → EngineCoordinator.emit()
  ├─ FeatureComputeHook → RustFeatureEngine → 141 features (+ 13 CSV data sources)
  ├─ StatePipeline → RustStateStore (state update)
  └─ DecisionBridge → AlphaDecisionModule.decide(snapshot)
      ├─ EnsemblePredictor: Ridge(60%)+LGBM(40%), Rust-native inference preferred
      ├─ SignalDiscretizer: z-score → z-clamp → fixed deadzone → min-hold
      ├─ Force exits: ATR 3-phase stop, quick loss, z-reversal, 4h reversal, alignment exit
      ├─ 4h direction filter: 1h entry blocked when 4h signal opposes
      ├─ Direction alignment: ETH follows BTC (entry + holding)
      └─ AdaptivePositionSizer: equity-tier × IC × vol (Rust delegate)
  └─ OrderEvent → ExecutionBridge → BybitExecutionAdapter (3x retry) → FillEvent

Real-time layer (parallel to bar flow):
  ├─ on_tick: wick detector (>0.8% move + 0.3% bounce → early entry)
  ├─ 60s monitor: z-score preview + limit order pre-placement
  └─ Instant signal on restart (no waiting for next bar)
```

## Rust Integration

- 141 features from RustFeatureEngine (105 base + 36 extensions: on-chain z-scores, interaction, IV, ETF returns, 4h BB)
- 202 PyO3 exports including push_cross_market() for ETF data
- State types use i64 fixed-point (Fd8, ×10^8); `_SCALE = 100_000_000`
- `RustStateStore` = position truth on Rust heap; Python gets snapshots via `get_*()`
- `RustFeatureEngine` = incremental features; `checkpoint()`/`restore_checkpoint()` persist as JSON
- `RustTickProcessor` = full hot-path (~80μs) **DISABLED**: z-score buffer diverges from Python InferenceBridge; Python pipeline is Rust-accelerated (~200μs/bar); not worth dual signal routing
- Event types: all 9 event classes backed by Rust PyO3 frozen classes
- `RustInferenceBridge`: z-score normalization with checkpoint/restore for preview without state mutation

## Key Files

- `decision/modules/alpha.py` — AlphaDecisionModule: decision logic + trade cooldown + duplicate bar guard
- `decision/signals/alpha_signal.py` — EnsemblePredictor (Rust-native preferred) + SignalDiscretizer
- `decision/sizing/adaptive.py` — AdaptivePositionSizer: equity-tier + IC health + vol (Rust delegate)
- `runner/alpha_main.py` — **PRODUCTION** entry (WS + real-time monitor + wick detector + limit orders)
- `runner/limit_order_manager.py` — Limit order pre-placement (z > 0.7×dz)
- `runner/builders/alpha_builder.py` — Coordinator builder + CsvCursor data sources + push_cross_market
- `engine/coordinator.py` — Main event loop orchestrator
- `engine/feature_hook.py` — Bridges RustFeatureEngine + 13 data sources + feature aliases
- `execution/adapters/bybit/execution_adapter.py` — BybitExecutionAdapter (3x retry)
- `strategy/config.py` — SYMBOL_CONFIG, MAX_ORDER_NOTIONAL_PCT, LEVERAGE_LADDER
- `features/batch_feature_engine.py` — Batch feature engine (192 features for training)
- `alpha/retrain/cli.py` — Retrain CLI (correct entry for `--sighup` hot-reload)
- `monitoring/ic_decay_monitor.py` — IC decay detection (GREEN/YELLOW/RED + auto-retrain trigger)
- `attribution/pnl_tracker.py` — PnL tracking (integrated into alpha_main fill observer)
- `rust_lib.rs` — Rust module registry + PyO3 exports

## Signal Pipeline

```
Ridge(60%) + LGBM(40%) ensemble → Rolling z-score → Z-clamp (|z|>3.5 → ±3.0)
  → Fixed deadzone (no vol-adaptive scaling) → Discretize (+1/-1/0)
  → Fixed min-hold → Trade cooldown (min_hold bars between flat→entry)
  → 4h direction filter (1h entry blocked when 4h opposes)
  → Direction alignment (ETH follows BTC, entry + holding)
  → Force exits (ATR 3-phase/quick_loss/z_reversal/4h_reversal/alignment_exit)
  → AdaptivePositionSizer (equity-tier × IC × leverage × z_scale)
  → Regime filter: inactive regime → deadzone × 1.5 (not blocked, just wider)
```

## Gotchas

**Build & environment**:
- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `pip install` requires `--break-system-packages` (no venv, system Python 3.12)
- Binary build requires `-lpython3.12` link flag (PyO3 symbols)

**Rust/Python interface**:
- Fd8: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- State types are Rust PyO3 objects (no Python dataclass layer); use `.to_dict()`/`.from_dict()` for serialization
- `RustFeatureEngine` uses own window sizes; checkpoint as bar history JSON
- `RustStateStore` = position truth; `OrderStateMachine` = execution audit trail only
- Feature hook source exceptions isolated via `_safe_call_source()` — NaN on failure, bar continues
- `RustGateChain` processes all gates in single FFI call — no per-gate Python↔Rust switching
- Feature aliases (`_RUST_ALIASES` in feature_hook.py): maps Rust names to model names (e.g. tnx_change_5d→treasury_10y_chg_5d)

**Trading & safety**:
- `MAX_ORDER_NOTIONAL_PCT = 250%` of equity (safety cap); dynamic via `get_max_order_notional(equity)`
- `_round_to_step()` applied in ALL sizing paths — prevents Bybit `Qty invalid` rejections
- SYMBOL_CONFIG: BTC+ETH 1h/4h active; 15m DISABLED (full-sample Sharpe 0.50, model decayed)
- `_NEUTRAL_DEFAULTS`: NaN features → neutral values (ls_ratio→1.0, rsi_14→50.0), not 0.0
- `reliable_close_position()` replaces bare `close_position()` calls
- Trade cooldown: `_last_trade_bar` prevents rapid-fire open/close cycles after warmup
- Duplicate bar guard: `last_ts` check prevents same bar triggering decide() twice

**Features & models**:
- ADX(14): Rust incremental tracker (PyAdxTracker); needs 2×14=28 bars warmup
- CrossAssetComputer: push benchmark (BTCUSDT) **before** altcoins each bar
- Ridge model uses own feature list (`ridge_features`) — may differ from LGBM
- BTC 1h: `deadzone=1.2, min_hold=6, max_hold=120`; ETH 1h: `deadzone=1.2, min_hold=6, long_only=false`
- Deadzone is FIXED (vol-adaptive disabled — backtest showed fixed outperforms by 43%)
- Regime filter: inactive → deadzone × 1.5 (not 999); strong signals can still trade at 0.6x size
- CsvCursor: loads full CSV history so each warmup bar gets time-appropriate values (not just latest)
- push_cross_market(): feeds ETF daily data (SPY/TLT/USO/GLD/COIN) to Rust engine
- OI CSV format: `_fix_oi_file()` required before batch feature computation (mixed 2/4 column format)
- Online Ridge: activated in EnsemblePredictor (forgetting_factor=0.99), auto-updates each bar

**Safety & security**:
- Model signing: HMAC-SHA256 via `QUANT_MODEL_SIGN_KEY`. Live mode **always** requires signatures; demo allows bypass
- Daily drawdown kill switch: `MAX_DAILY_DRAWDOWN_PCT` (default 5%) arms `RustKillSwitch` in main loop
- Leverage auto-detection: 3x for live (`api.bybit.com`), 10x for demo — see `strategy/config.py:_IS_LIVE`
- VPIN entry gate: reduces qty 30% when `vpin > 0.5` (microstructure toxicity)
- Decision audit: `data/runtime/decision_audit.jsonl` — every signal/entry/exit logged as JSON

**Startup & recovery**:
- Parallel warmup: 4 runners warm up concurrently via ThreadPoolExecutor (~4s total)
- Z-score checkpoint: `data/runtime/zscore_checkpoints/{runner}.json` — saved every 5min + on shutdown
- Instant signal: after warmup, emits latest confirmed bar immediately (no waiting for next bar)
- WS interval routing: bar dict includes `interval` field from topic (kline.60 vs kline.240)

**Deployment**:
- Production entry: `python3 -m runner.alpha_main` (systemd: `bybit-alpha.service`)
- Timers: health-watchdog (5min), data-refresh (6h), daily-retrain (daily 2am via `alpha.retrain.cli`), auto-retrain (3 days), ic-decay (daily 3am)
- `docs/deploy_truth.md` is deployment truth; `infra/systemd/` must sync via `infra/sync_systemd.sh`
- CI/CD: `.github/workflows/ci.yml` — lint + rust-test + python-test + security-scan
- Pre-commit hook: ruff lint + API key check + critical bug scan + core tests (~5s)
- Log rotation: `infra/logrotate.d/quant-system` (daily, 14 rotations, 50M max)
- Log archival: `scripts/archive_logs.py` (weekly, 6 month retention)
- Backup: `scripts/backup_remote.py` (daily 4am, 30 local + optional S3)

**Walk-forward baselines** (latest retrain 2026-03-27):
- 1h PASS: BTC (Sharpe 2.11), ETH (Sharpe 4.11)
- 4h PASS: BTC (Sharpe 5.34), ETH (Sharpe 4.88)
- 15m BTC DISABLED: full-sample Sharpe 0.50 (WF OOS inflated)
- 15m ETH FAIL: disabled
- Kelly optimal: 14x full / 7x half. Demo 10x. Production recommended 3x.

## Environment

```bash
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export BYBIT_BASE_URL=https://api-demo.bybit.com  # or https://api.bybit.com for live
# See .env.example for all optional vars (Binance, Polymarket, Telegram)
# Limit order config (optional):
export LIMIT_OFFSET_BPS=30       # 0.3% price offset for limit orders
export LIMIT_TTL_S=300           # 5 min TTL
export LIMIT_QTY_SCALE=0.5      # 50% of normal qty
```
