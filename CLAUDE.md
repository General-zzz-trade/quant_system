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
python3 -m alpha.auto_retrain --include-4h --force                # Retrain 1h + 4h models
python3 -m alpha.auto_retrain --daily --include-4h --sighup       # Daily lightweight retrain + hot-reload
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
python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)), 'exports')"  # expect 201
```

Rust builds must enable `python` feature: `maturin build --release --features python`

## Architecture

```
decision/        DecisionModule protocol, AlphaDecisionModule, signals, sizing
  modules/alpha.py   Framework-native decision logic + audit logging (~539 lines)
  signals/           EnsemblePredictor + SignalDiscretizer
  sizing/            AdaptivePositionSizer (equity-tier + IC + vol, Rust delegate)
  rust/              19 .rs (constraint pipeline, inference bridge, ML predict, sizer, exit mgr, micro alpha)

engine/          EngineCoordinator, Pipeline, Bridges, Dispatcher
  coordinator.py     Event orchestration hub
  pipeline.py        State transitions (→ RustStateStore)
  feature_hook.py    Bridges RustFeatureEngine into pipeline
  rust/              9 .rs (tick_processor ~80μs, pipeline, guards)

event/           Event types (Rust PyO3 driven, Python thin wrappers)
  events.py          8 event classes (MarketEvent, OrderEvent, FillEvent, etc.)
  rust/              9 .rs (EventHeader, event classes, validators)

state/           State management (Rust types, zero Python dataclass)
  snapshot.py        StateSnapshot container
  store.py           SQLite persistence + Rust to_dict/from_dict
  rust/              19 .rs (types, reducers, store)

execution/       Exchange adapters (Bybit, Hyperliquid, Binance)
  adapters/bybit/    Production adapter + execution_adapter (ExecutionAdapter protocol)
  safety/            CircuitBreaker, KillSwitch, OrderLimiter (Rust delegates)
  rust/              6 .rs (order state machine, WS client)

features/        Feature computation (185+ features, V1-V24)
  enriched_computer  Incremental features (Rust PyO3 trackers)
  batch_feature_engine  Batch computation
  rust/              26 .rs (FeatureEngine, indicators, incremental trackers, cross-asset, microstructure)

risk/            Risk gates (StagedRisk, AdaptiveStop, GateChain)
  rust/              14 .rs (gate_chain, risk engine, adaptive stop, drawdown, aggregator)

runner/          Runtime (alpha_main entry, backtest, recovery)
  alpha_main.py      PRODUCTION entry (EngineCoordinator + WS)
  strategy_config.py SYMBOL_CONFIG (BTC+ETH × 1h/4h)

alpha/           ML models (loader, online Ridge, auto-retrain)
monitoring/      Ops (watchdog, IC decay, data quality, Telegram, rolling Sharpe, decision audit)
portfolio/       Portfolio (allocator with Rust delegate, combiner, hedge)
attribution/     PnL tracking (Rust-backed PnLTracker)
data/            Data downloads + quality checks
regime/          Regime detection (CompositeRegime + ParamRouter)
infra/           Logging, config, systemd, errors
research/        Research scripts + Rust tools
```

**Data flow (Strategy H)**:
```
Bybit WS kline → MarketEvent → EngineCoordinator.emit()
  ├─ FeatureComputeHook → RustFeatureEngine → 185+ features (+ dominance + microstructure)
  ├─ StatePipeline → RustStateStore (state update)
  └─ DecisionBridge → AlphaDecisionModule.decide(snapshot)
      ├─ EnsemblePredictor: Ridge(60%)+LGBM(40%)
      ├─ SignalDiscretizer: z-score → z-clamp → deadzone → min-hold
      ├─ Force exits: ATR stop, quick loss, z-reversal, 4h reversal
      ├─ Direction alignment: ETH follows BTC
      └─ AdaptivePositionSizer: equity-tier × IC × vol
  └─ OrderEvent → ExecutionBridge → BybitExecutionAdapter → FillEvent
```

## Rust Integration

- 143 .rs files distributed in `{module}/rust/` dirs; build entry: `Cargo.toml` + `rust_lib.rs` at project root
- 201 PyO3 exports, 100% used by production Python
- State types use i64 fixed-point (Fd8, ×10^8); `_SCALE = 100_000_000`
- `RustStateStore` = position truth on Rust heap; Python gets snapshots via `get_*()`
- `RustFeatureEngine` = incremental features; `checkpoint()`/`restore_checkpoint()` persist as JSON
- `RustTickProcessor` = full hot-path (~80μs): features + predict + state in single FFI call
- Event types: all 9 event classes backed by Rust PyO3 frozen classes

## Key Files

- `decision/modules/alpha.py` — AlphaDecisionModule: framework-native decision logic
- `decision/signals/alpha_signal.py` — EnsemblePredictor (Ridge+LGBM) + SignalDiscretizer (z-score+deadzone)
- `decision/sizing/adaptive.py` — AdaptivePositionSizer: equity-tier + IC health + vol scaling
- `runner/alpha_main.py` — **PRODUCTION** entry point (EngineCoordinator + WS)
- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `engine/decision_bridge.py` — DecisionModule protocol + bridge
- `execution/adapters/bybit/execution_adapter.py` — BybitExecutionAdapter (ExecutionAdapter protocol)
- `execution/adapters/bybit/adapter.py` — Bybit REST V5 adapter
- `runner/strategy_config.py` — SYMBOL_CONFIG, MAX_ORDER_NOTIONAL_PCT, LEVERAGE_LADDER
- `state/snapshot.py` — StateSnapshot (frozen, duck-typed for Rust/Python)
- `state/store.py` — SQLite checkpoint persistence
- `features/enriched_computer.py` — Incremental feature computation (Rust trackers)
- `features/batch_feature_engine.py` — Batch feature engine (185+ features)
- `alpha/auto_retrain.py` — Auto-retrain orchestrator (weekly + daily + IC-triggered)
- `monitoring/ic_decay_monitor.py` — IC decay detection (GREEN/YELLOW/RED + Telegram)
- `monitoring/data_quality_check.py` — OHLC consistency + gap detection
- `attribution/pnl_tracker.py` — PnL tracking (Rust-backed, no Python fallback)
- `portfolio/allocator.py` — Portfolio allocation (Rust-delegated core math)
- `monitoring/decision_audit.py` — Structured JSONL audit log (signal/entry/exit)
- `monitoring/rolling_sharpe.py` — Per-symbol rolling Sharpe tracker (GREEN/YELLOW/RED)
- `infra/model_signing.py` — HMAC model signing + verification (enforced in live)
- `rust_lib.rs` — Rust module registry + PyO3 exports

## Signal Pipeline

```
Ridge(60%) + LGBM(40%) ensemble → Rolling z-score → Z-clamp (|z|>3.5 → ±3.0)
  → Vol-adaptive deadzone (dz × vol_ratio, clamped [0.5x, 2.0x])
  → Discretize (+1/-1/0) → Adaptive min-hold (base × vol_ratio^0.5)
  → Direction alignment (ETH follows BTC) → Force exits (ATR/quick_loss/z_reversal/4h_reversal)
  → AdaptivePositionSizer (equity-tier × IC × leverage × z_scale)
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

**Trading & safety**:
- `MAX_ORDER_NOTIONAL_PCT = 250%` of equity (safety cap); dynamic via `get_max_order_notional(equity)`
- `_round_to_step()` applied in ALL sizing paths — prevents Bybit `Qty invalid` rejections
- SYMBOL_CONFIG: BTC+ETH only; 15m disabled after WF FAIL
- `_NEUTRAL_DEFAULTS`: NaN features → neutral values (ls_ratio→1.0, rsi_14→50.0), not 0.0
- `reliable_close_position()` replaces bare `close_position()` calls
- `RustStateStore` initialized with real exchange balance (Fd8)

**Features & models**:
- ADX(14): Rust incremental tracker (PyAdxTracker); needs 2×14=28 bars warmup
- CrossAssetComputer: push benchmark (BTCUSDT) **before** altcoins each bar
- Ridge model uses own feature list (`ridge_features`) — may differ from LGBM
- Feature selection: greedy IC is optimal
- BTC 1h: `deadzone=1.0, min_hold=18, max_hold=120`; ETH 1h: `deadzone=0.9, min_hold=18, long_only=true`
- All hold/deadzone values are base — adapt via vol_ratio at runtime

**Safety & security**:
- Model signing: HMAC-SHA256 via `QUANT_MODEL_SIGN_KEY`. Live mode **always** requires signatures; demo allows bypass
- Daily drawdown kill switch: `MAX_DAILY_DRAWDOWN_PCT` (default 5%) arms `RustKillSwitch` in main loop
- Leverage auto-detection: 3x for live (`api.bybit.com`), 10x for demo — see `strategy_config.py:_IS_LIVE`
- VPIN entry gate: reduces qty 30% when `vpin > 0.5` (microstructure toxicity)
- Decision audit: `data/runtime/decision_audit.jsonl` — every signal/entry/exit logged as JSON

**Deployment**:
- Production entry: `python3 -m runner.alpha_main` (systemd: `bybit-alpha.service`)
- Timers: health-watchdog (5min), data-refresh (6h), daily-retrain (daily 2am), auto-retrain (Sun 2am), ic-decay (daily 3am)
- `docs/deploy_truth.md` is deployment truth; `infra/systemd/` must sync via `infra/sync_systemd.sh`
- CI/CD: `.github/workflows/ci.yml` — lint + rust-test + python-test + security-scan
- Pre-commit hook: ruff lint + API key check + critical bug scan + core tests (~5s)
- Log rotation: `infra/logrotate.d/quant-system` (daily, 14 rotations, 50M max)

**Walk-forward baselines** (T-1 corrected):
- 4h PASS: BTC (Sharpe 3.62), ETH (Sharpe 4.57)
- 1h PASS: BTC (Sharpe 2.43), ETH (Sharpe 3.92)
- 15m FAIL: disabled
- Kelly optimal: 14x full / 7x half. Demo 10x. Production recommended 3x.

## Environment

```bash
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export BYBIT_BASE_URL=https://api-demo.bybit.com  # or https://api.bybit.com for live
# See .env.example for all optional vars (Binance, Polymarket, Telegram)
```
