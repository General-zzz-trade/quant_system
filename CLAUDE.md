## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
pytest tests/ -x -q          # Run all tests (2654 Python tests)
pytest tests/unit/ -x -q     # Unit tests only
pytest -m benchmark          # Performance benchmarks
cd ext/rust && cargo test    # Rust unit tests (41 tests)
```

**CRITICAL after Rust build**: copy .so to local package (shadows system install):
```bash
cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/
```

## Architecture

```
engine/          Pipeline + coordinator (event -> state transitions)
features/        Feature computation (EnrichedFeatureComputer, 105 features)
decision/        Trading signals, ensemble, regime detection, rebalancing
alpha/           ML models + inference bridge
execution/       Order routing, state machine, dedup
state/           State types + Rust adapters
ext/rust/        Unified Rust crate -> _quant_hotpath (54 modules, ~16K LOC)
runner/          Live/paper/backtest entry points
regime/          Regime detection (volatility, trend)
risk/            Risk limits + kill switch
portfolio/       Allocator, rebalance, optimizer
monitoring/      Alerts, health checks, metrics, Prometheus, Grafana
infra/           Logging (structured JSON), networking
scripts/         Training, walk-forward validation, alpha research
```

**Data flow**: Market event -> FeatureComputeHook (RustFeatureEngine) -> Pipeline
  (RustStateStore) -> DecisionModule -> ExecutionPolicy -> OrderRouter

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 54 .rs modules, ~16,300 LOC
- Exports: 56 classes + 92 functions
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

## Key Files

- `engine/coordinator.py` — Main event loop orchestrator
- `engine/pipeline.py` — State transition pipeline (Rust fast path)
- `engine/feature_hook.py` — Bridges RustFeatureEngine into pipeline
- `features/enriched_computer.py` — 105 enriched feature definitions
- `ext/rust/src/lib.rs` — Rust module registry + PyO3 exports
- `runner/live_runner.py` — Production entry point

## Gotchas

- `_quant_hotpath/` at project root shadows pip-installed package — always copy .so after build
- `RustFeatureEngine` uses its own window sizes (not LiveFeatureComputer params)
- Tests require `_quant_hotpath` built; `pytest.importorskip("_quant_hotpath")` guards Rust tests
- Production models in `models_v8/` (LightGBM)
- CrossAssetComputer: must push benchmark (BTCUSDT) **before** altcoins each bar
- Fd8 conversion: Python `float * _SCALE` → Rust i64, Rust i64 → Python `/ _SCALE`
- `features/dynamic_selector.py` keeps `_rankdata`/`_spearman_ic` for scripts (not fallback)
