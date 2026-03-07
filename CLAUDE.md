## Commands

```bash
make rust                    # Build Rust crate (maturin + pip install)
pytest tests/ -x -q          # Run all tests
pytest tests/unit/ -x -q     # Unit tests only
pytest -m benchmark          # Performance benchmarks
```

**CRITICAL after Rust build**: copy .so to local package (shadows system install):
```bash
cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/
```

## Architecture

```
engine/          Pipeline + coordinator (event -> state transitions)
features/        Feature computation (EnrichedFeatureComputer, 105 features)
decision/        Trading signals, regime detection, rebalancing
alpha/           ML models + inference bridge
execution/       Order routing, state machine, dedup
state/           State types + Rust adapters
ext/rust/        Unified Rust crate -> _quant_hotpath (53 modules, ~16K LOC)
runner/          Live/paper/backtest entry points
regime/          Regime detection (volatility, trend)
risk/            Risk limits + kill switch
```

**Data flow**: Market event -> FeatureComputeHook (RustFeatureEngine) -> Pipeline
  (RustStateStore) -> DecisionModule -> ExecutionPolicy -> OrderRouter

## Rust Crate (`ext/rust/`)

- Single crate `_quant_hotpath`, 53 .rs modules, ~16,300 LOC
- Naming: `cpp_*` = C++ migration functions, `rust_*` = new kernel modules
- State types use i64 fixed-point (Fd8, x10^8); `_SCALE = 100_000_000`
- feature_hook.py always uses Rust (no Python fallback)
- `RustStateStore` keeps state on Rust heap, Python gets snapshots via `get_*()`

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

## Task Management

1. **Plan First**: Write plan to tasks/todo.md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Capture Lessons**: Update tasks/lessons.md after corrections

## Core Principles

* **Simplicity First**: Make every change as simple as possible. Impact minimal code.
* **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
* **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
