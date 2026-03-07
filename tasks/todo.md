# Python → Rust Deep Migration: 4 Phases

## Phase 1: Eliminate Python State Types from Pipeline [DONE]
- [x] Add float-returning properties to Rust state types (state_types.rs)
- [x] Rebuild Rust crate
- [x] Update PipelineOutput to hold Rust types directly (skip market_from_rust conversion)
- [x] Update coordinator.py internal state to Rust types
- [x] Update snapshot.py to accept Rust types
- [x] Update ml_decision.py to read Rust type fields
- [x] Update decision/sizing modules
- [x] Update backtest_runner.py and live_runner.py for Rust types
- [x] Run all tests and fix breakages (2550 passed)

## Phase 2: Event Hot-Path Rust Types [DONE]
- [x] Add RustMarketEvent, RustFillEvent, RustFundingEvent structs in Rust (rust_events.rs)
- [x] Fast-path reducers: reduce_rust_market, reduce_rust_fill, reduce_rust_funding (state_reducers.rs)
- [x] RustStateStore.process_event() fast path via downcast (state_store.rs)
- [x] rust_pipeline_apply() fast path via downcast (pipeline.rs)
- [x] detect_kind_inner() fast path via downcast (pipeline.rs)
- [x] Keep Python event compat path (getattr slow path unchanged)
- [x] Register new types in lib.rs
- [x] 12 parity tests pass (test_rust_events.py), 23.9x speedup
- [x] All 2550 tests pass
- [~] WebSocket Rust events: deferred — RustMarketEvent lacks trades/taker_buy fields, benefit minimal

## Phase 3: Risk Rules Rust Migration [DONE]
- [x] RustRiskEvaluator: unified hot-path evaluator (risk_engine.rs)
- [x] 6 rules: max_position, leverage_cap, max_drawdown, gross_exposure, net_exposure, concentration
- [x] Auto-reduce support (max_qty calculation)
- [x] Quick check methods: check_drawdown(), check_leverage()
- [x] RustRiskResult return type for violations
- [x] 21 parity tests pass (test_rust_risk_evaluator.py)
- [x] Python rules remain for Protocol interface + meta extraction (aggregator unchanged)
- [x] All 2583 tests pass

## Phase 4: Feature Engine Rust Migration [DONE]
- [x] Create RustFeatureEngine with rolling state (wraps BarState, cached features + prev_momentum tracking)
- [x] 105 features computed incrementally via push_bar() → get_features()/get_features_array()
- [x] Registered in lib.rs, exposed as RustFeatureEngine class
- [x] 5 parity tests pass (test_rust_feature_engine.py): batch-vs-incremental, smoke, names, warmup, perf
- [x] All 2588 tests pass
- [x] FeatureComputeHook Rust integration: per-symbol RustFeatureEngine, dict→flat args mapping
- [x] live_runner.py + live_paper_runner.py pass use_rust=True
- [x] 6 hook parity tests pass (test_feature_hook_rust.py)
- [x] All 2594 tests pass
