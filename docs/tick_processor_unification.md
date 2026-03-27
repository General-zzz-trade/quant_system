# RustTickProcessor Unification Analysis

**Date**: 2026-03-27
**Status**: Long-term optimization direction (analysis only, no code changes)

## Background

`RustTickProcessor` achieves ~80us/bar by merging features + predict + state update
into a single Rust struct, eliminating ~10 Python-to-Rust boundary crossings per tick.
It is currently **DISABLED** because its internal z-score buffer diverges from the
`RustInferenceBridge` z-score buffer used by the Python pipeline (~200us/bar).

## Problem Analysis

### Two Independent Z-Score Buffers

1. **RustTickProcessor** (`engine/rust/tick_signal.inc.rs:84-106`)
   - Maintains its own `bridge_states: HashMap<String, SymbolState>` (line 56, tick_processor.rs)
   - Calls `zscore_from_buf()` from within `apply_signal_pipeline()`
   - Buffer lives on the Rust heap, never exposed to Python

2. **RustInferenceBridge** (`decision/rust/inference_bridge.rs:98-103`)
   - Maintains `symbols: HashMap<String, SymbolState>` with identical `SymbolState` struct
   - Exposed to Python via PyO3 (`zscore_normalize()`, `apply_constraints()`)
   - Used by `SignalDiscretizer` in `decision/signals/alpha_signal.py:388`
   - Supports checkpoint/restore for preview-without-mutation and crash recovery

### Why They Diverge

Both buffers use the same `zscore_from_buf()` function (shared code), but:

- **Append timing**: The Python pipeline calls `zscore_normalize()` then separately
  `apply_constraints()`, which both append to the buffer (guarded by `hour_key`
  dedup). TickProcessor calls `apply_signal_pipeline()` once, doing both in sequence.
  If the Python path ever calls zscore_normalize without apply_constraints (e.g.,
  for preview/monitoring), the buffers accumulate different entries.

- **Python-side extras**: `SignalDiscretizer.discretize()` applies z-clamp
  (`|z| > 3.5 -> +/-3.0`) and regime-adaptive deadzone widening (`dz * 1.5`)
  *before* calling `apply_constraints()`. TickProcessor's `apply_signal_pipeline()`
  does neither -- it passes the raw z directly to `discretize()`.

- **Force exits**: The Python `AlphaDecisionModule.decide()` layer
  (`decision/modules/alpha.py`) applies ATR 3-phase stops, quick loss exits,
  z-reversal exits, 4h reversal exits, and alignment exits *after* discretization.
  TickProcessor has no equivalent -- it only outputs `ml_score` and `ml_short_score`.

- **Checkpoint/restore**: InferenceBridge supports `checkpoint()`/`restore_checkpoint()`
  for z-score preview (limit order manager uses this). TickProcessor has no
  preview capability without mutating state.

### Impact

If both were active simultaneously, they would produce different signals for the same
bar because:
1. Z-clamp and regime deadzone widening are missing from TickProcessor
2. Force exit logic is entirely absent from TickProcessor
3. Preview operations in the Python path would cause buffer content divergence

## Unification Options

### Option A: Shared SymbolState (TickProcessor delegates to InferenceBridge)

**Approach**: TickProcessor holds a reference to (or embeds) a single
`RustInferenceBridge`, sharing the same `SymbolState` buffers.

**Pros**: One source of truth for z-score buffers. Preview/checkpoint works.

**Cons**: Rust ownership model makes shared mutable state difficult. Would need
`Rc<RefCell<>>` or `Arc<Mutex<>>`, adding overhead. The InferenceBridge is a
`#[pyclass]` owned by Python -- Rust cannot hold a reference to it without
unsafe lifetime hacks.

**Feasibility**: Low. Architectural mismatch with PyO3 ownership model.

### Option B: TickProcessor Outputs Full Signal (bypass decide())

**Approach**: Move z-clamp, regime deadzone, and all force exits into Rust inside
TickProcessor. Python's `decide()` becomes a thin passthrough.

**Pros**: Maximum performance (~80us). Single code path.

**Cons**: Very large scope. Force exits require snapshot access (ATR values,
position P&L, 4h signals) that currently live in Python. Would need to port
`AlphaDecisionModule.decide()` (~300 lines of evolving logic) to Rust. High
maintenance burden -- every new exit rule would need Rust implementation.

**Feasibility**: Low. Too much evolving Python logic to replicate in Rust.

### Option C: TickProcessor Does Features + Predict Only, Z-Score in Python

**Approach**: Strip the signal pipeline from TickProcessor entirely. It outputs only
`raw_prediction` (ensemble score). Python's existing `SignalDiscretizer` +
`AlphaDecisionModule` handles z-score, discretization, and exits.

**Pros**: Clean separation. TickProcessor becomes a "feature engine + predictor"
accelerator. No z-score buffer divergence (only one buffer exists). All signal
logic stays in Python where it evolves rapidly.

**Cons**: Loses some of the 80us advantage -- Python still does z-score +
constraints (~50us). Net improvement ~100us vs current 200us (feature computation
and ML prediction are the expensive Rust parts).

**Feasibility**: High. Minimal changes needed.

### Option D: TickProcessor Outputs Raw Prediction, Python Handles Rest (Recommended)

**Approach**: Same as Option C but with a refined interface:

1. TickProcessor keeps: `push_bar()` -> `get_features()` -> `predict_ensemble()`
2. TickProcessor drops: `apply_signal_pipeline()`, `predict_short()`,
   `bridge_states` (z-score buffers)
3. New PyO3 method: `compute_prediction(symbol, bar_data) -> (raw_score, features_dict)`
4. Python pipeline: `raw_score` -> `RustInferenceBridge.zscore_normalize()` ->
   `SignalDiscretizer` -> `AlphaDecisionModule.decide()`

**Pros**:
- Eliminates z-score divergence (single buffer in InferenceBridge)
- Keeps the expensive work in Rust (features ~60us, ML predict ~15us = ~75us)
- Python only adds z-score + constraints (~25us) + decide() (~20us) = ~120us total
- 40% improvement over current 200us
- All signal logic stays in Python (easy to iterate)
- Checkpoint/restore works unchanged
- Short model prediction can also use this pattern

**Cons**:
- Not as fast as full-Rust 80us path
- Still crosses Python/Rust boundary for z-score

**Feasibility**: High. This is the recommended approach.

## Recommended Implementation Plan (Option D)

### Phase 1: Refactor TickProcessor to "Predict Only" Mode

1. Add a new method to `RustTickProcessor`:
   ```rust
   fn compute_raw(symbol, close, volume, high, low, open, hour_key, ts)
       -> (f64, HashMap<String, f64>)  // (raw_score, features)
   ```
2. This method calls `push_bar()` + `get_features()` + `predict_ensemble()` only
3. Does NOT call `apply_signal_pipeline()` or update `bridge_states`
4. Returns raw ensemble score + feature dict for Python consumption

### Phase 2: Wire Into Python Pipeline

1. In `engine/coordinator.py` or `engine/feature_hook.py`, replace the current
   multi-call sequence (push_bar + get_features + predict separately) with a
   single `compute_raw()` call
2. Feed `raw_score` into existing `SignalDiscretizer.discretize()` unchanged
3. Feed `features_dict` into existing state pipeline unchanged

### Phase 3: Remove Dead Code

1. Remove `apply_signal_pipeline()` and `predict_short()` from TickProcessor
2. Remove `bridge_states` field (z-score buffers)
3. Remove `CfgSnapshot` dependency from TickProcessor
4. Keep `RustInferenceBridge` as the sole z-score owner

### Phase 4: Validate

1. Compare raw predictions between old multi-call path and new `compute_raw()`
2. Verify z-scores match (they should -- same InferenceBridge buffer)
3. Benchmark: expect ~120us total (75us Rust + 45us Python) vs current 200us
4. Run backtest to confirm signal equivalence

### Estimated Effort

- Phase 1: 2-3 hours (Rust refactor + new PyO3 method)
- Phase 2: 1-2 hours (Python wiring)
- Phase 3: 1 hour (cleanup)
- Phase 4: 2-3 hours (validation + backtest)
- **Total: ~1 day**

### Expected Outcome

| Metric | Current | After Unification |
|--------|---------|-------------------|
| Latency per bar | ~200us | ~120us |
| Z-score buffers | 2 (divergent) | 1 (single truth) |
| Signal code paths | 2 (Rust + Python) | 1 (Python only) |
| Maintainability | Low (dual paths) | High (single path) |
| Preview/checkpoint | Python only | Works everywhere |

### Risks

- **Performance regression if Rust feature engine has bugs**: Mitigated by Phase 4
  comparison testing against current Python-orchestrated path.
- **Model reload**: TickProcessor currently loads models at construction time.
  Hot-reload (SIGHUP) would need to rebuild TickProcessor or add a `reload_models()`
  method.
- **State store coupling**: TickProcessor currently embeds `RustStateStore`. If we
  only use it for predict, we can decouple state store entirely, but this means
  the existing `process_tick()` state-update path is also removed. Need to ensure
  the Python state pipeline remains the source of truth.

## Files Referenced

- `engine/rust/tick_processor.rs` -- TickProcessor struct + process_tick
- `engine/rust/tick_signal.inc.rs` -- TickProcessor signal pipeline (z-score + constraints)
- `decision/rust/inference_bridge.rs` -- RustInferenceBridge (Python-side z-score)
- `decision/signals/alpha_signal.py` -- SignalDiscretizer (Python z-clamp + regime filter)
- `decision/modules/alpha.py` -- AlphaDecisionModule (force exits, alignment)
