# Rust Core Rewrite

## Goal

Make Rust the only implementation of the runtime kernel.

Python should remain only as:

- CLI and config loading
- exchange SDK / websocket integration
- monitoring and ops glue
- research / training / notebooks

The core runtime must no longer depend on Python implementations for:

- event typing and validation
- state mutation and snapshots
- engine dispatch / pipeline / coordination
- risk controls and execution safety
- deterministic decision logic
- backtest kernel
- hot-path feature computation

## Non-Goals

- Rewriting all research scripts to Rust
- Rewriting LightGBM / XGBoost / PyTorch training wrappers first
- Rewriting deploy manifests, alerts, dashboards, or infra scripts
- Replacing exchange connectivity with Rust in phase 1

## Current State

The repository already contains a substantial Rust crate under `ext/rust/`.

What exists today:

- Rust state types and reducers
- Rust pipeline normalization helpers
- Rust risk primitives
- Rust event validators and enums
- Rust feature/backtest/math accelerators
- Rust request-id / signer / parsing hot paths

What is still true today:

- Python remains the production orchestration path
- most Rust usage is optional acceleration or adapter-level delegation
- fallback logic keeps Python as the real source of truth
- the runtime boundary is still dynamic and Python-shaped (`Any`, ad-hoc objects, `SimpleNamespace`)

That means the project is not starting Rust migration from zero.
It is finishing an incomplete migration and removing the fallback model.

### Status Update: 2026-03-07

Implemented in the runtime path:

- `engine.pipeline` now prefers a coarse-grained Rust kernel boundary for event normalization.
- `StatePipeline` now defaults to Rust-backed market/account/position reducers via Python adapters.
- `engine.dispatcher` now uses Rust for both dedup and route classification.
- a locally built `_quant_hotpath` package is staged at the repository root so project-local runs load the new boundary first.

Still not fully migrated:

- `EngineCoordinator` is still Python orchestration.
- derived `portfolio` / `risk` state is not yet owned by the Rust pipeline path.
- Python fallback paths still exist and should be deleted only after broader parity coverage.

## Scope Definition

The following directories are part of the kernel rewrite.

### Rewrite to Rust

- `core/`
- `engine/`
- `event/`
- `state/`
- `risk/`
- `decision/` for deterministic decision modules and risk overlays
- `execution/` for ingress, routing, dedup, sequencing, state machine, reconcile core, safety
- `features/` for runtime feature computation and backtest feature engine
- `runner/backtest*` for deterministic simulation kernel

### Keep Python Shell Initially

- `runner/live_runner.py`
- `runner/paper_runner.py`
- `monitoring/`
- `infra/config/`
- `infra/logging/`
- exchange SDK clients under `execution/adapters/*` that depend on Python ecosystems

### Keep in Python Long-Term

- `research/`
- `scripts/`
- training wrappers in `alpha/models/*` around LightGBM/XGBoost/PyTorch
- ops / deploy / dashboards

## Hard Rules

1. Rust owns state.
2. Python does not mutate state directly.
3. Python does not implement fallback reducers once Rust parity is accepted.
4. Cross-language calls must be coarse-grained, not per-field and not per-small helper.
5. Event / state / decision / order contracts must be explicit DTOs, not dynamic objects.

## Target Architecture

Create a Rust workspace with these crates.

### `quant-kernel-types`

Owns all runtime DTOs:

- events
- headers
- states
- snapshots
- risk decisions
- order specs
- decision outputs

### `quant-kernel-engine`

Owns:

- dispatcher
- dedup
- sequencing
- pipeline
- coordinator
- backtest event loop

### `quant-kernel-state`

Owns:

- reducers
- snapshots
- checkpoint serialization
- state store contracts

### `quant-kernel-risk`

Owns:

- kill switch
- circuit breaker
- order limiter
- position / leverage / drawdown rules
- portfolio gates

### `quant-kernel-execution`

Owns:

- ingress sequencing
- idempotency
- order lifecycle state machine
- reconcile core
- execution safety policies

### `quant-kernel-features`

Owns:

- rolling windows
- technical indicators
- feature engine
- multi-timeframe aggregation
- selection hot paths

### `quant-kernel-decisions`

Owns deterministic runtime decisions:

- ML score -> target / order transform
- regime gates
- sizing and overlays

### `quant-bindings-py`

Thin PyO3 layer only.

No business logic here.
Only conversions and coarse-grained entry points.

## Required Interface Changes

Before the rewrite can complete, the Python runtime must stop passing dynamic objects into the kernel.

The following contracts need to become stable and language-neutral:

- `Event`
- `EventBatch`
- `StateSnapshot`
- `PipelineResult`
- `DecisionInput`
- `DecisionOutput`
- `OrderSpec`
- `RiskCheckResult`
- `BacktestConfig`
- `BacktestResult`

Use JSON-compatible and msgpack-friendly schemas for boundary stability.
Avoid exposing Python class instances inside Rust APIs.

## What To Replace First

### Phase 0: Freeze the Boundary

Deliverables:

- document canonical kernel DTOs
- define module ownership
- remove ad-hoc `Any` usage from runtime boundaries
- require `_quant_hotpath` in CI for kernel tests

Exit criteria:

- core runtime interfaces are explicit
- new Python-only kernel logic is blocked

### Phase 1: Event + State as Rust Source of Truth

Replace:

- event enums
- validators
- state structs
- reducers
- snapshots
- checkpoint serialization

Use existing work in:

- `ext/rust/src/state_types.rs`
- `ext/rust/src/state_reducers.rs`
- `ext/rust/src/event_types.rs`
- `ext/rust/src/event_validators.rs`

Exit criteria:

- Python reducers are wrappers only or removed
- replay and persistence use Rust-owned state transitions

### Phase 2: Engine Core

Replace:

- dispatcher
- pipeline
- coordinator
- loop/backtest event pump

Use existing work in:

- `ext/rust/src/pipeline.rs`
- `ext/rust/src/dedup_guard.rs`
- `ext/rust/src/route_match.rs`
- `ext/rust/src/engine_guards.rs`

Gap to close:

- current Python coordinator is still the real orchestrator
- Rust pipeline is only partially used

Exit criteria:

- the authoritative event flow is Rust
- Python no longer routes events between kernel modules

### Phase 3: Risk + Execution Safety

Replace:

- kill switch
- circuit breaker
- order limiter
- risk gate
- duplicate guards
- sequence buffers
- order state machine
- reconcile core

Use existing work in:

- `ext/rust/src/risk_engine.rs`
- `ext/rust/src/fill_dedup.rs`
- `ext/rust/src/sequence_buffer.rs`

Exit criteria:

- no Python-only risk decision in live path
- execution safety remains deterministic under replay

### Phase 4: Runtime Feature + Decision Core

Replace:

- feature engine
- technical indicators
- runtime feature transforms
- deterministic decision modules
- score-to-order translation

Use existing work in:

- `ext/rust/src/feature_engine.rs`
- `ext/rust/src/technical.rs`
- `ext/rust/src/multi_timeframe.rs`
- `ext/rust/src/ml_decision.rs`

Exit criteria:

- runtime feature computation is Rust-owned
- deterministic decisions do not cross the boundary multiple times per event

### Phase 5: Backtest Kernel

Replace:

- backtest event loop
- fill simulation kernel
- pnl/equity accumulation
- deterministic replay

Use existing work in:

- `ext/rust/src/backtest_engine.rs`

Exit criteria:

- backtest and replay share the same Rust state transition rules as live

### Phase 6: Remove Python Fallbacks

After parity is proven:

- delete Python reducers
- delete Python risk core
- delete Python dispatcher/pipeline implementations
- delete optional fallback toggles for core modules

Exit criteria:

- kernel cannot run without Rust
- Python runtime is orchestration only

## Modules To Target Immediately

The first concrete files to retire or demote are:

- `engine/pipeline.py`
- `engine/dispatcher.py`
- `engine/coordinator.py`
- `state/reducers/*`
- `state/store.py` for checkpoint/store adapter reshaping
- `risk/kill_switch.py`
- `risk/aggregator.py`
- `risk/rules/*`
- `decision/ml_decision.py`
- `runner/backtest_runner.py`

These are the highest-value files because they still define runtime truth in Python.

## What Not To Do

- Do not port one helper at a time and keep Python as the orchestrator forever.
- Do not keep dual implementations once parity is accepted.
- Do not expose dozens of tiny PyO3 functions and call them per event step.
- Do not start with exchange SDK rewrites; IO is not the kernel bottleneck.
- Do not start with research code; it will consume time without hardening production.

## Testing Strategy

Every phase must pass all four layers:

- unit parity tests: Rust vs previous Python behavior
- deterministic replay tests: same inputs, same outputs, same state hashes
- integration tests: live/paper/backtest wiring
- benchmark gates: lower latency and stable memory under realistic event rates

Required artifacts:

- golden event log fixtures
- state hash snapshots per replay step
- Rust-only benchmarks in CI
- Python-vs-Rust parity suite kept until fallback deletion

## Required Repo Changes

The repository should be reorganized so the Rust kernel is a first-class product, not just an acceleration folder.

Recommended direction:

- keep `ext/rust/` temporarily as the implementation root
- introduce workspace-style separation inside it
- rename `cpp_*` exports that are now Rust-owned
- update README and build flow to reflect Rust as mandatory for the kernel
- add CI job that builds the wheel before running kernel tests

## Recommended Execution Order

1. freeze DTO contracts
2. make event/state Rust-owned
3. move dispatcher/pipeline/coordinator into Rust
4. move risk/execution safety into Rust
5. move backtest kernel into Rust
6. delete Python fallbacks
7. only then evaluate whether live runtime shell should also move to Rust

## Decision

The right strategy is not "translate all Python to Rust".

The right strategy is:

- define the kernel boundary
- make Rust the single implementation of that boundary
- demote Python to shell code
- delete fallbacks aggressively once parity is proven

This project is already far enough along that a second, parallel rewrite would be wasteful.
The next move is integration and deletion, not another greenfield Rust effort.
