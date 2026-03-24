# Rust Replacement Matrix

> 状态: 当前 Rust 第二阶段迁移边界与替换优先级说明
> 更新时间: 2026-03-12
> 该文档是迁移规划，不是当前运行时真相源；运行时现状请优先参考 [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

## Purpose

This document defines what should be rewritten to Rust, what should stay in Python,
and what should remain hybrid during the migration.

The goal is not "rewrite all Python".
The goal is "make Rust the only runtime kernel implementation".

## Decision Rule

Rewrite to Rust when a module is:

- deterministic
- CPU-bound or stateful
- hot-path at runtime
- part of event/state/risk/execution causality
- easier to verify once contracts are explicit

Keep in Python when a module is:

- mostly I/O glue
- built around Python-only SDKs
- operational tooling
- research or experimentation
- model training / notebook / ad-hoc analysis

## Tier 1: Rewrite To Rust

These are the kernel. They should end up with Rust as the only implementation.

### `core/`

Rewrite:

- runtime DTOs
- fixed-point / decimal helpers
- deterministic clocks
- stable serialization contracts

Reason:

- these define kernel boundaries
- they must be language-neutral

Python should only keep compatibility wrappers during migration.

### `engine/`

Rewrite:

- `dispatcher.py`
- `pipeline.py`
- `coordinator.py`
- `replay.py`
- `guards.py`
- deterministic parts of `loop.py` and `tick_engine.py`

Reason:

- this is the runtime causality chain
- the kernel is not complete while Python still owns dispatch and coordination

Keep in Python for now:

- `bootstrap.py`
- `feature_hook.py`
- `module_reloader.py`

These are shell/integration concerns, not kernel truth.

### `event/`

Rewrite:

- `types.py`
- `header.py`
- `validators.py`
- `codec.py`
- `checkpoint.py`
- `store.py`
- replay/checkpoint event contracts

Reason:

- event identity and validation must be canonical
- event storage / replay should not depend on Python object shape

Keep hybrid temporarily:

- `runtime.py`
- `bus.py`
- `security.py`

These still touch Python callback wiring and integration policy.

### `state/`

Rewrite:

- `market.py`
- `account.py`
- `position.py`
- `portfolio.py`
- `risk.py`
- `snapshot.py`
- `reducers/*`
- `store.py`
- `diff.py`
- `versioning.py`

Reason:

- state is the center of the kernel
- reducers and snapshots must have one owner only

Current status:

- `market/account/position` reducers are already on the Rust path
- `portfolio/risk` still need to move into the main pipeline path

### `risk/`

Rewrite:

- `kill_switch.py`
- `interceptor.py`
- `decisions.py`
- `aggregator.py`
- `correlation_gate.py`
- `rules/*`
- deterministic portfolio risk checks

Reason:

- these are runtime gates
- they must be deterministic and cheap

Keep in Python:

- `margin_monitor.py`
- `reporting/*`

These are more operational / monitoring than kernel.

### `execution/`

Rewrite:

- `models/*`
- `ingress/*`
- `safety/*`
- `state_machine/*`
- `reconcile/*`
- `store/*`
- deterministic parts of `routing/*`
- deterministic parts of `sim/*`
- request-id / hashing / signer / validation helpers in `bridge/*`

Reason:

- this is the most critical runtime correctness surface
- ordering, idempotency, and state transitions should not remain in Python long-term

Keep in Python:

- `adapters/binance/*` transport and SDK wiring
- `adapters/generic/*`
- `live/*`
- operational pieces in `observability/*`

Those depend on exchange connectivity and ecosystem glue.

### `decision/`

Rewrite:

- `types.py`
- `engine.py`
- `composer.py`
- `gating.py`
- `selectors.py`
- `ml_decision.py`
- `allocators/*`
- `execution_policy/*`
- `intents/*`
- `risk_overlay/*`
- `sizing/*`
- deterministic candidate ranking / filtering

Reason:

- target-position and order derivation is kernel logic
- deterministic decision transforms belong in Rust

Keep in Python:

- Python ML wrappers
- sklearn/lightgbm/xgboost-facing glue
- meta-learning research code

Examples:

- `signals/ml/model_runner.py`
- `ensemble/meta_learner.py`

### `features/`

Rewrite:

- `rolling.py`
- `technical.py`
- `cross_sectional.py`
- `multi_timeframe.py`
- `multi_resolution.py`
- `live_computer.py`
- `enriched_computer.py`
- `batch_feature_engine.py`
- deterministic feature pipeline code

Reason:

- feature computation is hot-path and deterministic
- this is one of the highest ROI Rust targets

Keep in Python:

- store wrappers
- research-oriented feature auto-generation

Examples:

- `offline_store/*`
- `online_store/*`
- `auto/*`

### `runner/backtest*`

Rewrite:

- `runner/backtest_runner.py`
- deterministic backtest loop
- event stepping
- PnL accounting / metrics kernel

Reason:

- this is high-ROI and low external dependency
- parity is measurable

## Tier 2: Hybrid Or Replace Later

These are allowed to stay Python for a while, but should not be mistaken for
final kernel design.

### `runner/`

Keep Python shell:

## Current Phase 4 Candidates

These are the highest-signal candidates for the next Rust migration wave based
on current code ownership, available Rust modules, and testability.

### Candidate 1: Execution Timeout / Sequencing Helpers

Python today:

- [`execution/safety/timeout_tracker.py`](/quant_system/execution/safety/timeout_tracker.py)
- [`execution/safety/out_of_order_guard.py`](/quant_system/execution/safety/out_of_order_guard.py)

Rust already available:

- `rust/src/sequence_buffer.rs`
- `rust/src/fill_dedup.rs`
- `rust/src/payload_dedup.rs`

Why this is a good target:

- pure logic
- explicit contracts already documented in `docs/execution_contracts.md`
- recovery tests already exist for timeout / duplicate / late fill paths

### Candidate 2: Dispatcher Routing Core

Python today:

- [`engine/dispatcher.py`](/quant_system/engine/dispatcher.py)

Rust already available:

- `rust/src/route_match.rs`
- `rust/src/event_types.rs`
- `rust/src/event_validators.rs`

Why this is a good target:

- deterministic routing logic
- hot-path on every event
- parity is measurable through live/replay routing tests

Keep in Python for now:

- callback wiring
- handler registration
- integration shell around the route result

### Candidate 3: Order Lifecycle Projection / Reconcile Kernel

Python today:

- [`execution/state_machine/projection.py`](/quant_system/execution/state_machine/projection.py)
- [`execution/state_machine/reconciliation_rules.py`](/quant_system/execution/state_machine/reconciliation_rules.py)

Rust already available:

- `rust/src/order_state_machine.rs`
- `rust/src/execution_store.rs`

Why this is a good target:

- deterministic state derivation
- already close to kernel semantics
- directly affects replay / reconcile correctness

### Candidate 4: Backtest Signal Constraint Kernel

Python today:

- [`decision/backtest_module.py`](/quant_system/decision/backtest_module.py)
- [`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py)

Rust already available:

- `rust/src/inference_bridge.rs`
- `rust/src/ml_decision.rs`
- `rust/src/backtest_engine.rs`

Why this is a good target:

- live side already leans on Rust constraints
- parity tests now exist for `deadzone`, `min_hold`, `trend_hold`, `monthly_gate`, `vol_target`
- moving more backtest constraint logic to Rust reduces live/backtest divergence risk

## Not Recommended Yet

These areas should not be the next migration wave even if they eventually move:

- [`runner/live_runner.py`](/quant_system/runner/live_runner.py)
- [`engine/coordinator.py`](/quant_system/engine/coordinator.py)
- exchange transport glue under [`execution/adapters`](/quant_system/execution/adapters)

Reason:

- they are orchestration-heavy
- they mix IO and dependency wiring
- parity is harder to lock than for the deterministic kernels above

- `live_runner.py`
- `paper_runner.py`
- `replay_runner.py`
- `preflight.py`
- `graceful_shutdown.py`

Reason:

- mostly orchestration, CLI, lifecycle, ops behavior

Possible later Rust migration:

- only after event/state/execution kernel is fully stabilized

### `event/runtime.py` and callback wiring

Keep hybrid until:

- actor/permission model is frozen
- Python callback ecosystem is reduced

### `execution/bridge/*`

Split:

- deterministic request id / rate limiting / validation -> Rust
- retries / timeouts / transport orchestration -> Python

### `portfolio/`

Not part of the runtime kernel by default.

Only move to Rust if:

- the portfolio optimizer becomes part of live runtime
- latency or determinism becomes a real issue

Otherwise keep in Python.

### `regime/`

Split by function:

- deterministic runtime gating can move to Rust
- offline model selection / experimentation should stay Python

## Tier 3: Keep In Python

These do not need Rust replacement.

### Research / Experimentation

- `research/*`
- `scripts/*`
- notebooks and experiment reporting

Reason:

- low ROI to rewrite
- iteration speed matters more than runtime latency

### Model Training / Inference Wrappers

- `alpha/training/*`
- most of `alpha/models/*`
- model registry glue around Python ML libraries

Reason:

- the heavy work is already done by native libraries
- the wrapper layer is not the bottleneck

### Data / External IO

- `data/collectors/*`
- `data/live/*`
- `data/loaders/*`
- exchange / REST / websocket glue

Reason:

- network I/O dominates
- vendor SDK constraints dominate

### Monitoring / Ops / Infra

- `monitoring/*`
- `infra/config/*`
- `infra/logging/*`
- `infra/metrics/*`
- `infra/tracing/*`
- `deploy/*`

Reason:

- operational glue is better kept flexible

### Tests

Keep most tests in Python.

Add Rust unit tests only for:

- reducers
- state store
- pipeline/coordinator kernels
- event validation
- execution state machine

## Recommended Migration Order

### Phase A: Freeze Contracts

Define and lock:

- `Event`
- `EventHeader`
- `StateSnapshot`
- `PipelineResult`
- `DecisionOutput`
- `OrderSpec`
- `RiskResult`

### Phase B: Finish Event + State Ownership

Complete:

- Rust event/header/validator ownership
- Rust `portfolio/risk` reducers on the main path
- Rust snapshot/state-store ownership

Exit condition:

- Python no longer owns state transitions

### Phase C: Finish Engine Ownership

Complete:

- Rust dispatcher
- Rust pipeline
- Rust coordinator
- Rust replay loop

Exit condition:

- Python becomes orchestration shell only

### Phase D: Finish Decision + Execution Ownership

Complete:

- deterministic decision transforms
- order construction
- execution safety
- ingress sequencing
- state machine and reconcile core

Exit condition:

- order causality is entirely inside Rust

### Phase E: Finish Backtest Ownership

Complete:

- Rust backtest event loop
- Rust accounting path
- feature + decision + execution simulation parity

Exit condition:

- Python backtest runner becomes a shell around Rust

### Phase F: Delete Fallbacks

Delete:

- Python reducer fallbacks
- duplicate pipeline code
- dual contract adapters that only exist for migration

This is the real completion point.

## Immediate Next Steps

The next practical Rust targets are:

1. move `state.portfolio` and `state.risk` into the Rust pipeline path
2. replace Python `EngineCoordinator` internals with a coarse-grained Rust engine entrypoint
3. move deterministic `decision` transforms into Rust DTO-based APIs
4. move `execution.state_machine`, `execution.ingress`, and `execution.safety` into Rust

## What Not To Do

Do not spend time rewriting these early:

- notebooks
- training scripts
- deployment manifests
- monitoring dashboards
- websocket client plumbing
- Python-only model wrappers

That work looks large, but it does not move the kernel boundary.
