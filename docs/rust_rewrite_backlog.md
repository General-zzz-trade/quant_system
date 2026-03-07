# Rust Rewrite Backlog

## Near-Term Kernel Targets

### P0

- `state.portfolio`
- `state.risk`
- `state.reducers.portfolio`
- `state.reducers.risk`
- snapshot/store parity for derived state

### P1

- coarse-grained Rust coordinator entrypoint
- Python `EngineCoordinator` reduced to shell/wiring
- replay loop parity under Rust coordinator

### P2

- `decision.types`
- `decision.engine`
- `decision.gating`
- `decision.sizing`
- `decision.allocators`
- `decision.execution_policy`

### P3

- `execution.ingress`
- `execution.safety`
- `execution.state_machine`
- `execution.reconcile`
- `execution.store`

### P4

- backtest kernel end-to-end
- delete Python reducer fallbacks
- delete duplicate event/state contracts
