# Decision Layer

This package implements an institutional-grade decision layer:

`StateSnapshot -> DecisionOutput (targets/intents/orders + explain)`

## Contracts
- **No IO / no side effects** in `DecisionEngine.run()`.
- **Deterministic**: same snapshot + same config => same output (including IDs).
- **Explainable**: every decision produces a stable explain record.
- **Composable**: plug-in signals / allocators / sizing / execution_policy.

## Typical integration
- `engine.DecisionBridge` calls a `DecisionModule.decide(snapshot)` which can wrap `DecisionEngine`
  and emit `event.types.OrderEvent` or `execution.models.commands.SubmitOrderCommand`.

## Invariants
- Output orders must be validated (qty > 0, price >= 0 if provided).
- Risk overlay can force output to be empty (halted/blocked).
