# Wiring Truth — Module Integration Status

> 更新时间: 2026-03-17
> 目标: 记录每个功能模块是否已接入生产路径
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

## Production-Wired Modules

| Module | File | Wired Via | Notes |
|--------|------|-----------|-------|
| CompositeRegimeDetector | regime/composite.py | decision/regime_bridge.py | Default detector in RegimeAwareDecisionModule |
| RegimeParamRouter | regime/param_router.py | decision/regime_bridge.py | Active when enable_regime_sizing=True |
| VolatilityRegimeDetector | regime/volatility.py | regime/composite.py | Used internally by CompositeRegimeDetector |
| TrendRegimeDetector | regime/trend.py | regime/composite.py | Used internally by CompositeRegimeDetector |
| RegimePositionSizer | portfolio/regime_sizer.py | monitoring_builder + gate_chain | RegimeSizerGate in gate chain |
| StagedRiskManager | risk/staged_risk.py | monitoring_builder + gate_chain | StagedRiskGate; equity-based 5-stage risk |
| BurninGate | runner/preflight.py | portfolio_builder.py | Blocks live trading until phases complete |
| ADX(14) | features/enriched_computer.py | _ADXTracker incremental | 28-bar warmup; feeds TrendRegimeDetector |
| RustFeatureEngine | ext/rust/ | engine/feature_hook.py | 120 features, always Rust (no Python fallback) |
| RustInferenceBridge | ext/rust/ | alpha_runner.py | z-score + deadzone + min-hold + max-hold |
| 12x Rust Components | ext/rust/ | scripts/ops/alpha_runner.py | All 12 in production (see CLAUDE.md) |

## Partially Wired / Optional

| Module | File | Status | Notes |
|--------|------|--------|-------|
| StrategyRegistry | strategies/registry.py | Utility | Not in critical path; for strategy discovery |
| AvellanedaStoikovMaker | polymarket/strategies/maker_5m.py | Standalone | Needs manual integration into runner |
| InventoryManager | polymarket/strategies/inventory_manager.py | Standalone | Paired with maker strategy |

## Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| enable_regime_gate | True | Wraps decision modules with RegimeAwareDecisionModule |
| enable_regime_sizing | False | Activates: RegimePositionSizer, StagedRiskManager, ParamRouter |
| enable_burnin_gate | False | Blocks live trading until burn-in phases pass |
| enable_alpha_health | True | IC tracking + position scaling per symbol |
| enable_portfolio_risk | False | Cross-asset gross/net/concentration limits |
| initial_equity | 500.0 | Starting equity for StagedRiskManager stage determination |
