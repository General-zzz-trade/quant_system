# Wiring Truth — Module Integration Status

> 更新时间: 2026-03-17
> 目标: 记录每个功能模块是否已接入生产路径
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

## 两条生产路径

当前系统有两条独立的生产路径，接线状态需要分别追踪：

| 路径 | 入口 | 部署 | 状态 |
|------|------|------|------|
| **AlphaRunner** | `scripts/ops/run_bybit_alpha.py` | systemd `bybit-alpha.service` | **当前活跃** |
| **LiveRunner** | `runner/live_runner.py` | docker `quant-framework` / `quant-runner.service` | 可用（非默认） |

## Production-Wired Modules

### AlphaRunner 路径（当前生产主路径）

| Module | File | Wired Via | Notes |
|--------|------|-----------|-------|
| RustFeatureEngine | ext/rust/ | alpha_runner.py `__init__` | 120 features, always Rust |
| RustInferenceBridge | ext/rust/ | alpha_runner.py `process_bar` | z-score + deadzone + min-hold + max-hold |
| RustRiskEvaluator | ext/rust/ | alpha_runner.py `process_bar` | Drawdown + leverage check |
| RustKillSwitch | ext/rust/ | alpha_runner.py (shared) | Global emergency stop |
| RustOrderStateMachine | ext/rust/ | alpha_runner.py | Order lifecycle tracking |
| RustCircuitBreaker | ext/rust/ | alpha_runner.py | 3-failure/120s backoff |
| RustStateStore | ext/rust/ | alpha_runner.py (shared) | Position truth |
| CompositeRegimeDetector | regime/composite.py | alpha_runner.py `_check_regime` | **仅 BTC** (`use_composite_regime=True`) |
| RegimeParamRouter | regime/param_router.py | alpha_runner.py `_apply_composite_regime` | **仅 BTC**: params → deadzone/min_hold 闭环 |
| ADX(14) | features/enriched_computer.py | RustFeatureEngine feeds feat_dict | 28-bar warmup; feeds TrendRegimeDetector via feat_dict |
| PortfolioCombiner | scripts/ops/portfolio_combiner.py | run_bybit_alpha.py | AGREE ONLY for ETH 1h+15m |

### LiveRunner 路径（框架层）

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
| RustInferenceBridge | ext/rust/ | feature_hook + inference_bridge | z-score + deadzone + min-hold + max-hold |

## Partially Wired / Optional

| Module | File | Status | Notes |
|--------|------|--------|-------|
| StrategyRegistry | strategies/registry.py | Utility | Not in critical path; for strategy discovery |
| AvellanedaStoikovMaker | polymarket/strategies/maker_5m.py | Standalone | Needs manual startup; not in systemd |
| InventoryManager | polymarket/strategies/inventory_manager.py | Standalone | Paired with maker strategy |

## Configuration Flags

### AlphaRunner (SYMBOL_CONFIG in scripts/ops/config.py)

| Flag | Default | Effect |
|------|---------|--------|
| use_composite_regime | False | Enables CompositeRegimeDetector + ParamRouter; **仅 BTC 启用** |

### LiveRunner (LiveRunnerConfig in runner/config.py)

| Flag | Default | Effect |
|------|---------|--------|
| enable_regime_gate | True | Wraps decision modules with RegimeAwareDecisionModule |
| enable_regime_sizing | False | Activates: RegimePositionSizer, StagedRiskManager, ParamRouter |
| enable_burnin_gate | False | Blocks live trading until burn-in phases pass |
| enable_alpha_health | True | IC tracking + position scaling per symbol |
| enable_portfolio_risk | False | Cross-asset gross/net/concentration limits |
| initial_equity | 500.0 | Starting equity for StagedRiskManager stage determination |
