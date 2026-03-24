# Wiring Truth — Module Integration Status

> 更新时间: 2026-03-22
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
| RustFeatureEngine | rust/ | alpha_runner.py `__init__` | 120 features, always Rust |
| RustInferenceBridge | rust/ | alpha_runner.py `process_bar` | z-score + deadzone + min-hold + max-hold |
| RustRiskEvaluator | rust/ | alpha_runner.py `process_bar` | Drawdown + leverage check |
| RustKillSwitch | rust/ | alpha_runner.py (shared) | Global emergency stop |
| RustOrderStateMachine | rust/ | alpha_runner.py | Order lifecycle tracking |
| RustCircuitBreaker | rust/ | alpha_runner.py | 3-failure/120s backoff |
| RustStateStore | rust/ | alpha_runner.py (shared) | Position truth |
| CompositeRegimeDetector | regime/composite.py | alpha_runner.py `_check_regime` | **仅 BTC** (`use_composite_regime=True`) |
| RegimeParamRouter | regime/param_router.py | alpha_runner.py `_apply_composite_regime` | **仅 BTC**: params → deadzone/min_hold 闭环 |
| ADX(14) | features/enriched_computer.py | RustFeatureEngine feeds feat_dict | 28-bar warmup; feeds TrendRegimeDetector via feat_dict |
| PortfolioCombiner | scripts/ops/portfolio_combiner.py | run_bybit_alpha.py | AGREE ONLY for ETH 1h+15m |
| MultiTFConfluenceGate | runner/gates/multi_tf_confluence_gate.py | alpha_runner.py `_evaluate_gates` | 1h vs 4h alignment; prefers model IC=0.29 over indicator voting IC=0.05 |
| LiquidationCascadeGate | runner/gates/liquidation_cascade_gate.py | alpha_runner.py `_evaluate_gates` | Cascade protection (zscore>3 block) |
| CarryCostGate | runner/gates/carry_cost_gate.py | alpha_runner.py `_evaluate_gates` | Funding+basis carry cost adjustment |
| OnlineRidge | alpha/online_ridge.py | alpha_runner.py `_ensemble_predict` | RLS incremental weight updates (λ=0.997) |
| OnlineRidge (4h) | alpha/online_ridge.py | run_bybit_alpha.py `enable_online_ridge` | RLS for 4h runners |
| OptionsFlowComputer | features/options_flow.py | enriched_computer.py `on_bar` | 7 Deribit options features |
| BTCUSDT_4h Runner | alpha_runner.py | run_bybit_alpha.py | **WIRED**: Strategy H primary, cap 15% |
| ETHUSDT_4h Runner | alpha_runner.py | run_bybit_alpha.py | **WIRED**: Strategy H primary, cap 10% |
| SIGHUP Hot-Reload | run_bybit_alpha.py | signal handler | **WIRED**: <200ms reload |
| Per-Runner Checkpoint | alpha_runner.py | `_runner_key` | **WIRED**: per-runner state persistence |
| Dynamic Leverage | alpha_runner.py | DD-based scaling | **WIRED**: drawdown-based leverage adjustment |
| Vol-Adaptive Deadzone | alpha_runner.py | rv/vol_median | **WIRED**: realized vol / median vol scaling |
| BB Entry Scaler | alpha_runner.py | `_compute_entry_scale` | **WIRED**: Bollinger Band entry scaling |
| 4h Z-Score Stop | alpha_runner.py | reads `_consensus_signals` | **WIRED**: cross-timeframe stop signal |
| VenueRouter | execution/adapters/venue_router.py | run_bybit_alpha.py | **WIRED**: optional via `HYPERLIQUID_PRIVATE_KEY` |
| IC Decay Monitor | monitoring/ic_decay_monitor.py | systemd timer | **WIRED**: automated IC decay alerting |
| Signal Reconciliation | scripts/ops/signal_reconcile.py | manual / cron | **WIRED**: live vs expected signal comparison |
| Shadow A/B Compare | scripts/ops/shadow_compare.py | manual / cron | **WIRED**: shadow mode A/B signal comparison |
| ETHRegimeProxy | regime/eth_regime_proxy.py | Available, not yet wired | BTC regime → ETH param routing |
| PipelineMetrics | monitoring/pipeline_metrics.py | Available, not yet wired | Thread-safe pipeline counters |
| InputValidation | core/validation.py | Available, not yet wired | NaN/Inf price/qty validation |
| DailyReconciliation | scripts/ops/daily_reconciliation.py | Manual / cron | Live vs backtest signal comparison |

### Feature Pipeline (Enriched Features)

| Version | Features | File | Wired Via | Notes |
|---------|----------|------|-----------|-------|
| V1-V19 | ADX, OI, dominance, DVOL, options | features/enriched_computer.py | RustFeatureEngine | 157 enriched features |
| V22 | IV features | features/batch_feature_engine.py | batch pipeline | **WIRED**: T-1 shifted |
| V23 | Stablecoin features | features/batch_feature_engine.py | batch pipeline | **WIRED**: T-1 shifted |
| V24 | ETF Volume features | features/batch_feature_engine.py | batch pipeline | **WIRED**: T-1 shifted |

### Venue Adapters

| Venue | Status | File | Notes |
|-------|--------|------|-------|
| Bybit | **WIRED** (primary) | execution/adapters/ | REST V5 + WS |
| Hyperliquid | **WIRED** (optional) | execution/adapters/hyperliquid/ | Via VenueRouter, needs `HYPERLIQUID_PRIVATE_KEY` |
| Binance | **WIRED** (data only) | scripts/data/ | OI/klines/funding data source |
| Polymarket | **WIRED** (standalone) | polymarket/ | CLOB collector + RSI dry-run |

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
| RustFeatureEngine | rust/ | engine/feature_hook.py | 120 features, always Rust (no Python fallback) |
| RustInferenceBridge | rust/ | feature_hook + inference_bridge | z-score + deadzone + min-hold + max-hold |

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
