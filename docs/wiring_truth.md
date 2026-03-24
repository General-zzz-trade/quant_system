# Wiring Truth — Module Integration Status

> 更新时间: 2026-03-24
> 目标: 记录每个功能模块是否已接入生产路径
> 上位真相源: [`CLAUDE.md`](/quant_system/CLAUDE.md)

## 生产路径

当前生产路径已统一为框架原生路径 (EngineCoordinator + AlphaDecisionModule):

| 路径 | 入口 | 部署 | 状态 |
|------|------|------|------|
| **Framework-native (Strategy H)** | `runner/alpha_main.py` | systemd `bybit-alpha.service` | **当前活跃** (4 runners, 2 WS) |
| **LiveRunner** | `runner/live_runner.py` | docker `quant-framework` / `quant-runner.service` | 可用（非默认） |
| ~~AlphaRunner~~ | ~~`runner/alpha_runner.py`~~ | — | **DEPRECATED** (kept for rollback) |

## Production-Wired Modules

### Framework-native 路径（当前生产主路径）

Data flow: `Bybit WS kline → MarketEvent → EngineCoordinator.emit() → FeatureComputeHook → StatePipeline → DecisionBridge → AlphaDecisionModule → ExecutionBridge`

| Module | File | Wired Via | Notes |
|--------|------|-----------|-------|
| RustFeatureEngine | {module}/rust/ | engine/feature_hook.py (FeatureComputeHook) | 120+ features, always Rust (no Python fallback) |
| RustInferenceBridge | {module}/rust/ | decision/signals/alpha_signal.py (SignalDiscretizer) | z-score + deadzone + min-hold + max-hold |
| RustRiskEvaluator | {module}/rust/ | runner/gate_chain.py | Drawdown + leverage check |
| RustKillSwitch | {module}/rust/ | runner/gate_chain.py | Global emergency stop |
| RustOrderStateMachine | {module}/rust/ | execution/ | Order lifecycle tracking |
| RustCircuitBreaker | {module}/rust/ | execution/ | 3-failure/120s backoff |
| RustStateStore | {module}/rust/ | engine/pipeline.py (StatePipeline) | Position truth (Rust heap, Python gets snapshots) |
| RustGateChain | {module}/rust/ | runner/gate_chain.py | 9 gate types in single FFI call |
| CompositeRegimeDetector | regime/composite.py | decision/regime_bridge.py | **仅 BTC** (`composite_regime_symbols` config) |
| RegimeParamRouter | regime/param_router.py | decision/regime_bridge.py | **仅 BTC**: params → deadzone/min_hold |
| ADX(14) | features/enriched_computer.py | RustFeatureEngine feeds feat_dict | 28-bar warmup; feeds TrendRegimeDetector via feat_dict |
| PortfolioCombiner | scripts/ops/portfolio_combiner.py | runner/alpha_main.py | AGREE ONLY mode |
| MultiTFConfluenceGate | runner/gates/multi_tf_confluence_gate.py | runner/gate_chain.py | 1h vs 4h alignment |
| LiquidationCascadeGate | runner/gates/liquidation_cascade_gate.py | runner/gate_chain.py | Cascade protection (zscore>3 block) |
| CarryCostGate | runner/gates/carry_cost_gate.py | runner/gate_chain.py | Funding+basis carry cost adjustment |
| OnlineRidge | alpha/online_ridge.py | decision/signals/alpha_signal.py | RLS incremental weight updates (λ=0.997, 4h runners) |
| OptionsFlowComputer | features/options_flow.py | enriched_computer.py `on_bar` | 7 Deribit options features |
| AlphaDecisionModule | decision/modules/alpha.py | engine/coordinator.py via DecisionBridge | Framework-native decision logic (~300 lines) |
| EnsemblePredictor | decision/signals/alpha_signal.py | AlphaDecisionModule | Ridge(60%) + LGBM(40%) |
| SignalDiscretizer | decision/signals/alpha_signal.py | AlphaDecisionModule | z-score → deadzone → discretize |
| AdaptivePositionSizer | decision/sizing/adaptive.py | AlphaDecisionModule | equity-tier × IC × vol |
| BTCUSDT_4h Runner | runner/alpha_main.py | runner/strategy_config.py SYMBOL_CONFIG | **WIRED**: Strategy H primary |
| ETHUSDT_4h Runner | runner/alpha_main.py | runner/strategy_config.py SYMBOL_CONFIG | **WIRED**: Strategy H primary |
| SIGHUP Hot-Reload | runner/alpha_main.py | signal handler | **WIRED**: <200ms reload all 4 models |
| Per-Runner Checkpoint | state/checkpoint.py | CheckpointManager | **WIRED**: per-runner state persistence |
| Dynamic Leverage | decision/sizing/adaptive.py | vol_scale | **WIRED**: base 10x × vol_scale |
| Vol-Adaptive Deadzone | decision/signals/alpha_signal.py | rv/vol_median | **WIRED**: realized vol / median vol scaling |
| BB Entry Scaler | decision/modules/alpha.py | continuous tanh [0.75, 1.2] | **WIRED**: Bollinger Band entry scaling |
| 4h Z-Score Stop | decision/modules/alpha.py | force exits | **WIRED**: cross-timeframe stop signal |
| VenueRouter | execution/adapters/venue_router.py | runner/alpha_main.py | **WIRED**: optional via `HYPERLIQUID_PRIVATE_KEY` |
| IC Decay Monitor | monitoring/ic_decay_monitor.py | systemd timer (daily 3am) | **WIRED**: automated IC decay alerting + auto-retrain on RED |
| Signal Reconciliation | scripts/ops/signal_reconcile.py | manual / cron | **WIRED**: live vs expected signal comparison |
| Shadow A/B Compare | scripts/ops/shadow_compare.py | manual / cron | **WIRED**: shadow mode A/B signal comparison |
| ETHRegimeProxy | regime/eth_regime_proxy.py | decision/modules/alpha.py | **WIRED**: BTC regime → ETH param routing |
| PipelineMetrics | monitoring/pipeline_metrics.py | engine/pipeline.py | **WIRED**: Thread-safe pipeline counters |
| InputValidation | core/validation.py | core/ | **WIRED**: NaN/Inf price/qty validation |
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

### Framework-native (SYMBOL_CONFIG in runner/strategy_config.py)

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
