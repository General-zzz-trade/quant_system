# Quant System 代码库研究报告

> 更新时间: 2026-03-12
> 研究方式: 基于当前源码、运行时真相源、契约文档、改造计划、模型资产和测试结构逐层核对
> 结论基准: 以当前代码和已落地文档为准，不以历史 README、旧计划或过时注释为准

---

## 1. 执行摘要

这个仓库已经不是“策略脚本集合”，而是一个完整的量化交易平台雏形。它覆盖了：

- 数据采集与质量校验
- 事件模型与事件存储
- 状态推进与快照
- 特征计算与在线推理
- 决策、风控、执行、对账、恢复
- 回测、回放、paper、live
- 研究、训练、模型注册、模型加载
- 监控、告警、部署、运维脚本

当前最准确的判断是：

> 代码库已经形成了功能闭环和较强的生产候选能力，但仍处于“持续收口”的中后期，而不是完全定型的最终形态。

最重要的现状有 6 条：

1. 默认生产主路径仍然是 [`runner/live_runner.py`](/quant_system/runner/live_runner.py)
2. 当前运行时是“Python 主编排 + Rust 热路径内核”的混合架构
3. 状态推进已经显著收口到 Rust `process_event()` 写通道
4. live / backtest / replay / scripts 的关键约束已经做过一轮系统收口
5. execution recovery、incident policy、model governance 已从隐式习惯变成显式文档和测试
6. Rust 第二阶段迁移已开始围绕 parity tests 和小范围 fallback 删除推进，但还没有完成 runtime 总替换

当前客观规模：

| 指标 | 当前值 |
|---|---:|
| Python 文件 | 971 |
| Rust 文件 | 67 |
| Markdown 文档 | 29 |
| `tests/test_*.py` 文件 | 230 |
| 顶层主功能目录 | `engine/`, `event/`, `state/`, `decision/`, `execution/`, `risk/`, `runner/`, `scripts/`, `research/`, `ext/rust/` |

---

## 2. 当前真实定位

这个仓库的真实产品形态是：

- 场景: 加密永续合约 / 多资产量化交易
- 形态: 平台型系统，不是单策略回测仓库
- 架构: 事件驱动、事实推进状态、decision/execution 与 state 解耦
- 运行: live、backtest、replay、paper 均存在
- 技术方向: Python 控制面 + Rust 运行内核逐步上收

更准确地说，它现在不是“Rust-only runtime”，也不是“Python-only monolith”，而是：

> Python 负责装配、生态集成、研究训练和大量运行控制；Rust 负责越来越多的热路径、状态、特征、校验与确定性逻辑。

---

## 3. 代码库地图

从目录结构看，仓库已经形成了清晰的平台分层。

### 3.1 核心运行层

- [`engine`](/quant_system/engine): 协调器、dispatcher、pipeline、loop、replay
- [`event`](/quant_system/event): 事件类型、header、codec、runtime、store、checkpoint
- [`state`](/quant_system/state): 状态视图、Rust adapter、snapshot、versioning
- [`runner`](/quant_system/runner): live/backtest/replay/paper 等运行入口

### 3.2 决策与智能层

- [`features`](/quant_system/features): 技术指标、跨周期、微观结构、batch/live feature engine
- [`alpha`](/quant_system/alpha): 模型封装、在线推理、training、model loader
- [`decision`](/quant_system/decision): 信号、intents、allocators、execution policy、sizing、risk overlay
- [`regime`](/quant_system/regime): regime 判定与桥接

### 3.3 安全与执行层

- [`risk`](/quant_system/risk): kill switch、aggregator、correlation gate、rules
- [`portfolio`](/quant_system/portfolio): 组合分配、优化、风险模型
- [`execution`](/quant_system/execution): adapters、ingress、state machine、safety、reconcile、tca、latency、routing

### 3.4 研究与运维层

- [`research`](/quant_system/research): walk-forward、hyperopt、registry、retrain
- [`scripts`](/quant_system/scripts): 训练、验证、研究、数据刷新、ops 工具层
- [`monitoring`](/quant_system/monitoring): health、alerts、metrics、dashboards
- [`deploy`](/quant_system/deploy): docker、k8s、prometheus、grafana、systemd
- [`ext/rust`](/quant_system/ext/rust): `_quant_hotpath` PyO3 crate 和 standalone Rust trader

结论不是“execution 很大”，而是：

> 这个仓库已经拥有交易系统的完整平台横截面，而不是一个研究仓库外挂执行器。

---

## 4. 当前运行时真相

当前运行时真相以 [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md) 为准。

### 4.1 默认生产入口

当前默认生产入口是：

- [`runner/live_runner.py`](/quant_system/runner/live_runner.py)

其他路径的定位：

- [`runner/backtest_runner.py`](/quant_system/runner/backtest_runner.py): 历史回测入口
- [`runner/replay_runner.py`](/quant_system/runner/replay_runner.py): 回放与一致性验证入口
- [`ext/rust/src/bin/main.rs`](/quant_system/ext/rust/src/bin/main.rs): 重要的 standalone Rust trader 演进路径，但不是当前默认生产入口

### 4.2 `LiveRunner` 的真实定位

[`runner/live_runner.py`](/quant_system/runner/live_runner.py) 不是一个薄入口，而是完整 runtime 装配器。当前会装配的关键部件包括：

- `EngineCoordinator`
- `EngineLoop`
- `DecisionBridge`
- `ExecutionBridge`
- `FeatureComputeHook`
- `KillSwitch`
- `MarginMonitor`
- `ReconcileScheduler`
- `SystemHealthMonitor`
- `AlertManager`
- `LatencyTracker`
- `OrderTimeoutTracker`
- `OrderStateMachine`
- `ModelRegistry` 自动加载相关 loader
- user stream / venue client / health endpoint / persistence 等外围能力

这意味着：

> 生产 runtime 的复杂度主要在 Python 装配层，而不是已经完全下沉到了 Rust binary。

---

## 5. 事件链与状态链

### 5.1 当前统一事件链

当前标准闭环可以概括为：

```text
Market / Replay / Backtest input
  -> coordinator.emit()
  -> EventDispatcher route
  -> StatePipeline / RustStateStore.process_event()
  -> snapshot
  -> DecisionBridge
  -> IntentEvent / OrderEvent
  -> ExecutionBridge
  -> venue adapter / algo / reconcile path
  -> fill / reject / report-like result
  -> dispatcher reinjection
  -> pipeline 再推进状态
```

这条链上的约束很清楚：

- decision 不直接改状态
- execution 不直接改状态
- 状态由事实事件推进
- snapshot 是 decision 的标准输入

### 5.2 `EventDispatcher`

[`engine/dispatcher.py`](/quant_system/engine/dispatcher.py) 现在已经明显朝“Rust route matcher + Python handler dispatch”收口：

- 路由决策使用 Rust `rust_route_event`
- 去重使用 Rust `DuplicateGuard`
- Python 负责 handler 注册和顺序执行

并且此前残留的 `EventType` 导入 fallback 已删除，说明 dispatcher routing 现在已经以 Rust 路由器为真相源。

### 5.3 `StatePipeline`

[`engine/pipeline.py`](/quant_system/engine/pipeline.py) 是当前最关键的制度边界之一。

当前事实：

- `normalize_to_facts()` 走 Rust `rust_normalize_to_facts`
- `event kind` 检测走 Rust `rust_detect_event_kind`
- 真正状态推进走 `store.process_event(inp.event, inp.symbol_default)`
- Python 负责快照生成、Rust state lazy conversion、向决策层暴露可读视图

这说明系统已经不是“Python reducer 为主、Rust 做优化”，而是：

> Rust 已经是状态推进和事实归一化的真正内核；Python 主要承担外层编排、导出和桥接。

---

## 6. Ownership Matrix 结论

基于当前代码和文档，ownership 可以归纳为：

| 子系统 | 当前 owner | 说明 |
|---|---|---|
| 运行时总控 | Python | `LiveRunner`, `EngineCoordinator`, `EngineLoop` |
| 事件路由与去重热路径 | Rust 主导，Python 包装 | dispatcher 已显著上收 |
| 状态推进 | Rust 主导 | `RustStateStore.process_event()` 是核心写通道 |
| pipeline 外层 | Python | snapshot、导出、桥接 |
| 特征热路径 | 混合，Rust 更重 | `FeatureComputeHook` 管理 Rust engines |
| 推理约束状态 | Rust 主导，Python 编排 | live inference bridge 已与 Rust 强耦合 |
| 决策编排 | Python | `DecisionBridge`, modules, strategy assembly |
| execution transport / adapter | Python | 交易所 IO 和生态 glue 仍在 Python |
| execution 纯逻辑 | 混合，向 Rust 迁移 | safety / sequence / route / state machine 已有候选 |
| 研究训练 | Python | `research/`, `scripts/`, model training |
| 监控与运维 | Python | dashboards、alerts、deploy、ops |

---

## 7. 执行层评估

[`execution`](/quant_system/execution) 是当前最大的生产子系统，也是最像真正实盘平台的一层。

它包含：

- `adapters/`: Binance、generic venue、stream transport、REST/WS glue
- `ingress/`: payload dedup、router、sequence handling
- `safety/`: duplicate guard、out-of-order guard、timeout tracker、risk gate
- `state_machine/`: transitions、machine、projection
- `reconcile/`: drift 检测、scheduler、healer
- `routing/`, `algos/`, `sim/`, `latency/`, `tca/`, `observability/`

### 7.1 当前 execution contract

[`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md) 已经把以下东西显式化：

- 合法订单状态
- `pending_cancel -> filled` 是允许路径
- timeout 是本地观测超时，不是 venue 未成交的证明
- `CanonicalFill` 是执行层标准成交事实对象
- incident matrix 定义了 timeout、late fill、duplicate fill、restart drift 的默认动作

### 7.2 当前 recovery 现状

此前已经补齐的恢复类测试，意味着 execution recovery 已经不只是“模块存在”，而是有组合测试护栏：

- [`tests/integration/test_execution_recovery_e2e.py`](/quant_system/tests/integration/test_execution_recovery_e2e.py)
- [`tests/integration/test_execution_timeout_restart_recovery.py`](/quant_system/tests/integration/test_execution_timeout_restart_recovery.py)
- [`tests/execution_safety/test_late_execution_report.py`](/quant_system/tests/execution_safety/test_late_execution_report.py)
- [`tests/execution_safety/test_cancel_replace_flow.py`](/quant_system/tests/execution_safety/test_cancel_replace_flow.py)

这说明系统已经开始从“能恢复”转向“恢复行为可验证”。

### 7.3 仍未完全收口的点

execution 仍然有一个重要未完事项：

- 公共事件层 `FillEvent` 与执行层 `CanonicalFill` 还没有完全统一成单一事实模型

因此 execution 已经很强，但还没到“语义完全收官”的状态。

---

## 8. 决策、约束与跨路径一致性

这部分是最近收口最明显的区域。

### 8.1 当前三层真相源

当前关键约束语义由三层构成：

- live 语义源: [`alpha/inference/bridge.py`](/quant_system/alpha/inference/bridge.py)
- backtest 对齐实现: [`decision/backtest_module.py`](/quant_system/decision/backtest_module.py)
- scripts 共享后处理: [`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py)

### 8.2 已收口的约束

基于最近的改造，以下约束已经做过跨路径对齐：

- `deadzone`
- `min_hold`
- `trend_follow / trend_hold`
- `monthly_gate`
- `vol_target`
- 离散退出规则 `should_exit_position`
- `rolling_zscore`

### 8.3 已落地的 parity / contract 护栏

关键测试包括：

- [`tests/unit/decision/test_backtest_live_parity.py`](/quant_system/tests/unit/decision/test_backtest_live_parity.py)
- [`tests/unit/decision/test_backtest_module_constraints.py`](/quant_system/tests/unit/decision/test_backtest_module_constraints.py)
- [`tests/unit/scripts/test_backtest_engine_constraints.py`](/quant_system/tests/unit/scripts/test_backtest_engine_constraints.py)
- [`tests/contract/test_runtime_contracts.py`](/quant_system/tests/contract/test_runtime_contracts.py)
- [`tests/replay/test_replay_vs_live_equivalence.py`](/quant_system/tests/replay/test_replay_vs_live_equivalence.py)

结论：

> 这部分已经不再是“大家大概一致”，而是开始有系统的行为级护栏。

---

## 9. `scripts/` 模块评估

[`scripts`](/quant_system/scripts) 当前不是线上主 runtime，而是围绕主系统的工具层。

### 9.1 当前定位

[`scripts/README.md`](/quant_system/scripts/README.md) 已明确：

- `train`
- `validate`
- `research`
- `data`
- `ops`
- `shared`

六大逻辑分组，并由 [`scripts/catalog.py`](/quant_system/scripts/catalog.py) 作为分类真相源。

### 9.2 当前主力入口

当前已经被显式标记为 `official` / `recommended` / `specialized` / `legacy-reference` 的主脚本包括：

- `train_v11.py`
- `backtest_engine.py`
- `walkforward_validate.py`
- `run_paper_trading.py`
- `testnet_smoke.py`
- `refresh_data.py`
- 多个 specialized 训练和验证脚本

### 9.3 当前共享 helper

[`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py) 已经成为 scripts 层的后处理真相源，承载：

- `rolling_zscore`
- `_apply_monthly_gate`
- `_apply_trend_hold`
- `_apply_vol_target`
- `_enforce_min_hold`
- `should_exit_position`
- `_compute_bear_mask`

### 9.4 结论

`scripts/` 仍然是历史沉积最重的目录之一，但和此前不同的是：

> 它现在已经开始被治理，而不是继续无边界增长。

---

## 10. 模型层与模型治理

### 10.1 模型类型

代码层支持的模型类型包括：

- [`LGBMAlphaModel`](/quant_system/alpha/models/lgbm_alpha.py)
- [`XGBAlphaModel`](/quant_system/alpha/models/xgb_alpha.py)
- [`EnsembleAlphaModel`](/quant_system/alpha/models/ensemble.py)
- [`LSTMAlphaModel`](/quant_system/alpha/models/lstm_alpha.py)
- [`TransformerAlphaModel`](/quant_system/alpha/models/transformer_alpha.py)

实际仓库资产的主力仍然是 tree-based / ensemble 路径，而不是深度序列模型。

### 10.2 当前模型资产

[`models_v8`](/quant_system/models_v8) 下的主资产仍然集中在：

- BTC gate 系列
- ETH gate 系列
- 4h / 15m / 30m 等 specialized 资产
- SOL gate / bear 路径

当前 registry 中标记为 production 的模型有 2 个：

- `alpha_v8_4h_BTCUSDT`
- `alpha_v8_BTCUSDT`

### 10.3 当前治理现状

[`docs/model_governance.md`](/quant_system/docs/model_governance.md) 已经明确了：

- `ModelRegistry` 负责注册、元数据、production 标记
- `ProductionModelLoader` 负责加载当前 production model 并在 `model_id` 变化时 reload
- `shadow_compare.py` 负责 candidate vs production 的准入比较

最近已落地的关键收口：

- loader 增加了 feature schema mismatch 拒绝加载
- registry / loader / rollback / reload 已有测试

这说明模型治理已经从“脚本习惯”开始转向“制度 + 测试”。

---

## 11. Rust 内核现状

[`ext/rust`](/quant_system/ext/rust) 不是装饰层，而是真正的热路径内核。

### 11.1 当前已深度接管的区域

从代码和替换矩阵看，Rust 已深度进入：

- dispatcher routing
- event kind / fact normalization
- state store / reducers
- feature hot path
- inference constraint 部分逻辑
- duplicate / payload dedup / sequence buffer
- 一部分 execution store / hashing / request id / signer

### 11.2 当前 Phase 4 候选

[`docs/rust_replacement_matrix.md`](/quant_system/docs/rust_replacement_matrix.md) 已明确第二阶段候选：

- timeout / sequencing helpers
- dispatcher routing core
- order projection / reconcile kernel
- backtest signal constraint kernel

### 11.3 当前真实状态

必须强调：

- Rust 已经非常深入
- 但系统还没有切到“Rust-only runtime”
- Python 仍然掌管 runtime 装配、adapter IO、研究训练、ops 生态

---

## 12. 测试结构评估

当前测试结构已经明显超出普通量化仓库水平，至少覆盖了：

- unit
- contract
- replay
- integration
- persistence
- regression
- performance
- execution safety

其中更高价值的测试类型包括：

- live/backtest parity
- replay vs live equivalence
- timeout / restart / late fill 恢复
- dispatcher / sequence buffer 的 Rust parity
- model loader / registry / rollback / schema mismatch

结论不是“测试很多”，而是：

> 当前测试已经开始承担架构收口工具的角色，而不是只做功能回归。

---

## 13. 完成度评估

### 13.1 已经完成的部分

- 功能闭环已经形成
- 默认生产主路径已经明确
- 运行时真相源已建立
- execution incident policy 已成文档
- model governance 已开始成体系
- scripts 已开始治理
- Rust 第二阶段迁移已有候选和 parity tests

### 13.2 尚未完全完成的部分

- `CanonicalFill` 与公共 `FillEvent` 的彻底统一
- Python/Rust 双栈边界的进一步简化
- 更多 execution 纯逻辑向 Rust 上收
- 部分研究脚本的局部差异语义是否长期保留
- standalone Rust trader 何时取代 Python live runtime，当前仍未完成

### 13.3 最准确的整体判断

如果必须用一句话总结：

> 这是一个已经具备真实量化交易平台骨架、测试护栏和生产候选能力的中大型代码库；当前主要任务已从“补功能”转为“继续收口真相源、统一语义并降低双栈维护成本”。

---

## 14. 当前风险与技术债

最重要的技术债已经不再是“缺模块”，而是下面这些收口问题：

1. execution 公共事实模型仍未完全统一
2. Python runtime 与 standalone Rust runtime 并存，长期 ownership 解释成本仍高
3. `scripts/` 虽已治理，但历史沉积仍深
4. 一些 specialized / research 语义仍有局部包装，不是完全统一实现

但相比前一阶段，当前明显改善的是：

- 真相源已经成文档
- 关键差异已开始有 parity tests
- 恢复类 case 已开始有组合集成测试
- 模型上线不再完全依赖人工约定

---

## 15. 结论

当前代码库的真实结论如下：

- 它已经形成了研究、验证、回测、paper、live、恢复、运维的完整闭环
- 它当前默认仍以 Python live runtime 为生产主路径
- Rust 已经深度接管运行内核的关键热路径
- 最近一轮改造已经把约束状态机、execution incident policy、scripts 治理、model governance、Rust parity 护栏明显收口
- 它还没有完全开发收官，但已经进入“收尾与定型”阶段，而不是“早期搭框架”阶段

因此，当前最准确的总体评价是：

> 这是一个功能闭环已形成、生产候选能力很强、并正在系统性收口的量化交易平台代码库。
