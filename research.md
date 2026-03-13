# Quant System 代码库研究报告

> 更新时间: 2026-03-13
> 研究方式: 基于当前源码、真相源文档、测试结构、CI 与部署工件逐层核对
> 判断原则: 以当前代码和已落地契约为准，不以历史 README 口径、旧脚本名或未接线清单为准

---

## 1. 执行摘要

这不是一个“策略脚本仓库”，而是一个已经具备完整横截面的量化交易平台代码库。它已经覆盖：

- 市场数据接入、事件建模、状态推进、快照读取
- 特征工程、在线推理、决策编排、风险裁决、执行与对账
- 回测、回放、paper、testnet、live wiring
- 研究分析、模型注册、模型热加载、重训练、运维控制面
- 监控、告警、审计、CI、Docker、K8s、Argo Rollouts

当前最准确的总体判断是：

> 这是一个“Python 主编排 + Rust 深度接管热路径”的中大型量化平台；功能闭环已经形成，强项在运行时制度、风险与执行语义、以及测试护栏，主要问题已经从“缺功能”转成“收口真相源、消化脚本/部署分叉、降低双栈维护成本”。

本轮核对后，最关键的 8 条结论如下：

1. 当前唯一默认生产主路径仍然是 [`runner/live_runner.py`](/quant_system/runner/live_runner.py)，不是 Rust standalone binary。
2. Rust 已经深度接管状态推进、路由、特征、约束状态机、去重和部分执行原语，但 Python 仍拥有 runtime 装配、交易所 IO、研究训练和 ops glue。
3. [`decision/engine.py`](/quant_system/decision/engine.py) 和 [`engine/decision_bridge.py`](/quant_system/engine/decision_bridge.py) 已形成比较清晰的“只读快照 -> 意见事件”制度中心。
4. [`execution/`](/quant_system/execution) 已经不是 adapter 集合，而是包含 canonical model、ingress、state machine、reconcile、safety、observability、store、sim 的完整执行子平台。
5. [`risk/`](/quant_system/risk) 的成熟度高于常见量化仓库水位，尤其是 [`risk/aggregator.py`](/quant_system/risk/aggregator.py) 与 [`risk/kill_switch.py`](/quant_system/risk/kill_switch.py) 体现出明显的制度化设计。
6. `research/`、`scripts/`、`model governance` 并非空壳，但成熟度不均衡；其中一部分已产品化，另一部分仍带明显历史沉积。
7. 测试强项在 Python runtime wiring、契约、恢复链路和 operator/control 面；弱项在部署工件验证、CI 自身、Rust 默认门禁和部分遗留 research 入口活性。
8. 当前最需要正视的不是功能缺失，而是具体的漂移点：部署脚本与 compose 不一致、Dockerfile 工件疑似过时、配置 schema 与 runtime 解析存在偏差、以及部分测试不在默认 CI 门里。

---

## 2. 仓库规模与复杂度

### 2.1 客观规模

按 2026-03-13 当前工作树统计：

| 指标 | 当前值 |
|---|---:|
| Python 文件 | 996 |
| Python 代码行 | 154,645 |
| Rust 文件 | 67 |
| Rust 代码行 | 25,493 |
| Markdown 文档 | 30 |
| `tests/` 下 `test_*.py` | 247 |
| `execution/tests/` 下额外 `test_*.py` | 28 |
| 已发现测试文件合计 | 275 |

这意味着：

> 这个仓库已经进入“中大型单仓平台”量级，复杂度来源并不只在 runtime，本地研究工具、运维脚本和测试护栏也占了非常高的体量。

### 2.2 Python 代码热点

按目录统计，当前最重的 Python 区域如下：

| 目录 | Python 文件数 | 代码行 | 结论 |
|---|---:|---:|---|
| `scripts/` | 115 | 46,432 | 最大的历史沉积区，也是当前研究/运维工作层 |
| `tests/` | 252 | 44,231 | 测试已经是架构护栏，不只是回归补丁 |
| `execution/` | 202 | 16,296 | 最大的生产子系统 |
| `runner/` | 15 | 6,899 | 入口不多，但装配责任非常集中 |
| `decision/` | 88 | 5,507 | 决策框架与回测兼容逻辑并存 |
| `portfolio/` | 78 | 5,002 | 能力面很强，但 live 接线深度不如 execution/risk |

从体量上看，当前仓库的复杂度分布不是“所有难点都在 engine”，而是：

- 生产复杂度集中在 [`runner/`](/quant_system/runner)、[`engine/`](/quant_system/engine)、[`execution/`](/quant_system/execution)、[`risk/`](/quant_system/risk)
- 维护复杂度集中在 [`scripts/`](/quant_system/scripts) 与测试体系
- 演进复杂度集中在 [`ext/rust/`](/quant_system/ext/rust) 与 Python/Rust 双栈边界

---

## 3. 当前真实定位

### 3.1 产品形态

当前代码库的真实产品定位是：

- 场景: 加密永续合约量化交易
- 架构: 事件驱动、事实推进状态、快照驱动决策
- 形态: 平台型系统，不是单策略研究仓
- 运行模式: live、backtest、replay、paper、testnet 都存在
- 技术路线: Python 控制面 + Rust 热路径内核

### 3.2 当前真相源

当前应优先相信以下文档，而不是历史规划文档：

- 运行时真相: [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md)
- 跨路径契约: [`docs/runtime_contracts.md`](/quant_system/docs/runtime_contracts.md)
- 执行制度: [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md)
- 模型治理: [`docs/model_governance.md`](/quant_system/docs/model_governance.md)

### 3.3 不应继续当作“当前事实”的说法

以下说法现在都不够准确：

- “系统已经 Rust-only runtime”
- “Python 只剩配置和胶水”
- “standalone Rust binary 已经是默认生产入口”
- “所有 fallback 都已删除”

更准确的当前描述是：

> Python 仍控制 runtime 装配、exchange IO、研究训练与运维生态；Rust 已经深度掌管状态、路由、特征、部分约束状态机与多个执行/去重原语。

---

## 4. 运行时主链与 Ownership

### 4.1 当前入口分工

当前几条关键入口的真实定位如下：

| 路径 | 当前定位 |
|---|---|
| [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | 默认生产主路径 |
| [`runner/backtest_runner.py`](/quant_system/runner/backtest_runner.py) | 历史回测与验证入口 |
| [`runner/replay_runner.py`](/quant_system/runner/replay_runner.py) | 事件重放与一致性验证 |
| [`runner/live_paper_runner.py`](/quant_system/runner/live_paper_runner.py) | 更接近 live 语义的 paper 路径 |
| [`runner/paper_runner.py`](/quant_system/runner/paper_runner.py) | 简化 demo / stdin paper runner |
| [`ext/rust/src/bin/main.rs`](/quant_system/ext/rust/src/bin/main.rs) | 候选 Rust runtime，不是默认生产入口 |

### 4.2 当前统一事件链

当前标准闭环可以概括为：

```text
Market / Replay / Backtest input
  -> EngineLoop.submit()
  -> EventDispatcher.dispatch()
  -> StatePipeline / RustStateStore.process_event()
  -> StateSnapshot
  -> DecisionBridge
  -> IntentEvent / OrderEvent
  -> ExecutionBridge / LiveExecutionBridge
  -> Fill / Reject / synthetic ingress event
  -> dispatcher reinjection
  -> pipeline 再次推进状态
```

关键边界已经比较清楚：

- decision 读 snapshot，不直接写状态
- execution 不直接写状态
- state 只能经由 pipeline/store 的事实写通道推进
- replay 与 live 共用事件语义，但并不是完整等价装配器

### 4.3 `LiveRunner` 的真实角色

[`runner/live_runner.py`](/quant_system/runner/live_runner.py) 不是薄入口，而是当前最重的运行时装配器。它会把以下能力接到一个 runtime 中：

- [`engine/coordinator.py`](/quant_system/engine/coordinator.py) 与 [`engine/loop.py`](/quant_system/engine/loop.py)
- [`engine/feature_hook.py`](/quant_system/engine/feature_hook.py) 与在线推理桥
- [`engine/decision_bridge.py`](/quant_system/engine/decision_bridge.py)
- [`engine/execution_bridge.py`](/quant_system/engine/execution_bridge.py) / live execution wiring
- [`risk/kill_switch.py`](/quant_system/risk/kill_switch.py)、[`risk/margin_monitor.py`](/quant_system/risk/margin_monitor.py)、相关性和组合风险聚合
- [`execution/reconcile/scheduler.py`](/quant_system/execution/reconcile/scheduler.py)、timeout tracker、order state machine
- [`monitoring/health.py`](/quant_system/monitoring/health.py)、[`monitoring/health_server.py`](/quant_system/monitoring/health_server.py)、[`monitoring/alerts/manager.py`](/quant_system/monitoring/alerts/manager.py)
- model loader、SIGHUP 热重载、operator control、ops timeline、user stream 重连

这带来两个判断：

1. 当前 runtime 能力非常完整，说明系统已经过了“只会跑策略”的阶段。
2. [`runner/live_runner.py`](/quant_system/runner/live_runner.py) 也形成了明显的责任集中点，它既是优势，也是后续收口时最大的复杂度汇聚点。

### 4.4 Ownership Matrix

| 子系统 | 当前 owner | 说明 |
|---|---|---|
| 运行时总控 | Python | `LiveRunner` / `EngineLoop` / `EngineCoordinator` |
| 路由与 dispatcher dedup | Rust 主导，Python 包装 | [`engine/dispatcher.py`](/quant_system/engine/dispatcher.py) |
| 状态推进 | Rust 主导 | [`engine/pipeline.py`](/quant_system/engine/pipeline.py) + `RustStateStore` |
| pipeline 外层快照桥 | Python | snapshot、懒转换、导出视图 |
| 特征热路径 | Rust 主导，Python 编排 | per-symbol `RustFeatureEngine` |
| 在线约束状态机 | Rust 主导，Python 编排 | [`alpha/inference/bridge.py`](/quant_system/alpha/inference/bridge.py) |
| 决策编排 | Python | [`decision/engine.py`](/quant_system/decision/engine.py)、[`engine/decision_bridge.py`](/quant_system/engine/decision_bridge.py) |
| 风控聚合与熔断动作 | Python | [`risk/aggregator.py`](/quant_system/risk/aggregator.py)、[`risk/kill_switch.py`](/quant_system/risk/kill_switch.py) |
| 执行 transport / adapter | Python 主导 | Binance REST/WS、user stream、reconcile wiring |
| 执行纯逻辑原语 | 混合 | state machine、dedup、payload guards、sequence buffer 已有 Rust 接入 |
| 研究训练与模型治理 | Python | [`research/`](/quant_system/research)、[`scripts/`](/quant_system/scripts) |
| 监控与运维 | Python | [`monitoring/`](/quant_system/monitoring)、[`infra/`](/quant_system/infra)、[`deploy/`](/quant_system/deploy) |

---

## 5. Rust 内核现状

[`ext/rust/Cargo.toml`](/quant_system/ext/rust/Cargo.toml) 当前同时产出：

- PyO3 扩展库 `_quant_hotpath`
- 独立二进制 `quant_trader`

`lib.rs` 当前导出/组织了 60+ 个 Rust 模块，能力覆盖：

- state types / reducers / store / pipeline
- route matcher / duplicate guard / payload dedup / sequence buffer
- feature engine / cross asset / microstructure
- inference bridge / unified predictor / tick processor
- risk engine / decision math / portfolio allocator
- order state machine / execution store / signer / request id
- standalone WS client / backtest engine / attribution 等

这说明 Rust 的地位已经不是“优化器”，而是真正的内核层。

### 5.1 Rust 已深度接管的区域

当前已经能从 Python 主路径上明确看到 Rust ownership 的区域包括：

- dispatcher 路由与 dedup: [`engine/dispatcher.py`](/quant_system/engine/dispatcher.py)
- event kind / fact normalization: [`engine/pipeline.py`](/quant_system/engine/pipeline.py)
- state store 与 event apply: [`engine/coordinator.py`](/quant_system/engine/coordinator.py)
- live feature engine: [`engine/feature_hook.py`](/quant_system/engine/feature_hook.py)
- inference 约束状态机: [`alpha/inference/bridge.py`](/quant_system/alpha/inference/bridge.py)
- ingress payload dedup: [`execution/ingress/router.py`](/quant_system/execution/ingress/router.py)
- order status / sequence / dedup parity 支撑: [`execution/state_machine/machine.py`](/quant_system/execution/state_machine/machine.py) 等

### 5.2 当前真实状态

必须强调两点：

- Rust 深度已经很高，这是真的。
- Rust 完全取代 Python runtime 这件事还没有发生，这也是真的。

最准确的表述仍然是：

> 当前系统是“Python 主编排、Rust 深度接管热路径”的混合运行时，而不是已经切换到 Rust-only runtime。

---

## 6. 核心业务层分析

### 6.1 `features/` 与 `alpha/`

这两个目录已经深度 live-wired，而不是纯研究代码。

关键链路是：

- [`engine/feature_hook.py`](/quant_system/engine/feature_hook.py) 负责从 `MarketEvent` 驱动 per-symbol feature engine
- `FeatureComputeHook` 不只接 OHLCV，还接 funding、OI、long/short ratio、spot、FGI、IV、PCR、on-chain、liquidation、mempool、macro、sentiment 等外源
- [`alpha/inference/bridge.py`](/quant_system/alpha/inference/bridge.py) 在特征完成后直接注入 `ml_score`

[`alpha/inference/bridge.py`](/quant_system/alpha/inference/bridge.py) 的成熟度尤其高，它不只是“跑模型”：

- 支持 ensemble averaging
- 支持 `deadzone`
- 支持 `min_hold`
- 支持 `trend_follow`
- 支持 `monthly_gate`
- 支持 bear model / short model
- 支持 vol targeting
- 支持 checkpoint / restore
- 支持 model hot swap
- 关键约束状态由 Rust bridge 保存

因此：

> `alpha/` 的真实定位不是模型类集合，而是 live 语义和研究语义之间最关键的桥。

最大的风险在这里也很明确：

- 特征空间很大，schema 漂移风险高
- live / backtest / research 若不持续共用 helper，很容易“名字对上了，语义却偏了”

### 6.2 `decision/`、`regime/`、`strategies/`

[`decision/engine.py`](/quant_system/decision/engine.py) 当前是业务制度中心之一。它把决策过程拆成：

1. risk overlay
2. universe selection
3. signal generation
4. candidate generation / filtering
5. allocation / constraints
6. sizing
7. intent -> order spec -> execution policy -> validate

这层的优点很明确：

- 强调 determinism
- 强调 no side effects
- explain/audit 结构完整
- `DecisionBridge` 只重新注入意见事件，不直接触发执行副作用

[`decision/backtest_module.py`](/quant_system/decision/backtest_module.py) 则体现了另一个现实：

- live 语义已经相当复杂
- backtest 需要一个兼容实现去逼近 live
- 仓库当前仍然维护“平台决策框架”和“历史回测兼容模块”两条线

`regime` 当前更像 gating / scaling 原语，而不是 runtime 的统一状态层。[`decision/regime_bridge.py`](/quant_system/decision/regime_bridge.py) 说明 regime 主要用于策略准入，而不是最底层事实状态。

[`strategies/`](/quant_system/strategies) 的真实定位更接近“可插拔策略库 + 实验层”，而不是系统真相源。这里的抽象层级并不完全一致：

- 有些模块偏平台式 decision module
- 有些模块更像研究代码搬入主仓
- 有些模块强调 HFT / order book / stat arb 实验

因此应明确：

> `decision/` 是制度中心，`strategies/` 更像实验实现库，不应把后者误写成运行时真相源。

### 6.3 `risk/` 与 `portfolio/`

[`risk/aggregator.py`](/quant_system/risk/aggregator.py) 和 [`risk/kill_switch.py`](/quant_system/risk/kill_switch.py) 显示出明显的生产系统设计意识：

- rule protocol + meta builder 解耦
- enable/disable 和快照
- 统计与错误计数
- fail-safe
- 多作用域 kill switch
- `HARD_KILL` / `REDUCE_ONLY`
- 审计元信息与 TTL

这不是“几条 if 语句”级别的风控。

[`portfolio/`](/quant_system/portfolio) 的能力面也很完整：

- allocator / rebalance / risk budget
- Black-Litterman
- factor constraints
- transaction-cost-aware objective
- risk model registry

但它的现状更像“强平台能力储备 + 部分已接线 live 能力”，而不是像 `execution`、`risk`、`decision` 那样能直接证明自己处于当前生产主链中心。

因此：

> `portfolio/` 的设计成熟度很高，但当前 live ownership 相对弱于 `decision/risk/execution`，是典型的“平台能力大于当前主路径接入深度”的模块。

---

## 7. 执行子系统评估

[`execution/`](/quant_system/execution) 是当前最大的生产子系统，也是最接近独立平台的一层。

### 7.1 当前结构

按职责看，`execution/` 已经拆成完整分层：

| 分层 | 代表文件 | 当前职责 |
|---|---|---|
| adapters | [`execution/adapters/binance/venue_client_um.py`](/quant_system/execution/adapters/binance/venue_client_um.py)、[`execution/adapters/binance/user_stream_processor_um.py`](/quant_system/execution/adapters/binance/user_stream_processor_um.py) | 交易所接入、payload 解析、venue client |
| bridge | [`execution/bridge/execution_bridge.py`](/quant_system/execution/bridge/execution_bridge.py)、[`execution/bridge/live_execution_bridge.py`](/quant_system/execution/bridge/live_execution_bridge.py) | 下单、ack 归一化、synthetic fill / rejection 观察面 |
| ingress | [`execution/ingress/router.py`](/quant_system/execution/ingress/router.py)、[`execution/ingress/order_router.py`](/quant_system/execution/ingress/order_router.py) | 事实回流、去重、payload mismatch fail-fast |
| state machine | [`execution/state_machine/machine.py`](/quant_system/execution/state_machine/machine.py)、[`execution/state_machine/transitions.py`](/quant_system/execution/state_machine/transitions.py) | 订单生命周期制度 |
| reconcile | [`execution/reconcile/controller.py`](/quant_system/execution/reconcile/controller.py)、[`execution/reconcile/scheduler.py`](/quant_system/execution/reconcile/scheduler.py) | 本地与 venue 收敛 |
| safety | [`execution/safety/risk_gate.py`](/quant_system/execution/safety/risk_gate.py)、[`execution/safety/timeout_tracker.py`](/quant_system/execution/safety/timeout_tracker.py) | 下单前后安全护栏 |
| models | [`execution/models/fills.py`](/quant_system/execution/models/fills.py)、[`execution/models/rejections.py`](/quant_system/execution/models/rejections.py) | canonical execution model |
| observability | [`execution/observability/incidents.py`](/quant_system/execution/observability/incidents.py)、[`execution/observability/rejections.py`](/quant_system/execution/observability/rejections.py) | incident、alert、审计 |
| algos / sim | [`execution/algo_adapter.py`](/quant_system/execution/algo_adapter.py)、[`execution/algos/`](/quant_system/execution/algos)、[`execution/sim/`](/quant_system/execution/sim) | TWAP/VWAP/Iceberg 与 paper/sim |
| store | [`execution/store/event_log.py`](/quant_system/execution/store/event_log.py) | ack / dedup / event persistence |

### 7.2 当前核心架构

当前执行层最重要的结构不是“下单”，而是“双通道收敛”：

1. `LiveExecutionBridge` 负责命令发送，以及 direct ack -> synthetic fill / rejection
2. Binance user stream 负责把真实 venue order/fill 事实重新送回 ingress
3. ingress + dedup + state machine + reconcile 负责把 synthetic path 与 venue fact path 收敛起来

这个设计非常像真实生产系统，而不是回测仓外挂 execution adapter。

### 7.3 当前优势

执行层最强的地方有 4 个：

1. 语义制度化。状态机、timeout、late fill、rejection、incident taxonomy 已成代码和文档。
2. 恢复意识强。`pending_cancel -> filled`、timeout -> cancel -> restart -> late fill 这类真实场景已经被显式建模。
3. 观测面完整。rejection 和 synthetic fill 已进入统一 incident / alert 链路。
4. Rust 已开始接管关键热路径，如 dedup、payload guard、state machine parity 支撑。

### 7.4 当前未完全收口的点

执行层最大的剩余结构问题有 3 个：

1. `CanonicalFill`、公共 `FillEvent`、ingress fill event 仍然不是单一类型，只是通过映射 helper 收口。
2. synthetic fill、真实 fill、order update、timeout/restart/reconcile 并存，语义强但复杂度也高。
3. 抽象上支持多 venue，但从代码成熟度看，当前明显是 Binance-heavy 实现。

因此对执行层最准确的评价是：

> 它已经具备完整执行平台雏形，强项在制度边界和恢复意识，主要风险在跨模型身份一致性、双通道收敛复杂度，以及 generic abstraction 与 Binance 实际成熟度之间的不对称。

---

## 8. 研究层、脚本层与模型治理

### 8.1 `research/` 的真实定位

[`research/`](/quant_system/research) 不是单一平台，而是两类东西的混合：

- 已经较产品化、且接入模型生命周期的模块
  - [`research/model_registry/registry.py`](/quant_system/research/model_registry/registry.py)
  - [`research/model_registry/artifact.py`](/quant_system/research/model_registry/artifact.py)
  - [`research/retrain/pipeline.py`](/quant_system/research/retrain/pipeline.py)
  - [`research/retrain/scheduler.py`](/quant_system/research/retrain/scheduler.py)
- 更偏研究工具箱的统计/验证模块
  - combinatorial CV
  - overfit detection
  - walk-forward optimizer
  - hyperopt
  - factor evaluation / significance / monte carlo

因此应写成：

> `research/` 当前是一组“部分已产品化、部分仍偏工具化”的研究与治理模块集合。

### 8.2 `scripts/` 的真实状态

[`scripts/README.md`](/quant_system/scripts/README.md) 与 [`scripts/catalog.py`](/quant_system/scripts/catalog.py) 已经给 flat `scripts/` 建立了治理骨架：

- 有逻辑分组: train / validate / research / data / ops / shared
- 有主入口状态: `current` / `experimental` / `archive-adjacent`
- 有共享后处理真相源: [`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py)
- 有 CLI 入口: [`scripts/cli.py`](/quant_system/scripts/cli.py)

这是非常明确的正向变化。

但也必须说清楚：

- `scripts/` 仍然是仓库中最大的历史沉积区
- 当前 catalog 只治理主力入口，不等于整个平铺目录已经收口
- 许多 specialized / legacy / archive-adjacent 路径仍然并存

因此最准确的判断是：

> `scripts/` 已经从“平铺工具箱”进入“受治理工作层”，但还没有完成结构性收口。

### 8.3 模型治理现状

当前模型治理链条已经比较清晰：

- registry: [`research/model_registry/registry.py`](/quant_system/research/model_registry/registry.py)
- artifact store: [`research/model_registry/artifact.py`](/quant_system/research/model_registry/artifact.py)
- production loader: [`alpha/model_loader.py`](/quant_system/alpha/model_loader.py)
- compare/promote/rollback CLI: [`scripts/cli.py`](/quant_system/scripts/cli.py)
- shadow compare: [`scripts/shadow_compare.py`](/quant_system/scripts/shadow_compare.py)

目前已经做到：

- register / promote / rollback / history audit
- production model pointer 查询
- loader `reload_if_changed()`
- feature schema mismatch 拒绝加载
- ops audit 把 model ops 和 runtime control timeline 合并观察

这表明模型治理已经从“人工约定”升级到“代码 + 文档 + 测试”的阶段。

### 8.4 模型资产的现实特征

[`models_v8/`](/quant_system/models_v8) 直接存放了当前主力模型资产及多个 backup 目录。这说明：

- monorepo 同时承载代码与模型运维历史
- operator 使用体验直接
- 仓库重量和资产分支历史也随之变重

这不是错误，但它会放大：

- 资产命名治理成本
- backup/rollback 语义解释成本
- 仓库级别的操作复杂度

---

## 9. 监控、配置、安全与部署

### 9.1 监控与 operator control

监控层的成熟度是这个仓库的亮点之一。

关键组件包括：

- [`monitoring/health.py`](/quant_system/monitoring/health.py): 系统健康评估
- [`monitoring/health_server.py`](/quant_system/monitoring/health_server.py): `/health`、`/operator`、`/control`、`/ops-audit`
- [`monitoring/alerts/manager.py`](/quant_system/monitoring/alerts/manager.py): 规则告警引擎
- [`monitoring/metrics/prometheus.py`](/quant_system/monitoring/metrics/prometheus.py): Prometheus exporter
- [`monitoring/engine_hook.py`](/quant_system/monitoring/engine_hook.py): engine 侧接线

这意味着控制面已经不是外部文档，而是 runtime 内建能力。

### 9.2 配置与安全

[`infra/config/loader.py`](/quant_system/infra/config/loader.py) 已经体现出安全意识：

- 支持 YAML/JSON
- 支持 plaintext secret 扫描
- 支持 env-based credential resolution
- 支持 schema validation

这层已经具备“生产级配置入口”的基本意识，而不是随手 `yaml.safe_load`。

### 9.3 部署现状

仓库当前并存至少 4 套部署叙事：

1. 根目录 [`docker-compose.yml`](/quant_system/docker-compose.yml)
   - 当前更像 paper/testnet/runtime 组合
2. 根目录 [`Dockerfile`](/quant_system/Dockerfile) 与 [`Dockerfile.trader`](/quant_system/Dockerfile.trader)
   - 分别对应 Python+Rust paper 运行时与 Rust trader
3. [`deploy/docker/docker-compose.yml`](/quant_system/deploy/docker/docker-compose.yml)
   - 另一套 engine + timescaledb + prometheus + grafana 栈
4. [`deploy/k8s/deployment.yaml`](/quant_system/deploy/k8s/deployment.yaml) 与 [`deploy/argocd/rollout.yaml`](/quant_system/deploy/argocd/rollout.yaml)
   - 候选生产 / GitOps / canary 形态

结论不是“支持很多部署方式”，而是：

> 当前部署层存在明显多轨并存现象，且它们并不完全共享同一条当前主路径。

### 9.4 CI 现状

[`/.github/workflows/ci.yml`](/quant_system/.github/workflows/ci.yml) 与 [`/.github/workflows/deploy.yml`](/quant_system/.github/workflows/deploy.yml) 当前反映出以下事实：

- CI 只跑在 self-hosted `quant-server`
- 默认测试门是 `pytest tests/`
- performance tests 被排除
- lint 只跑 Ruff 的 E/W/F
- 没看到显式 `cargo test`
- 覆盖率门槛是 57%

这是一套“可用的基本护栏”，但还不是“强约束发布门”。

---

## 10. 测试结构评估

### 10.1 测试分布

`tests/` 当前分层相当清晰：

| 类别 | 文件数 |
|---|---:|
| unit | 209 |
| integration | 16 |
| replay | 5 |
| execution_safety | 5 |
| contract | 4 |
| performance | 4 |
| regression | 3 |
| persistence | 1 |

此外，[`execution/tests/`](/quant_system/execution/tests) 还有 28 个额外测试文件。

### 10.2 当前最有价值的测试护栏

当前最能体现架构成熟度的测试不是“数量多”，而是下列类型已经落地：

- runtime contract: [`tests/contract/test_runtime_contracts.py`](/quant_system/tests/contract/test_runtime_contracts.py)
- replay vs live equivalence: [`tests/replay/test_replay_vs_live_equivalence.py`](/quant_system/tests/replay/test_replay_vs_live_equivalence.py)
- live/backtest signal parity: [`tests/unit/decision/test_backtest_live_parity.py`](/quant_system/tests/unit/decision/test_backtest_live_parity.py)
- execution timeout / restart / late fill recovery: [`tests/integration/test_execution_timeout_restart_recovery.py`](/quant_system/tests/integration/test_execution_timeout_restart_recovery.py)
- execution rejection contract: [`tests/integration/test_execution_rejection_contract.py`](/quant_system/tests/integration/test_execution_rejection_contract.py)
- production wiring / model loader / operator control: [`tests/integration/test_production_integration_e2e.py`](/quant_system/tests/integration/test_production_integration_e2e.py)、[`tests/integration/test_operator_control_recovery_flow.py`](/quant_system/tests/integration/test_operator_control_recovery_flow.py)
- health server / CLI / model ops / scripts catalog

因此：

> 当前测试体系已经开始承担“架构收口工具”的角色，而不是只做功能回归。

### 10.3 当前测试盲区

测试的主要弱项也非常明确：

1. [`execution/tests/`](/quant_system/execution/tests) 不在默认 `pytest tests/` 与 `pyproject.toml` 的 `testpaths = ["tests"]` 门里。
2. performance tests 默认被 CI 忽略。
3. 部署清单和 GitHub workflow 本身没有看到对应验证。
4. Rust 默认发布门里没有显式 `cargo test`。
5. 部分遗留 research 入口没有活性测试，因此旧 import 漂移不会自动暴露。

---

## 11. 已确认的具体漂移与风险

这部分不是抽象技术债，而是已经能从当前代码直接确认的偏差。

### 11.1 Deploy workflow 与当前 compose 不一致

[`scripts/deploy.sh`](/quant_system/scripts/deploy.sh) 和 [`/.github/workflows/deploy.yml`](/quant_system/.github/workflows/deploy.yml) 会滚动重启：

- `paper-btc`
- `paper-sol`
- `paper-eth`

但当前根 [`docker-compose.yml`](/quant_system/docker-compose.yml) 实际定义的是：

- `paper-multi`

这说明部署自动化和当前 compose 真相并不一致，是明确的运维风险。

### 11.2 Docker 工件存在过时迹象

已经能直接确认两个具体问题：

1. 根 [`Dockerfile`](/quant_system/Dockerfile) 中有 `COPY _quant_hotpath/ /app/_quant_hotpath/`，但当前仓库根目录并不存在 `_quant_hotpath/` 路径。
2. [`deploy/docker/Dockerfile`](/quant_system/deploy/docker/Dockerfile) 依赖 `requirements.txt`，但当前仓库并没有对应 `requirements*.txt` 文件。

这说明：

> 仓库中的多套 Docker 工件至少有一部分已经脱离当前代码现实，不能把它们都当成当前生产真相源。

### 11.3 Config schema 与 runtime 解析存在偏差

[`infra/config/schema.py`](/quant_system/infra/config/schema.py) 当前记录了 `monitoring.metrics_port`，而 [`runner/live_runner.py`](/quant_system/runner/live_runner.py) 的 `from_config()` 实际处理的是：

- `health_port`
- `health_host`
- `health_auth_token_env`

[`infra/config/examples/live.yaml`](/quant_system/infra/config/examples/live.yaml) 里又使用了 `monitoring.metrics_port`。

这代表配置文档、schema 与 runtime 解析之间存在明显漂移。

### 11.4 `research/experiment.py` 仍引用已迁入 archive 的训练入口

[`research/experiment.py`](/quant_system/research/experiment.py) 当前仍然：

- `from scripts.train_lgbm import ...`

但仓库中实际存在的是：

- [`scripts/archive/train_lgbm.py`](/quant_system/scripts/archive/train_lgbm.py)

这是一个非常典型的“遗留 research 入口未纳入当前治理和测试护栏”的信号。

### 11.5 一部分 execution tests 不在默认 CI 门里

当前默认 CI 执行的是：

- `pytest tests/`

而不是：

- `pytest tests/ execution/tests/`

所以像 [`execution/tests/e2e/test_um_user_stream_trade_updates_state.py`](/quant_system/execution/tests/e2e/test_um_user_stream_trade_updates_state.py) 这类 execution 子树测试，默认并不在当前主测试门中。

### 11.6 K8s / Argo 工件更像候选生产形态

[`deploy/argocd/application.yaml`](/quant_system/deploy/argocd/application.yaml) 仍包含 `repoURL` placeholder，[`deploy/argocd/rollback-config.yaml`](/quant_system/deploy/argocd/rollback-config.yaml) 仍包含 Slack webhook placeholder。

这说明 K8s/GitOps 清单的成熟度不低，但当前更像候选生产形态，而不是唯一已落地真相源。

---

## 12. 总体完成度评估

### 12.1 已经形成的东西

当前已经明确形成的能力包括：

- 完整事件闭环
- 默认生产主路径
- Python + Rust 混合运行时
- 决策、风险、执行、恢复、监控、operator control 的制度边界
- model registry / loader / promote / rollback / audit
- script catalog 与共享 postprocess helper
- 跨路径 parity / contract / recovery 测试护栏

### 12.2 尚未完全收官的东西

当前最重要的未完成项包括：

1. 执行公共事实模型尚未完全统一成单一 fill / rejection 语义
2. Python/Rust 双栈 ownership 仍然需要持续收口
3. `scripts/` 的 flat 目录仍然很重，治理还未覆盖全部历史沉积
4. 部署工件与 CI 门的统一程度明显落后于 runtime 和 execution 代码本身
5. `portfolio/`、generic venue abstraction、standalone Rust runtime 等能力强于当前主路径整合度

### 12.3 最准确的阶段判断

如果必须用一句话总结当前所处阶段：

> 这个仓库已经从“搭平台”阶段进入“平台已经成立、但需要系统性收口真相源和维护边界”的阶段。

---

## 13. 下一阶段最值得做的事

从代码现状看，下一阶段最值得优先处理的不是再加功能，而是继续做 5 件收口工作：

1. 把部署真相源统一下来，修复 compose / deploy script / Dockerfile / K8s 清单之间的漂移。
2. 把 `execution/tests/` 和 Rust tests 纳入默认 CI 门，提升真实发布可信度。
3. 继续收敛 live/backtest/research 共享约束状态机，减少重复实现。
4. 继续统一 `CanonicalFill`、公共事件层和 ingress 事实模型。
5. 明确长期 runtime 方向: Python 主装配长期保留，还是逐步让 Rust binary 成为默认入口。

---

## 14. 结论

当前对整个项目最准确、也最不失真的总结如下：

- 这是一个已经具备完整交易平台横截面的代码库，而不是策略脚本集合。
- 当前默认生产路径仍是 [`runner/live_runner.py`](/quant_system/runner/live_runner.py) 主导的 Python runtime。
- Rust 已经深度进入 state / route / feature / constraint / execution 原语，但尚未完成对 Python runtime 的总替换。
- `decision`、`risk`、`execution`、`monitoring` 这几层已经具有明显的制度化和生产候选气质。
- `research`、`scripts`、`deploy` 则体现出典型的“功能很强，但仍在治理收口中”的特征。
- 项目当前最大的风险不是功能缺失，而是多轨工件、遗留入口、配置与部署漂移、以及双栈带来的长期维护成本。

因此，当前最准确的总体评价是：

> 这是一个功能闭环已形成、生产候选能力很强、执行与风险子系统明显成熟、但仍需继续收口脚本层与部署层真相源的量化交易平台代码库。
