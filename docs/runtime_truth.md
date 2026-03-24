# Runtime Truth

> 更新时间: 2026-03-20
> 目标: 作为当前运行时真相源，回答“现在系统到底怎么跑、谁拥有什么、哪些是现状、哪些是演进方向”
> 角色: 当前代码库关于 runtime 路径和 ownership 的最高优先级文档

---

## 1. 当前活跃运行时

当前仓库不是“单入口单服务”状态，而是三条路径并存：

| 路径 | 代码入口 | 部署 / 启动方式 | 当前定位 |
|---|---|---|---|
| 方向性 alpha | [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) （由 [`scripts/run_bybit_alpha.py`](/quant_system/scripts/run_bybit_alpha.py) 转发） | systemd `bybit-alpha.service` | **当前活跃的多品种 Bybit demo 交易路径** |
| 高频做市 | [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) | systemd `bybit-mm.service` | **当前活跃的专用 ETHUSDT 做市路径** |
| Framework live runtime | [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | `quant-framework` / `infra/systemd/quant-runner.service` | 完整 engine 框架路径；候选 / 收敛路径，不是当前主机上的默认交易服务 |

结论：

- 当前主机上的方向性交易真相以 `scripts/ops/run_bybit_alpha.py` 为准
- 当前主机上的做市交易真相以 `scripts/run_bybit_mm.py` 为准
- `runner/live_runner.py` 仍然是 framework runtime 的真相源，但不是当前默认 host trading service
- `rust/src/bin/main.rs` 仍是重要候选路径，不是当前默认入口

---

## 2. 当前默认框架路径与当前活跃交易路径

需要明确区分两个概念：

### 2.1 当前活跃交易路径

- `bybit-alpha.service` 调的是 `python3 -m scripts.run_bybit_alpha --symbols ... --ws`
- `scripts/run_bybit_alpha.py` 只是兼容包装层，真正逻辑在 [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py)
- 当前 `run_bybit_alpha.py` 实现仍然走 `AlphaRunner` / `PortfolioManager` / `PortfolioCombiner` 这条轻量路径
- 当前 `run_bybit_alpha.py` 的 `--ws`、REST poll 和 `--once` 已共享同一套 `PortfolioManager / PortfolioCombiner / kill enforcement` 语义；`--ws` 仍是 host service 的默认模式
- 当前 `run_bybit_alpha.py` 不再提供 `--legacy` / `LiveRunner` CLI 切换；如果需要 framework runtime，应直接使用 [`runner/live_runner.py`](/quant_system/runner/live_runner.py)
- 当前 kill 语义以 active alpha path 为准：kill 后禁止新开仓，并在后续 bar 上强制把 COMBO 与本地 runner 状态收回 flat
- active host services 现在还会把 hard-kill 持久化到 `data/runtime/kills/`；重启不会绕过这层风控，必须人工 clear

### 2.2 当前默认框架路径

`runner/live_runner.py` 仍然是 framework 层的标准装配入口：

```text
runner/live_runner.py
  -> EngineCoordinator
  -> EventDispatcher
  -> StatePipeline / RustStateStore
  -> FeatureComputeHook
  -> DecisionBridge
  -> ExecutionBridge
  -> venue adapter / reconcile / monitoring / risk gates
```

它的角色是：

- framework live 语义真相源
- backtest / replay / contract / recovery 对齐工作的上位参考
- 长期收敛目标

它当前不是：

- 当前默认的 directional alpha host service
- 当前默认的做市 service

---

## 3. 路径定位

| 路径 | 当前定位 | 真相源级别 |
|---|---|---|
| [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) | **当前活跃方向性生产入口** | 最高 |
| [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) | **当前活跃做市入口** | 最高 |
| [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | framework live 入口（候选 / 收敛路径） | 高 |
| [`runner/backtest_runner.py`](/quant_system/runner/backtest_runner.py) | 历史回测入口 | 高 |
| [`runner/replay_runner.py`](/quant_system/runner/replay_runner.py) | 回放与一致性验证 | 高 |
| [`runner/paper_runner.py`](/quant_system/runner/paper_runner.py) | 简化模拟 / 演示路径 | 中 |
| [`rust/src/bin/main.rs`](/quant_system/rust/src/bin/main.rs) | 候选独立运行时 | 中 |

说明：

- 文档里如果谈“当前活跃交易服务”，优先以 `run_bybit_alpha.py` / `run_bybit_mm.py` 为准
- 文档里如果谈“framework live / backtest / replay 统一契约”，优先以 `runner/live_runner.py` 为准
- 文档里如果谈“当前默认部署方式”，必须区分 systemd host services 与 compose / GitHub Actions

---

## 4. Ownership Matrix

| 子系统 | 当前 owner | 说明 |
|---|---|---|
| 事件语义 | Python 主导，Rust 有验证 / 解析辅助 | `event/` 是制度源 |
| 状态存储与推进 | Rust 主导 | `RustStateStore.process_event()` 是核心写通道 |
| pipeline 外层封装 | Python | `engine/pipeline.py` 负责桥接、快照、懒转换 |
| framework 运行时总控 | Python | `EngineCoordinator`, `LiveRunner`, `EngineLoop` |
| 活跃方向性交易总控 | Python | `scripts/ops/run_bybit_alpha.py`, `AlphaRunner`, `PortfolioManager` |
| 活跃做市总控 | Python | `scripts/run_bybit_mm.py`, `BybitMMRunner` |
| 特征热路径 | Rust 主导，Python 编排 | `FeatureComputeHook` / `RustFeatureEngine` / AlphaRunner 内的 RustFeatureEngine |
| 推理约束状态 | Rust 主导，Python 编排 | `LiveInferenceBridge` / `RustInferenceBridge` |
| 决策编排 | Python | `DecisionBridge`, `decision/engine.py`，以及 AlphaRunner 内部策略逻辑 |
| 风控原语 | 混合 | Rust primitives + Python wiring / aggregator / gates |
| 执行适配器 | Python 主导 | Binance / Bybit / Polymarket IO、adapter、reconcile |
| WS 热路径 | 混合 | Python adapter + Rust transport / parser / ws client |
| 独立交易二进制 | Rust | `quant_trader`，但当前不是默认主路径 |
| 研究训练 | Python | `research/`, `alpha/training/`, `scripts/` |
| 监控与运维 | Python | `monitoring/`, `infra/`, `deploy/` |

---

## 5. Framework 路径上的统一事件链

标准事件闭环：

```text
Market data / Replay / Backtest input
  -> coordinator.emit()
  -> dispatcher route
  -> pipeline/store.process_event()
  -> snapshot
  -> decision modules
  -> IntentEvent / OrderEvent
  -> execution bridge
  -> fill/reject-like result
  -> dispatcher reinjection
  -> pipeline state advance
```

约束：

- 决策不直接改状态
- 执行不直接改状态
- 事实事件推进状态
- snapshot 是 decision 的标准输入

说明：

- 这条事件链是 framework / contract / replay / recovery 的制度中心
- 当前 `bybit-alpha.service` 并不等价于这条完整链路；它复用了部分 Rust 组件和 execution adapter，但不是同一装配面
- 当前 `bybit-mm.service` 是完全独立的做市运行时，不应被误写成走 `LiveRunner`

---

## 6. 契约基线

这部分不是完整 schema，而是当前开发必须共同遵守的最小基线。

### 6.1 Event

最小要求：

- 必须有 `event_type`
- 必须有 `header`
- `header` 至少应稳定提供 `event_id`
- `header` 的稳定时间字段当前是 `ts_ns`
- 某些事件本体另外提供业务时间字段，例如 `MarketEvent.ts`
- 事实事件与意见事件必须语义分离

核心类型：

- 事实: `MARKET`, `FILL`, `FUNDING`
- 意见: `SIGNAL`, `INTENT`, `ORDER`
- 控制 / 风险: `RISK`, `CONTROL`

### 6.2 Snapshot

决策输入最小字段：

- `markets`
- `account`
- `positions`
- `portfolio`
- `risk`
- `event_id`
- `ts`

约束：

- decision 模块只读 snapshot
- snapshot 不应由 decision / execution 原地修改

### 6.3 OrderSpec / OrderEvent

最小字段：

- `order_id`
- `intent_id`
- `symbol`
- `side`
- `qty`
- `price` 可空

### 6.4 Fill

最小字段：

- `fill_id`
- `order_id`
- `symbol`
- `qty`
- `price`

说明：

- 通用 `event.types.FillEvent` 当前不含强制 `side`
- 部分 execution 子系统内部 fill-like 对象会额外携带 `side`
- 因此 `side` 目前不能作为所有运行路径都成立的强制基线

### 6.5 一致性要求

framework live / backtest / replay 至少需要对齐：

- 事件语义
- snapshot 最小字段
- 订单与成交最小字段
- 线上信号约束
  - `min_hold`
  - `deadzone`
  - trend gate
  - monthly gate
  - vol target

---

## 7. 当前不是事实的说法

以下说法不应继续作为“当前现状”写进文档：

- “系统已经完全 Rust-only runtime”
- “Python 仅剩配置和胶水”
- “当前默认生产入口是 standalone Rust binary”
- “`scripts.run_bybit_alpha` 默认已经切到 `LiveRunner`”
- “当前所有交易服务都走同一套 recovery / health / control plane”

更准确的现状表述是：

> 当前系统是 Python 主编排、Rust 深度接管热路径和核心状态推进的混合运行时；host 上活跃交易服务仍是 `run_bybit_alpha` 与 `run_bybit_mm`，而 `LiveRunner` 是 framework 真相源与收敛路径。

---

## 8. 本文档的用途

本文档用于约束后续开发：

- 修改 README / operations / api / deploy 文档时，以此为准
- 讨论 active service 与 framework path 时，先区分 `bybit-alpha` / `bybit-mm` / `LiveRunner`
- 设计 live / backtest / replay 一致性测试时，以 framework contract 为准

后续若 `LiveRunner` 真正取代 `run_bybit_alpha` 成为默认 host service，应先更新本文档，再改其他文档。
