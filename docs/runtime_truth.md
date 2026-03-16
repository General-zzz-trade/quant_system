# Runtime Truth

> 更新时间: 2026-03-12
> 目标: 作为当前运行时真相源，回答“现在系统到底怎么跑、谁拥有什么、哪些是现状、哪些是演进方向”
> 角色: 当前代码库关于 runtime 路径和 ownership 的最高优先级文档

---

## 1. 当前唯一生产主路径

当前仓库的默认生产主路径是：

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

结论：

- `runner/live_runner.py` 是当前 Python 运行时真相源（engine 层默认入口）
- `scripts/ops/run_bybit_alpha.py` 是当前实际生产 alpha 入口（轻量 runner，直接调 Bybit/Binance API，不经过完整 engine 层）
- `runner/backtest_runner.py`、`runner/replay_runner.py` 是共享事件语义的验证路径
- `ext/rust/src/bin/main.rs` 是重要的 standalone Rust trader 演进路径，但不是当前默认生产入口

当前实际部署两套入口并存：

| 入口 | 用途 | 部署方式 | 状态 |
|------|------|---------|------|
| `scripts/ops/run_bybit_alpha.py` | 多品种 alpha 交易 (ETH 1h+15m, SUI, AXS) | systemd `bybit-alpha.service` | **活跃运行** |
| `runner/live_runner.py` | 完整 engine 层入口 | docker `paper-multi` | 停用 |

---

## 2. 运行路径定位

| 路径 | 当前定位 | 真相源级别 |
|---|---|---|
| `runner/live_runner.py` | 当前默认生产入口 | 高 |
| `runner/backtest_runner.py` | 历史回测入口 | 高 |
| `runner/replay_runner.py` | 回放与一致性验证 | 高 |
| `runner/paper_runner.py` | 简化模拟/演示路径 | 中 |
| `ext/rust/src/bin/main.rs` | 候选独立运行时 | 中 |

说明：

- 当前“生产运行规则”应优先以 `live_runner` 为准
- Rust binary 能力很强，但尚未替代 Python 主装配路径
- 文档中若出现“系统现在完全由 Rust runtime 驱动”的说法，应视为演进目标而非当前事实

---

## 3. Ownership Matrix

| 子系统 | 当前 owner | 说明 |
|---|---|---|
| 事件语义 | Python 主导，Rust 有验证/解析辅助 | `event/` 是制度源，Rust 提供 validators/parsers |
| 状态存储与推进 | Rust 主导 | `RustStateStore.process_event()` 是核心写通道 |
| pipeline 外层封装 | Python | `engine/pipeline.py` 负责桥接、快照、懒转换 |
| 运行时总控 | Python | `EngineCoordinator`, `LiveRunner`, `EngineLoop` |
| 特征热路径 | Rust 主导，Python 编排 | `FeatureComputeHook` 管理 `RustFeatureEngine` |
| 推理约束状态 | Rust 主导，Python 编排 | `LiveInferenceBridge` 依赖 `RustInferenceBridge` |
| 决策编排 | Python | `DecisionBridge`, `decision/engine.py` |
| 风控原语 | 混合 | Rust 有 risk primitives，Python 有 aggregator/gates/live wiring |
| 执行适配器 | Python 主导 | 交易所 IO、adapter (Binance/IB/Polymarket/CCXT)、reconcile 主要在 Python |
| WS 热路径 | 混合 | Python adapter + Rust transport / parser / ws client |
| 独立交易二进制 | Rust | `quant_trader`，但当前不是默认主路径 |
| 研究训练 | Python | `research/`, `alpha/training/`, `scripts/` |
| 监控与运维 | Python | `monitoring/`, `infra/`, `deploy/` |

---

## 4. 当前统一事件链

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

---

## 5. 契约基线

这部分不是完整 schema，而是当前开发必须共同遵守的最小基线。

### 5.1 Event

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
- 控制/风险: `RISK`, `CONTROL`

### 5.2 Snapshot

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

### 5.3 OrderSpec / OrderEvent

最小字段：

- `order_id`
- `intent_id`
- `symbol`
- `side`
- `qty`
- `price` 可空

### 5.4 Fill

最小字段：

- `fill_id`
- `order_id`
- `symbol`
- `qty`
- `price`

说明：

- 通用 `event.types.FillEvent` 当前不含 `side`
- 部分 execution 子系统内部 fill-like 对象会额外携带 `side`
- 因此 `side` 目前不能作为所有运行路径都成立的强制基线

### 5.5 一致性要求

live / backtest / replay 至少需要对齐：

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

## 6. 当前不是事实的说法

以下说法不应继续作为“当前现状”写进文档：

- “系统已经完全 Rust-only runtime”
- “Python 仅剩配置和胶水”
- “当前默认生产入口是 standalone Rust binary”
- “所有 fallback 都已移除”

更准确的现状表述是：

> 当前系统是 Python 主编排、Rust 深度接管热路径和核心状态推进的混合运行时。

---

## 7. 本文档的用途

本文档用于约束后续开发：

- 修改 README / operations / api 文档时，以此为准
- 讨论 Rust 替换范围时，以此为起点
- 设计 live/backtest/replay 一致性测试时，以本契约基线为准

后续若 Rust binary 成为默认生产入口，应更新本文档，而不是让 README 单独漂移。
