# Runtime Contracts

> 更新时间: 2026-03-12
> 目标: 对比 live / backtest / replay 三条路径的当前契约差异，并给出统一基线
> 适用范围: 当前默认 Python runtime 及其验证路径
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 三条路径当前定位

| 路径 | 入口 | 当前定位 |
|---|---|---|
| live | [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | 默认生产路径 |
| backtest | [`runner/backtest_runner.py`](/quant_system/runner/backtest_runner.py) | 历史回测与结果验证 |
| replay | [`runner/replay_runner.py`](/quant_system/runner/replay_runner.py) | 历史事件重放与一致性验证 |

---

## 2. 当前共同点

三条路径都围绕同一组核心概念运转：

- 事件驱动
- `EngineCoordinator` 作为总控
- `EventDispatcher` 作为路由器
- `StatePipeline` / Rust-backed store 作为状态写通道
- `StateSnapshot` 作为决策读取边界

共同基线：

- `IntentEvent` / `OrderEvent` 是意见事件
- `MarketEvent` / `FillEvent` / `FundingEvent` 是事实事件
- 决策层原则上读取 snapshot，不直接写状态
- 执行结果应重新注入事件链，而不是直接改状态

---

## 3. 当前差异

### 3.1 live

live 路径额外具备：

- `FeatureComputeHook`
- `LiveInferenceBridge` / `unified_predictor`
- `ExecutionBridge`
- venue client / user stream
- `RiskGate` / `CorrelationGate` / `KillSwitchBridge`
- `MarginMonitor` / `ReconcileScheduler` / `HealthServer`

结论：

- live 是最完整的运行路径
- 也是当前默认生产真相源

### 3.2 backtest

backtest 路径具备：

- `EngineCoordinator`
- `DecisionBridge`
- `ExecutionBridge`
- `BacktestExecutionAdapter`
- 指标汇总、fills、summary

当前特点：

- 使用相同事件概念
- 但执行端是模拟 adapter
- 策略模块可能使用更轻量的 snapshot 访问方式

### 3.3 replay

replay 路径当前更偏“事件再注入器”：

- `run_replay()` 创建 `EngineCoordinator`
- 使用 `EventReplay` 将历史事件重新注入 dispatcher
- 默认不自动装配 live 的 decision / execution / monitoring 全栈

结论：

- replay 当前重点是重放和一致性验证
- 它不是完整 live 栈的等价装配器
- 因此“replay 与 live 完全等价”目前不能只靠入口结构保证，必须靠测试约束

---

## 4. 统一字段基线

### 4.1 Header

当前稳定基线：

- `event_id`
- `event_type`
- `version`
- `ts_ns`
- `source`

### 4.2 IntentEvent

最小字段：

- `intent_id`
- `symbol`
- `side`
- `target_qty`
- `reason_code`
- `origin`

### 4.3 OrderEvent

最小字段：

- `order_id`
- `intent_id`
- `symbol`
- `side`
- `qty`
- `price`

### 4.4 FillEvent

最小字段：

- `fill_id`
- `order_id`
- `symbol`
- `qty`
- `price`

注意：

- `side` 不是当前所有 fill 形态的统一强制字段

### 4.5 StateSnapshot

决策读取最小字段：

- `symbol`
- `ts`
- `event_id`
- `event_type`
- `bar_index`
- `markets`
- `positions`
- `account`
- `portfolio`
- `risk`

---

## 5. 下一步收口要求

为了让三条路径更一致，后续开发应满足：

1. live / backtest / replay 共享同一组最小字段测试
2. replay 不再长期保留占位测试
3. 线上信号约束应有明确对齐测试
4. execution 返回的 fill-like 结果要逐步收敛到统一规范

---

## 6. 本轮已完成的约束对齐

已完成：

- replay 占位测试替换为真实测试
- 新增 runtime contract 测试，锁定 snapshot / intent / order / fill 最小字段
- `decision/backtest_module.py` 已补充以下 live-like 约束开关：
  - `trend_follow`
  - `trend_indicator`
  - `trend_threshold`
  - `monthly_gate`
  - `monthly_gate_window`
  - `vol_target`
  - `vol_feature`
- `decision/backtest_module.py` 的 `trend_follow` 已从“入场门禁”收敛为更接近 live 的“deadzone 场景下持仓延续”语义
- `decision/backtest_module.py` 的 `monthly_gate` 已从“仅限制开仓”收敛为更接近 live 的“门限失效时平仓”语义
- `scripts/backtest_engine.py` 已从模型 `config.json` 透传上述回测约束
- 已新增 cross-path parity tests，直接对比 live bridge 与 backtest module 的：
  - trend hold
  - monthly gate flatten
  - min_hold flip timing
  - deadzone entry blocking
  - deadzone fade flattening
  - vol_target sizing scale
- `decision/backtest_module.py` 的持仓退出语义已进一步收口：
  - 当离散化后的信号回到 deadzone 内时，如果未触发 trend hold 且已满足 `min_hold`，则按 live 语义主动平仓

仍未完全对齐：

- backtest 模块的信号约束实现仍是 Python 近似实现，不等价于 `LiveInferenceBridge` 的 Rust 约束状态机
- trend hold / monthly gate / vol targeting 在不同回测脚本之间仍有重复实现
- live 与部分研究脚本之间仍存在约束逻辑散落问题

已开始收口：

- `scripts/signal_postprocess.py` 已作为研究脚本共享后处理模块落地
- `walkforward_validate.py`、`walkforward_validate_1m.py`、`backtest_alpha_v8.py`、`backtest_portfolio.py`、`train_short_production.py`、`train_multi_horizon.py`、`backtest_small_cap.py`、`backtest_honest.py`、`train_btc_v9b.py`、`train_eth_v9.py` 已开始共享 monthly gate / trend hold / vol_target / min_hold / rolling z-score / discrete exit rule 相关 helper
- `ic_analysis_short_features.py` 已改为复用共享 bear mask helper，并通过本地 wrapper 保持原有 warmup 区间全 False 的研究语义

---

## 7. 恢复链路第一轮加固

已完成：

- `LiveRunner` 新增定向恢复测试：
  - user stream step 异常后重连
  - main loop 中 timeout tracker 持续检查
  - startup reconcile 的 position / balance mismatch 检测
- 修复 `LiveRunner.stop()` 在 `_running` 已提前清零时可能跳过资源清理的问题
- 修复 `LiveRunner._apply_perf_tuning()` 在部分机器上读取 `nohz_full=(null)` 时崩溃的问题
- 修复 startup reconcile 本地余额读取错误，改为读取 `account.balance`

仍未完成：

- 断连、重复 fill、乱序 fill、reconcile、healer 的完整端到端联动测试
- restart 后 checkpoint restore 与 periodic reconcile 的更强一致性校验
