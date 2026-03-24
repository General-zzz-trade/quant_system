# Quant System 架构研究报告

> 更新时间: 2026-03-24
> 判断原则: 以当前源码为准；当前活跃交易服务以 CLAUDE.md 和 systemd 配置为准

---

## 1. 执行摘要

这是一个 **"Python 编排 + Rust 热路径"** 的量化交易平台。经过 2026-03-24 的全面重构：

- **框架原生**: AlphaDecisionModule 通过 EngineCoordinator 驱动，替代了旧的 AlphaRunner god class
- **Rust 深度接管**: 198 个 PyO3 导出，100% 被 Python 生产代码调用
- **零死代码**: 所有 Python 库文件都有生产引用
- **零超限文件**: 所有生产文件 ≤800 行硬上限

核心结论：

1. 生产入口是 `runner/alpha_main.py`（EngineCoordinator + AlphaDecisionModule）
2. Rust 已接管：事件类型、状态管理、特征计算、信号约束、风控 gate、执行安全
3. Python 负责：运行时编排、交易所 IO、模型训练、运维监控、研究分析
4. event/ 和 state/ 模块是 Rust 主导（76% / 84%），Python 只是薄包装
5. 框架调用率 100%：所有框架模块（coordinator, pipeline, bridges）都在生产路径上

---

## 2. 仓库规模

按 2026-03-24 统计：

| 指标 | 当前值 |
|------|-------:|
| Python 生产文件 | 483 |
| Python 生产代码行 | ~102,000 |
| Rust 源文件 | 102 |
| Rust 代码行 | ~35,000 |
| PyO3 导出 | 198 (100% 使用) |
| 测试文件 | 334 |
| 测试通过 | 3,033 |

### 模块 Rust 化程度

| 模块 | Python | Rust | Rust% | 状态 |
|------|--------|------|-------|------|
| state/ | 549 | 3,048 | 84% | Rust 主导 |
| event/ | 550 | 1,773 | 76% | Rust 主导 |
| risk/ | 2,320 | 4,434 | 65% | 混合 |
| features/ | 4,696 | 7,060 | 60% | 混合 |
| regime/ | 436 | 597 | 57% | 混合 |
| engine/ | 3,172 | 2,704 | 46% | 混合 |
| decision/ | 5,825 | 4,621 | 44% | 混合 |
| execution/ | 12,982 | 1,339 | 9% | 轻度 Rust |
| runner/ | 15,044 | 0 | 0% | 纯 Python |
| alpha/ | 3,619 | 0 | 0% | 纯 Python |
| 其他 | ~20,000 | 0 | 0% | 纯 Python |

---

## 3. 生产架构

### 3.1 数据流

```
Bybit WS kline → MarketEvent → EngineCoordinator.emit()
  │
  ├─ EventDispatcher → Route.PIPELINE
  ├─ FeatureComputeHook.on_event() → RustFeatureEngine → 120+ 特征
  ├─ StatePipeline.apply() → RustStateStore (Rust 堆上状态)
  └─ DecisionBridge.on_pipeline_output()
      └─ AlphaDecisionModule.decide(snapshot)
          ├─ Regime filter (自适应 p20/p25 百分位)
          ├─ EnsemblePredictor: Ridge(60%)+LGBM(40%)
          ├─ SignalDiscretizer: z-score → z-clamp → deadzone → min-hold
          ├─ Force exits: ATR 止损, 快速止损, z 反转, 4h 反转
          ├─ Direction alignment: ETH 跟随 BTC
          └─ AdaptivePositionSizer: 资金分层 × IC × vol
  │
  └─ OrderEvent → ExecutionBridge → BybitExecutionAdapter
      └─ FillEvent → StatePipeline (状态更新)
```

### 3.2 入口分工

| 入口 | 定位 | 状态 |
|------|------|------|
| `runner/alpha_main.py` | **生产**: 框架原生 alpha | ✅ ACTIVE |
| `runner/live_runner.py` | 框架 LiveRunner (替代运行时) | 备用 |
| `runner/backtest_runner.py` | 回测引擎 | ✅ ACTIVE |

### 3.3 关键文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `decision/modules/alpha.py` | 415 | 核心决策逻辑 |
| `decision/signals/alpha_signal.py` | 186 | 预测 + z-score 离散化 |
| `decision/sizing/adaptive.py` | 147 | 自适应仓位计算 |
| `runner/alpha_main.py` | 427 | 生产入口 + Coordinator 接线 |
| `engine/coordinator.py` | 397 | 事件编排中心 |
| `engine/pipeline.py` | 290 | 状态转换管线 |
| `engine/feature_hook.py` | 489 | 特征计算桥接 |
| `execution/adapters/bybit/execution_adapter.py` | 96 | Bybit 执行桥接 |

---

## 4. Rust 集成

### 4.1 架构

```
项目根/
  Cargo.toml              ← 构建配置
  rust_lib.rs             ← 编译入口 (#[path] 引用各模块)

  event/rust/             ← 9 个 .rs (EventHeader, 事件类型, 验证器)
  state/rust/             ← 19 个 .rs (类型, reducer, store)
  features/rust/          ← 15 个 .rs (FeatureEngine, 指标, 增量追踪器)
  decision/rust/          ← 11 个 .rs (约束管线, 推理桥, ML 预测)
  engine/rust/            ← 9 个 .rs (tick_processor, pipeline, guards)
  execution/rust/         ← 6 个 .rs (订单状态机, WS 客户端)
  risk/rust/              ← 6 个 .rs (gate_chain, 风险引擎, 自适应止损)
  regime/rust/            ← 1 个 .rs (CompositeRegimeDetector)
  research/rust/          ← 8 个 .rs (回测, PnL, 特征选择)
  common_rust/            ← 15 个 .rs (签名, 哈希, saga, SPSC ring)
```

### 4.2 关键 Rust 组件

| 组件 | 用途 | 性能 |
|------|------|------|
| RustTickProcessor | 完整热路径: 特征+预测+状态 | ~80μs/tick |
| RustFeatureEngine | 120+ 增量特征计算 | ~35μs/bar |
| RustStateStore | 状态真相源 (Fd8 定点数) | ~15μs/event |
| RustInferenceBridge | z-score + 约束管线 | ~5μs |
| RustAdaptiveStopGate | ATR 三阶段追踪止损 | ~1μs |
| RustGateChain | 配置化 gate 管线 | ~3μs |

### 4.3 Python↔Rust 边界

- **Rust 主导**: event 类型、state 类型、特征计算、信号约束、风控 gate
- **Python 主导**: 运行时编排、交易所 API、模型训练/加载、运维监控
- **混合**: engine (热路径 Rust, 编排 Python), execution (安全 Rust, API Python)
- **零桥接层**: state/ 删除了 rust_adapters.py，Rust 类型直接使用

---

## 5. 信号管线

```
Raw prediction (Ridge 60% + LightGBM 40% ensemble, T-1 cross-market features)
  → Rolling z-score (4h: window=180/warmup=45, 1h: 720/180)
  → Z-score clamp: |z|>3.5 with no position → cap ±3.0
  → Vol-adaptive deadzone: dz × (rv_20 / rolling_vol_median), clamped [0.5x, 2.0x]
  → Discretize: z > deadzone → +1, z < -deadzone → -1, else 0
  → Adaptive min-hold (base × vol_ratio^0.5)
  → Direction alignment: ETH blocked if opposing BTC consensus
  → Force exits: ATR stop, quick loss, z-reversal, 4h reversal
  → AdaptivePositionSizer: equity-tier × IC × leverage × z_scale
```

---

## 6. 风控体系

| 层级 | 组件 | 实现 |
|------|------|------|
| 信号层 | Regime filter | 自适应 p20/p25 百分位（Python） |
| 信号层 | Z-score clamp | |z|>3.5 → ±3.0（Python） |
| 信号层 | Direction alignment | ETH 跟随 BTC（Python） |
| 仓位层 | Equity-tier cap | <$500: 35%, $500-10K: 25%, >$10K: 18%（Python） |
| 仓位层 | IC health scaling | GREEN=1.2x, YELLOW=0.8x, RED=0.4x（Python） |
| 仓位层 | Portfolio leverage check | >5x 告警（Python） |
| 退出层 | ATR 三阶段止损 | 初始→保本→追踪（Python + RustAdaptiveStopGate） |
| 退出层 | Quick loss | -1% 逆向 → 平仓（Python） |
| 退出层 | Z-reversal | z < -0.3 平多, z > 0.3 平空（Python） |
| 退出层 | 4h reversal | 4h 信号反转 → 1h 平仓（Python） |
| 安全层 | Circuit breaker | RustCircuitBreaker (3 失败/120s) |
| 安全层 | Kill switch | RustKillSwitch (lock-free) |
| 安全层 | Order limiter | RustOrderLimiter (频率+额度) |
| 安全层 | Duplicate guard | RustDedupStore (持久化去重) |

---

## 7. Alpha 研究结论 (T-1 修正)

| 方向 | Sharpe | 结论 |
|------|--------|------|
| **4h alpha** | **3.62-4.57** | 最强时间框架 |
| 1h alpha | 2.43-3.92 | 调节器角色 |
| 15m alpha | -1.36-0.27 | 失败，已禁用 |
| 5m/1m HFT | -5 到 -25 | 成本 > 信号 |
| 做市 | 不可行 | 逆向选择 > 价差 |
| 神经网络 | 不如 Ridge | Ridge > MLP > LGBM (4h OOS) |
| 跨所套利 | 不可行 | 价差 < 费用 |
| 最强特征 | DVOL zscore (IC=0.074), ETF volume (IC=0.11), funding zscore (IC=0.052) |

---

## 8. 运维自动化

| 定时器 | 频率 | 用途 |
|--------|------|------|
| health-watchdog | 每 5 分钟 | 服务健康 + 数据新鲜度 + 自动重启 + Telegram |
| data-refresh | 每 6 小时 | K 线 + 资金费率 + OI 同步 |
| daily-retrain | 每天 2am UTC | 4h 模型轻量重训 + SIGHUP 热加载 |
| auto-retrain | 每周日 2am UTC | 1h 模型走步验证重训 |
| ic-decay-monitor | 每天 3am UTC | IC 衰减检测 + 自动重训触发 |

---

## 9. 代码质量指标

| 指标 | 值 |
|------|------|
| 死库文件 | 0 |
| 超 800 行文件 | 0 (生产) |
| Rust 导出使用率 | 198/198 = 100% |
| 测试通过 | 3,033 / 3,058 (99.2%) |
| 框架调用率 | 100% |
| Python↔Rust 桥接层 | 0 (直接使用 Rust 类型) |

---

## 10. 交易所适配

| 交易所 | 协议 | 状态 | 费率 |
|--------|------|------|------|
| Bybit | REST V5 + WS | ✅ 生产 | 0.02% maker |
| Hyperliquid | REST | 已接入 | 0% maker |
| Binance | REST + WS | 已搭建 | 0.02% maker |
| Polymarket | CLOB REST | dry-run | 0% |
