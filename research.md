# Quant System — 深度代码库研究报告

> 版本: v3.0.0 (Rust 内核完成)
> 语言: Python 3.12 + Rust (PyO3)
> 定位: 事件驱动的加密货币永续合约量化交易系统
> 规模: 942 Python 文件 + 58 Rust 模块 (~18,100 LOC)
> 测试: 2,644 Python + 52 Rust 测试
> 交易所: Binance, Bitget

---

## 目录

1. [系统概览与设计哲学](#1-系统概览与设计哲学)
2. [顶层架构与模块地图](#2-顶层架构与模块地图)
3. [Rust 内核 (`ext/rust/`)](#3-rust-内核-extrust)
4. [引擎层 (`engine/`)](#4-引擎层-engine)
5. [状态管理 (`state/`)](#5-状态管理-state)
6. [事件系统 (`event/`)](#6-事件系统-event)
7. [特征工程 (`features/`)](#7-特征工程-features)
8. [决策系统 (`decision/`)](#8-决策系统-decision)
9. [Alpha 模型 (`alpha/`)](#9-alpha-模型-alpha)
10. [风控系统 (`risk/`)](#10-风控系统-risk)
11. [执行层 (`execution/`)](#11-执行层-execution)
12. [投资组合 (`portfolio/`)](#12-投资组合-portfolio)
13. [市场体制识别 (`regime/`)](#13-市场体制识别-regime)
14. [基础设施 (`infra/`, `monitoring/`)](#14-基础设施-infra-monitoring)
15. [运行器 (`runner/`)](#15-运行器-runner)
16. [研究与脚本 (`research/`, `scripts/`)](#16-研究与脚本-research-scripts)
17. [测试体系](#17-测试体系)
18. [端到端数据流](#18-端到端数据流)
19. [关键设计模式](#19-关键设计模式)
20. [Rust 迁移历程](#20-rust-迁移历程)
21. [Walk-Forward 策略表现](#21-walk-forward-策略表现)

---

## 1. 系统概览与设计哲学

### 1.1 定位

**机构级加密货币永续合约量化交易系统**，支持 Binance / Bitget 等交易所：

- **回测** (backtest)：读取历史 CSV OHLCV 数据，模拟完整交易生命周期
- **模拟盘** (paper trading)：与实时数据连接但不实际下单
- **实盘** (live trading)：通过交易所 API 实际执行订单

### 1.2 核心设计哲学

| 原则 | 实现方式 |
|------|----------|
| **Rust 热路径** | 所有计算密集型代码在 Rust 中执行，Python 仅做编排 |
| **事件驱动** | 所有状态变更由事件触发，通过 Dispatcher 路由 |
| **确定性** | 相同事件序列 → 相同状态结果 |
| **不可变状态** | 所有 State/Event 都是 frozen dataclass 或 Rust i64 定点数 |
| **单写通道** | 状态仅通过 `rust_pipeline_apply()` 修改 |
| **可审计** | EventHeader 因果链追溯、事件去重、全链路日志 |

### 1.3 双语代码风格

- 类/函数名：英文
- 注释和文档：中文
- 事件术语：事实事件(MARKET/FILL)、意见事件(SIGNAL/INTENT)、命令事件(ORDER)

---

## 2. 顶层架构与模块地图

```
quant_system/
├── ext/rust/        # Rust PyO3 内核 (58 模块, ~18,100 LOC)
├── _quant_hotpath/  # 编译后的 Rust .so 文件
├── engine/          # 引擎层：Coordinator, Pipeline, Dispatcher, FeatureHook
├── state/           # 状态层：Rust 适配器 + Python 类型定义
├── event/           # 事件层：事件类型、Header、存储、回放、安全
├── features/        # 特征工程：105 特征定义、跨资产、微观结构
├── decision/        # 决策层：信号、定量、分配、执行策略
├── alpha/           # Alpha 模型：LightGBM, XGBoost, LSTM, Transformer
├── risk/            # 风控层：规则聚合器、Kill Switch、熔断器
├── execution/       # 执行层：交易所适配器、订单状态机、安全机制
├── portfolio/       # 组合层：优化器、风险模型
├── regime/          # 体制识别：波动率、趋势
├── strategies/      # HFT 策略：imbalance scalper
├── runner/          # 运行器：live, paper, backtest
├── monitoring/      # 监控：Prometheus, Grafana, Telegram
├── infra/           # 基础设施：配置、日志、模型签名
├── scripts/         # 训练、验证、数据下载、研究
├── research/        # 实验追踪、Monte Carlo、灵敏度分析
├── core/            # 基础抽象：时钟、Effects、类型
├── attribution/     # PnL 归因
├── deploy/          # Docker, K8s, Prometheus, Grafana 配置
└── tests/           # 2,644 测试 (212 文件)
```

### 模块依赖层级

```
Layer 0 (Rust Kernel):  ext/rust/ → _quant_hotpath
Layer 1 (Foundation):   core/, event/, state/
Layer 2 (Engine):       engine/ (pipeline, coordinator, dispatcher, feature_hook)
Layer 3 (Intelligence): features/ → decision/ → alpha/
Layer 4 (Safety):       risk/ → portfolio/
Layer 5 (Execution):    execution/ (adapters, state_machine, reconcile)
Layer 6 (Integration):  runner/ → monitoring/ → infra/
```

---

## 3. Rust 内核 (`ext/rust/`)

### 3.1 概述

单一 Rust crate `_quant_hotpath`，通过 PyO3 + maturin 构建，覆盖所有计算热路径。

| 指标 | 值 |
|------|-----|
| 模块数 | 58 .rs 文件 |
| 代码量 | ~18,100 LOC |
| 导出数 | 161 (58 类 + 103 函数) |
| Rust 测试 | 52 |
| Cargo 依赖 | pyo3 0.23, serde 1, sha2, hmac, uuid |

### 3.2 模块分类

#### 状态与管道 (核心)

| 模块 | 职责 | 关键导出 |
|------|------|----------|
| `state_types.rs` | Fd8 定点数状态类型 | `RustMarketState`, `RustAccountState`, `RustPositionState` |
| `state_reducers.rs` | 纯函数状态转换 | `RustMarketReducer`, `RustAccountReducer`, `RustPositionReducer` |
| `state_store.rs` | 状态存储（Rust 堆） | `RustStateStore` — 状态常驻 Rust，按需导出 |
| `pipeline.rs` | 事件→状态管道 | `rust_pipeline_apply()` — 单次 FFI 调用 |
| `fixed_decimal.rs` | i64×10^8 定点数 | `Fd8` — 消除 Decimal 解析开销 |
| `rust_events.rs` | 原生事件类型 | `RustMarketEvent`, `RustFillEvent`, `RustFundingEvent` |

#### 特征计算

| 模块 | 职责 | 关键导出 |
|------|------|----------|
| `feature_engine.rs` | 105 特征增量计算 | `RustFeatureEngine` — `push_bar()` 增量推送 |
| `cross_asset.rs` | 跨资产特征 | `RustCrossAssetComputer` — beta, correlation, EMA |
| `factor_signals.rs` | 因子信号 | `rust_momentum_score`, `rust_volatility_score`, `rust_adx` |
| `indicators.rs` | 技术指标内核 | EMA, RSI, MACD, Bollinger, ATR |
| `microstructure.rs` | 微观结构 | `RustVPINCalculator`, `RustStreamingMicrostructure` |
| `feature_selector.rs` | 特征选择 | IC/ICIR 计算、贪心选择 |
| `fast_1m_features.rs` | 1 分钟高频特征 | 快速 1m bar 特征计算 |
| `multi_timeframe.rs` | 多时间框架 | 4h 特征计算 |

#### 风控与决策

| 模块 | 职责 | 关键导出 |
|------|------|----------|
| `risk_engine.rs` | 6 条风控规则 | `RustRiskEvaluator` |
| `decision_math.rs` | 仓位定量 | `rust_fixed_fraction_qty()`, `rust_volatility_adjusted_qty()` |
| `decision_signals.rs` | 策略数学 | `rust_rolling_sharpe`, `rust_max_drawdown`, `rust_strategy_weights` |
| `decision_policy.rs` | 执行策略 | 限价/被动单策略 |
| `portfolio_allocator.rs` | 组合分配 | `rust_allocate_portfolio()` |
| `regime_buffer.rs` | 体制检测缓冲 | `RustRegimeBuffer` |

#### 基础设施

| 模块 | 职责 |
|------|------|
| `event_store.rs` | 事件存储 |
| `checkpoint_store.rs` | 检查点存储 |
| `dedup_guard.rs` | 事件去重 |
| `rate_limiter.rs` | 限速器 |
| `signer.rs` | HMAC 签名 |
| `order_state_machine.rs` | 订单状态机 |
| `sequence_buffer.rs` | 序列缓冲 |

### 3.3 命名约定

- `cpp_*` 函数：C++ 迁移而来（保留原名兼容）
- `rust_*` 函数：新写 Rust 内核
- `Rust*` 类：Rust 原生类型

### 3.4 Fd8 定点数

所有价格/数量字段使用 `i64` 定点数（×10^8），消除 Python `Decimal` 的解析和格式化开销：

```
Python float → × 100_000_000 → Rust i64
Rust i64 → ÷ 100_000_000 → Python float
```

`_SCALE = 100_000_000` 定义在 `state/rust_adapters.py`。

---

## 4. 引擎层 (`engine/`)

### 4.1 核心组件

| 文件 | 职责 |
|------|------|
| `coordinator.py` | 总控：持有状态，注册路由 handler，`emit()` 统一入口 |
| `pipeline.py` | 状态管道：`rust_pipeline_apply()` 快速路径 + Python 后备 |
| `dispatcher.py` | 事件路由：按类型分发到 PIPELINE/DECISION/EXECUTION |
| `feature_hook.py` | 特征钩子：`RustFeatureEngine` 桥接 |
| `loop.py` | 事件循环：线程安全入队 + 单线程批量处理 |
| `tick_engine.py` | Tick 引擎：HFT tick 级处理 |
| `execution_bridge.py` | 执行桥接：OrderEvent → 交易所 |
| `module_reloader.py` | 模块热重载 |

### 4.2 StatePipeline

状态更新的唯一合法路径：

1. **事件归一化** (`normalize_to_facts`)：MARKET/FILL/ORDER → 标准化事实事件
2. **Rust 快速路径**：`rust_pipeline_apply()` — 单次 FFI 调用完成所有 reducer
3. **快照生成**：仅在状态变更时生成（`build_snapshot_on_change_only=True`）
4. **event_index 推进**：仅有事实事件时 +1

当无自定义 reducer 时自动使用 Rust 路径（`RustStateStore`）。

### 4.3 FeatureHook

`feature_hook.py` 将 `RustFeatureEngine` 嵌入管道：

- 每个 MarketEvent 触发 `push_bar()`
- 增量计算 105 个特征
- 无 Python fallback — 必须有 Rust crate

### 4.4 EventDispatcher 路由规则

| 事件类型 | 路由 |
|----------|------|
| MARKET, FILL, ORDER_UPDATE | `Route.PIPELINE` → 状态更新 |
| SIGNAL, INTENT, RISK | `Route.DECISION` → 决策模块 |
| ORDER | `Route.EXECUTION` → 执行层 |

### 4.5 EngineCoordinator

运行时总控，生命周期：INIT → RUNNING → STOPPED

核心职责：
- 持有所有引擎状态
- 注册三条路由的 handler
- Pipeline 输出后触发 DecisionBridge
- DecisionBridge 产生的 OrderEvent 经 Dispatcher 路由到 ExecutionBridge
- ExecutionBridge 产生的 FillEvent 经 Dispatcher 路由回 Pipeline

---

## 5. 状态管理 (`state/`)

### 5.1 双层状态

**Rust 层** (热路径)：
- `RustStateStore` 持有 `RustMarketState`, `RustAccountState`, `RustPositionState`
- 所有字段为 i64 定点数 (Fd8)
- 状态常驻 Rust 堆，通过 `get_*()` 方法按需导出到 Python

**Python 层** (兼容)：
- `MarketState`, `AccountState`, `PositionState` — frozen dataclass
- 用于决策模块输入（只读快照）
- `state/rust_adapters.py` 提供 Rust↔Python 转换

### 5.2 Reducer 模式

纯函数状态转换：输入旧状态 + 事件 → 输出新状态 + changed 标志

Rust reducer 接受 Python 事件对象 (`&Bound<'_, PyAny>`)，返回 `RustReducerResult`。

### 5.3 Snapshot

`build_snapshot()` 将 Market + Account + Positions 打包为只读快照，供决策系统使用。

---

## 6. 事件系统 (`event/`)

### 6.1 事件类型

| 事件 | 类型 | 用途 |
|------|------|------|
| `MarketEvent` | 事实 | OHLCV K线/行情 |
| `FillEvent` | 事实 | 成交回报 |
| `FundingEvent` | 事实 | 资金费率结算 |
| `SignalEvent` | 意见 | 策略信号 (side + strength) |
| `IntentEvent` | 意见 | 交易意图 (target_qty + reason) |
| `OrderEvent` | 命令 | 订单指令 (price + qty) |
| `RiskEvent` | 裁决 | 风控结果 |
| `ControlEvent` | 控制 | 系统控制 (halt/resume/shutdown) |

### 6.2 EventHeader

每个事件携带 `EventHeader`：
- `event_id`: UUID 唯一标识
- `root_event_id`: 根因事件 ID
- `parent_event_id`: 父事件 ID
- `ts_ns`: 纳秒级时间戳

### 6.3 Rust 原生事件

`RustMarketEvent`, `RustFillEvent`, `RustFundingEvent` 在 Rust 侧直接构造，避免 Python 对象创建开销（23.9x 加速）。

### 6.4 其他组件

| 文件 | 职责 |
|------|------|
| `store.py` | 事件持久化存储 |
| `checkpoint.py` | 检查点保存/恢复 |
| `replay.py` | 事件重放 |
| `security.py` | 事件安全（白名单授权） |
| `bootstrap.py` | 事件系统初始化 |

---

## 7. 特征工程 (`features/`)

### 7.1 核心组件

| 文件 | 职责 |
|------|------|
| `enriched_computer.py` | 105 特征定义 (`ENRICHED_FEATURE_NAMES`) |
| `live_computer.py` | 在线特征计算（实盘/模拟盘） |
| `batch_feature_engine.py` | 批量特征计算（回测） |
| `cross_asset_computer.py` | 跨资产特征 → 委托 `RustCrossAssetComputer` |
| `dynamic_selector.py` | 动态特征选择（IC/ICIR） |
| `cross_sectional.py` | 截面特征 |
| `multi_timeframe.py` | 多时间框架特征（4h） |
| `multi_resolution.py` | 多分辨率特征 |
| `rolling.py` | 滚动窗口 → `RollingWindow` (Rust) |

### 7.2 特征计算架构

**实盘路径**：
```
MarketEvent → FeatureHook → RustFeatureEngine.push_bar() → 105 features (增量)
```

**回测路径**：
```
DataFrame → batch_feature_engine.compute_features_batch() → cpp_compute_all_features() → DataFrame
```

### 7.3 105 特征分类

- **价格类**: 多窗口收益率、log 收益率、z-score
- **波动率类**: 实现波动率、Parkinson、ATR
- **动量类**: RSI、MACD、ROC、ADX
- **成交量类**: VWAP 偏离、OBV、成交量 z-score
- **微观结构**: Kyle Lambda、VPIN、order imbalance
- **跨资产**: BTC beta、correlation、relative strength
- **体制**: 波动率体制、趋势同步
- **多时间框架**: 4h EMA/RSI/volatility

### 7.4 CrossAssetComputer

跨资产特征计算委托给 `RustCrossAssetComputer`：
- 滚动 beta、correlation、EMA
- **关键约束**：必须先推送 benchmark (BTCUSDT)，再推送 altcoins

---

## 8. 决策系统 (`decision/`)

### 8.1 决策引擎流水线

```
Snapshot → Risk Overlay Gate → Signal Generation → Candidate Generation
    → Candidate Filtering → Allocation → Constraints → Sizing
    → Target Position → Intent Building → Execution Policy → OrderSpec
```

### 8.2 信号模型

```
decision/signals/
├── technical/        # MA交叉, 突破, RSI, MACD, 布林带, 均值回归, 网格
├── factors/          # momentum, volatility, liquidity, carry, volume_price_div
│                     # (全部委托 Rust: rust_*_score)
├── feature_signal.py # 特征信号 → rust_compute_feature_signal()
└── ensemble.py       # 信号集成
```

### 8.3 仓位定量

- `rust_fixed_fraction_qty()` — 固定比例定量 (Rust)
- `rust_volatility_adjusted_qty()` — 波动率调整定量 (Rust)
- `rust_apply_allocation_constraints()` — 约束应用 (Rust)

### 8.4 执行策略

- `marketable_limit.py` — 可成交限价单（市价 + 滑点偏移）
- `passive.py` — 被动挂单

### 8.5 多策略管理

`multi_strategy.py` 支持多策略权重分配：
- `rust_rolling_sharpe()` — 滚动 Sharpe ratio
- `rust_max_drawdown()` — 最大回撤
- `rust_strategy_weights()` — 权重计算 (equal/sharpe/inverse_vol)

### 8.6 Regime Bridge

`regime_bridge.py` 使用 `RustRegimeBuffer` 进行体制检测缓冲，替代 Python `_PriceBuffer`。

---

## 9. Alpha 模型 (`alpha/`)

### 9.1 模型架构

| 文件 | 模型 | 用途 |
|------|------|------|
| `models/lgbm_alpha.py` | LightGBM | **生产主力** (models_v8/) |
| `models/xgb_alpha.py` | XGBoost | 备选 |
| `models/lstm_alpha.py` | LSTM | 深度学习实验 |
| `models/transformer_alpha.py` | Transformer | 深度学习实验 |
| `models/ensemble.py` | 集成模型 | 多模型融合 |
| `models/ma_cross.py` | 均线交叉 | 基线策略 |

### 9.2 推理管道

```
alpha/inference/bridge.py → model_loader.py → LightGBM predict
```

- 模型签名验证 (`infra/model_signing.py`)：HMAC-SHA256
- 漂移检测 (`monitoring/drift_adapter.py`)
- OOD 检测 (`monitoring/ood_detector.py`)

### 9.3 训练

- `training/purged_split.py` — Purged K-Fold（防前视偏差）
- `training/regime_split.py` — 按体制分割
- `signal_transform.py` — 信号变换

---

## 10. 风控系统 (`risk/`)

### 10.1 RustRiskEvaluator (6 条规则)

在 Rust 中实现，单次 FFI 调用评估所有规则：

| 规则 | 职责 |
|------|------|
| MaxPosition | 单品种最大持仓 |
| MaxLeverage | 杠杆上限 |
| MaxDrawdown | 最大回撤限制 |
| MaxExposure | 总敞口限制 |
| Concentration | 集中度限制 |
| CircuitBreaker | 连续亏损熔断 |

### 10.2 Python 编排层

| 文件 | 职责 |
|------|------|
| `aggregator.py` | 规则聚合器（可插拔、短路优化） |
| `kill_switch.py` | 熔断开关（HARD_KILL / REDUCE_ONLY，支持 TTL） |
| `rules/max_drawdown.py` | 最大回撤规则（Python，触发后允许减仓） |
| `margin_monitor.py` | 保证金监控 |
| `meta_builder_live.py` | 实盘 meta 构建 |

### 10.3 风控决策模型

```
RiskAction: ALLOW / REJECT / REDUCE / KILL
RiskScope: SYMBOL / STRATEGY / PORTFOLIO / ACCOUNT / GLOBAL
合并策略: KILL > REJECT > REDUCE > ALLOW
```

---

## 11. 执行层 (`execution/`)

### 11.1 交易所适配器

**Binance** (`execution/adapters/binance/`):
- REST API 调用（下单/查询/取消）
- WebSocket 实时推送
- K线/深度/资金费率/清算/OI 数据源
- HMAC 签名认证
- 限速管理

**Bitget** (`execution/adapters/bitget/`):
- REST + WebSocket
- 订单/成交/仓位/余额映射器
- 去重机制

### 11.2 订单状态机

`execution/state_machine/machine.py`:
```
PENDING_NEW → NEW → PARTIALLY_FILLED → FILLED
    │          │
    ↓          ↓
  REJECTED   CANCELED / EXPIRED
```

线程安全，严格状态转换验证，历史追踪。

### 11.3 安全机制

| 文件 | 职责 |
|------|------|
| `safety/duplicate_guard.py` | 重复订单检测 |
| `safety/risk_gate.py` | 执行前风控闸门 |
| `safety/timeout_tracker.py` | 订单超时追踪 |
| `reconcile/scheduler.py` | 定期对账 |

### 11.4 模拟执行

`execution/sim/` 提供回测执行适配器：即时成交、滑点模拟、手续费计算、embargo 机制。

---

## 12. 投资组合 (`portfolio/`)

### 12.1 优化器

- Black-Litterman 模型
- 目标函数：最小方差、最大夏普、风险平价
- 约束：权重范围、杠杆限制、资产数

### 12.2 风险模型

```
portfolio/risk_model/
├── volatility/     # 波动率估计 (历史, EWMA)
├── correlation/    # 滚动相关性
├── covariance/     # 协方差矩阵 (EWMA, 样本)
├── factor/         # 因子模型 (暴露, 协方差, 特异风险)
├── stress/         # 压力测试
├── calibration/    # 模型校准
└── aggregation/    # 风险聚合
```

### 12.3 Rust 组合分配

`rust_allocate_portfolio()` 在 Rust 中执行约束数学：
- 杠杆上限缩放
- 单品种名义值上限
- 换手率上限
- weight → notional → qty 转换

---

## 13. 市场体制识别 (`regime/`)

| 文件 | 职责 |
|------|------|
| `base.py` | `RegimeLabel` 基类 |
| `state.py` | `RegimeState` — 存储最新体制标签 |
| `volatility.py` | 波动率体制检测（阈值比较） |
| `trend.py` | 趋势体制检测（阈值比较） |

`RustRegimeBuffer` 在 Rust 中维护价格缓冲区，`regime_bridge.py` 使用它进行体制判定。

---

## 14. 基础设施 (`infra/`, `monitoring/`)

### 14.1 基础设施

| 文件 | 职责 |
|------|------|
| `config/load.py` | 配置加载（YAML + 环境变量） |
| `logging/setup.py` | 结构化 JSON 日志 |
| `model_signing.py` | 模型签名与验证（HMAC-SHA256） |
| `runtime/run_context.py` | 运行时上下文 |
| `threading_utils.py` | 线程安全工具 |
| `messaging/zmq_backend.py` | ZMQ 消息后端 |

### 14.2 监控

| 组件 | 文件 | 职责 |
|------|------|------|
| Prometheus | `monitoring/metrics.py` | 指标导出 |
| Grafana | `deploy/grafana/` | 交易仪表盘 |
| Telegram | `monitoring/alerts/telegram.py` | 告警推送 |
| Health | `monitoring/health_server.py` | 健康检查 HTTP 端点 (127.0.0.1) |
| SLO | `monitoring/slo.py` | SLO/SLI 追踪 |
| Drift | `alpha/monitoring/drift_adapter.py` | 模型漂移检测 |

### 14.3 部署

```
deploy/
├── k8s/          # Kubernetes: deployment, HPA, leader election, secrets
├── argocd/       # ArgoCD: rollout, analysis template, rollback
├── prometheus/   # Prometheus: alerts, alertmanager
├── grafana/      # Grafana: dashboards, datasources
└── systemd/      # systemd service + logrotate
```

Docker 多阶段构建：builder (Rust 编译) → runtime (非 root 用户)。

---

## 15. 运行器 (`runner/`)

| 文件 | 模式 | 说明 |
|------|------|------|
| `live_runner.py` | 实盘 | 连接交易所 API，实际下单 |
| `live_paper_runner.py` | 模拟盘 | 实时数据 + 模拟执行 |
| `backtest_runner.py` | 回测 | CSV 数据 + 模拟执行 |
| `graceful_shutdown.py` | 优雅关闭 | 信号处理 + 状态保存 |
| `testnet_validation.py` | Testnet 验证 | 测试网冒烟测试 |

---

## 16. 研究与脚本 (`research/`, `scripts/`)

### 16.1 研究工具

| 文件 | 职责 |
|------|------|
| `monte_carlo.py` | Monte Carlo 模拟 |
| `sensitivity.py` | 参数灵敏度分析 |
| `significance.py` | 统计显著性检验 |
| `experiment.py` | 实验追踪框架 |
| `model_registry/artifact.py` | 模型版本管理 |

### 16.2 关键脚本

| 脚本 | 用途 |
|------|------|
| `backtest_alpha_v8.py` | V8 回测主入口 |
| `walkforward_validate.py` | Walk-forward 验证 |
| `train_v8_production.py` | BTC 生产模型训练 |
| `train_eth_production.py` | ETH 生产模型训练 |
| `train_sol_production.py` | SOL 生产模型训练 |
| `ic_analysis_v9.py` | IC 分析 |
| `download_binance_klines.py` | 历史数据下载 |
| `sweep_params.py` | 参数扫描 |
| `oos_eval.py` | 样本外评估 |

---

## 17. 测试体系

### 17.1 测试规模

| 类型 | 数量 |
|------|------|
| Python 测试 | 2,644 |
| Rust 测试 | 52 |
| 测试文件 | 212 |
| 测试分类 | unit, integration, regression |

### 17.2 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── engine/              # 管道、调度、特征钩子、状态存储
│   ├── features/            # 特征计算、Rust 一致性、选择器
│   ├── decision/            # 信号、定量、rebalance
│   ├── alpha/               # 模型、推理、漂移
│   ├── execution/           # 适配器、状态机、Bitget
│   ├── state/               # 状态类型、Rust 适配器
│   ├── risk/                # Rust 风控评估器
│   ├── portfolio/           # 优化器、风险模型
│   ├── event/               # 事件存储、检查点
│   ├── runner/              # 运行器、预检
│   ├── scripts/             # 脚本单元测试
│   └── ...
├── integration/             # 集成测试
│   ├── engine_state/        # 引擎-状态一致性
│   ├── test_production_integration_e2e.py
│   └── ...
└── regression/              # 回归测试
    ├── test_known_bug_cases.py
    ├── test_pnl_regression.py
    └── test_strategy_regression.py
```

### 17.3 关键一致性测试

- `test_rust_parity.py` / `test_rust_parity_v2.py` — Rust vs Python 计算一致性
- `test_feature_engine_parity.py` — C++ 批量引擎 vs Python 特征一致性
- `test_rust_technical_parity.py` — Rust 技术指标一致性
- `test_cross_asset.py` — Rust 跨资产计算一致性
- `test_pipeline_consistency.py` — 管道状态一致性

---

## 18. 端到端数据流

### 18.1 实盘数据流

```
Binance WebSocket → MarketEvent
    ↓
EngineCoordinator.emit()
    ↓
EventDispatcher.dispatch()
    ↓
StatePipeline → rust_pipeline_apply() → RustStateStore 状态更新
    ↓
FeatureHook → RustFeatureEngine.push_bar() → 105 features
    ↓
ML Inference → LightGBM predict
    ↓
DecisionBridge → Signal → Intent → OrderSpec
    ↓
RiskEvaluator → RustRiskEvaluator (6 rules)
    ↓
ExecutionBridge → Binance REST API
    ↓
FillEvent → 回注 Dispatcher → Pipeline 更新 Account + Position
```

### 18.2 回测数据流

```
CSV OHLCV → MarketEvent → [同上流程]
    ↓
BacktestExecutionAdapter (即时成交, 滑点, 手续费)
    ↓
equity_curve.csv + fills.csv + summary.json
```

---

## 19. 关键设计模式

| 模式 | 位置 | 说明 |
|------|------|------|
| **Rust FFI 加速** | `ext/rust/` | Python 编排 + Rust 计算的双层架构 |
| **事件溯源** | `engine/pipeline.py` | 状态由事件序列确定性推导 |
| **CQRS** | `engine/coordinator.py` | 写路径 (Pipeline) 与读路径 (Snapshot) 分离 |
| **Reducer** | `state/reducers/` | 纯函数状态转换 (Redux-like) |
| **Protocol Design** | 全局 | Python Protocol 定义接口契约 |
| **薄包装模式** | `features/`, `decision/signals/factors/` | Python 提取数据 → Rust 计算 → Python 包装结果 |
| **固定点数** | `fixed_decimal.rs` | i64×10^8 消除 Decimal 开销 |

---

## 20. Rust 迁移历程

### P0: C++ → Rust (完成)

5,744 LOC C++ pybind11 → ~5,100 LOC Rust PyO3。`ext/rolling/` 已归档。

### P1: 内核迁移 (完成)

+3,631 LOC Rust — state types, reducers, pipeline, guards, validators。

### P2: 深度集成 (完成)

4 个子阶段：JSON 消除 (5.67x)、Fd8 定点数、决策数学、RustStateStore (5.11x)。

### P3: 热路径迁移 (完成)

事件 (23.9x)、风控 (RustRiskEvaluator)、特征引擎 (RustFeatureEngine 105 features)。

### P4: Fallback 删除 + 新模块 (完成)

删除 Python fallback、添加微观结构/regime/signal Rust 模块。

### P5: 最终内核迁移 (完成)

CrossAsset (582 LOC Rust)、Portfolio Allocator、Multi-Strategy 数学。

### P6: 因子信号 + 清理 (完成)

Factor signals (280 LOC Rust)、RollingWindow/multi_timeframe fallback 删除、全量死代码清理。

**最终状态**：58 模块，~18,100 LOC Rust，零 Python fallback。

---

## 21. Walk-Forward 策略表现

### 当前最佳结果

| 品种 | WF 通过率 | Sharpe | 累计收益 | 策略 | 特征选择 |
|------|-----------|--------|----------|------|----------|
| **BTC** | 18/21 (86%) | 2.39 | +262% | Strategy F | stable_icir |
| **ETH** | 15/21 (71%) | 1.19 | +189.5% | Strategy F | stable_icir |
| **SOL** | 13/17 (76%) | 1.80 | +301% | Strategy F | greedy (dz=1.0, mh=48) |

### 验证方法

Walk-forward 验证确保策略的样本外有效性：
- 滚动训练/测试窗口
- 每个窗口独立训练模型
- 测试窗口为纯样本外
- 通过率 > 60% 视为合格

### 生产模型

- 位置：`models_v8/`
- 类型：LightGBM
- 特征：105 enriched features
- 训练脚本：`scripts/train_v8_production.py`
