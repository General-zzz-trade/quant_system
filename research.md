# Quant System — 深度代码库研究报告

> 版本: v0.2.0
> 语言: Python 3.11+
> 定位: 事件驱动的加密货币永续合约量化交易系统
> 文件数: 555 个 Python 文件
> 外部依赖: **核心逻辑零外部依赖**（仅 stdlib）

---

## 目录

1. [系统概览与设计哲学](#1-系统概览与设计哲学)
2. [顶层架构与模块地图](#2-顶层架构与模块地图)
3. [核心基础层 (`core/`)](#3-核心基础层-core)
4. [事件系统 (`event/`)](#4-事件系统-event)
5. [状态管理 (`state/`)](#5-状态管理-state)
6. [引擎层 (`engine/`)](#6-引擎层-engine)
7. [决策系统 (`decision/`)](#7-决策系统-decision)
8. [风控系统 (`risk/`)](#8-风控系统-risk)
9. [执行层 (`execution/`)](#9-执行层-execution)
10. [投资组合与风险模型 (`portfolio/`)](#10-投资组合与风险模型-portfolio)
11. [上下文状态 (`context/`)](#11-上下文状态-context)
12. [Alpha 模型 (`alpha/`)](#12-alpha-模型-alpha)
13. [特征工程 (`features/`)](#13-特征工程-features)
14. [归因分析 (`attribution/`)](#14-归因分析-attribution)
15. [市场体制识别 (`regime/`)](#15-市场体制识别-regime)
16. [基础设施 (`infra/`, `monitoring/`)](#16-基础设施-infra-monitoring)
17. [运行器 (`runner/`)](#17-运行器-runner)
18. [测试体系 (`tests/`, `tests_unit/`)](#18-测试体系-tests-tests_unit)
19. [端到端数据流](#19-端到端数据流)
20. [关键设计模式与架构决策](#20-关键设计模式与架构决策)
21. [系统优势与改进空间](#21-系统优势与改进空间)

---

## 1. 系统概览与设计哲学

### 1.1 定位

这是一个**机构级加密货币永续合约量化交易系统**，专为 Binance 等交易所设计，支持：

- **回测** (backtest)：读取历史 CSV OHLCV 数据，模拟完整交易生命周期
- **模拟盘** (paper trading)：与实时数据连接但不实际下单
- **实盘** (live trading)：通过交易所 API 实际执行订单

### 1.2 核心设计哲学

| 原则 | 实现方式 |
|------|----------|
| **事件驱动** | 所有状态变更由事件触发，通过 EventBus/Dispatcher 路由 |
| **确定性** | 相同事件序列 → 相同状态结果；SimulatedClock + DeterministicRandom |
| **纯函数业务逻辑** | Effects 模式注入所有 I/O，业务层不碰副作用 |
| **不可变状态** | 所有 State/Event 都是 `frozen=True, slots=True` 的 dataclass |
| **可审计** | W3C 分布式追踪、事件因果链、全链路日志 |
| **零外部依赖核心** | 核心逻辑仅使用 Python stdlib（`websocket-client` 等为可选依赖） |
| **制度优先 ("冻结版")** | 关键模块标记为"冻结版 v1.0"，修改需走制度审批 |

### 1.3 中文代码注释风格

代码中大量使用中文注释，术语体系为：
- **事实事件** = 实际发生的市场/成交事件 (MARKET, FILL)
- **意见事件** = 策略产生的信号/意图 (SIGNAL, INTENT)
- **命令事件** = 发往交易所的订单 (ORDER)
- **制度** = 系统不可逾越的规则/约束
- **冻结版** = 版本锁定，不允许随意修改

---

## 2. 顶层架构与模块地图

```
quant_system/
├── core/           # 基础层：时钟、Effects、类型、事件总线、配置、拦截器
├── event/          # 事件层：事件类型定义、EventBus、编解码、运行时
├── state/          # 状态层：MarketState、AccountState、PositionState + Reducers
├── engine/         # 引擎层：Coordinator、Pipeline、Dispatcher、Guard、Loop、Saga
├── decision/       # 决策层：信号生成、候选筛选、仓位定量、分配、执行策略
├── risk/           # 风控层：规则聚合器、Kill Switch、风控决策、拦截器
├── execution/      # 执行层：订单状态机、交易所适配器(Binance)、安全机制
├── portfolio/      # 组合层：优化器、风险模型（波动率、相关性、流动性、尾部风险）
├── context/        # 上下文层：快照系统、约束管理、视图模式、审计日志
├── alpha/          # Alpha 模型：策略信号基类、MA 交叉、突破策略
├── features/       # 特征工程：特征定义、在线/离线存储、转换管道
├── attribution/    # 归因分析：PnL 归因、成本分析、报告生成
├── regime/         # 体制识别：市场状态分类与检测
├── policy/         # 策略层：交易策略制度约束
├── infra/          # 基础设施：认证、CI、配置、日志、指标、运行时
├── monitoring/     # 监控：告警规则、仪表盘、运维手册
├── runner/         # 运行器：回测运行器（含 CLI）、模拟盘运行器
├── research/       # 研究：实验、笔记本、报告
├── tools/          # 工具脚本
├── tests/          # 测试：集成、性能、回归、重放、合约、故障
└── tests_unit/     # 单元测试
```

### 模块依赖层级（从下到上）

```
Layer 0 (Foundation):  core/
Layer 1 (Domain):      event/ → state/
Layer 2 (Engine):      engine/ (pipeline, dispatcher, coordinator, guard, saga)
Layer 3 (Intelligence): decision/ → risk/ → portfolio/
Layer 4 (Execution):   execution/ (adapters, state_machine, reconcile)
Layer 5 (Integration): context/ → runner/ → infra/
```

---

## 3. 核心基础层 (`core/`)

### 3.1 文件清单与职责

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `clock.py` | 时间抽象 | `Clock` (Protocol), `SystemClock`, `SimulatedClock`, `ReplayClock` |
| `effects.py` | 副作用边界 | `Effects`, `LogEffect`, `MetricsEffect`, `PersistEffect`, `RandomEffect` |
| `types.py` | 代数类型系统 | `Envelope[E]`, `EventKind`, `Route`, `Priority`, `TraceContext`, `EventMetadata`, `Symbol` |
| `errors.py` | 异常层次 | `QuantError` → `StateError`/`RiskError`/`ExecutionError`/`VenueError`/`PipelineError` |
| `interceptors.py` | 管道拦截链 | `PipelineInterceptor` (Protocol), `InterceptorChain`, `InterceptAction` (CONTINUE/REJECT/KILL) |
| `observability.py` | 可观测拦截器 | `TracingInterceptor`, `LoggingInterceptor`, `MetricsInterceptor` |
| `config.py` | 分层配置 | `ConfigService` (Defaults < File < Env < Runtime), 支持热更新 |
| `bus.py` | 有界事件总线 | `BoundedEventBus` (优先级 min-heap, 背压信号, DROP_LOWEST/REJECT 溢出策略) |
| `bootstrap.py` | 系统装配 | `SystemContext`, `bootstrap()`, `bootstrap_test()` |

### 3.2 关键设计细节

#### 时钟三态

```python
Clock (Protocol)
├── SystemClock      # 生产环境：真实 UTC 时间 + time.monotonic()
├── SimulatedClock   # 测试/回测：advance(delta), set(t) 手动控制时间
└── ReplayClock      # 事件重放：feed(ts) 逐事件推进时间
```

#### Effects 模式（六边形架构）

所有 I/O 通过 `Effects` 容器注入，业务逻辑保持纯函数：

```python
@dataclass(frozen=True)
class Effects:
    clock: Clock
    log: LogEffect
    metrics: MetricsEffect
    persist: PersistEffect
    random: RandomEffect
```

- `live_effects()` → 生产环境（真实时钟、真实日志、无操作指标）
- `test_effects(seed=42)` → 测试环境（模拟时钟、内存指标、确定性随机）

#### 有界事件总线

- **容量**: 默认 10,000 条
- **优先级**: CRITICAL(0) < HIGH(1) < NORMAL(2) < LOW(3)，基于 min-heap
- **背压**: 80% 水位线触发 BACKPRESSURE 信号
- **溢出策略**: DROP_LOWEST（驱逐最低优先级）或 REJECT（拒绝新事件）
- **线程安全**: 持锁操作 heap/handlers，锁外分发（避免回调死锁）

#### 拦截器链

管道中每个事件经过 `before_reduce → reducer → after_reduce` 流程：
- `before_reduce` 可以 REJECT（跳过 reducer）或 KILL（紧急停机）
- `after_reduce` 可以 KILL（状态不一致时紧急停机）
- 可观测拦截器（tracing/logging/metrics）纯记录，永不阻断

#### 分层配置

```
Defaults(代码硬编码) < File(JSON/YAML) < Env(QS_ 前缀) < Runtime(热更新)
```

支持 `watch(key, callback)` 监听热更新，`hot_update(key, value)` 运行时修改。

---

## 4. 事件系统 (`event/`)

### 4.1 事件类型层次

```python
EventType(Enum):
    MARKET   # 市场行情（K线/tick）
    SIGNAL   # 策略信号
    INTENT   # 交易意图
    ORDER    # 订单命令
    FILL     # 成交事实
    RISK     # 风控裁决
    CONTROL  # 系统控制（halt/resume/shutdown）
```

### 4.2 事件定义

所有事件继承 `BaseEvent(ABC, frozen=True, slots=True)`：

| 事件类 | 关键字段 | 用途 |
|--------|----------|------|
| `MarketEvent` | ts, symbol, OHLCV | K线/行情数据 |
| `SignalEvent` | signal_id, symbol, side, strength | 策略信号 |
| `IntentEvent` | intent_id, symbol, side, target_qty, reason_code, origin | 交易意图 |
| `OrderEvent` | order_id, intent_id, symbol, side, qty, price | 订单指令 |
| `FillEvent` | fill_id, order_id, symbol, qty, price | 成交回报 |
| `RiskEvent` | rule_id, level, message | 风控裁决 |
| `ControlEvent` | command, reason | 系统控制 |

### 4.3 富领域类型

```python
Side(Enum): BUY / SELL
Venue(Enum): BINANCE / OKX / BYBIT / SIM
OrderType(Enum): MARKET / LIMIT / STOP / STOP_LIMIT
TimeInForce(Enum): GTC / IOC / FOK / GTX
Symbol(frozen): value (e.g. "BTCUSDT")
Qty(frozen): value (Decimal)
Price(frozen): value (Decimal)
Money(frozen): amount (Decimal) + currency
```

### 4.4 EventHeader

每个事件携带 `EventHeader`，包含：
- `event_id`: UUID 唯一标识
- `root_event_id`: 根因事件 ID（因果链追溯）
- `parent_event_id`: 父事件 ID
- `correlation_id`: 业务关联 ID
- `ts_ns`: 纳秒级时间戳
- `source`: 事件来源
- `event_type`: 事件类型枚举
- `version`: Schema 版本

支持 `EventHeader.new_root()` 和 `EventHeader.from_parent()` 构造因果链。

### 4.5 EventBus

事件层自己也有一个 `EventBus`（不同于 `core/bus.py` 的 `BoundedEventBus`）：
- 三路路由：按 `event_type` 字符串、按 `event_cls` 类型、`subscribe_any` 全量
- 支持继承匹配（`isinstance` 检查）
- 线程安全：锁内快照 handler 列表，锁外执行

### 4.6 EventRuntime

`event/runtime.py` 提供事件运行时：
- 管理 WebSocket 连接到交易所
- 将原始市场数据转换为 `MarketEvent`
- 通过 handler 回调分发事件

### 4.7 Codec 系统

`event/codec.py` 提供事件序列化/反序列化：
- 注册表模式：`EventType → BaseEvent 子类`
- 支持 `encode(event) → dict` 和 `decode(dict) → BaseEvent`
- 用于持久化、重放和网络传输

---

## 5. 状态管理 (`state/`)

### 5.1 状态三元组

系统状态由三个不可变 dataclass 组成：

#### MarketState（市场事实）
```python
@dataclass(frozen=True, slots=True)
class MarketState:
    symbol: str
    last_price: Optional[Decimal]
    open/high/low/close/volume: Optional[Decimal]
    last_ts: Optional[datetime]
```
- `with_tick()`: 更新最新价
- `with_bar()`: 更新完整 OHLCV

#### AccountState（账户事实）
```python
@dataclass(frozen=True, slots=True)
class AccountState:
    currency: str
    balance: Decimal
    margin_used/margin_available: Decimal
    realized_pnl/unrealized_pnl/fees_paid: Decimal
    last_ts: Optional[datetime]
```

#### PositionState（持仓事实）
```python
@dataclass(frozen=True, slots=True)
class PositionState:
    symbol: str
    qty: Decimal
    avg_price: Optional[Decimal]
    last_price: Optional[Decimal]
    last_ts: Optional[datetime]
```

### 5.2 Reducer 模式

每个状态类型对应一个 Reducer：
- `MarketReducer.reduce(state, event) → ReduceResult(state, changed)`
- `AccountReducer.reduce(state, event) → ReduceResult(state, changed)`
- `PositionReducer.reduce(state, event) → ReduceResult(state, changed)`

Reducer 是纯函数：
- 输入旧状态 + 事件
- 输出新状态 + 是否变更标志
- 不可变转换（创建新对象，不修改原对象）

### 5.3 Snapshot

`state/snapshot.py` 提供 `build_snapshot()` 函数，将 Market + Account + Positions 打包为 `StateSnapshot`，用于：
- 决策模块输入（只读）
- 审计日志记录
- 回测重放对比

---

## 6. 引擎层 (`engine/`)

### 6.1 总体架构

引擎是系统的中枢，负责将**事件**转化为**状态变更**和**决策输出**：

```
事件源 → EngineLoop → EngineCoordinator → EventDispatcher → {
    PIPELINE handler  → StatePipeline → State + Snapshot
    DECISION handler  → DecisionBridge → 意见事件
    EXECUTION handler → ExecutionBridge → 交易所适配器
}
```

### 6.2 EventDispatcher（交通警察）

路由规则（`engine/dispatcher.py:144-203`）：

| 事件类型 | 路由 | 说明 |
|----------|------|------|
| MARKET, FILL, ORDER_UPDATE | `Route.PIPELINE` | 事实事件 → 状态更新 |
| SIGNAL, INTENT, RISK | `Route.DECISION` | 意见事件 → 决策模块 |
| ORDER (命令) | `Route.EXECUTION` | 命令事件 → 执行层 |
| 其他 | `Route.DROP` | 明确丢弃 |

关键特性：
- **事件去重**: 基于 `event_id` 的 TTL 去重（24小时，500K 上限）
- **顺序保证**: 内部序列号 `_seq` 保证事件顺序
- **锁策略**: 锁内路由 + 去重，锁外执行 handler

### 6.3 StatePipeline（状态唯一写通道）

`engine/pipeline.py` 是状态更新的唯一合法路径：

1. **事件归一化** (`normalize_to_facts`):
   - MARKET → 行情事实
   - FILL → 成交事实（强制 side 规范化，qty 取绝对值）
   - ORDER_UPDATE → 订单状态更新
   - 其他 → 空（不推进 event_index）

2. **Reducer 链**:
   - `MarketReducer.reduce(market_state, fact_event)`
   - `AccountReducer.reduce(account_state, fact_event)`
   - `PositionReducer.reduce(position_state[symbol], fact_event)`

3. **快照生成**:
   - 仅在 `any_changed=True` 时生成（可配置）
   - 快照是决策系统的唯一输入

4. **event_index 推进**:
   - 仅在有事实事件时 +1
   - 用于重放和审计对齐

### 6.4 EngineCoordinator（总控）

`engine/coordinator.py:73-374` 是运行时总控：

**生命周期**: INIT → RUNNING → STOPPED

**核心职责**:
1. 持有所有引擎状态（Market, Account, Positions, event_index）
2. 注册三条路由的 handler
3. `emit(event, actor)` 是统一事件入口
4. Pipeline 输出后触发 DecisionBridge
5. DecisionBridge 产生的 OrderEvent 经 Dispatcher 路由到 ExecutionBridge
6. ExecutionBridge 产生的 FillEvent 经 Dispatcher 路由回 Pipeline

**因果链闭环**:
```
MarketEvent → Pipeline → Snapshot → DecisionBridge → IntentEvent + OrderEvent
    ↑                                                        ↓
    └─── FillEvent ←── ExecutionBridge ←── OrderEvent ───────┘
```

### 6.5 EngineLoop（生产级事件循环）

`engine/loop.py:76-306` 实现线程安全的事件循环：

- **Inbox**: `queue.Queue(maxsize=100_000)` 作为背压边界
- **submit()**: 线程安全入队（IO/回调线程调用）
- **drain()**: 单线程批量处理（保证确定性）
- **Guard 集成**: before_event / on_error / after_event 三阶段守卫
- **重试机制**: RETRY 决策 → 回入 inbox（有次数上限和等待间隔）
- **后台模式**: `start_background()` 启动 daemon 线程

### 6.6 Guard 系统（错误裁决）

`engine/guards.py` 提供错误处理策略：

```python
GuardAction(Enum):
    ALLOW   # 放行
    DROP    # 丢弃事件
    RETRY   # 建议重试
    STOP    # 停止引擎
```

`BasicGuard` 的裁决逻辑：
1. FATAL 错误 → 直接 STOP
2. INVARIANT 违规 → 直接 STOP
3. 连续执行错误 ≥ 2 → STOP
4. 连续领域错误 ≥ 5 → STOP
5. 全局连续错误 ≥ 5 → STOP
6. RETRYABLE/TIMEOUT → RETRY（带 0.2s 等待）
7. 默认 → DROP（保守，防止状态污染）

### 6.7 错误分类系统

`engine/errors.py` 实现策略无关的错误分类：

```python
ErrorSeverity: INFO / WARNING / ERROR / FATAL
ErrorDomain: ENGINE / PIPELINE / DISPATCHER / DECISION / EXECUTION / IO / RISK / DATA / INVARIANT

classify_exception(exc) → ClassifiedError(severity, domain, code, message, ctx)
```

异常层次：
```
EngineException
├── RecoverableError
├── RetryableError
├── FatalError
├── InvariantViolation
├── ExecutionError
└── DataError
```

### 6.8 Saga 模式（订单生命周期）

`engine/saga.py` 追踪订单全生命周期：

```
PENDING → SUBMITTED → ACKED → PARTIAL_FILL → FILLED
              │          │         │
              │          ↓         │
              │      CANCELLED     │
              ↓                    ↓
           REJECTED            EXPIRED
              │
              ↓
          COMPENSATING → COMPENSATED
              │
              ↓
           FAILED
```

`SagaManager` 特性：
- 线程安全
- 严格状态转换验证
- 填充追踪（加权平均价格）
- 补偿动作（CancelOrderAction）
- 完成订单环形缓冲（默认保留 10,000 条）
- 按 intent_id / symbol 查询

### 6.9 DecisionBridge（状态→意见）

`engine/decision_bridge.py` 连接 Pipeline 输出到决策模块：

```python
class DecisionModule(Protocol):
    def decide(self, snapshot: Any) -> Iterable[Any]:
        """只读 snapshot → 意见事件"""
```

铁律：
- 永远不修改 state
- 永远不调用 execution
- 只处理 snapshot，不处理原始 state

### 6.10 ExecutionBridge（意见→现实）

`engine/execution_bridge.py` 连接 Dispatcher 到交易所：

```python
class ExecutionAdapter(Protocol):
    def send_order(self, order_event: Any) -> Iterable[Any]:
        """OrderEvent → FillEvent / RejectEvent"""
```

铁律：
- 永远不修改 state
- 永远不生成 snapshot
- 永远不绕过 dispatcher

---

## 7. 决策系统 (`decision/`)

### 7.1 决策引擎流水线

`decision/engine.py` 实现完整的决策流水线：

```
Snapshot → Risk Overlay Gate → Universe Selection → Signal Generation
    → Candidate Generation → Candidate Filtering → Allocation
    → Constraints → Sizing → Target Position → Intent Building
    → Execution Policy → Validation → OrderSpec
```

### 7.2 核心组件

#### 信号模型 (`decision/signals/`)

```
signals/
├── base.py            # SignalModel(Protocol), NullSignal
├── ensemble.py        # 信号集成
├── technical/
│   ├── ma_cross.py    # 均线交叉
│   ├── breakout.py    # 突破策略
│   ├── rsi_signal.py  # RSI
│   ├── macd_signal.py # MACD
│   ├── bollinger_band.py  # 布林带
│   ├── mean_reversion.py  # 均值回归
│   └── grid_signal.py     # 网格策略
├── statistical/
│   ├── zscore.py      # Z-Score
│   └── cointegration.py  # 协整
└── ml/
    ├── features_contract.py  # 特征契约
    └── model_runner.py       # ML 模型运行器
```

#### 候选筛选 (`decision/candidates/`)

- `score_rank.py`: 按信号得分排名生成候选
- `filters.py`: 候选过滤（最低分数、最低置信度、最大数量）

#### 仓位定量 (`decision/sizing/`)

- `fixed_fraction.py`: 固定比例（账户权益 × 风险比例 / 价格 = 数量）

#### 分配 (`decision/allocators/`)

- `single_asset.py`: 单资产分配器（100% 权重给最佳候选）
- `constraints.py`: 约束应用（最大持仓数）

#### 执行策略 (`decision/execution_policy/`)

- `marketable_limit.py`: 可成交限价单（市价 + 滑点）
- `passive.py`: 被动挂单

#### 风控叠加 (`decision/risk_overlay/`)

- `kill_conditions.py`: 基本熔断条件检查

#### 治理 (`decision/governance/`)

- `feature_flags.py`: 功能开关（策略级/信号级启停）

### 7.3 决策输出类型

```python
@dataclass
class DecisionOutput:
    ts: datetime
    strategy_id: str
    targets: Tuple[TargetPosition, ...]
    orders: Tuple[OrderSpec, ...]
    explain: DecisionExplain

@dataclass
class DecisionExplain:
    # 完整决策路径解释：gates, universe, signals, candidates, targets, orders
```

### 7.4 DecisionGraph

`decision/graph.py` 实现轻量级 DAG 容器，支持拓扑排序执行依赖节点。

---

## 8. 风控系统 (`risk/`)

### 8.1 风控决策模型

`risk/decisions.py` 定义标准化风控决策：

```python
RiskAction: ALLOW / REJECT / REDUCE / KILL
RiskScope: SYMBOL / STRATEGY / PORTFOLIO / ACCOUNT / GLOBAL

RiskCode:
    # 账户级: INSUFFICIENT_MARGIN, MAX_LEVERAGE, MAX_DRAWDOWN, LIQUIDATION_RISK
    # 仓位级: MAX_POSITION, MAX_NOTIONAL, MAX_DELTA, MAX_GROSS, MAX_NET
    # 市场级: VOLATILITY_SPIKE, LIQUIDITY_DRY, PRICE_GAP
    # 系统级: STALE_DATA, EXCHANGE_DOWN, OMS_DEGRADED
```

### 8.2 RiskAggregator（规则聚合器）

`risk/aggregator.py` 是风控核心：

- **可插拔规则集**: `RiskRule(Protocol)` 定义标准接口
- **Meta 解耦**: 上层把事实打包成 meta 传入，风控层不直接依赖 Context/AccountState
- **可热禁用**: `enable(rule)` / `disable(rule)`
- **可观测**: 每条规则的调用次数、动作分布、异常次数、耗时
- **Fail-safe**: 规则异常时保守失败（默认 REJECT，可配置 KILL）
- **短路优化**: 遇到 KILL 可立即终止后续规则评估
- **合并策略**: KILL > REJECT > REDUCE > ALLOW

### 8.3 Kill Switch（熔断开关）

`risk/kill_switch.py` 实现多层级熔断：

**熔断模式**:
- `HARD_KILL`: 完全禁止新单
- `REDUCE_ONLY`: 只允许减仓

**作用域优先级**:
```
GLOBAL > ACCOUNT/PORTFOLIO > STRATEGY > SYMBOL
```

**特性**:
- TTL 支持自动恢复
- 可审计（source, reason, tags, meta）
- `allow_order()` 闸门接口供执行层调用

### 8.4 Risk 规则 (`risk/rules/`)

规则目录包含具体的风控规则实现：
- 保证金检查
- 杠杆限制
- 最大回撤
- 持仓限制
- 敞口限制
- 波动率突变检测

### 8.5 Risk Interceptor

`risk/interceptor.py` 实现管道拦截器接口，将风控决策嵌入事件处理管道。

---

## 9. 执行层 (`execution/`)

### 9.1 订单状态机

`execution/state_machine/machine.py` 实现严格的订单生命周期管理：

```
PENDING_NEW → NEW → PARTIALLY_FILLED → FILLED
    │          │          │
    ↓          ↓          ↓
  REJECTED  CANCELED   EXPIRED
```

**特性**:
- 线程安全（RLock）
- 严格状态转换验证
- 历史追踪（Transition 列表）
- 终态自动归档

### 9.2 订单模型 (`execution/models/`)

| 文件 | 职责 |
|------|------|
| `orders.py` | 订单数据模型 |
| `fills.py` | 成交数据模型 |
| `commands.py` | 订单命令（提交/取消/修改） |
| `intents.py` | 交易意图模型 |
| `balances.py` | 余额模型 |
| `positions.py` | 持仓模型 |
| `instruments.py` | 交易品种模型 |
| `venue.py` | 交易所模型 |
| `validation.py` | 输入验证 |
| `serialization.py` | 序列化 |
| `errors.py` | 执行错误类型 |
| `transfers.py` | 资金划转 |

### 9.3 交易所适配器 (`execution/adapters/binance/`)

Binance 适配器实现：
- REST API 调用（下单、查询、取消）
- WebSocket 实时推送处理
- 签名认证
- 限速管理

### 9.4 安全机制 (`execution/safety/`)

执行安全层包含：
- 订单频率限制
- 最大委托量检查
- 重复订单检测
- 网络异常重试

### 9.5 对账 (`execution/reconcile/`)

执行后对账机制：
- 本地状态 vs 交易所状态对比
- 孤儿订单检测
- 缺失成交补偿

### 9.6 模拟执行 (`execution/sim/`)

用于回测和模拟盘的执行适配器，实现即时成交（价格来自市场状态）。

---

## 10. 投资组合与风险模型 (`portfolio/`)

### 10.1 优化器 (`portfolio/optimizer/`)

```
optimizer/
├── base.py          # 优化器基类
├── objectives.py    # 目标函数（最小方差、最大夏普、风险平价）
├── constraints.py   # 约束条件（权重范围、杠杆限制、资产数限制）
├── input.py         # 优化输入数据结构
├── postprocess.py   # 优化结果后处理
├── diagnostics.py   # 优化诊断
├── exceptions.py    # 优化异常
└── solvers/         # 求解器实现
```

### 10.2 风险模型 (`portfolio/risk_model/`)

```
risk_model/
├── volatility/     # 波动率估计（历史、EWMA、GARCH-like）
├── correlation/    # 相关性估计
├── covariance/     # 协方差矩阵
├── factor/         # 因子模型
├── liquidity/      # 流动性评估
├── tail/           # 尾部风险（VaR、CVaR）
├── stress/         # 压力测试
├── calibration/    # 模型校准
├── aggregation/    # 风险聚合
└── diagnostics/    # 风险诊断
```

---

## 11. 上下文状态 (`context/`)

### 11.1 Context 系统

`context/` 提供跨模块的只读状态视图：

```
context/
├── context.py              # 主上下文容器
├── snapshot.py             # 上下文快照
├── versioning.py           # 版本管理
├── registry.py             # 注册表
├── reducer.py              # 上下文级 Reducer
├── validators.py           # 验证器
├── constraints/            # 约束管理
│   ├── exchange_constraints.py   # 交易所约束（最小数量、价格精度）
│   ├── portfolio_constraints.py  # 组合约束（最大持仓、最大敞口）
│   └── strategy_constraints.py   # 策略约束
├── views/                  # 视图模式
│   ├── base.py             # 视图基类
│   ├── strategy_view.py    # 策略视图
│   ├── risk_view.py        # 风控视图
│   ├── execution_view.py   # 执行视图
│   └── monitoring_view.py  # 监控视图
├── market/                 # 市场状态子系统
│   ├── market_state.py
│   ├── market_snapshot.py
│   ├── market_rules.py
│   └── account/
├── portfolio/              # 组合状态
│   ├── portfolio_state.py
│   ├── portfolio_snapshot.py
│   └── exposure_state.py
├── risk/                   # 风控状态
│   ├── risk_state.py
│   ├── risk_snapshot.py
│   └── limits_state.py
├── regime/                 # 体制状态
│   ├── regime_state.py
│   └── regime_snapshot.py
├── execution/              # 执行状态
│   ├── execution_state.py
│   └── execution_snapshot.py
└── audit/                  # 审计系统
    ├── audit_log.py
    └── diff.py
```

### 11.2 视图模式

不同模块通过专用视图访问上下文：
- **StrategyView**: 策略需要的市场数据 + 持仓
- **RiskView**: 风控需要的敞口 + 限制
- **ExecutionView**: 执行需要的订单状态 + 市场规则
- **MonitoringView**: 监控需要的全局概览

---

## 12. Alpha 模型 (`alpha/`)

### 12.1 结构

```
alpha/
├── base.py           # AlphaModel 基类
├── registry.py       # Alpha 注册表
├── models/
│   ├── breakout.py   # 突破策略 Alpha
│   └── ma_cross.py   # 均线交叉 Alpha
├── signals/          # 信号生成
├── inference/        # 推理/预测
├── training/         # 训练管道
└── validation/       # 验证框架
```

### 12.2 设计

Alpha 模型框架将**信号生成**与**交易决策**分离：
- Alpha 模型专注于预测方向和强度
- Decision 引擎负责将预测转化为交易

---

## 13. 特征工程 (`features/`)

```
features/
├── definitions/      # 特征定义（特征注册、Schema）
├── online_store/     # 在线特征存储（低延迟服务）
├── offline_store/    # 离线特征存储（批量训练用）
└── transforms/       # 特征转换管道
```

---

## 14. 归因分析 (`attribution/`)

```
attribution/
├── pnl.py      # PnL 归因（策略级、因子级）
├── cost.py     # 成本分析（手续费、滑点、市场冲击）
└── report.py   # 归因报告生成
```

---

## 15. 市场体制识别 (`regime/`)

```
regime/
├── __init__.py
├── detector.py        # 体制检测器
├── hmm.py            # 隐马尔可夫模型
├── volatility.py     # 波动率体制
└── trend.py          # 趋势体制
```

与风控系统联动：特定体制下可触发 VOLATILITY_SPIKE 等风控码。

---

## 16. 基础设施 (`infra/`, `monitoring/`)

### 16.1 基础设施 (`infra/`)

```
infra/
├── auth/       # 认证（API Key 管理、签名）
├── ci/         # CI/CD 配置
├── config/     # 部署配置
├── logging/    # 日志配置
├── metrics/    # 指标采集
└── runtime/    # 运行时管理
```

### 16.2 监控 (`monitoring/`)

```
monitoring/
├── alerts/       # 告警规则定义
├── dashboards/   # 仪表盘配置
└── runbooks/     # 运维手册
```

---

## 17. 运行器 (`runner/`)

### 17.1 回测运行器 (`runner/backtest_runner.py`)

这是系统中最完整的可运行入口，功能包括：

#### CSV 数据读取
- 自动识别列名（ts/timestamp/time/datetime 等）
- 支持多种时间格式（ISO、Unix 秒/毫秒/微秒/纳秒）
- 容错处理（跳过无效行、重复表头）

#### BacktestExecutionAdapter
- 即时成交模拟
- 滑点模拟（买入价上调、卖出价下调）
- 手续费计算
- 已实现 PnL 计算（加权平均价格法）
- 持仓状态追踪

#### MovingAverageCrossModule
内置均线交叉策略：
- 收盘价 > MA → 开多
- 收盘价 < MA → 平多
- 产生 IntentEvent + OrderEvent

#### 指标计算
- 权益曲线（equity curve）
- 最大回撤（max drawdown）
- CAGR（年化复合回报）
- 夏普比率（Sharpe ratio, 年化）
- 卡尔玛比率（Calmar ratio）
- 胜率、盈亏比、利润因子
- 交易频率、持仓时间统计
- 最大连续亏损次数

#### 滚动窗口验证（Walk-Forward）
`run_walk_forward()` 支持：
- 训练/测试窗口滚动切割
- 独立评估每个测试窗口
- 防止前视偏差（look-ahead bias）

#### CLI 入口
```bash
python runner/backtest_runner.py \
    --csv data/binance/ohlcv/BTCUSDT_1m_ohlcv.csv \
    --symbol BTCUSDT \
    --starting-balance 10000 \
    --ma 20 \
    --qty 0.01 \
    --fee-bps 4 \
    --slippage-bps 2 \
    --out out/btcusdt_default
```

输出：`equity_curve.csv`, `fills.csv`, `trades.csv`, `summary.json`

### 17.2 模拟盘运行器 (`runner/paper_runner.py`)

连接实时数据源但不实际执行订单。

---

## 18. 测试体系 (`tests/`, `tests_unit/`)

### 18.1 测试分类

```
tests/
├── contract/               # 合约测试（接口契约验证）
├── execution_safety/       # 执行安全测试
├── failure/                # 故障注入测试
├── integration/            # 集成测试
│   ├── engine_state/       # 引擎-状态集成
│   ├── execution_state/    # 执行-状态集成
│   ├── risk_portfolio/     # 风控-组合集成
│   └── strategy_engine/    # 策略-引擎集成
├── performance/            # 性能测试
├── persistence/            # 持久化测试
├── regression/             # 回归测试
├── replay/                 # 重放测试
└── unit/                   # 单元测试
    ├── engine/
    ├── event/
    ├── execution/
    ├── portfolio/
    ├── risk/
    └── state/
```

### 18.2 测试工具

```
tests_unit/                 # 独立单元测试目录
```

### 18.3 测试配置

```ini
# pytest.ini
testpaths = ["tests", "tests_unit"]
python_files = "test_*.py"
addopts = "-q --tb=short"
```

### 18.4 测试哲学

- **确定性**: `test_effects(seed=42)` 保证每次运行结果相同
- **隔离性**: Effects 模式使测试无需真实 I/O
- **分层验证**: 单元 → 集成 → 合约 → 性能 → 回归 → 重放
- **安全优先**: 专门的 `execution_safety/` 测试
- **故障注入**: `failure/` 测试系统在异常条件下的行为

---

## 19. 端到端数据流

### 19.1 实盘数据流

```
Binance WebSocket
    │
    ↓
EventRuntime (event/runtime.py)
    │ 原始数据 → MarketEvent
    ↓
EngineLoop.submit() [线程安全入队]
    │
    ↓
EngineLoop.drain() [单线程串行处理]
    │
    ↓
EngineCoordinator.emit()
    │
    ↓
EventDispatcher.dispatch()
    │
    ├─[MARKET/FILL]──→ StatePipeline.apply()
    │                       │
    │                       ├── normalize_to_facts()
    │                       ├── MarketReducer.reduce()
    │                       ├── AccountReducer.reduce()
    │                       ├── PositionReducer.reduce()
    │                       └── build_snapshot()
    │                              │
    │                              ↓
    │                       DecisionBridge.on_pipeline_output()
    │                              │
    │                              ↓
    │                       DecisionModule.decide(snapshot)
    │                              │
    │                              ├── Signal Generation
    │                              ├── Candidate Selection
    │                              ├── Position Sizing
    │                              └── Order Generation
    │                                     │
    │                                     ↓
    │                              IntentEvent + OrderEvent
    │                                     │
    │                                     ↓ [重新注入 Dispatcher]
    │
    ├─[SIGNAL/INTENT]─→ _handle_decision_event() [v1.0 no-op]
    │
    └─[ORDER]─────────→ ExecutionBridge.handle_event()
                              │
                              ↓
                        ExecutionAdapter.send_order()
                              │  (Binance REST API)
                              ↓
                        FillEvent / RejectEvent
                              │
                              ↓ [重新注入 Dispatcher → PIPELINE]
                              │
                        StatePipeline 更新 Account + Position
```

### 19.2 回测数据流

```
CSV OHLCV 文件
    │
    ↓
iter_ohlcv_csv() → OhlcvBar
    │
    ↓
MarketEvent (with EventHeader)
    │
    ↓
EngineCoordinator.emit(event, actor="replay")
    │
    ↓
[同上的 Dispatcher → Pipeline → Decision → Execution 流程]
    │
    ↓ (BacktestExecutionAdapter 即时成交)
    │
    ↓
equity_curve.csv + fills.csv + trades.csv + summary.json
```

---

## 20. 关键设计模式与架构决策

### 20.1 设计模式清单

| 模式 | 位置 | 实现 |
|------|------|------|
| **依赖注入 (DI)** | `core/effects.py`, `core/bootstrap.py` | Effects 容器注入所有 I/O |
| **事件溯源 (Event Sourcing)** | `engine/pipeline.py`, `state/` | 状态由事件序列确定性推导 |
| **CQRS** | `engine/coordinator.py` | 写路径 (Pipeline) 与读路径 (Views) 分离 |
| **Saga 模式** | `engine/saga.py` | 订单生命周期管理与补偿事务 |
| **Chain of Responsibility** | `core/interceptors.py` | 拦截器链顺序执行 |
| **Reducer (Redux-like)** | `state/reducers/` | 纯函数状态转换 |
| **Protocol-Based Design** | 全局 | Python Protocol 定义接口契约 |
| **Strategy Pattern** | `decision/signals/`, `decision/execution_policy/` | 可插拔策略 |
| **Observer Pattern** | `event/bus.py`, `core/bus.py` | 事件订阅/发布 |
| **Factory Method** | `event/factory/`, `core/effects.py` | `live_effects()`, `test_effects()` |
| **Hexagonal Architecture** | `core/` | 端口和适配器 (Effects = 端口, 具体实现 = 适配器) |
| **Ring Buffer** | `core/observability.py`, `engine/saga.py` | 有界缓冲防止内存泄漏 |
| **Layered Configuration** | `core/config.py` | 分层优先级配置 |
| **DAG Execution** | `decision/graph.py` | 拓扑排序执行依赖图 |

### 20.2 关键架构决策

#### 决策 1: 零外部依赖核心
- **选择**: 核心逻辑仅使用 Python stdlib
- **理由**: 减少供应链风险、简化部署、提高可审计性
- **代价**: 部分功能需自行实现（如优先级队列）

#### 决策 2: 不可变状态 (`frozen=True, slots=True`)
- **选择**: 所有 State/Event 都是冻结 dataclass
- **理由**: 线程安全、无副作用、易于测试、支持值相等性
- **代价**: 每次状态更新创建新对象（内存压力）

#### 决策 3: 事件驱动 + 单线程处理
- **选择**: IO 线程只负责入队，处理线程串行执行
- **理由**: 保证确定性（相同事件序列 = 相同结果）
- **代价**: 处理吞吐受限于单线程

#### 决策 4: Dispatcher 路由 vs 直接调用
- **选择**: 所有模块间通信必须经过 Dispatcher
- **理由**: 解耦、可审计、事件去重、统一入口
- **代价**: 间接调用增加延迟

#### 决策 5: "冻结版"制度
- **选择**: 关键模块标记版本号，修改需走制度审批
- **理由**: 防止核心逻辑被意外修改
- **代价**: 演进速度受制度约束

#### 决策 6: 双事件总线
- **选择**: `core/bus.py` (BoundedEventBus) 和 `event/bus.py` (EventBus) 并存
- **观察**: 两套事件总线设计略有重叠
  - `core/bus.py`: 优先级感知、背压、有界
  - `event/bus.py`: 类型路由、继承匹配
- **可能原因**: 架构演进中不同层级的需求

#### 决策 7: 中文注释 + 英文代码
- **选择**: 类/函数名用英文，注释和文档用中文
- **理由**: 面向中文开发团队，降低沟通成本

---

## 21. 系统优势与改进空间

### 21.1 优势

1. **架构成熟度高**: 分层清晰、职责明确、模块解耦彻底
2. **安全意识强**: Kill Switch、Guard、Safety 层层防护
3. **可测试性好**: Effects 模式 + 确定性时钟 + 确定性随机 = 完全可复现
4. **可观测性全面**: W3C tracing、结构化日志、指标、审计日志
5. **风控体系完整**: 规则聚合、熔断开关、风控决策标准化
6. **回测基础扎实**: 完整的回测流水线、指标计算、Walk-Forward 验证
7. **订单生命周期管理**: Saga 模式 + 状态机双重保障
8. **错误处理细致**: 分类系统 + Guard 策略 + 重试机制

### 21.2 待完善领域

1. **Decision handler 为 no-op**: `_handle_decision_event()` 当前空实现，风控门控未在此集成
2. **Portfolio/Risk Model**: 目录结构完整但需验证实现深度
3. **多品种支持**: 当前回测以单品种为主，多品种组合需要更多集成
4. **异步执行**: 当前单线程串行处理，高频场景可能成为瓶颈
5. **持久化**: `InMemoryPersist` 重启后丢失，生产环境需要持久化方案
6. **双事件总线**: `core/bus.py` 和 `event/bus.py` 的职责边界可进一步厘清
7. **ML 管道**: `alpha/training/`、`features/` 等模块目录存在但内容待填充

### 21.3 代码质量观察

- **类型标注**: 广泛使用 `frozen=True, slots=True`，但 `disallow_untyped_defs = false` 说明还在渐进推进
- **Decimal 精度**: 金融计算全面使用 `Decimal`，避免浮点误差
- **线程安全**: 关键组件都有 `threading.Lock`/`RLock` 保护
- **异常处理**: 明确区分 Retryable / Fatal / Invariant 等错误类型
- **代码风格**: ruff 格式化，120 字符行宽，lint 规则严格

---

## 附录 A: 文件行数统计（按模块）

| 模块 | 估计 Python 文件数 | 核心职责 |
|------|-------------------|---------|
| core/ | 10 | 基础抽象（时钟、效果、类型、总线、配置） |
| event/ | ~20 | 事件定义、编解码、运行时、检查点 |
| state/ | ~10 | 状态模型与 Reducer |
| engine/ | 18 | 协调、管道、调度、守卫、循环、Saga |
| decision/ | ~40 | 信号、候选、定量、分配、策略 |
| risk/ | ~10 | 规则、聚合、熔断、决策 |
| execution/ | ~50 | 订单模型、状态机、交易所适配器、安全 |
| portfolio/ | ~30 | 优化器、风险模型 |
| context/ | ~30 | 上下文、视图、约束、审计 |
| tests/ + tests_unit/ | ~100+ | 全面测试覆盖 |
| 其他 | ~230 | alpha, features, attribution, regime, infra, monitoring, runner, tools |

## 附录 B: 关键配置默认值

```python
_DEFAULT_CONFIG = {
    "bus.capacity": 10_000,
    "bus.high_watermark": 0.8,
    "bus.overflow_policy": "drop_lowest",
    "pipeline.snapshot_on_change_only": True,
    "pipeline.fail_on_missing_symbol": False,
    "risk.fail_safe_action": "reject",
    "risk.reject_on_reduce": False,
    "risk.kill_switch.halt_pipeline": True,
    "saga.max_completed": 10_000,
    "observability.max_spans": 10_000,
    "observability.max_log_entries": 5_000,
    "observability.log_continue": False,
}
```

## 附录 C: 依赖清单

### 核心（零依赖）
- Python ≥ 3.11 stdlib only

### 可选
- `websocket-client ≥ 1.6` — 实盘 WebSocket
- `python-dotenv ≥ 1.0` — 环境变量加载
- `pyyaml ≥ 6.0` — YAML 配置

### 开发
- `pytest ≥ 7.4` + `pytest-cov` + `pytest-timeout`
- `ruff ≥ 0.4` — Linter/Formatter
- `mypy ≥ 1.10` — 类型检查
