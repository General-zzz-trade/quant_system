# Distinguished-Level Architecture Redesign

> 日期: 2026-02-23
> 目标: 将 quant_system 从 Staff 级提升到 Distinguished 级架构

---

## 一、当前架构诊断

### 已有的 Staff 级优点
- Event Sourcing + Immutable State (Reducer 模式)
- Protocol-based 依赖注入
- 5 层 Kill Switch 风控
- Frozen dataclass + slots 类型安全
- 零外部依赖核心

### 需要突破的架构天花板

| 维度 | 当前问题 | Distinguished 目标 |
|------|---------|-------------------|
| **类型安全** | `Any` 散落各处，event routing 靠字符串匹配 | 让非法状态在编译期不可表达 |
| **副作用隔离** | I/O（网络、磁盘、时钟）隐式散落 | 纯函数核心 + Effect Boundary |
| **事件流** | 同步单线程、无背压、无流控 | 有界队列 + 背压 + 优先级 |
| **风控集成** | Risk 与 Engine 松散耦合 | Risk 作为 Pipeline 内置拦截层 |
| **可观测性** | 无结构化日志、无 trace | OpenTelemetry 级别 Trace + Metrics |
| **模块边界** | 跨模块 import 自由、`platform/` 命名冲突 | 严格分层 + Anti-Corruption Layer |
| **工作流** | 单条因果链、无补偿事务 | Saga Pattern + 补偿回滚 |
| **时间模型** | `time.monotonic()` 硬编码 | 统一 Clock 抽象、支持回放/模拟 |
| **配置** | 散落在各 dataclass 默认值 | 集中式分层配置 + 运行时热更新 |
| **测试架构** | 手工 mock | Property-based + 确定性重放 + Chaos |

---

## 二、Distinguished 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                          │
│   CLI · Dashboard · API · Alerts                                   │
├─────────────────────────────────────────────────────────────────────┤
│                        Orchestration Layer                         │
│   Saga Coordinator · Lifecycle Manager · Scheduler                 │
├──────────────┬──────────────┬───────────────┬───────────────────────┤
│   Ingress    │    Core      │   Egress      │   Cross-cutting       │
│              │  (Pure)      │               │                       │
│ ┌──────────┐ │ ┌──────────┐ │ ┌───────────┐ │ ┌───────────────────┐ │
│ │ Market   │ │ │ Signal   │ │ │ Execution │ │ │ Observability     │ │
│ │ Feed     │ │ │ Engine   │ │ │ Gateway   │ │ │ (Trace/Metrics)   │ │
│ ├──────────┤ │ ├──────────┤ │ ├───────────┤ │ ├───────────────────┤ │
│ │ Event    │ │ │ Risk     │ │ │ Venue     │ │ │ Config Service    │ │
│ │ Ingress  │ │ │ Engine   │ │ │ Adapters  │ │ │ (Hot reload)      │ │
│ ├──────────┤ │ ├──────────┤ │ ├───────────┤ │ ├───────────────────┤ │
│ │ Replay   │ │ │ Decision │ │ │ Reconcile │ │ │ Clock Service     │ │
│ │ Store    │ │ │ Engine   │ │ │ Engine    │ │ │ (Unified time)    │ │
│ └──────────┘ │ ├──────────┤ │ └───────────┘ │ ├───────────────────┤ │
│              │ │ State    │ │               │ │ Secret Vault      │ │
│              │ │ Machine  │ │               │ └───────────────────┘ │
│              │ └──────────┘ │               │                       │
├──────────────┴──────────────┴───────────────┴───────────────────────┤
│                     Foundation Layer                                │
│   Event Store · WAL · Snapshot Store · Audit Log                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、12 项 Distinguished 级架构改造

### 改造 1: 消除 `Any` — 代数类型系统

**问题**: 当前 event routing 依赖字符串匹配 (`"MARKET" in et_u`)，pipeline I/O 大量使用 `Any`。

**方案**: 用 Tagged Union + Generic Protocol 让非法状态不可表达。

```python
# BEFORE: 运行时字符串匹配
def _route_for(self, event: Any) -> Route:
    et = getattr(event, "event_type", None)
    if "MARKET" in str(et).upper():
        return Route.PIPELINE

# AFTER: 编译期类型保证
@dataclass(frozen=True, slots=True)
class Envelope[E: BaseEvent]:
    """类型安全的事件信封 — 编译期即知道内容类型"""
    event: E
    metadata: EventMetadata
    trace_id: TraceId

# 路由通过类型分发，非法路由在编译期报错
class EventRouter:
    @singledispatchmethod
    def route(self, envelope: Envelope) -> Route:
        return Route.DROP

    @route.register
    def _(self, envelope: Envelope[MarketEvent]) -> Route:
        return Route.PIPELINE

    @route.register
    def _(self, envelope: Envelope[OrderEvent]) -> Route:
        return Route.EXECUTION
```

**改动范围**: `event/types.py`, `engine/dispatcher.py`, `engine/pipeline.py`

---

### 改造 2: Effect Boundary — 纯函数核心

**问题**: I/O（`time.monotonic()`, `time.sleep()`, 网络调用）散落在核心逻辑中，无法确定性测试。

**方案**: 所有副作用通过 Effect Protocol 注入，核心逻辑为纯函数。

```python
# 统一 Effect 接口
class Effects(Protocol):
    """所有副作用的唯一出口"""
    clock: ClockEffect        # 时间
    random: RandomEffect      # 随机数
    log: LogEffect            # 日志
    metrics: MetricsEffect    # 指标
    persist: PersistEffect    # 持久化

# 纯核心：输入 -> 输出，零副作用
def evaluate_signal(
    snapshot: StateSnapshot,
    config: SignalConfig,
) -> SignalResult:
    """纯函数 — 给定相同输入永远返回相同输出"""
    ...

# Effect 边界：仅在系统最外层
class LiveEffects:
    """生产环境的副作用实现"""
    clock = SystemClock()
    random = SecureRandom()
    log = StructuredLogger()
    metrics = PrometheusMetrics()
    persist = PostgresPersist()

class TestEffects:
    """测试环境：所有副作用可控"""
    clock = FakeClock(start=datetime(2024, 1, 1))
    random = DeterministicRandom(seed=42)
    log = CaptureLogger()
    metrics = InMemoryMetrics()
    persist = InMemoryPersist()
```

**改动范围**: 新增 `core/effects.py`，重构所有使用 `time`/`random` 的模块

---

### 改造 3: 有界事件总线 + 背压

**问题**: 当前 EventBus 同步发布、无背压。高频行情下 handler 阻塞会导致事件堆积。

**方案**: 引入有界环形缓冲区 + 背压信号 + 优先级队列。

```python
@dataclass(frozen=True, slots=True)
class BusConfig:
    capacity: int = 10_000          # 有界队列
    high_watermark: float = 0.8     # 80% 触发背压
    priority_classes: int = 3       # CRITICAL > NORMAL > LOW

class BoundedEventBus:
    """有界事件总线 — 背压 + 优先级 + 溢出策略"""

    def publish(self, envelope: Envelope) -> PublishResult:
        if self._ring.utilization > self._cfg.high_watermark:
            return PublishResult.BACKPRESSURE  # 上游减速
        self._ring.push(envelope, priority=envelope.priority)
        return PublishResult.ACCEPTED

    def drain(self, batch_size: int = 64) -> list[Envelope]:
        """批量消费 — 高优先级先出"""
        return self._ring.pop_batch(batch_size)
```

**改动范围**: `event/bus.py` 重写，`engine/coordinator.py` 适配

---

### 改造 4: Risk 内置于 Pipeline (Interceptor Chain)

**问题**: Risk 当前是外挂模块，与 Pipeline 松散耦合。Decision 事件的 risk check 路径不清晰。

**方案**: Risk 作为 Pipeline 的内置拦截器链（Interceptor Pattern）。

```python
class PipelineInterceptor(Protocol):
    """Pipeline 拦截器 — 在 state 变更前/后执行"""
    def before_reduce(self, event: Envelope, state: StateSnapshot) -> InterceptResult: ...
    def after_reduce(self, event: Envelope, old: StateSnapshot, new: StateSnapshot) -> InterceptResult: ...

class InterceptResult(Enum):
    CONTINUE = "continue"     # 继续处理
    REJECT = "reject"         # 拒绝此事件
    KILL = "kill"             # 触发熔断

# Risk 规则作为拦截器注册
class RiskInterceptor(PipelineInterceptor):
    def before_reduce(self, event, state):
        if isinstance(event.event, OrderEvent):
            decision = self._aggregator.evaluate_order(event.event, meta=state)
            if decision.action == RiskAction.REJECT:
                return InterceptResult.REJECT
        return InterceptResult.CONTINUE

# Pipeline 变为拦截器链
class StatePipeline:
    def __init__(self, interceptors: list[PipelineInterceptor]):
        self._chain = interceptors

    def apply(self, event: Envelope, state: StateSnapshot) -> PipelineResult:
        for i in self._chain:
            result = i.before_reduce(event, state)
            if result != InterceptResult.CONTINUE:
                return PipelineResult(state=state, intercepted_by=i, result=result)
        new_state = self._reduce(event, state)
        for i in self._chain:
            i.after_reduce(event, state, new_state)
        return PipelineResult(state=new_state, advanced=True)
```

**改动范围**: `engine/pipeline.py`, `risk/aggregator.py`, 新增 `engine/interceptors.py`

---

### 改造 5: Saga Pattern — 订单生命周期管理

**问题**: 当前订单流 `Intent → Order → Fill` 无补偿事务。失败后无回滚机制。

**方案**: Saga Coordinator 管理订单生命周期 + 补偿事务。

```python
@dataclass(frozen=True, slots=True)
class OrderSaga:
    """订单 Saga — 管理完整订单生命周期"""
    saga_id: str
    state: SagaState  # CREATED -> RISK_CHECKED -> SUBMITTED -> FILLED/FAILED/COMPENSATED

    # 每一步都有补偿动作
    steps: tuple[SagaStep, ...] = (
        SagaStep(action=validate_intent, compensate=noop),
        SagaStep(action=check_risk, compensate=release_risk_reservation),
        SagaStep(action=submit_order, compensate=cancel_order),
        SagaStep(action=await_fill, compensate=cancel_and_reconcile),
    )

class SagaCoordinator:
    """Saga 编排器 — 保证最终一致性"""

    async def execute(self, saga: OrderSaga) -> SagaResult:
        completed_steps = []
        for step in saga.steps:
            result = step.action(saga)
            if result.failed:
                # 反向补偿所有已完成步骤
                for completed in reversed(completed_steps):
                    completed.compensate(saga)
                return SagaResult.COMPENSATED
            completed_steps.append(step)
        return SagaResult.COMPLETED
```

**改动范围**: 新增 `engine/saga.py`, `engine/saga_steps.py`

---

### 改造 6: 统一 Clock + 时间旅行

**问题**: `time.monotonic()` 硬编码，回放和模拟依赖不同时间源但无统一抽象。

**方案**: 统一 Clock Protocol，支持系统时间、交易所时间、模拟时间。

```python
class Clock(Protocol):
    """统一时钟 — 系统中所有时间获取的唯一来源"""
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...

class SystemClock(Clock):
    """生产: 系统时钟"""

class ExchangeClock(Clock):
    """生产: 交易所时钟 (NTP 校准)"""

class SimulatedClock(Clock):
    """回测: 可控时钟 — 支持时间跳跃"""
    def advance(self, delta: timedelta) -> None: ...
    def set(self, t: datetime) -> None: ...

class ReplayClock(Clock):
    """回放: 从事件流中恢复时间"""
    def feed(self, event_ts: datetime) -> None: ...
```

**改动范围**: 新增 `core/clock.py`，替换所有 `time.monotonic()` / `time.time()` 调用

---

### 改造 7: 结构化可观测性 (OpenTelemetry 模式)

**问题**: 无 trace、无结构化 metrics、日志格式不统一。

**方案**: 基于 W3C Trace Context 的结构化可观测性。

```python
@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str       # 全链路唯一
    span_id: str        # 当前操作
    parent_id: str      # 父操作
    baggage: Mapping[str, str]  # 业务上下文

class Tracer(Protocol):
    def start_span(self, name: str, parent: TraceContext | None = None) -> Span: ...

class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, attributes: dict) -> None: ...
    def end(self) -> None: ...

# 每个事件自带 TraceContext
@dataclass(frozen=True, slots=True)
class EventMetadata:
    event_id: str
    timestamp: datetime
    trace: TraceContext      # 全链路追踪
    source: str              # 事件来源
    causation_id: str        # 因果链

# 使用示例：完整链路追踪
# MarketEvent(trace=T1) → Signal(trace=T1) → Decision(trace=T1) → Order(trace=T1) → Fill(trace=T1)
# 任何一环出问题，trace_id=T1 可追溯全链路
```

**改动范围**: 新增 `core/observability.py`，所有事件增加 `TraceContext`

---

### 改造 8: 严格分层 + Anti-Corruption Layer

**问题**: 模块间自由 import，`platform/` 与 stdlib 冲突，边界不清。

**方案**: 四层架构 + 显式依赖方向 + ACL。

```
Foundation (向下依赖: 无)
    core/types.py, core/clock.py, core/effects.py, core/errors.py

Domain (向下依赖: Foundation)
    state/, risk/rules/, decision/signals/

Application (向下依赖: Domain + Foundation)
    engine/, risk/aggregator.py, decision/engine.py

Infrastructure (向下依赖: Application + Domain + Foundation)
    execution/adapters/, infra/ (renamed from platform/), monitoring/
```

```python
# Anti-Corruption Layer: 外部 API → 内部模型
class BinanceACL:
    """Binance 反腐败层 — 隔离外部 API 变化"""

    @staticmethod
    def to_internal_fill(raw: dict) -> FillEvent:
        """Binance JSON → 内部 FillEvent (验证 + 转换)"""
        ...

    @staticmethod
    def from_internal_order(order: OrderCommand) -> dict:
        """内部 OrderCommand → Binance API 参数"""
        ...
```

**改动范围**: 重命名 `platform/` → `infra/`，调整 import 层级，新增 `core/` 基础层

---

### 改造 9: Write-Ahead Log + 崩溃恢复

**问题**: 当前状态在内存中，崩溃后丢失所有状态。

**方案**: WAL (Write-Ahead Log) 保证崩溃恢复 + 精确一次语义。

```python
class WriteAheadLog(Protocol):
    """WAL — 状态变更先写日志，再应用"""
    def append(self, entry: WALEntry) -> int: ...      # 返回 LSN
    def read_from(self, lsn: int) -> Iterator[WALEntry]: ...
    def truncate_before(self, lsn: int) -> None: ...

@dataclass(frozen=True, slots=True)
class WALEntry:
    lsn: int                    # Log Sequence Number
    event: Envelope             # 原始事件
    state_before: bytes         # 状态快照 (压缩)
    state_after: bytes          # 变更后状态
    checksum: str               # 完整性校验

class RecoverableStatePipeline:
    """可恢复的 Pipeline — 崩溃后从 WAL 恢复"""

    def apply(self, event: Envelope) -> PipelineResult:
        # 1. 写 WAL
        entry = WALEntry(event=event, state_before=self._snapshot())
        lsn = self._wal.append(entry)

        # 2. 应用变更
        result = self._inner_pipeline.apply(event)

        # 3. 确认 WAL
        self._wal.confirm(lsn)
        return result

    def recover(self) -> None:
        """崩溃恢复: 从最后一个 checkpoint 重放 WAL"""
        for entry in self._wal.read_from(self._last_checkpoint):
            self._inner_pipeline.apply(entry.event)
```

**改动范围**: 新增 `core/wal.py`，`engine/pipeline.py` 包装

---

### 改造 10: Property-Based Testing 架构

**问题**: 当前测试为手工 example-based，无法覆盖边界条件。

**方案**: Hypothesis + 确定性重放 + 不变量检查。

```python
# 系统级不变量 — 任何操作序列都必须满足
INVARIANTS = [
    "总资产 = 现金 + 持仓市值 + 冻结保证金",
    "任意时刻: gross_leverage <= max_leverage_cap",
    "Kill Switch 激活后: 仅允许 REDUCE 类型订单",
    "event_index 严格单调递增",
    "同一 idempotency_key 只产生一次副作用",
]

# Property-based test
@given(events=event_sequences())
def test_state_machine_invariants(events):
    engine = build_test_engine()
    for event in events:
        engine.emit(event)
        snapshot = engine.get_state_view()
        assert_invariants(snapshot, INVARIANTS)

# Deterministic replay test
def test_replay_produces_identical_state():
    """回放历史事件流，状态必须逐字节一致"""
    original = run_with_events(events, clock=FakeClock(t0))
    replayed = run_with_events(events, clock=FakeClock(t0))
    assert original.state == replayed.state
```

**改动范围**: 新增 `tests/properties/`，`tests/invariants.py`

---

### 改造 11: 插件架构 (Strategy/Venue/Indicator)

**问题**: 策略、交易所适配器、指标都是硬编码，新增需要改核心代码。

**方案**: 声明式插件注册 + 运行时发现。

```python
# 策略插件声明
@strategy_plugin(
    name="bollinger_band",
    version="1.0",
    params={"window": 20, "std_dev": 2.0},
)
class BollingerBandStrategy:
    def generate_signal(self, snapshot: StateSnapshot) -> SignalResult: ...

# 交易所适配器插件
@venue_plugin(name="binance_um", version="1.0")
class BinanceUMAdapter:
    def submit_order(self, cmd: OrderCommand) -> VenueAck: ...
    def cancel_order(self, cmd: CancelCommand) -> VenueAck: ...

# 插件注册表
class PluginRegistry:
    def discover(self, package: str) -> list[Plugin]: ...
    def get_strategy(self, name: str) -> StrategyPlugin: ...
    def get_venue(self, name: str) -> VenuePlugin: ...
```

**改动范围**: 新增 `core/plugins.py`，重构 `decision/signals/`, `execution/adapters/`

---

### 改造 12: 集中式分层配置 + 热更新

**问题**: 配置散落在各模块 dataclass 默认值中，无法运行时调整。

**方案**: 分层配置 (defaults < file < env < runtime) + 变更通知。

```python
class ConfigService:
    """分层配置服务 — 支持热更新"""

    _layers: tuple[ConfigLayer, ...] = (
        DefaultsLayer(),           # 代码内默认值
        FileLayer("config.yaml"),  # 文件配置
        EnvLayer(),                # 环境变量覆盖
        RuntimeLayer(),            # 运行时热更新
    )

    def get(self, key: str, type_: type[T]) -> T:
        """类型安全的配置读取"""
        for layer in reversed(self._layers):
            if layer.has(key):
                return layer.get(key, type_)
        raise ConfigKeyError(key)

    def watch(self, key: str, callback: Callable[[T], None]) -> None:
        """配置变更通知 — 用于热更新"""
        ...

    def hot_update(self, key: str, value: Any) -> None:
        """运行时热更新 (仅 RuntimeLayer)"""
        self._layers[-1].set(key, value)
        self._notify_watchers(key)
```

**改动范围**: 新增 `core/config.py`，重构所有 `*Config` dataclass

---

## 四、实施顺序

```
Phase 0 (Day 1)        Phase 1 (Week 1)       Phase 2 (Week 2-3)      Phase 3 (Week 4-5)
┌──────────────┐      ┌───────────────┐      ┌───────────────┐       ┌───────────────┐
│  Foundation   │  →   │  Core Reform  │  →   │  Integration  │   →   │  Advanced     │
│              │      │               │      │               │       │               │
│ #8 分层重构  │      │ #1 类型系统   │      │ #4 Risk拦截链 │       │ #9 WAL恢复    │
│ platform→    │      │ #2 Effect边界 │      │ #5 Saga模式   │       │ #10 Property   │
│   infra      │      │ #6 统一Clock  │      │ #7 可观测性   │       │    测试       │
│ 建立 core/   │      │ #3 有界EventBus│     │ #12 配置服务  │       │ #11 插件架构  │
└──────────────┘      └───────────────┘      └───────────────┘       └───────────────┘
```

### Phase 0: Foundation (1 天) — 先清路障
- [ ] 重命名 `platform/` → `infra/` (解决 stdlib 冲突)
- [ ] 创建 `core/` 基础层目录结构
- [ ] 定义依赖方向规则（Foundation → Domain → Application → Infrastructure）
- [ ] 确保 309 个测试仍全部通过

### Phase 1: Core Reform (1 周) — 核心改造
- [ ] 改造 1: 代数类型系统 — Envelope[E] + singledispatch 路由
- [ ] 改造 2: Effect Boundary — 纯函数核心 + Effect Protocol
- [ ] 改造 6: 统一 Clock — 替换所有 time.monotonic()/time.time()
- [ ] 改造 3: 有界 EventBus — 背压 + 优先级 + 环形缓冲区
- [ ] 测试全部通过 + 新增 core/ 模块测试

### Phase 2: Integration (2 周) — 集成改造
- [ ] 改造 4: Risk 拦截器链 — Pipeline Interceptor Pattern
- [ ] 改造 5: Saga Pattern — 订单生命周期管理
- [ ] 改造 7: 结构化可观测性 — TraceContext + 结构化日志
- [ ] 改造 12: 集中式配置服务 — 分层 + 热更新
- [ ] 端到端集成测试通过

### Phase 3: Advanced (2 周) — 高级特性
- [ ] 改造 9: WAL + 崩溃恢复
- [ ] 改造 10: Property-Based Testing — Hypothesis + 不变量检查
- [ ] 改造 11: 插件架构 — Strategy/Venue/Indicator 插件化
- [ ] 全量回归测试 + 性能基准

---

## 五、验收标准

### Distinguished 级检查清单
- [ ] 零 `Any` 在核心路径 (engine, risk, decision, state)
- [ ] 所有副作用通过 Effect Protocol 注入
- [ ] 事件流有背压保护，不会 OOM
- [ ] Risk 作为 Pipeline 内置拦截器，无法绕过
- [ ] 订单生命周期有 Saga 补偿事务
- [ ] 全链路 Trace 可追踪任意事件因果链
- [ ] 崩溃后可从 WAL 恢复到精确状态
- [ ] 策略/交易所/指标可插件化扩展
- [ ] Property-based 测试覆盖核心不变量
- [ ] 配置支持运行时热更新
- [ ] `pytest` 从项目目录直接运行（无 stdlib 冲突）
- [ ] 所有模块严格遵守分层依赖方向

### 量化指标
| 指标 | Staff (当前) | Distinguished (目标) |
|------|-------------|---------------------|
| `Any` 在核心路径 | ~50 处 | 0 |
| 测试覆盖率 | <5% | >80% |
| 副作用隔离 | 0% | 100% |
| 崩溃恢复时间 | ∞ (丢失) | <1s |
| 事件延迟 P99 | 未测 | <10ms |
| 配置热更新 | 不支持 | <100ms 生效 |
