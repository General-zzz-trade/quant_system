# 目标架构（阶段 2）

- 依据：`docs/arch/current-state.md`（HEAD `e3cd2d6`）。
- 状态：**提议，待批准**。本文件里的每个取舍都有对应的 ADR，ADR 里列了备选方案。凡是会改变交易或风控行为的地方都单独标注，未经批准不执行。

## 1. 目标与范围

### 1.1 要解决什么

`current-state.md` 第 7 节的五个结构性结论，对应五个目标：

| 现状 | 目标 |
|---|---|
| 候选运行时被已关闭的 ML 系统拖着 | 候选运行时放进一个独立的包，不 import 任何遗留代码 |
| 退出侧不可靠（R1–R7） | 退出路径有明确语义、有测试、失败时有告警 |
| 没有端口，订单库被 7 个以上文件直接写 SQL | 交易所、行情、存储、时钟、通知都经过端口；每张表只有一个写入者 |
| 约一半代码不在运行路径上 | 新包里只有在用的代码；遗留代码在边界之外，按批次处置 |
| 安全网不均匀，没有架构规则 | 每条不变量对应一个自动检查，全部进 CI |

### 1.2 范围

- **范围内**：Binance USDT-M 上的目标仓位运行时（现在的 `runner.binance_target`）及其依赖的策略、风控、执行、持久化和研究撮合。
- **范围外，原地冻结**：ML 线（`engine`、`decision`、`features`、`alpha`、`runner/binance_main.py`、`runner/offline` 的 ML 部分）、Bybit / OKX / Polymarket、`rust_bin`。它们不迁入新包，只接受安全修改和删除。去留见 ADR-0008。
- 新包名用 `trading/`。仓库里没有同名目录。

## 2. 分层与依赖规则

### 2.1 四层

依赖只能向内：`infrastructure → adapters → application → domain`。

| 层 | 放什么 | 允许依赖 | 禁止 |
|---|---|---|---|
| `trading/domain` | 策略函数、再平衡、订单与状态转移表、持仓投影、风控规则。全部是纯计算 | 标准库的纯计算模块、numpy、pandas | 任何 I/O：`sqlite3`、`urllib`、`socket`、`subprocess`、`time`、`logging`、`os`；Rust 扩展 `_quant_hotpath` |
| `trading/application` | 用例（每小时决策、下单生命周期、止损维护、急停平仓、对账）和端口定义 | `domain` | 同上。时间只能经 `Clock` 端口取得 |
| `trading/adapters` | 端口的实现：Binance REST、SQLite、文件、模拟交易所、通知、系统时钟 | `application` 的端口、`domain` | 读环境变量；适配器之间互相 import |
| `trading/infrastructure` | 配置加载（唯一读环境变量的地方）、组装根、命令行入口、进程租约、日志设置 | 以上全部 | — |

研究代码（现在的 `lab/*_study.py`）是另一个入口：它依赖 `trading.domain` 和 `trading.adapters.simulated`，不属于 `trading/`。

### 2.2 包结构

```
trading/
  domain/
    market.py            Bar、资金费、可见时间（available_at）语义
    strategy/overlay.py  StrategySpec、compute_targets、latest_target
    rebalance.py         Intent、plan
    orders.py            OrderIntent、OrderStatus、合法转移表、Fill
    position.py          StrategyPosition
    risk/exposure.py     3 倍上限常量与判定函数
    risk/drawdown.py     周期键、高水位、触发与闩锁规则
    risk/limits.py       单笔 / 日累计 / 频率限额规则
    risk/kill.py         KillState
  application/
    ports.py             全部端口（Protocol）
    session.py           每小时一步：行情 → 目标 → 意图 → 提交
    runtime.py           主循环编排：预热、轮询、对账、风险动作
    execution/lifecycle.py   意图预留、发单、查询、成交入账、结算
    execution/stops.py       交易所侧止损维护
    execution/flatten.py     急停与回撤平仓
    risk/entry_gate.py       入场组合门
    reconcile.py             账本与交易所核对
  adapters/
    binance/             rest、market_data、account、order_gateway、stop_gateway
    sqlite/              order_journal、drawdown_store、limit_store、paper_store
    files/               kill_file、event_log、status_file、signed_spec
    simulated/           bar_broker（研究撮合模型，纸面与研究共用）
    notify/              telegram、null
    system/              clock、sleeper
  infrastructure/
    settings.py          环境变量 → 带校验的不可变配置对象
    bootstrap.py         组装根
    cli/target.py        运行时入口
    cli/kill.py          急停命令行
    lease.py
```

### 2.3 规则的自动化检查

用 import-linter 在 CI 里检查（草案，落地时以实际运行输出为准）：

```ini
[importlinter]
root_package = trading
include_external_packages = True

[importlinter:contract:layers]
name = trading 的依赖只能向内
type = layers
layers =
    trading.infrastructure
    trading.adapters
    trading.application
    trading.domain

[importlinter:contract:no-legacy]
name = trading 不依赖遗留包
type = forbidden
source_modules = trading
forbidden_modules = runner, shared, execution, engine, decision, features, alpha,
    strategy, event, state, risk, data, monitoring, attribution, infra,
    polymarket, scripts, research, lab

[importlinter:contract:core-no-io]
name = domain 与 application 不做 I/O
type = forbidden
source_modules = trading.domain, trading.application
forbidden_modules = sqlite3, urllib, http, socket, subprocess, time, _quant_hotpath

[importlinter:contract:adapters-independent]
name = 适配器之间互不依赖
type = independence
modules = trading.adapters.binance, trading.adapters.sqlite, trading.adapters.files,
    trading.adapters.simulated, trading.adapters.notify, trading.adapters.system
```

import-linter 查不了的两条规则用 AST 测试补：

- `os.environ` / `os.getenv` 只出现在 `trading/infrastructure/`。
- `trading/domain` 不 import `logging` 和 `os`。

迁移期间，旧位置的模块变成只做重导出的薄壳（例如 `lab/strategy.py` 重导出 `trading.domain.strategy.overlay`）。所以"遗留包 import 新包"是允许的，反方向被 `no-legacy` 契约禁止。

## 3. 端口

端口的方法集合来自现有代码的实际调用点（`current-state.md` 2.2 的 V8–V10，以及订单生命周期追踪）。全部用 `typing.Protocol` 定义在 `trading/application/ports.py`。

| 端口 | 方法 | 现在的直接调用点 | 目标适配器 |
|---|---|---|---|
| `MarketData` | `closed_klines(symbol, interval, limit, now_ms)`、`funding_history(symbol, limit)` | `runner/binance_target.py:277,279,328,334` | `adapters.binance.market_data`；研究用 CSV 快照 |
| `ExchangeAccount` | `equity()`、`positions(symbol=None)`、`open_orders(symbol)`、`filters(symbol)`、`blockers()`、`mark_price(symbol)` | `shared/exposure_gate.py:26,30,34`；`execution_adapter.py:165,260-277`；`runner/binance_target.py:236,291` | `adapters.binance.account`；`adapters.simulated` |
| `OrderGateway` | `send_market(...)`、`send_limit(...)`、`get_order(...)`、`cancel_order(...)`、`trades(symbol, order_id)` | `shared/confirmed_execution.py:176,185,189,226,249,254` | `adapters.binance.order_gateway`；`adapters.simulated` |
| `StopOrderGateway` | `place_stop(...)`、`get_stop(...)`、`cancel_stop(...)` | `protective_stops.py:87,103,106,125` | `adapters.binance.stop_gateway` |
| `OrderJournal` | `reserve(intent)`、`transition(intent_id, to)`、`record_fill(...)`、`settle(...)`、`unresolved(...)`、`undelivered(owner)`、`consume(owner, fill_id, apply)`、`checkpoint(owner)`、`adopt(...)`、止损行的读写 | 7 个以上文件直接写 SQL（V8） | `adapters.sqlite.order_journal`，三张订单表和止损表的唯一写入者 |
| `DrawdownStore` | `load(scope, period)`、`save(...)`、`observe(...)` | `shared/drawdown_guard.py:45` | `adapters.sqlite.drawdown_store` |
| `LimitStore` | `load_day(scope, day)`、`add(...)` | 现在在 Rust 进程内存里 | `adapters.sqlite.limit_store` |
| `KillSwitch` | `state()` | `runner/binance_target.py:258,316` | `adapters.files.kill_file` |
| `SpecSource` | `load_verified()` | `runner/binance_target.py:52-58,386` | `adapters.files.signed_spec` |
| `Clock`、`Sleeper` | `now()`、`sleep(s)` | `time.time` / `time.sleep` 散落在 5 个文件 | `adapters.system`；测试用假时钟 |
| `Notifier` | `alert(level, title, detail)` | 资金路径上没有任何调用 | `adapters.notify.telegram`、`null` |
| `EventLog`、`StatusSink` | `write(kind, **fields)`、`publish(status)` | `runner/binance_main.py:130-133`；`runner/binance_target.py:363-365` | `adapters.files` |
| `PaperStore` | `load()`、`save(state)` | 不存在，纸面状态只在内存 | `adapters.sqlite.paper_store` |

约束：

- 应用层的构造参数全部用端口类型，不再出现 `Any`。
- 执行模式下，入场门的 kill、回撤、敞口、限额四个组件必须齐全，缺任何一个在构造时抛错。
- 适配器不读环境变量。所有配置由 `infrastructure.settings` 读一次，校验后以不可变对象传入。

## 4. C4 容器图

```mermaid
flowchart TB
  OP(["操作员"])
  RS(["研究者"])

  subgraph HOST["交易主机"]
    RT["目标仓位运行时<br/>python -m runner.binance_target<br/>（systemd，默认纸面）"]
    KC["急停命令行<br/>python -m trading.infrastructure.cli.kill"]
    MON["监控与对账定时任务<br/>（只读，告警）"]
    subgraph STATE["状态目录（受管）"]
      DB[("orders.sqlite<br/>drawdown.sqlite<br/>paper.sqlite")]
      FILES["kill_switch.json<br/>status.json<br/>events.jsonl<br/>manifest.json"]
    end
    SPEC["已签名的策略 spec<br/>final_spec.json + .sig"]
    LEG["遗留运行时（冻结）<br/>alpha_main / shadow_main / polymarket"]
  end

  subgraph RESEARCH["研究环境"]
    ST["研究 study<br/>python -m lab.*_study"]
    SNAP[("不可变数据快照<br/>data_files/binance_public_日期")]
    REG["trials.jsonl<br/>forward_looks.jsonl"]
  end

  BPUB["Binance 公共行情<br/>（主网 REST）"]
  BACC["Binance 账户与下单<br/>（测试网或主网 REST）"]
  TG["Telegram"]

  OP -->|"写入急停"| KC --> FILES
  OP -->|"启停、看状态"| RT
  RT -->|"已收盘 K 线、资金费"| BPUB
  RT -->|"权益、持仓、订单、止损"| BACC
  RT --> DB
  RT --> FILES
  RT -->|"验签后读取"| SPEC
  RT -->|"告警"| TG
  MON -->|"只读"| BACC
  MON --> FILES
  MON --> TG
  RS --> ST
  ST --> SNAP
  ST --> REG
  ST -->|"发布并签名"| SPEC
```

说明：

- 运行时是单进程、单线程。纸面和执行是同一个进程的两种组装方式。
- systemd 的 `ExecStart` 不变，仍是 `python3 -m runner.binance_target`。这个模块在迁移后是一个转发到 `trading.infrastructure.cli.target` 的薄壳。
- 状态目录是"受管"的：带清单和账户身份，执行模式下缺库拒绝启动（ADR-0007）。
- 遗留运行时与新运行时不共享代码路径，也不共享状态目录。

## 5. C4 组件图（目标仓位运行时内部）

```mermaid
flowchart LR
  subgraph INFRA["infrastructure"]
    CLI["cli/target.py"]
    SET["settings.py<br/>环境变量 → 不可变配置"]
    BOOT["bootstrap.py<br/>组装根"]
    LEASE["lease.py"]
  end

  subgraph APP["application"]
    RUN["runtime.py<br/>主循环"]
    SES["session.py<br/>每小时一步"]
    GATE["risk/entry_gate.py<br/>入场门"]
    LIFE["execution/lifecycle.py<br/>下单生命周期"]
    STOPS["execution/stops.py"]
    FLAT["execution/flatten.py<br/>急停与回撤平仓"]
    REC["reconcile.py"]
    PORTS{{"ports.py"}}
  end

  subgraph DOM["domain"]
    STRAT["strategy/overlay.py"]
    REB["rebalance.py"]
    ORD["orders.py<br/>状态与转移表"]
    POS["position.py"]
    RISK["risk/*<br/>敞口、回撤、限额、kill"]
  end

  subgraph ADP["adapters"]
    BIN["binance/*"]
    SQL["sqlite/*"]
    FIL["files/*"]
    SIMX["simulated/bar_broker"]
    NOT["notify/*"]
    SYS["system/clock"]
  end

  CLI --> SET --> BOOT
  CLI --> LEASE
  BOOT --> RUN
  BOOT -.->|"构造并注入"| ADP

  RUN --> SES
  RUN --> REC
  RUN --> FLAT
  RUN --> STOPS
  SES --> STRAT
  SES --> REB
  SES --> GATE
  SES --> LIFE
  FLAT --> LIFE
  GATE --> RISK
  LIFE --> ORD
  LIFE --> POS
  STOPS --> ORD

  SES --> PORTS
  GATE --> PORTS
  LIFE --> PORTS
  STOPS --> PORTS
  FLAT --> PORTS
  REC --> PORTS

  BIN -.->|"实现"| PORTS
  SQL -.->|"实现"| PORTS
  FIL -.->|"实现"| PORTS
  SIMX -.->|"实现"| PORTS
  NOT -.->|"实现"| PORTS
  SYS -.->|"实现"| PORTS
```

要点：

- 入场订单到达 `OrderGateway` 的唯一路径是 `session → entry_gate → lifecycle`。`adapters.binance` 只在 `bootstrap` 里被构造，应用层拿不到具体类。
- 减仓不经过入场门（与现在一致），但经过同一个 `lifecycle`，所以"至多发一次、只认交易所成交"对减仓同样成立。
- 纸面模式把 `BIN` 换成 `SIMX`，其余组件不变（ADR-0006）。

## 6. 现有文件 → 目标位置

| 现有 | 目标 | 方式 |
|---|---|---|
| `lab/strategy.py`、`lab/data.py` 的聚合函数 | `trading/domain/strategy/overlay.py`、`trading/domain/market.py` | 移动，旧位置留薄壳 |
| `lab/rebalance.py` | `trading/domain/rebalance.py` | 移动 |
| `shared/strategy_position.py` | `trading/domain/position.py` | 移动 |
| `lab/orders.py`（依赖 Rust 事件类型） | `trading/domain/orders.py`（纯 Python 数据类） | 重写，约 40 行 |
| `shared/exposure_gate.py` | 规则 → `domain/risk/exposure.py`；取数 → `application/risk/entry_gate.py` 经 `ExchangeAccount` | 拆分 |
| `shared/drawdown_guard.py` | 规则 → `domain/risk/drawdown.py`；存储 → `adapters/sqlite/drawdown_store.py` | 拆分 |
| `execution/safety/limits.py` + Rust 限额器 | `domain/risk/limits.py` + `adapters/sqlite/limit_store.py` | 重写为纯 Python（ADR-0008） |
| `shared/entry_risk.py` | `application/risk/entry_gate.py` | 移动并加类型 |
| `shared/confirmed_execution.py` | `application/execution/lifecycle.py` | 移动；SQL 移入适配器 |
| `shared/execution_journal.py` | `adapters/sqlite/order_journal.py` | 移动并扩展为唯一写入者 |
| `execution/adapters/binance/execution_adapter.py` | 减仓定量与对账 → `application`；其余并入 `lifecycle` | 拆分 |
| `execution/adapters/binance/protective_stops.py` | `application/execution/stops.py` | 移动；SQL 移入适配器 |
| `execution/adapters/binance/adapter.py`、`rest.py` | `adapters/binance/*`，按端口拆成 4 个类 | 移动并拆分 |
| `shared/kill_file.py` | `adapters/files/kill_file.py` + `infrastructure/cli/kill.py` | 拆分 |
| `infra/model_signing.py` 的验签 | `adapters/files/signed_spec.py` | 移动在用部分 |
| `runner/offline/broker.py` 的 `BarBroker` | `adapters/simulated/bar_broker.py` | 移动 |
| `lab/simulate.py` | 研究侧保留 `simulate()`；`Sleeve` 的职责由 `application/session.py` + 模拟适配器承担 | 见 ADR-0006 |
| `runner/binance_target.py` | `application/runtime.py`、`application/session.py`、`infrastructure/bootstrap.py`、`infrastructure/cli/target.py` | 拆分；旧文件留薄壳 |
| `runner/binance_main.py` 的 6 个符号 | 各归其位（`EquitySource` → application，`EventJournal` → adapters.files，`build_adapters` → bootstrap） | 复制后切断依赖 |
| `shared/execution_lease.py` | `infrastructure/lease.py` | 移动 |

## 7. 不变量清单与自动化检查

"现状"一列：✓ 表示已有测试且成立；✗ 表示不成立或没有检查。

### 7.1 风控上限

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| L1 | 总名义敞口加拟下单名义不超过 `min(配置, 3) × 权益`；3 是 `domain` 里的常量，配置只能调低 | 单元测试（参数化边界值）；架构测试：常量只定义一次 | ✓ |
| L2 | 每张入场单在到达交易所之前都经过入场门 | 分层契约（应用层拿不到具体适配器）；用记录型假网关的测试：没有通过记录就没有发单调用 | ✗ 靠调用方自觉（R14） |
| L3 | 执行模式下入场门的四个组件缺一不可 | 构造期断言 + 单元测试 | ✗（R14） |
| L4 | 权益、持仓、状态库任何一项未知或不可读时，入场被拒 | 现有 `tests/offline/test_risk_safety.py` 迁入 | ✓ |
| L5 | 回撤闩锁和日限额计数跨重启保持；状态库缺失或账户身份不符时执行模式拒绝启动 | 重启测试；缺库测试 | ✗（R6、R12） |
| L6 | 在 spec 验签、研究闸门和 scope 校验全部通过之前，不对账户做任何写操作 | 记录型假账户测试：闸门不过时 POST 次数为 0 | ✗（R11） |
| L7 | 没有自动调高上限的代码路径 | 配置对象不可变（frozen dataclass）；架构测试：`settings` 之外不读环境变量 | 部分 |

### 7.2 Kill-switch

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| K1 | kill 激活后不再有新的入场单 | 现有测试迁入 | ✓ |
| K2 | kill（平仓）或回撤闩锁激活后：先撤销全部入场挂单，再提交 reduce-only 单，直到交易所持仓为 0 或重试耗尽并告警 | 由探针改写的场景测试（场景 a、e） | ✗（R1、R2） |
| K3 | 命令行与运行时解析出同一个 kill 文件路径 | 单元测试：两者调用同一个解析函数 | ✗（R5） |
| K4 | 运行时从不改写操作员写入的 kill 文件 | 架构测试：`KillSwitch` 端口没有写方法；写方法只在 `cli/kill.py` | ✗（R17） |
| K5 | 有持仓却没有有效的交易所侧止损时，入场被拒并告警 | 场景测试（探针 C 段） | ✗（R3） |
| K6 | kill 文件损坏或不可读时按激活处理 | 现有测试迁入 | ✓ |

### 7.3 无前视偏差

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| F1 | 时刻 T 的目标权重只依赖 `available_at ≤ T` 的数据 | 性质测试：改动 T 之后的任意 K 线，T 及之前的目标不变 | 部分（有截尾一致性测试） |
| F2 | 截尾窗口算出的值等于全历史算出的值 | 现有 `tests/unit/lab/test_lab_engine.py` 迁入 | ✓ |
| F3 | 行情适配器只返回已收盘 K 线 | 现有 `test_klines_closed_only_and_ascending` 迁入 | ✓ |
| F4 | 聚合只保留完整且对齐 UTC 的分组 | 单元测试 | ✓ |
| F5 | 前向窗口的数据只在登记过的查看中读取 | 现有 `lab/forward.py` 的拒绝逻辑测试；数据加载函数加日期上界参数（范围外，见 ADR-0008 的后续项） | 部分（R21） |

### 7.4 订单状态机

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| O1 | 同一意图至多向交易所提交一次 | 现有 `tests/offline/test_execution_safety.py`、`test_market_execution.py` 迁入 | ✓ |
| O2 | 状态只能沿转移表变化，非法转移抛错 | 单元测试：枚举全部状态对 | ✗ 状态是自由字符串 |
| O3 | 订单表和止损表只有一个写入者 | 契约：`sqlite3` 只允许在 `trading.adapters.sqlite` 被 import | ✗（V8） |
| O4 | 只有交易所成交记录才成为成交；每笔成交恰好入账一次 | 现有测试迁入 | ✓ |
| O5 | 五个崩溃点重启后不重复下单、不丢已入库成交 | 现有子进程崩溃测试迁入；补 `TargetRuntime` 带未决订单重启的测试 | 部分 |
| O6 | 减仓被拒绝预留时写日志并告警 | 场景测试 | ✗（R7） |

### 7.5 回测 / 实盘一致性

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| P1 | 纸面运行的成交与研究模拟逐笔相同 | 现有 `test_paper_runtime_reproduces_lab_simulation` 迁入 | ✓ |
| P2 | 纸面与执行运行同一份 `session` 代码，只换适配器 | 分层契约 + 同一套端口契约测试分别跑在模拟交易所和假 Binance 上 | ✗ 两条代码路径 |
| P3 | 纸面状态跨重启连续 | 重启测试 | ✗（R10） |
| P4 | 研究与实盘共用的参数（限价偏移、最小名义额）只有一个来源 | 架构测试：常量只定义一次 | ✗ 两处硬编码 |

### 7.6 结构

| ID | 不变量 | 检查方式 | 现状 |
|---|---|---|---|
| S1 | 依赖只能向内；不依赖遗留包；核心层不做 I/O | import-linter 四条契约 | ✗ |
| S2 | 资金相关测试全部在 CI 门禁里 | CI 配置；一个测试断言门禁清单覆盖 `tests/architecture`、`tests/characterization`、迁入的安全测试 | ✗（R22） |

## 8. 关键决策

每条都有 ADR，状态都是"提议"。

| ADR | 决策 | 建议 |
|---|---|---|
| [0001](../adr/0001-migration-scope-and-strangler-boundary.md) | 迁移范围与新旧边界 | 新建 `trading/` 包，只迁候选运行时；遗留代码原地冻结 |
| [0002](../adr/0002-layering-and-enforcement.md) | 分层方式与强制手段 | 四层 + import-linter；组装根在最外层 |
| [0003](../adr/0003-ports.md) | 端口粒度 | 按能力拆成小端口，用 `Protocol` |
| [0004](../adr/0004-order-state-machine-and-journal-ownership.md) | 订单状态机与订单库所有权 | 状态枚举加转移表放在 `domain`；SQLite 适配器是唯一写入者 |
| [0005](../adr/0005-exit-path-semantics.md) | 急停、平仓、止损的语义 | **改变风控行为，需要您逐项确认** |
| [0006](../adr/0006-one-session-for-paper-and-live.md) | 纸面与实盘是否同一份会话代码 | 同一份，换适配器 |
| [0007](../adr/0007-state-durability.md) | 状态持久化 | 受管状态目录；缺库拒启；限额计数与纸面状态落盘 |
| [0008](../adr/0008-legacy-code-and-rust.md) | 遗留代码与 Rust 扩展的处置 | 新路径不依赖 Rust；死代码分批删除；冻结线加执行硬开关 |

## 9. 明确不做的事

- 不改策略数学、不改任何风控阈值的数值。
- 不迁移 ML 线和其他交易所的代码。
- 不引入新的外部依赖作为运行时依赖。import-linter 只在开发与 CI 环境使用。
- 不引入消息总线、异步框架或多进程。运行时保持单进程单线程轮询。
- 不改 systemd 单元的 `ExecStart`。
