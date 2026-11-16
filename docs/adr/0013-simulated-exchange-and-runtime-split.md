# ADR-0013：模拟交易所适配器、模拟盘走执行路径、运行时按职责拆分

- 状态：**已批准**（2026-11-15，技术负责人"现在全部批准"）。原为提议稿；下面的内容是批准时的版本，实施中如有变动在本文更正并记入 `execution-log.md`。
- 日期：2026-11-15

## 背景

【代码】

- 模拟盘和执行在会话里是两条路径：`trading/application/session.py:132,139,173,226,238` 有 5 处 `if self.sleeve is not None`。模拟盘绕过了入场门、订单库、止损和对账。
- 因此运行时入口会加载研究包（`runner/binance_target.py:28-29`），完成标准第 1 条不成立【实际运行结果，见设计第 1 节】。
- `trading/application/runtime.py` 有 351 行，一个类同时管：预热、轮询和退避、对账、止损要求、急停/回撤/击穿、暂停、模拟盘存取和状态。
- 运行时靠在问题文字里找 `"stop_breached"` 决定平仓（`runtime.py:142`）。类型化见 ADR-0014。

独立审计（`design-2026-11-m5.md` 附录 A，H1、H2）证明：初稿说的"模拟盘和执行唯一的区别是注入的交易所适配器"做不到。下面这些差异都必须保留，否则模拟盘与研究不再逐笔一致，或者违反已批准的 ADR-0007：

| 差异 | 模拟盘 | 执行 | 依据 |
|---|---|---|---|
| 行情恢复后的补放 | 逐根补放 | 只用最新一根决策 | ADR-0007 补充批准；`runtime.py:83,282-285`；`tests/unit/lab/test_target_runtime_resilience.py:213` |
| 限价入场单的过期 | 由撮合器按 K 线过期（研究里先撮合、后过期，`runner/offline/broker.py:246`） | 由生命周期按秒过期（`lifecycle.py:293-302`，`target.py:150`） | 审计 H1-1 |
| 定仓用的权益 | 每个品种的模拟账户用自己的权益（与研究相同） | 账户权益乘以 1/品种数（`session.py:142`，`target.py:152`） | 审计 H1-3，实测最多差 14.99 USDT |
| 新 K 线的处理顺序 | 撮合 → 消费成交 → 决策 | 退出侧 → 行情 → 决策；成交在对账时消费 | 审计 H1-4 |
| 每秒限额的时钟 | K 线时间 | 真实的单调时钟（`target.py:140-142`） | 审计 H1-5 |
| 止损 | 模拟交易所按 K 线最高价、最低价触发（决定 L） | 交易所侧真实止损 | 审计 H1-2：研究撮合器在概览策略的配置下挂不了止损 |

## 决定

1. **模拟交易所**：`trading/adapters/simulated/exchange.py` 的 `SimulatedExchange`。
   - 包装研究撮合器 `BarBroker`；撮合器从 `runner/offline/broker.py` 原样搬到 `trading/adapters/simulated/bar_broker.py`。
   - 实现 `ExchangeAccount`（包括 `filters`、`get_ticker`）、`OrderGateway`、`StopOrderGateway`、`AccountSnapshot`，以及新端口 `BarListener`。
   - 止损由 `StopModel` 模拟：K 线最高价或最低价越过触发价就成交，成交价是触发价加止损滑点。
   - 签名见设计第 4.2 节。
2. **模式差异集中在一处**：`trading/infrastructure/modes.py` 的 `ModePolicy`。
   - 上表每一行都是它的一个字段，由组装根按模式设定。
   - 会话、生命周期、退出监督者只读 `ModePolicy`，不判断"现在是不是模拟盘"。
3. **会话只有一条路径**：删除会话里的全部模拟盘分支。
4. **运行时拆分**：
   - `MarketPoller`：预热、轮询、退避、行情问题，按 `ModePolicy.catch_up` 补放；
   - `ExitSupervisor`：对账、ADR-0011 的整账户检查、暂停、止损同步、急停/回撤/击穿；另有 `settle(symbol)`，供模拟盘在新 K 线撮合之后、决策之前消费成交和处理过期；
   - `StatusReporter`：状态快照。
   - 两种模式下，退出侧都在每轮开始时先执行一次。
5. **研究与实盘共用的参数只定义一次**：限价偏移现在写了三份（`lab/simulate.py:39`、`runner/offline/broker.py:38`、`trading/infrastructure/target.py:48`），改为一个定义，各处引用。
6. **先验证可行性**（设计第 10 节 S9）：
   - 在一个丢弃型分支上实现到验收门能跑的程度；
   - 用审计员的探针（限价单过期、权益、限额时钟）和验收门判断是否可行；
   - 由技术负责人决定是否正式实施（S10）。
7. **切换时机**：
   - 模拟盘没有部署、不产出前向证据（2026-11-14 决定），所以 ADR-0006 的排法 1、2 现在没有区别。
   - S10 要么在 M5 的 4 周计时开始之前合并，要么在 M5 通过之后合并：往测试网主机部署资金路径的改动会让计时从头开始（设计第 6 节）。

## 状态未知与失败方向

- 模拟交易所的状态读不出时，与真实交易所一样返回错误，由同一套逻辑失败即关闭。
- 模拟盘组装时：
  - 不构造任何带凭据的适配器，不发出任何签名请求，不读账户相关的环境变量（审计 M6）；
  - 这三点由测试固定。

## 有意的分叉

- 入场门和止损在模拟盘里生效，研究里没有。
- 每次分叉写一条 `paper_divergence` 事件，记下原因：入场门拒绝、止损触发、暂停。对比工具（设计 G6）把它们列为"已解释的差异"。
- 止损触发后，模拟盘与实盘一样暂停该品种入场，由人解除（决定 L）。
- 挂单入场跨 K 线挂着时，执行路径把它当作未了结订单，同品种的新入场（例如翻向后的新方向）被拒，写一条 `paper_divergence`（`entry_refused`）；研究里不会拒。当前规格是市价入场，不受影响；以后改用挂单入场时再处理（技术负责人 2026-11-16，见下文"实施更正（S10）"第 3a 部分）。

## 备选方案

- **维持两条路径**（ADR-0006 方案 A）：实盘专有的代码只有单元测试和测试网覆盖；完成标准第 1 条永远不成立。
- **模拟盘的止损只占位、不触发**：保持与研究一致，但检验不到止损路径。技术负责人没有选。
- **重写撮合器**：失去"研究与模拟盘逐笔一致"由构造保证的性质。
- **只拆运行时、不做模拟交易所**：可以先做（就是 S3），但第 1 条标准和 P2 不变量仍不成立。

## 后果

- 模拟盘会执行入场门、订单库、止损和对账的代码。部署后，这些代码在真实时间上运行。
- 模拟盘的状态 = 模拟交易所的快照（由 `PaperStore` 保存）+ 订单库。ADR-0009 写明，方案 B 之后模拟盘目录也要初始化，使用同一个 `state.sqlite`。
- `ModePolicy` 成为模拟盘与实盘差异的唯一清单。增加一项差异要改它，并补测试。

## 验收

不改断言、必须通过：
- `tests/unit/lab/test_target_runtime.py::test_paper_runtime_reproduces_lab_simulation`，包括 `("maker", 1)` 在内的所有参数化组合；
- `tests/unit/lab/test_target_runtime_resilience.py` 的三条重启测试（第 68、172、192 行）；
- `tests/unit/lab/test_simulation_golden.py` 的两个指纹。

新测试：
- 会话里没有模拟盘分支（架构测试）；
- 加载运行时入口时不加载 `lab`、`runner.offline`（完成标准第 1 条）；
- 限价偏移只定义一次；
- 模拟盘不构造带凭据的适配器，签名请求数为 0；
- 行情全断时，急停平仓和止损同步照常执行；
- 同一组端口契约测试，分别在模拟交易所和假 Binance 上通过；
- 模拟止损在最高价、最低价越过触发价时成交，并写分叉事件。

## 实施更正（2026-11-15，S3c）

- **运行时拆分已实施**，模块是 `trading/application/market_poller.py`、`exit_supervisor.py`、`status.py`。
  - `TargetRuntime` 只负责组装三个部分、保存模拟盘状态、执行每一轮。
  - 入口和测试原来调用的方法（`reconcile`、`poll`、`status`、`refresh_equity`、`risk_actions`）都转给对应的部分。
- **三块互相隔离**，执行模式下生效：
  - 某一块抛出异常，记为 `CYCLE_FAILED`。立即阻断全部入场，之后每次对账都重新阻断，直到这块恢复。这一轮里另外两块照常执行。
  - 某个品种处理 K 线出错，记为 `DECISION_FAILED`：只阻断这个品种，下一根 K 线重新决策。
  - 模拟盘照旧直接抛出，不能悄悄和研究结果分叉。
- **每轮顺序没有改成"退出侧先跑"**（技术负责人 2026-11-15 选择"保持现在的顺序"）。
  - 实测：退出侧先跑时，新仓位成交后要等下一轮（约 5 秒）才挂止损，12 条已有测试因此失败。
  - 现在的顺序是行情 → 退出 → 状态。决策时看到的对账结果最多旧一轮，但入场前的风控闸门本身实时读取急停、回撤和暂停状态。
  - 上文第 4 条"两种模式下，退出侧都在每轮开始时先执行一次"作废。S10 的模拟盘撮合顺序到时另行设计。

## 实施更正（2026-11-15，S10）

技术负责人 2026-11-15 批准 S10（方案 B 正式实施），在 M5 的 4 周计时开始之前合并，分 2–3 个 PR。

**四个决定**（技术负责人 2026-11-15）。第 1 部分只记录，由后面的部分实施：
- **翻向规则只在模拟盘放宽**：同一品种有一笔已确认、还在等撮合的减仓时，模拟盘允许再预留一笔入场（翻向后的新方向）。执行模式保持现在更严的规则。
  - 原因【S9 原型提交 `e61bad0`】：模拟盘的市价减仓要等下一根 K 线开盘才成交，研究却在同一根 K 线上先减仓、再挂新方向的入场。按执行的规则，订单库会拒绝这笔入场，模拟盘就和研究分叉。
- **每秒 1 笔的限额在模拟盘保留**：被推迟的入场在模拟时间 5 秒后重试。
- **模拟盘用代码里的默认风控限额，不读任何账户相关的环境变量**。规格签名检查读取的 `BINANCE_TESTNET` 可以接受：它只会让检查更严。
- **模拟盘状态必须原子保存**：要么只用一个存储，要么从订单库重建模拟交易所的状态。

**第 1 部分**（分支 `s10-move-broker`）：
- 撮合器搬到 `trading/adapters/simulated/bar_broker.py`。上文第 1 条和设计第 4.1 节写的是"原样搬入"，实际有三处不同：
  - `Bar` 改从 `trading.domain.bars` 导入。这是同一个类；`trading` 不得导入 `data`。
  - 为 `mypy trading/` 的严格检查加了类型标注，不改行为。
  - 新增公开方法 `cancel(order_id)`，供模拟交易所撤单。研究不调用它，研究结果不变。
- `SymbolRules`、`SYMBOL_RULES`、`broker_config` 从 `lab/simulate.py` 搬到同一个模块。`lab.simulate` 导入并转出这三个名字。
- 旧路径 `runner/offline/broker.py` 留作转发薄壳，因为撮合器的两个单元测试文件还从这里导入。
- 第 5 条更正：限价偏移原来写了**四**份，不是三份，漏了 `lab/breakout_study.py:73`（已关闭的突破线）。
  - 现在只在 `trading/domain/orders.py` 的 `LIMIT_OFFSET_BPS` 定义一次，四处都引用它，值不变。
  - 由架构测试 `tests/architecture/test_shared_parameters.py` 固定。
- 导入契约"adapters do not import each other"加入 `trading.adapters.simulated`。这是收紧，没有放开任何导入。

**第 2 部分**（分支 `s10-simulated-exchange`）：
- 新增模拟交易所 `trading/adapters/simulated/exchange.py`（`SimulatedExchange`、`StopModel`、`BarStopModel`）和新端口 `BarListener`。
- 新增一组接口契约测试 `tests/contract/test_exchange_ports.py`，在模拟交易所和假 Binance 上都通过；两边合理的差异列在它的文档字符串里。
- 还没有接进运行时，那是第 3 部分。
- 与设计第 4.2 节签名草案的不同：
  - 没有 `clock`，成交时间取 K 线时间；
  - `filters` 的类型是 `SimFilters`，因为适配器之间不得互相导入；
  - taker 费率取各品种撮合器的配置；
  - 拒绝逐仓模式的撮合器。
- 详见执行记录"S10 第 2 部分"。

**第 3a 部分**（分支 `s10-paper-path`）：模拟盘改走执行路径。
- `ModePolicy`：类型在 `trading/application/mode_policy.py`（应用层不得导入组装层），两个实例和 `policy_for` 在 `trading/infrastructure/modes.py`。字段与 S9 原型的报告一致：
  - `catch_up`（`replay_all` / `latest_only`）、`entry_ttl`（`venue` / `lifecycle`）、`sizing_equity`（`per_symbol_venue` / `account_share`）；
  - `limiter_clock`（`bar_time` / `monotonic`）：管入场闸门的全部时钟，包括限额器的日界和每秒窗口、"处理过晚"、被推迟入场的重试；
  - `settle_new_bar_before_decision`、`isolate_failures`（模拟盘出错直接抛出）、`entry_beside_working_reduction`（决定 1）；
  - 另有只用于状态文件的 `name`。两个实例的每个字段都不同，由测试逐项固定。
  - 不在字段里、只在组装时不同的：交易所（模拟交易所）、默认限额（决定 3）、没有整账户检查和告警、下单后不等待。写在 `modes.py` 的说明里。
- 会话只有一条路径（上文第 3 条）：删掉 5 处模拟盘分支和会话里那份 `BROKER_STATE`（第 2 部分的待定事项）。由架构测试 `tests/architecture/test_paper_path.py` 固定：会话里没有读 sleeve、mode 或名字含 paper 的条件；应用层没有任何与 "paper"/"execution" 的比较。
- 行情侧按时间顺序跨品种逐根补放：推进窗口 → 模拟交易所撮合 → `ExitSupervisor.settle()` → 按 K 线收盘时间决策 → 推迟的入场在模拟时间 5 秒后重试（决定 2）。
  - 与设计第 4.2 节草案的不同：`settle()` 不带品种参数，一次对账全部品种。原因：对账结束后统一施加阻断（S3 审计第 2 条），只对一个品种对账做不到这一点。
  - 执行模式的每轮顺序仍是行情 → 退出 → 状态（S3c 的更正），模拟盘同一轮也是这个顺序；撮合和 settle 发生在行情侧里面。
- `paper_divergence` 只由 `PaperDivergenceLog`（`trading/infrastructure/paper.py`）写，三种原因各一处、各写一次（第 2 部分的待定事项）：被拒的入场（`blocked` 且不是"等下一秒重试"）→ `entry_refused`；止损触发或击穿的问题事件 → `stop_triggered` / `stop_breached`；新的暂停（`DivergencePauses`）→ `pause`。模拟交易所本身不写事件。
- 模拟盘不做 ADR-0010/0011 的整账户检查（第 2 部分的待定事项，技术负责人 2026-11-15 决定）。
- **新增一条有意的分叉**（技术负责人 2026-11-16 决定保持现状、记录在案）：执行路径把挂着的入场单当作"未了结订单"，每次对账写 `unresolved_order` 问题并阻断本品种入场（S10 之前执行模式就是这样，`exit_supervisor.alert_wait` 只推迟了告警）。
  - 后果：一张挂单入场跨 K 线挂着时，同品种的新入场（例如翻向后的新方向）被拒，写一条 `paper_divergence`（`entry_refused`）；研究里不会拒。
  - 当前规格是市价入场，不受影响：模拟盘的市价单在下一根 K 线撮合后、决策之前就已了结。
  - 以后改用挂单入场时再处理。证据见执行记录"S10 第 3a 部分"（80 根 K 线的数字）。

**第 3b 部分**（分支 `s10-paper-state`）：模拟盘状态原子保存（决定 4）、入口变薄、模拟盘安全测试、验收补充。
- **原子保存：一个存储，与订单库同一事务**。
  - 模拟盘状态 = 模拟交易所的快照 + 每个品种已决策到的 K 线，存在模拟盘目录 `state.sqlite` 的 `paper_state` 表里，和订单库同一个文件。
  - 订单库的每个事务提交前，只要模拟交易所或游标自上次写入后变过，就在同一个事务里写入当前文档（`PaperOrderJournal`、`SqlitePaperState`）。所以库里任何一个已提交的状态都是运行中某一时刻的完整状态，交易所和订单库不可能来自不同时刻。崩溃时回到最后一次提交：
    - 交易所的变化还没有被任何提交带上，就随崩溃丢掉，等于没发生过。订单库已预留、但下单结果没提交的订单，交易所里也就没有；生命周期在宽限期之后按 `not_found` 了结，和真实交易所相同。
    - 已撮合、还没决策的 K 线，重启后照常决策，不会再撮合一次（`SimulatedExchange.has_matched`）。
  - 每次保存后照旧同步写一份 `paper_state.json`（迁移计划第 10 步：在第 15 步之后的审计确认可以停写之前一直写）。它只写不读：删除或修改它不改变模拟盘状态。
  - 没有选的办法：
    - 每轮结束时保存快照、订单库照旧单独提交：崩溃落在两者之间时会不一致，不满足决定 4。
    - 从订单库重建模拟交易所：订单库里没有资金费记录、撮合器拒绝的订单、挂单的剩余量和年龄等，做不到与研究逐笔一致。
    - 整轮一个事务（所有存储共用一个连接）：也满足决定 4，而且崩溃后能逐笔复现研究；但要改连接工厂和各存储的 `BEGIN IMMEDIATE`，执行模式也会受影响。没有采用。
- **入口变薄**（ADR-0017 推迟到这里的一项）：启动检查、两种模式的组装和主循环搬到 `trading/infrastructure/cli/target.py`。`runner/binance_target.py` 只剩两件 `trading` 不能导入的东西：签名的规格加载器（`infra.model_signing`）和交易成本日志（`monitoring.tca`），以及原有名字的转发。
