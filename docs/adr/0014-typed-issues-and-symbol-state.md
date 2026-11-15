# ADR-0014：类型化的问题与事件，每个品种一个状态机

- 状态：**已批准**（2026-11-15，技术负责人"现在全部批准"）。原为提议稿；下面的内容是批准时的版本，实施中如有变动在本文更正并记入 `execution-log.md`。
- 日期：2026-11-15

## 背景

【代码】

- 问题是字符串列表，交易行为靠子串匹配决定：`runtime.py:142` 用 `any("stop_breached" in i for i in issues)` 决定击穿平仓。还有一处按字符串结尾判断等待时长（`runtime.py:156`，`issue.endswith(UNRESOLVED_ORDERS)`）。
- 事件类型是散落的字符串字面量：`issue`、`alert`、`blocked`、`intent`、`fill`、`warmup`、`paper_*` 等。
- 品种的状态（预热中、可交易、暂停、平仓中、急停）分散在会话和运行时的多个字段里：`entry_ready`、`deferred`、暂停记录、平仓失败计数等。

措辞改动会悄悄改变交易行为，而且没有任何测试能发现。

## 决定

1. **`Issue` 记录**（`trading/application/issues.py`）：
   ```python
   class IssueKind(Enum):
       STOP_BREACHED; STOP_TRIGGERED; STOP_MISSING; STOP_UNCERTAIN
       LEDGER_MISMATCH; FOREIGN_HOLDING; UNRESOLVED_ORDER; UNDELIVERED_FILL
       RECONCILE_FAILED; ACCOUNT_UNREADABLE; MARKET_DATA; MARKET_REFUSED
       FLATTEN_FAILED; PAUSE_NOT_SAVED; KILL_ACTIVE; DRAWDOWN_LATCHED; FOREIGN_FILL
   class Effect(Enum):
       BLOCK_SYMBOL_ENTRIES; BLOCK_ALL_ENTRIES; CANCEL_RESTING_ENTRIES; FLATTEN_SYMBOL; PAUSE_SYMBOL; ALERT_ONLY
   @dataclass(frozen=True)
   class Issue:
       kind: IssueKind; symbol: str | None; detail: str
       @property
       def effects(self) -> frozenset[Effect]: ...   # 由一张固定的表从 kind 推出
   ```
   - "问题 → 后果"是一张表，在领域层定义，由测试逐条固定。
   - `detail` 只用于展示，任何交易判断都不读它。
2. **事件类型**：`EventKind` 枚举；写事件的接口只接受枚举值。
3. **品种状态机**（`trading/application/symbol_state.py`）：
   - 状态：`WARMING`、`READY`、`BLOCKED`、`PAUSED`、`FLATTENING`、`HALTED`（急停或回撤锁定中）。
   - 转移由本轮的问题集合加急停、回撤状态决定。转移表与订单状态机一样，写死并逐条测试。
   - 只有 `READY` 允许入场；任何状态都允许减仓。
4. 与 ADR-0013 的运行时拆分在同一步实施：`ExitSupervisor.run()` 返回 `list[Issue]`。
5. 执行顺序保持现状（审计 L5）：先写暂停（`runtime.py:147`），再在同一轮里撤掉挂着的入场单、执行平仓（`runtime.py:215-217`），避免暂停的品种多等一轮才撤单。
6. `CANCEL_RESTING_ENTRIES` 是独立的后果：急停不平仓时只撤单不平仓，已有测试固定了这一点（`tests/unit/trading/test_exit_robustness.py:127,144`）。

## 状态未知与失败方向

- 无法分类的异常一律映射为能阻断入场的类型：读取类失败为 `ACCOUNT_UNREADABLE`，对账失败为 `RECONCILE_FAILED`。
- 表里没有登记的问题类型，在测试中直接失败；生产中按阻断全部入场处理。

## 备选方案

- **继续用字符串，加测试固定措辞**：成本最低，但每次改措辞都要同步改测试，而且同一个词出现在不同问题里时仍会误判。

## 后果

- 告警按 `kind` 去重，不再依赖文字是否相同。
- 状态文件里会出现每个品种的状态，人更容易看懂当前为什么不交易。

## 验收

- 架构测试：`trading/application` 里不再出现"在问题文字里找子串"的判断（AST 规则，并配一个能被它抓到的反例）。
- 状态转移表：多一条或少一条，测试都会失败。
- 现有运行时测试的断言不变；只能改写字符串断言的写法（例如改为断言问题类型），每处都要记录。
- 变异自检：把任何一个问题类型的后果去掉，都有测试失败。

## 实施更正（2026-11-15，S3a）

S3 分 4 个 PR 合并，每个都能单独验证：
- S3a：类型化问题和后果表；
- S3b：事件类型改为枚举，订单状态变化写事件；
- S3c：运行时拆分，三块互相隔离；
- S3d：品种状态机。

S3a 与上文"决定"的不同之处，以代码为准：

- **`Issue` 是 `str` 的子类，不是冻结数据类。**
  - 文字与原来完全相同，所以事件日志、告警，以及大约 90 处按文字断言问题列表的已有测试都不用改。
  - 种类和品种是它的属性。交易判断只看种类，架构测试 `tests/architecture/test_issue_rules.py` 禁止应用层做任何文字查找。
- **文件位置。** 后果表在领域层 `trading/domain/issues.py`，`Issue` 在 `trading/application/issues.py`。
  - 表里没有登记的种类，生产中按"阻断全部入场"处理；测试要求每个种类都已登记。
- **S3a 的种类。**
  - 新增两种：`VENUE_BLOCKED`（交易所适配器拒绝执行，例如未连接、双向持仓模式）和 `CANCEL_FAILED`（撤掉挂着的入场单失败）。
  - 推迟到产生它们的步骤：`ACCOUNT_UNREADABLE` 和 `FOREIGN_FILL` 在 S5；`KILL_ACTIVE` 和 `DRAWDOWN_LATCHED` 在 S3c、S3d。
  - `FOREIGN_HOLDING` 现在只用于对账发现的"不认识的挂单"，阻断这个品种。S5 的整账户检查会决定它是否改为阻断全部。
- **`PAUSE_SYMBOL` 仍从持久的止损记录写入**，所以重启后不会丢。表决定哪些止损状态会暂停入场：`stops.triggers()` 只返回后果里含暂停的那些。
- **行为变化（更严格）：** 止损同步发现的问题（触发、结果未知、缺失、击穿）现在当轮就阻断这个品种的入场。原来只是间接阻断：靠暂停和"有仓位但没有止损"那条检查。
- **告警按种类去重。** 问题还在、只是文字变了（数量、订单号），不会重新计时；同一轮里同一种问题有多条时，合成一条发出。
- **事件日志的 `issue` 事件多了 `issue_kind` 字段。**

## 实施更正（2026-11-15，S3b）

- **`EventKind`** 放在领域层 `trading/domain/events.py`，取值就是日志里一直用的词。
  - 新增两种事件：`order` 记录订单的每次状态变化，`stop` 记录止损的每次状态变化。这是数据与状态架构讨论里技术负责人的决定。
  - 止损事件是顺带加的，与订单事件同理。
- **写事件的接口只接受 `EventKind`。**
  - JSONL 事件日志收到别的类型会抛 `TypeError`。
  - 架构测试 `tests/architecture/test_event_kind_rules.py` 禁止在生产代码里用字符串当事件种类。
- **状态变化事件由 `RecordedOrderJournal` 写出。** 它包在执行用的订单库外面（`trading/application/execution/recorded_journal.py`），订单库仍然是状态本身。
  - 写事件失败只记日志，不会打断下单路径。
  - 手工工具（`manual`）直接操作订单库，它们的改动还不写事件。

## 实施更正（2026-11-15，S3d）

- **`SymbolState`** 放在领域层 `trading/domain/symbol_state.py`，共 6 个状态。
  - 状态是按固定优先级算出来的分类：预热中 > 平仓中 > 急停或回撤锁定 > 暂停 > 阻断 > 可交易。
  - 允许的转移写成一张表：除了"已经预热完又回到预热中"以外都允许，共 31 种。测试把表逐条固定。
  - 表外的转移按"阻断"处理并告警。对同一个被挡住的目标状态只告警一次。
- **计算位置。** 退出侧每轮在风险动作之前、之后各算一次（`ExitSupervisor.update_states`），所以即将执行的平仓会先显示为"平仓中"。
  - 输入是上一次对账的问题（含整块失败）、急停、回撤和暂停状态。
  - 状态变化写 `symbol_state` 事件，状态文件里也显示每个品种的状态。
- **只有"可交易"能入场。** 这一条放在入场闸门原有检查之后，作为兜底：
  - 原有检查拒绝时，保留它自己的拒绝原因；
  - 原有检查放行但状态不是"可交易"时，拒绝原因为 `state: <状态>`；
  - 这种情况下，限额器已经为这笔入场预留的额度会被计入但不会用掉，方向偏保守。
- **`KILL_ACTIVE`、`DRAWDOWN_LATCHED` 两个问题种类不再需要**，状态机直接读急停和回撤状态，与上文第 3 条"转移由本轮的问题集合加急停、回撤状态决定"一致。

## 实施更正（2026-11-15，S3 独立审计之后）

- **第 4 条没有实施**：`ExitSupervisor.run()` 不返回问题列表，状态和问题都通过属性读取（`last_issues`、`session.state`）。
- **第 6 条的后果"撤掉挂着的入场单"已从枚举里删除。** 它从来没有任何代码读取，放进表里也不会起作用。急停不平仓时只撤单、不平仓，以及暂停时撤单，仍由风险动作根据急停和暂停状态直接执行（由 `test_exit_robustness.py:127,144` 固定）。
- **新增种类 `EVENT_LOG_FAILED`**（阻断全部入场）：事件日志写不进时报出，平仓、止损、对账照常（技术负责人对审计第 3 条的决定）。
- **状态的计算时机**：行情侧每轮开头也算一次，入场闸门看到的是这一轮的状态。整个预热成功之前，所有品种都是"预热中"。权益不明时显示"阻断"。
- **"阻断全部入场"在所有品种对账完之后统一施加**，确保对排在后面的品种也有效。
