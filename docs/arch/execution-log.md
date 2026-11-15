# 迁移执行记录（阶段 4）

- 分支：第 1–6 步在 `arch-migration` 上完成，2026-11-14 经 PR #7（合并提交 `e1bad0a`，CI 8 项全绿）合入 `master`，旧分支已删除。此后每一步用临时分支加 PR，合并后立即删除分支。
- 批准（2026-11-14）：技术负责人回复"全部执行"，批准了当时摘要里列出的两批建议（第一批：ADR-0001–0004 的结构决策、ADR-0008 子决策 1、行数口径；第二批：ADR-0005 子决策 1–7、ADR-0007 子决策 1、3、4），并确认交易账户专用于本运行时、连续执行全部步骤。各 ADR 的状态行逐条写明了批准范围和仍未回答的事项。
- 每一步的"通用验收"是：全量 pytest、`ruff check --select E,W,F .`、`scripts/check_local_safety.py`，第 2 步之后加 `lint-imports`。下面贴的是原始输出的尾部。
- 基线（HEAD `e3cd2d6`）：`2920 passed, 34 skipped, 24 deselected`；ruff 通过；本地安全检查 `Ran 347 tests ... OK (skipped=1)`。

## 阶段 1 补查（只读）

在独立 worktree 里跑了 3 个只读审查，结果并入 `current-state.md`：

- 资金缺陷独立复核：见 `current-state.md` 5.4。12 条里 10 条成立，2 条部分成立，0 条被推翻；新增 X1、X2。

## 第 1 步：特征化测试（退出路径与订单生命周期）

**改了什么**

- 新增 `tests/characterization/`（4 个文件，372 行，全部是测试）：16 个测试。其中 12 个带 `known_defect` 标记，把 R1、R2、R3、R5、R6、R8、R9、R11、R12（两个）、R13、R20 的现状固定下来；另外 4 个固定应当保持的正常行为（kill 平仓在交易所接受时能平掉、回撤闩锁同文件同 scope 重启后仍在、kill 文件语义、带订单号的止损成交能入账）。
- `pytest.ini` 注册 `known_defect` 标记（1 行）。
- 没有改任何生产代码，没有改任何已有测试。

**通用验收（原始输出尾部）**

```
2936 passed, 34 skipped, 24 deselected, 9 warnings, 312 subtests passed in 790.26s (0:13:10)
exit=0
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
```

2936 = 基线 2920 + 新增 16。耗时比基线长是因为同时有 3 个只读审查在跑。

**没改什么 / 遗留**

- 原计划第 1 步里的"状态转移表特征化"挪到第 5 步一起做：它只服务第 5 步，放在一起更容易核对。
- 带 `known_defect` 的测试断言的是缺陷本身。它们只会在第 1b、10、11、12、13 步按已批准的修复逐条改断言，每次改动都在对应步骤里列出。

## 第 1b 步：纸面运行不因行情失败退出；纸面状态落盘（ADR-0007 子决策 3、4，已批准提前）

**改了什么**（`runner/binance_target.py`，+125 / -12 行；新增 `tests/unit/lab/test_target_runtime_resilience.py`，7 个测试）

- 行情请求（K 线、资金费）失败时不再让进程退出：本轮跳过行情步骤，写一行 `issue` 事件，按 5、10、20 … 秒退避（上限 300 秒）重试；同一周期里对账、止损维护、kill / 回撤平仓照常执行。两种模式都生效。只捕获 `BinanceRestError` 和 `OSError`，其他异常仍然退出进程。
- 纸面模式：每个周期处理完 K 线后，把两个品种的模拟账户（持仓、现金流、挂单、成交记录）原子写入 `<state_dir>/paper_state.json`（写临时文件、fsync、rename）。重启预热时恢复它，并把停机期间错过的 K 线按研究撮合逐根补算。
- 保存的状态与当前 spec、品种或撮合配置不一致，或者停机太久、行情接口已经取不到恢复点那根 K 线时，拒绝启动并提示；新增 `--reset-paper-state`，把旧记录改名归档（不删除）后重新开始。
- 两处因补算而必须的因果性修正：
  - 目标计算只使用"结算时间 ≤ 当前 K 线可见时间"的资金费（以前用字典里最新的若干笔）。正常实时运行时两者相同；只在一次追赶多根 K 线时不同。当前候选 spec 的 `funding_filter` 为 False，资金费不进入目标，对它没有影响。
  - 追赶多根 K 线时，资金费请求数量随缺口增大（以前固定 3 笔，缺口超过一天会漏掉结算）。

**新增测试的证据**

- 纸面运行 150 根 K 线后停机 7 根再重启：两个品种的全部成交与研究模拟 `lab.simulate` 逐笔相同、权益误差 < 1e-9；市价入场与挂单 4 小时两种配置都测了。
- 运行中连续 4 根 K 线行情失败：进程不退出，恢复后补处理，最终仍与研究逐笔相同；事件日志里有 4 行 `issue`，失败计数 1→4。
- 执行模式下行情失败时，kill(平仓) 仍然把仓位平掉（独立复核的 X1）。
- 退避期内不发请求；spec 不一致时拒绝启动，`--reset-paper-state` 归档旧文件；恢复点早于可取得的历史时拒绝启动。

**通用验收（原始输出尾部）**

```
2943 passed, 34 skipped, 24 deselected, 9 warnings, 312 subtests passed in 570.16s (0:09:30)
exit=0
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
```

2943 = 2936 + 新增 7。已有测试的断言一行未改；`tests/unit/lab/test_target_runtime.py` 的纸面等于研究测试照常通过。

**没改什么 / 遗留**

- 预热阶段的行情失败仍然让进程退出（启动失败由 systemd 重启）。
- 连续失败没有主动告警，只有事件日志和状态文件里的 `market_failures` 计数。告警出口在第 12 步加。
- 部署影响：主机上第一次用新代码启动时没有 `paper_state.json`，从零开始，与现在相同；之后的重启都会续上。

## 第 2 步：门禁与架构规则骨架

**改了什么**

- 新建空包 `trading/`，四个子包 `domain`、`application`、`adapters`、`infrastructure`，每个只有一行说明。
- `.importlinter`：4 条契约——依赖只能向内；`trading` 不 import 任何遗留顶层包（19 个）；`domain`、`application` 不 import `os`、`time`、`sqlite3`、`urllib`、`http`、`socket`、`subprocess`、`_quant_hotpath`；`domain` 不 import `logging`。"适配器之间互不依赖"一条等第 8 步有了适配器再加。
- 【实际运行结果】契约不是摆设：临时放进一个同时违反 4 条规则的文件，`lint-imports` 输出 `Contracts: 0 kept, 4 broken`，并逐条列出违规的 import；删掉后恢复 `4 kept`。
- `tests/architecture/test_rules.py`：只有 `trading/infrastructure` 能读环境变量；CI 和 `make test` 的 pytest 清单必须包含 `tests/architecture`、`tests/characterization`、`tests/offline`、`tests/contract`、`tests/unit/lab`、`tests/unit/binance`；CI 必须运行 `lint-imports`。
- CI：核心测试步骤加入上面 4 个目录（`tests/offline` 以前只以 unittest 方式跑，`tests/contract` 以前不在 CI）；lint 任务安装 `import-linter==2.16` 并运行；ruff 清单补上 `trading/`、两个新测试目录、`tests/unit/lab/`、`shared/kill_file.py`、`shared/exposure_gate.py`。
- `Makefile`：`CORE_TESTS` 同步；新增 `lint-arch` 目标并纳入 `make test`。`pyproject.toml` 打包清单加入 `trading/`。
- 修正测试夹具 `tests/unit/lab/test_target_runtime.py` 的路由：不带 `symbol` 的止损查询发到持有该止损的品种，账户级持仓与挂单查询合并两个品种。并新增断言：每个品种任何时候最多一张有效止损；kill 平仓后两个品种都没有持仓和残留止损。这是让测试更严，已有断言未改。

**通用验收（原始输出尾部）**

```
2947 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 380.14s (0:06:20)
exit=0
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ lint-imports
Contracts: 4 kept, 0 broken.
```

**遗留**

- import-linter 是开发工具，本机装在独立目录里（`/opt/lint-pkgs`，用包装命令 `lint-imports-isolated` 调用），不影响系统 Python。本地运行：`make test LINT_IMPORTS=lint-imports-isolated`。
- CI 的改动要等推送后才能看到实际运行结果，本地只验证了它们调用的命令。

## 第 3 步：端口定义与契约测试

**改了什么**

- 新增 `trading/application/ports.py`（140 行）：`MarketData`、`ExchangeAccount`、`OrderGateway`、`StopOrderGateway`、`OrderJournal`、`KillSwitch`、`DrawdownGuard`、`OrderLimits`、`ExposureReader`、`EventLog`、`Notifier`，以及 `Clock`、`Sleeper` 两个可调用类型。方法名和签名沿用现有实现，所以现有类不用改就符合端口。`OrderJournal` 暂不含裸 SQL 的 `connect()`，第 6 步收口。
- 新增 `tests/contract/test_ports.py`（16 个测试）：现有的 Binance 适配器、订单日志、kill 文件、回撤闩锁、下单限额器、敞口门、事件日志逐一满足对应端口（方法齐全、端口声明的每个参数名都被接受）；交易所整体不可用时，9 个读方法都抛异常而不是返回空；敞口门在账户读不到时拒绝入场。另有一个测试证明签名检查能发现缺参数。
- `shared/entry_risk.py`、`shared/exposure_gate.py`、`shared/confirmed_execution.py` 的协作者参数从 `Any` 改为端口类型（只在类型检查时导入，运行时行为不变）。旧包引用新包是允许的方向。
- `tests/architecture/test_rules.py` 新增：`trading/application` 里函数参数不得标注为 `Any`。
- `pyproject.toml` 为 `trading` 加了严格的 mypy 配置（未标注函数、裸泛型都报错）；CI lint 任务和 `make lint-arch` 运行 `mypy trading/`。

**通用验收（原始输出尾部）**

```
2965 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 376.30s (0:06:16)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 6 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

2965 = 2947 + 16（端口契约）+ 2（架构规则）。

**遗留**

- `BinanceAdapter.get_balances` 出错时返回空快照，不在任何端口里，暂不处理。
- 测试用的 `FakeBinance` 是 HTTP 层的假交易所，契约测试检查的是"真实适配器 + 假 HTTP"。审查发现的假交易所与真实 Binance 的差异（成交查询忽略 `limit`、撤销已成交订单不报错等）在第 5、6 步补测试时一并处理。

## 第 4 步：风控规则与存储分离

**先补的测试**（`tests/characterization/test_risk_gates.py`，13 个，在改代码之前对旧代码全部通过）：按 `binance_target` 实际的组装方式（就绪门 → 组合门 → kill → 回撤 → 敞口 → 限额器）逐个触发，断言拒绝原因的前缀；减仓在 kill 激活时仍放行；敞口计入未结的入场意图；回撤库损坏时 `latched()` 读作已闩锁；重启后更宽的配置不能放松当前周期；kill 文件的 `active` 不是布尔值或文件损坏时按激活处理。这些对应变异实验里存活的 M12、M14、M15、M16 和入场门组合。

**改了什么**

- `trading/domain/risk/`：`exposure.py`（3 倍常量、配置只能调低的校验、超限判定）、`drawdown.py`（UTC 日 / ISO 周的周期键、高水位与"存储的限额只能收紧"的计算）、`kill.py`（`KillState`）。都是纯计算，没有存储和时钟。
- `trading/application/risk/`：`entry_gate.py`（组合门，`check()` 的函数体与原来逐字相同，已用 diff 核对）、`exposure_gate.py`（敞口门，通过 `ExchangeAccount`、`OrderJournal` 端口取数）。新增 `CompositeRiskGate.for_execution(...)`：kill、回撤、敞口三个组件任何一个缺失就在构造时抛错。原构造函数签名不变。
- `trading/adapters/sqlite/drawdown_store.py`：回撤闩锁的 SQLite 实现，表结构不变，计算调用 domain。时钟改为可注入，默认仍是 `time.time`。
- `shared/entry_risk.py`、`shared/exposure_gate.py`、`shared/drawdown_guard.py` 改为只做重导出的薄壳，所有旧的导入路径照常可用；`shared/kill_file.py` 的 `KillState` 改为从 domain 导入。
- `runner/binance_target.py`、`runner/binance_main.py` 的执行模式改用 `CompositeRiskGate.for_execution`（两处都本来就传齐了组件，行为不变）。
- 端口新增 `OrderRequest`、`EntryReservation`、`WalletCheck`（后两者只给冻结的 Bybit 运行时用）。
- 新增 `tests/unit/trading/test_domain_risk.py`（21 个）：3 倍上限的边界值（恰好 3 倍放行，多 0.01 拒绝）、权益未知时拒绝、配置只能调低、周期键、高水位与限额收紧、损坏的存储值被拒绝、`for_execution` 缺组件抛错。
- 架构测试新增：全仓库 `MAX_TOTAL_LEVERAGE =` 只定义一次，位置是 `trading/domain/risk/exposure.py`。

没有改任何阈值和默认值（日 / 周回撤 5% / 12%、单笔 5,000、每日 50 单、3 倍上限都不变）。`OrderLimiter` 没动。

**通用验收（原始输出尾部）**

```
3000 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 510.63s (0:08:30)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 15 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

3000 = 2965 + 13 + 21 + 1。`tests/offline/test_risk_safety.py`（21 个）、`tests/unit/binance/test_binance_risk.py` 和第 1 步的特征化测试一行未改，全部通过。`scripts/check_local_safety.py` 在只有标准库的环境里运行，`trading` 的这些模块只依赖标准库，所以它仍然通过。

**遗留**

- R6（状态库缺失时闩锁消失）、R12（回撤阈值没有硬上限）照旧，分别在第 13 步和"15 步之后"处理；对应的特征化测试未改。

## 第 5 步：订单状态机显式化

**先补的测试**（`tests/characterization/test_order_facts.py`，13 个，改代码前对旧代码全部通过）：只有"属于本订单、字段齐全有效"的交易所成交才会入账。查单返回的订单身份不符（client id、订单号、品种、方向任一不同）时不入账、订单保持未结；成交记录的 trade_id 为空、价格为 0、数量为 NaN 或 0、手续费为 NaN、方向相反时，同样不入账；别的订单或别的品种的成交被忽略。对应变异实验里存活的 M3、M4。

**改了什么**

- 新增 `trading/domain/orders.py`：`OrderStatus` 枚举（取值就是数据库里现有的字符串，已有数据不用迁移）、未结 / 已结集合、合法转移表 `TRANSITIONS`、`check_transition()`。转移表是按现有代码归纳的，文件头部逐条写明。
- `shared/confirmed_execution.py`：所有改状态的地方（`_set_status` 和新抽出的 `_settle`，后者合并了原来两处重复的结算 SQL）在写入前读出当前状态并对照转移表。**观察模式**：表外的转移在生产里只记 ERROR 日志并记入 `TRANSITION_VIOLATIONS`，照常继续，不影响下单。终态集合改为从 domain 导入。
- `tests/conftest.py` 新增一个对所有测试生效的检查：任何测试只要产生了表外转移就失败。
- `runner/binance_target.py`、`runner/binance_main.py` 的终态字面量集合改为从 domain 推导；已用脚本核对推导结果与原字面量完全相同。
- 新增 `tests/unit/trading/test_domain_orders.py`（116 个）：11×10 个"原状态→新状态"组合逐一判定允许或拒绝；已结状态永远不能再变；未结状态都能结算；未知字符串被拒绝；枚举值等于数据库字符串。
- 架构测试新增：Binance 下单路径的三个文件里不再出现终态集合的字面量。

**证据：检查真的会失败**【实际运行结果】临时从转移表里删掉 `reserved -> unknown` 一条，跑 `tests/unit/binance/test_binance_execution.py`，超时类测试立即失败，报 `order status transitions outside the table: [... 'reserved' -> 'unknown' is not allowed]`；恢复后 8 个全部通过。全量测试在严格检查下全部通过，说明全部现有测试里出现过的转移都在表内。

**通用验收（原始输出尾部）**

```
3130 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 605.21s (0:10:05)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 16 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

3130 = 3000 + 13 + 116 + 1。

**遗留**

- 观察模式何时改为"生产也拒绝表外转移"，留到第 6 步之后的审计根据日志决定。

## 第 6 步：订单库单一所有者

**改了什么**

- 新增 `trading/adapters/sqlite/order_journal.py`（`SqliteOrderJournal`）：接管原 `shared/execution_journal.py` 的全部内容（`orders`、`fills`、`checkpoint` 三个方法逐字相同，`consume` 只有参数类型标注不同，已用 diff 核对），并新增方法承接散落在别处的 SQL：意图预留（原来在 `shared/confirmed_execution.py` 的一个 `BEGIN IMMEDIATE` 事务里，规则和顺序原样搬过来，返回被拒原因）、存订单号、改状态 / 结算、按意图取成交、取单笔成交、写入成交、止损订单收编、止损表的建表与读写。
- `shared/confirmed_execution.py`、`execution/adapters/binance/execution_adapter.py`、`execution/adapters/binance/protective_stops.py` 改为调用这些方法；三个文件里不再有任何 `.execute(`。执行器新增 `track()`，收编止损时不再直接改它的内部字典。
- `shared/execution_journal.py` 改为薄壳。冻结的 Bybit / OKX 适配器、ML 运行时和测试仍可用原来的 `connect()` 直接访问。
- `OrderJournal` 端口补上这些方法；端口契约测试自动覆盖。
- 新增 `tests/contract/test_order_journal_compat.py`：用重构前（`e3cd2d6`）的建表语句造一个带未结订单、未投递成交和有效止损的库，新代码能原样读出、恢复未决订单（已确认数量和已见成交号都对），并且这张未结订单仍然挡住新的入场。
- 架构测试新增：Binance 下单路径上（执行器和 `execution/adapters/binance/`）不得出现 `.execute(`。

表结构不变；新旧代码读写同一个库文件。

**通用验收（原始输出尾部）**

```
3132 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 379.30s (0:06:19)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 17 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

3132 = 3130 + 1（兼容性）+ 1（架构规则）。订单相关的全部已有测试（含子进程强杀的崩溃恢复测试）与第 1、5 步的特征化测试一行未改，全部通过；全量测试在严格的状态转移检查下通过。

## 第 3 步之后的自检审计（独立审计员，审计对象 `e2bd2a4`）与处理

审计员在只读导出的副本上复现、注入缺陷做了验证。结论：没有致命问题；3 倍上限、kill-switch、无前视、订单状态机、纸面等于研究都没有被破坏；没有删除或放宽任何已有断言。发现的问题和处理如下。

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| H1 | 高 | 第 1b 步的批准没有写进设计文档；ADR-0007 要求 SQLite，实现是 JSON；计划第 10、14 步和不变量表已过时 | 每份 ADR 写明实际批准范围和未答事项；ADR-0007 记下 JSON 偏差，第 10 步改存 SQLite 并迁入；计划加入第 1b 步，修订第 10、14、15 步、遗留清单和不变量表；`CLAUDE.md` 两行过时描述已改 |
| H2 | 高 | 第 4、5 步在审计结束前就提交了，审计没起到放行作用 | 属实。我当时让审计与后续步骤并行。计划 0.5 节加了约定：从第 6 步之后的审计起，审计处理完之前不开始下一步代码改动。第 4–6 步由下一次审计覆盖 |
| M1 | 中 | 执行模式下行情恢复后的补算会按几小时前的 K 线发减仓单 | 已修：执行模式只交易新鲜的 K 线（收盘后 10 分钟内），错过的 K 线只用来补齐窗口，与重启时的预热一致；纸面仍然逐根补算以等于研究。新增测试，注入原行为后测试变红 |
| M2 | 中 | 被拒的 kill 平仓在整个行情故障期间都不会重试；日志说"X1 已解决"说过头了 | 更正：第 1b 步只保证了"交易所接受时，行情故障期间仍能平仓"。平仓被拒后的重试属于 R1，第 11 步的验收加入这个场景 |
| M3 | 中 | 第 1b 步测试缺口：重启时有挂单、非行情异常仍退出、资金费截断、资金费请求数随缺口放大 | 已补 6 个测试（见下）；其中两个用注入原缺陷的方式确认会变红 |
| M4 | 中 | 架构规则可绕过：不经 import 的 I/O、动态导入、别名读环境变量、`Optional[Any]` 等写法、`--ignore` 绕过门禁清单、`echo` 替代 `lint-imports` | 已加严：`.importlinter` 增加禁止项；AST 规则识别上述全部写法；门禁清单按命令的独立参数核对并拒绝 `--ignore`；CI 必须有一行恰好是 `run: lint-imports` 和 `run: mypy trading/`。把审计员给出的绕过写法放进 `trading/` 试跑：`lint-imports` 报 `2 kept, 2 broken`，架构测试 2 个失败；删掉后恢复 |
| M5 | 中 | 纸面存档直接绑定 `BarBroker` 的内部属性 | 新增测试：`BarBroker` 实例的属性集合必须等于"已保存 + 由配置重建 + 单独处理"三组之和，新增或改名属性会让测试失败 |
| L1 | 低 | 崩溃在"已写事件"和"已保存状态"之间时，重启补算会重复写纸面事件 | 记录：事件里都带 K 线时间，核对前向纸面记录时按 (品种, K 线时间) 去重 |
| L2 | 低 | 第一次在 `trading` 里 import numpy 时 mypy 会因 3.11 设定报错 | 第 7 步处理 |
| L3 | 低 | 端口清单与计划不完全一致；契约测试的异常类型过宽；CI ruff 清单缺 `tests/contract/` | 异常类型收窄为 `BinanceRestError`；CI 补上 `tests/contract/`、`tests/unit/trading/`。端口差异：`SpecSource`、`StatusSink`、`PaperStore`、`LimitStore` 在用到它们的第 10、13 步再加；`DrawdownGuard`、`OrderLimits`、`ExposureReader` 是第 4 步需要而新增的 |
| L4 | 低 | 日志个别表述过度 | 更正：第 2 步的"每个品种任何时候最多一张有效止损"实际只在一个时点断言；第 1b 步的"挂单 4 小时配置"并没有覆盖"重启时正有挂单"，现已补测试 |
| L5 | 低 | 存档目录没有 fsync；归档文件名同一秒会冲突；`--reset-paper-state` 若留在 systemd 里每次重启都会重置；拒绝启动时没有告警 | 前两项已修；第三项写入运行手册约束（不得写进 unit 文件）；告警在第 12、14 步 |
| L6 | 低 | 预热补算没有"等待资金费发布"的窗口，只影响 `funding_filter=True` | 记录，当前候选 spec 不受影响 |

### 第 1b 步审计后补丁（代码）

- `runner/binance_target.py`：执行模式补算只交易新鲜 K 线；归档文件名精确到毫秒且不覆盖已有文件；保存后对目录 fsync。
- `tests/unit/lab/test_target_runtime_resilience.py` 新增 6 个测试：重启时正有挂单、停机 40 小时加行情故障 40 小时（跨越 5 次资金费结算）、执行模式补算不交易过期 K 线、非行情异常仍然退出进程、目标只看到已结算的资金费、存档覆盖全部可变属性。
- 【实际运行结果】把"只交易新鲜 K 线"改回原行为，`test_execution_catch_up_after_an_outage_does_not_trade_stale_bars` 失败；把资金费请求数改回固定 3 笔，`test_paper_survives_a_downtime_and_an_outage_longer_than_a_day` 失败。
- `tests/contract/test_ports.py`：交易所不可用时要求抛 `BinanceRestError`，不再接受任意异常。
- `.importlinter`、`tests/architecture/test_rules.py`、`.github/workflows/ci.yml`：见上表 M4、L3。

**通用验收（原始输出尾部）**

```
3141 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 419.89s (0:06:59)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 17 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

## 第 6 步之后的自检审计（独立审计员，审计对象 `0347153`）与处理

结论：没有致命问题。第 4、6 步逐行核对与基线等价（入场门 `check()` 逐字相同；预留事务的规则顺序、事务边界、`created_at` 取值时点都一致；结算与改状态的 SQL 条件一致）；第 5 步的观察模式在生产中不改变控制流；没有放宽任何已有断言。审计期间第 4–6 步已经通过 PR #7 合入 `master`，这次审计没能在合并前放行，这一点如实记录。

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| H1 | 高 | `0347153` 上 `tests/unit/trading` 不在 CI 和 `make test` 里 | 合并前的 `5877ece` 已补上；`master` 合并后的 CI（`e1bad0a`）全绿 |
| H2 | 高 | 第 6 步搬走的几处 SQL 细节（结算条件、"已结不能再改"的 `settled=0` 条件、收编行与止损行的初始状态、有效止损的状态集合、`created_at` 时点、结算后移出 pending）注入后测试都发现不了 | 第 9 步之前直接对 `SqliteOrderJournal` 补测试 |
| M1 | 中 | 转移表 30 种转移里只有 12 种有测试证据；`test_domain_orders` 的 110 个用例由表本身推出，不固定表的内容 | 第 9 步之前补齐证据或收窄表；目标文档 O2 改为"部分" |
| M2 | 中 | "根据生产日志决定何时改为拒绝"的条件不会满足 | 写进 ADR-0004 实施记录 |
| M3 | 中 | 严格检查看不到子进程、unittest 方式的运行、模块级 fixture 和插入语句 | 记录；第 1 条的表述"任何测试只要产生表外转移就失败"更正为"在 pytest 进程内的函数级测试中" |
| M4 | 中 | 规则仍有绕过写法：只引用不调用 `datetime.now`、`sys.modules`、`threading`、带类型标注的 `MAX_TOTAL_LEVERAGE`、`typing.Any` 别名、`logging.FileHandler`、`getattr(os, "environ")`、`executemany`/`executescript`、门禁里的 `-k`/`-m`/`continue-on-error`、单引号字面量；`sqlite3` 规则没实现 | 全部补上。把这些写法放进 `trading/` 试跑：`lint-imports` 报 `2 kept, 2 broken`，架构测试 4 个失败；删掉后恢复 |
| M5 | 中 | "单一写入者"说过头：ML 运行时与冻结适配器仍经 `connect()` 写 | 代码注释、ADR-0004、目标文档 O3 如实改写 |
| M6 | 中 | CI 覆盖率门禁还统计已成薄壳的 `shared.*`，搬走的逻辑不在统计内 | 改为统计 `trading.application.risk`、`trading.adapters.sqlite`、`trading.domain.risk`、`trading.domain.orders`；本地实跑 `Total coverage: 80.73%`（门槛 75%） |
| M7 | 中 | 上次修补让执行模式在行情恢复后连最新一根 K 线的减仓也不下，超出 ADR-0007 的批准文字 | 技术负责人 2026-11-14 选择"只用最新一根决策，允许减仓"。已改：较早错过的 K 线只补窗口，最新一根照常决策，减仓立即下，开仓仍被"处理过晚"挡住。测试改为断言"只有最新一根 K 线产生意图、只发减仓单"；把行为改回"每根都交易"后该测试失败。已写入 ADR-0007 |
| M8 | 中 | ADR-0004 的实现与方案 B 文字不同，没有记录 | 写入 ADR-0004 实施记录 |
| L1 | 低 | 入场门的特征化测试用手工构造而非 `for_execution`，检查顺序与 `reduce_only` 的严格判断没被测 | 第 10 步（入场门迁入运行时组装）时补 |
| L2 | 低 | 薄壳丢了 `ABSENT`、私有 `_periods` 和类名 | 记录；仓库内无使用者 |
| L3 | 低 | 执行器从适配器导入拒绝原因常量，第 9 步会违反分层 | 第 9 步时移到 domain |
| L4 | 低 | `update_status` 的 SELECT 与 UPDATE 不在同一写事务 | 记录；只影响观察记录 |
| L5 | 低 | 计划、目标文档、执行记录、ADR-0007 有过时或不一致的表述 | 已修正 |

### 本次修补的改动

- `runner/binance_target.py`：行情恢复后只用最新一根 K 线决策（M7）。
- `tests/unit/lab/test_target_runtime_resilience.py`：上一条对应的测试按批准的新规则改断言。**这是本项目第一次修改已有测试的断言**，原因是技术负责人批准的行为变更；原断言"恢复周期内没有任何下单"改为"只有最新一根 K 线产生意图，且只发减仓单"。
- `.importlinter`、`tests/architecture/test_rules.py`：M4。
- `.github/workflows/ci.yml`：M6。
- 文档：L5、M5、M7、M8。

**通用验收（原始输出尾部）**

```
3142 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 418.18s (0:06:58)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 17 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

## 第 7 步：领域层——策略、再平衡、持仓、时间语义

**先补的测试**（`tests/unit/trading/test_strategy_causality.py`，8 个）：替换审计指出的弱因果测试。3 套 spec（只做多趋势；资金费、波动突变、4h 确认三层全开；scaled 信号加两层）× 2 个随机种子，每个种子在一天的 24 个小时相位上各取一个切点；把切点之后的 K 线换成随机生成的另一段行情、切点之后的资金费换成随机值，切点及之前的目标和就绪标志必须**逐位**不变；只用切点及之前的数据单独计算，结果也必须逐位相同。另有功效检查（改动在日线边界收盘的那根 K 线，目标确实会变，说明测试不是空转）和变异自检（把聚合 K 线的可见时间提前 1 小时，测试必须报出违规）。在改代码之前就对旧代码全部通过。

**改了什么**

- `trading/domain/market.py`：`frame_from_bars`、`aggregate`、`funding_series` 和 `HOUR`/`DAY`，从 `lab/data.py` 搬来。`frame_from_bars` 的参数从具体的 `Bar` 类改为只描述字段的 `BarLike` 协议，这样 domain 不依赖遗留的 `data` 包。
- `trading/domain/strategy/overlay.py`：`lab/strategy.py` 整体搬来；与原文件的差异只有导入路径和 3 处类型标注（已用 diff 核对）。
- `trading/domain/rebalance.py`、`trading/domain/position.py`：逐字搬来。
- `trading/domain/orders.py` 新增纯 Python 的 `OrderIntent`，第 9 步用它替换依赖 Rust 事件类型的订单对象。
- `lab/strategy.py`、`lab/rebalance.py`、`shared/strategy_position.py` 改为重导出薄壳；`lab/data.py` 只保留读文件的 `load_symbol`，其余从 domain 导入。
- 4 个研究 study（`lab/study.py`、`overlay_study.py`、`pair_study.py`、`breakout_study.py`）的"代码是否有未提交改动"检查加上 `trading/`。否则以后在 `trading/domain` 里改了策略却不提交，研究结果的代码指纹不会记成"有改动"。
- `pyproject.toml`：mypy 对 numpy、pandas 不做分析（它们的类型存根需要比 3.11 更新的语法，审计 L2）。

**通用验收（原始输出尾部）**

```
3150 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 430.78s (0:07:10)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 22 source files
$ lint-imports
Contracts: 4 kept, 0 broken.
```

3150 = 3142 + 8。`tests/unit/lab/`（含纸面等于研究、截尾窗口等于全历史）与原来的因果测试都一行未改，全部通过。`core-no-io` 契约从这一步起第一次约束真实代码（策略函数用 numpy/pandas，不读文件、不取时钟），仍然 4 条全部保持。

## 第 8 步：Binance 适配器迁入 `trading/adapters`

**改了什么**

- `trading/adapters/binance/`：`rest.py`、`urls.py`、`adapter.py` 从 `execution/adapters/binance/` 搬来，与原文件只差导入路径和类型标注（逐文件 diff 核对；为通过严格 mypy，补了泛型参数、一个 `NoReturn` 和三处把 `json.loads` 的结果先赋给带类型的变量（原写两处，第 9 步后审计 F8 更正），运行行为不变）。`config.py` 搬来时去掉了读环境变量的 `from_env`。
- 一个适配器类同时实现行情、账户、下单、止损四个端口，不拆类。
- REST 限流从 Rust 的 `RustRateLimitPolicy` 换成纯 Python 实现 `trading/adapters/binance/rate_limit.py`（容量、补充速度、端点权重、算术完全照搬，时钟可注入）。先写对拍测试：5 个随机种子各 3,000 次请求（含随机的权重同步），Python 与 Rust 每一次的放行 / 拒绝结果都相同。
- 交易所返回的数据对象（`CanonicalFill`、`VenuePosition`、`CanonicalBalance`、`BalanceSnapshot`）原样搬到 `trading/domain/venue.py`；`execution/models/` 里对应的模块改为重导出，所以旧代码里的 `isinstance` 判断看到的仍是同一个类。
- 读 API 密钥的逻辑搬到 `trading/infrastructure/settings.py`（`trading` 里唯一读环境变量的地方）。旧路径 `execution.adapters.binance.config.BinanceConfig` 保留 `from_env`，内部调用新函数，运行时和脚本照常可用。
- `execution/adapters/binance/` 下的 `rest.py`、`urls.py`、`adapter.py`、`config.py` 改为兼容薄壳。`ws_kline_client.py`（死代码）和旧的 Rust 限流包装没有搬。
- `.importlinter` 新增第 5 条契约：`trading.adapters.binance` 与 `trading.adapters.sqlite` 互不依赖。
- 架构测试新增：在一个干净的子进程里 import `trading.adapters.binance.adapter` 和 `trading.infrastructure.settings`，`sys.modules` 里没有任何遗留顶层包、`tests` 或 Rust 扩展。

**已有测试的改动（不是断言）**：`tests/unit/test_coverage_wave3.py` 里 15 处 `patch("execution.adapters.binance.rest.urlopen")` 改为 `patch("trading.adapters.binance.rest.urlopen")`。代码搬家后，打补丁的位置必须跟着搬，否则补丁打在薄壳上、不影响真正发请求的模块。断言一行未改。`tests/contract/test_ports.py` 的导入改为新路径，断言未改。

**通用验收（原始输出尾部）**

```
3158 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 432.75s (0:07:12)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 30 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

3158 = 3150 + 7（限流）+ 1（隔离）。`tests/unit/binance/`、`tests/unit/execution/test_binance_urls.py`、第 1 步的特征化测试一行未改，全部通过。

**遗留**：自己写的两个限流测试第一次跑失败，原因是我对浮点补充量（`0.1 秒 × 10` 略小于 1）和"首次请求才开始计时"的预期写错了；以 Rust 对拍为准修正了测试里的期望值，实现没改。

## 第 9 步准备：补齐第 6 步后审计要求的证据

**改了什么**

- `tests/contract/test_order_journal_sql.py`（6 个）：直接对 `SqliteOrderJournal` 固定审计指出的 SQL 细节——已结算的行不会被改状态；结算返回原状态，重复结算不报原状态；预留只在成功时读一次时钟，被拒绝时不读；收编的交易所订单是"未结、减仓、市价、submitted"；新止损行初始是 `unknown`，"有效止损"包括 `canceling`；订单结算后从执行器的待决集合里移除。
- 【实际运行结果】对审计列出的 7 个存活缺陷逐个注入：6 个现在会被测出（H、I、J、K、F、T）。G（重复结算的 UPDATE 加 `AND settled=0`）仍然测不出来：单进程下结算只对仍在待决集合里的订单发生，两种写法行为相同，所以不固定它。
- `tests/characterization/test_order_transitions.py`（12 个）：转移表按字面量固定；9 个场景通过真实的执行器产生转移，每个场景断言恰好产生哪些转移（包括审计点名的 `rejected_unconfirmed -> unknown`、`unknown -> rejected_unconfirmed`、重启后的 `reserved -> not_found`、`reserved -> filled`）；表里其余 16 种允许的转移各自写明保留的理由。表多出或少掉任何一条转移，这个测试都会失败（已试：往表里加一条 `submitted -> reserved`，测试失败）。
- 预留被拒的原因常量从 SQLite 适配器移到 `trading/domain/orders.py`（审计 L3），执行器改从 domain 导入。第 9 步把执行器迁入 `trading/application` 时不会违反分层。

**通用验收（原始输出尾部）**

```
3176 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 441.21s (0:07:21)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 30 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

3176 = 3158 + 6 + 12。

**遗留**：审计 M2（观察模式没有告警、何时改为拒绝）等第 12 步有告警出口后再处理。

## 第 9a 步：下单生命周期迁入 `trading/application/execution/lifecycle.py`

第 9 步拆成两个 PR：9a 只搬执行器（`ConfirmedExecutor`），9b 再搬 Binance 执行适配器里的减仓定量、对账和止损管理。

**前置测试（计划 0.6 第 9 步一行，先在旧代码上跑通）**

- `tests/unit/trading/test_domain_position.py`（14 个）：持仓记账的空头开仓与加仓、部分平仓、平到零、多翻空、空翻多、手续费按币种分开记（含负手续费）、8 种无效成交逐条被拒且不改状态、存储格式往返。
- 【实际运行结果】在旧代码上 14 个全过；对 `trading/domain/position.py` 逐个注入 6 个缺陷（翻转后不重置开仓价、平到零不清开仓价、空头盈亏符号写反、按整笔数量记已实现盈亏、手续费币种合并、不检查手续费是否有限），每个都至少有一个测试失败。生产代码已还原。
- 计划里第 9 步的其他前置测试（减仓不超过自有仓位、对账不一致或有外部挂单时阻断入场、`TargetFillConsumer` 拒绝与落库不符的事件、止损结果不明与撤销中状态）对应的代码在 9b 才搬，在 9b 开始时先补。

**改了什么**

- `trading/application/execution/lifecycle.py`：执行器原样搬来，类名改为 `OrderLifecycle`。与原文件相比只有 29 行不同：
  - 日志（`OrderJournal` 端口）、时钟（`Clock`）、下单后的等待（`Sleeper`）、锁由构造参数传入，模块本身不再 import `time`、`threading`、`pathlib`、`shared`。
  - 参数校验抽成静态方法 `check_settings`，校验内容和报错文字不变。
  - `_state` 的参数类型从 `Any` 改为 `object`（应用层参数不得是 `Any` 的架构规则）；成交载荷加了类型标注，限价单分支加了一行 `assert price is not None`（该分支里价格前面一定已赋值），都是给严格 mypy 看的，运行行为不变。
- `trading/application/ports.py`：新增 `Lock` 类型（可重入的上下文管理器，由组装根提供）。
- `shared/confirmed_execution.py` 改为兼容薄壳：`ConfirmedExecutor(adapter, journal_path, scope, ...)` 的签名不变，先校验参数（与搬家前一样，参数不合法时不会建出日志文件，已实测），再构造 SQLite 日志、`time.time`、`time.sleep`（每次调用时再查找，与原来一样，所以 `patch("shared.confirmed_execution.time.sleep")` 仍然有效）和 `threading.RLock()`。`.path`、`._connect()`（OKX 用）、`PendingOrder`、`TRANSITION_VIOLATIONS` 照旧可用。Binance、Bybit、OKX 三个执行适配器一行未改。
- 架构规则扩大范围：禁止字面量状态集合的规则加上 `lifecycle.py`；"币安下单路径上不得出现 SQL"的规则加上整个 `trading/application`。`core-no-io` 契约原本就覆盖 `trading.application`，计划里"对 `lifecycle` 生效"的要求由它自动满足（不得 import `time`、`sqlite3`、`threading`）。
- CI 覆盖率门禁加上 `trading.application.execution`，否则搬走的代码不再被计量。【实际运行结果】本地 82.14%（门槛 75%），其中 `lifecycle.py` 82%。

**新增测试（计划第 9 步验收要求的假时钟测试）**

`tests/unit/trading/test_order_lifecycle.py`（9 个）直接构造 `OrderLifecycle`，注入记录型的时钟、等待函数和锁，不再真的等待：

- 市价单的调用顺序恰好是"加锁 → 下单 → 等待 3 秒 → 查订单 → 查成交 → 解锁"，预留时间来自注入的时钟。
- 等待时间为 0 时不调用等待函数。
- 挂单的有效期用注入的时钟计算：差 0.1 秒不撤，到点才撤，撤后再查一次，结算为 `canceled`。
- 6 种不合法参数被拒。
- 【实际运行结果】对 `lifecycle.py` 注入 5 个缺陷（不等待、等待挪到下单前、轮询时忽略有效期、有效期比较差一、轮询不加锁），每个都至少有一个测试失败。

**已有测试的改动（不是断言）**

- `tests/conftest.py` 的自动夹具改为从 `trading.application.execution.lifecycle` 读 `TRANSITION_VIOLATIONS`（与薄壳里的是同一个列表对象）。
- `tests/characterization/test_order_transitions.py` 打补丁的位置从 `shared.confirmed_execution.check_transition` 改为 `trading.application.execution.lifecycle.check_transition`。代码搬家后，补丁必须打在真正调用它的模块上。薄壳故意不再导出 `check_transition`，以后谁再往旧位置打补丁会直接报错，而不是悄悄失效。
- 两处断言一行未改。

**行为上唯一可见的差别**：执行器的日志记录器名字从 `shared.confirmed_execution` 变成 `trading.application.execution.lifecycle`。【实际运行结果】仓库里没有任何配置或测试按这个名字过滤日志（`grep -rn confirmed_execution` 只在 CI 文件列表里出现）。

**行数**：手写约 293 行（`lifecycle.py` 与原文件的差异 29、薄壳 43、新测试 205、其余 16）；纯移动约 280 行。

**通用验收（原始输出尾部）**

```
3199 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 441.15s (0:07:21)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 32 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

3199 = 3176 + 14 + 9。第 1 步的特征化测试、第 5、6 步的订单测试、`tests/unit/binance/` 全部不改断言通过。

## 第 9b 步：止损管理器迁入 `trading/application/execution/stops.py`，并补齐第 9 步其余前置测试

第 9 步原计划的其余内容再拆两份：9c 把 Binance 执行适配器里的减仓定量和对账搬进应用层；9d 处理 TCA 日志改走 `EventLog` 端口、删除读 `RISK_SCOPE` / `ORDER_JOURNAL_PATH` 环境变量的代码、订单事件改用 `OrderIntent`。拆分原因：本步的前置测试已经约 250 行，再加代码会超过每个改动 400 行的上限。

**前置测试（先在旧代码上跑通）**

`tests/characterization/test_execution_boundaries.py`（23 个），覆盖计划 0.6 第 9 步一行剩下的四项：

- 减仓数量：交易所持仓大于本策略确认的仓位时，只平自己那部分（数量 0 的全平和数量过大的减仓两种写法）；交易所持仓更小时，按交易所持仓封顶；本策略没有确认仓位、方向相反、"减仓"实际会加仓、交易所没有持仓、同一品种有两个持仓，这几种情况都不发单，并标记 `blocked`。
- 对账：账本与交易所持仓不一致、有不是本系统下的挂单、对账本身失败、有结果不明的订单、有未交付的成交，每一种都阻断入场（不只是检查 `_ready` 标志，而是确认之后的入场单确实没发出去）；不一致消除后，下一次对账通过，入场恢复。
- 成交消费者：事件的订单号、品种、方向、数量、价格任何一项与落库的交易所事实不符都拒绝，持仓和检查点不变；伪造的成交号被拒；同一成交只记一次。
- 止损：下单回应丢失但交易所已接受时，下一轮查到并标为 `active`，不重复下单；根本没到交易所时，90 秒查询宽限内不重下，到点才把旧记录标为 `inactive` 并以下一个尝试号重下；撤销被确认但实际没生效时，旧止损保持 `canceling`，每一轮都再撤一次，不会被当成当前止损，撤销生效后标为 `inactive`。
- 假交易所会把只减仓单的数量就地截断，所以测试里另记一份"实际发出的参数"，否则"不按交易所持仓封顶"的缺陷看不出来。

【实际运行结果】在旧代码上 23 个全过；对将要搬动的代码逐个注入 20 个缺陷（减仓的三道上限、方向与唯一持仓检查、对账五类不一致、对账结果不阻断入场、入场不看就绪状态、消费者三项比对、止损的查询宽限及其差一、查到后不标 `active`、`canceling` 被当成当前止损、失效记录不标 `inactive`），20 个全部被测出。生产代码已还原。

**改了什么**

- `trading/application/execution/stops.py`：止损管理器原样搬来，与原文件只差 44 行（其中 20 行是两个新的端口类型）：
  - 时钟改为必须注入，模块不再 import `time`。
  - 交易所参数的类型从 `Any` 改为 `StopVenue`（止损端口加订单查询端口），执行器参数改为 `StopOwner`（日志、作用域、属主、收编交易所订单这四样）。现有的 `BinanceExecutionAdapter` 不用改就符合。
  - 私有方法 `_adopt` 改为直接接收触发后产生的订单号，而不是整个查询结果（应用层参数不得是 `Any`）；传的值和以前用到的那一个字段相同。
- `execution/adapters/binance/protective_stops.py` 改为兼容薄壳：子类只补上 `clock=time.time` 的默认值，并重导出 `ACTIVE`、`TRIGGERED`、`MIN_STOP_PCT`、`MAX_STOP_PCT`。`runner/binance_target.py`、`runner/binance_main.py`、测试和探针都不用改。
- 架构规则不用改：`core-no-io` 契约、"应用层参数不得是 `Any`"、"应用层不得出现 SQL" 都已经覆盖 `trading/application` 下的新文件；覆盖率门禁里的 `trading.application.execution` 也自动包含了它。

**已有测试的改动**：没有。

【实际运行结果】搬家后对 `trading/application/execution/stops.py` 重新注入止损相关的 5 个缺陷，5 个全部被测出；两个止损探针（`docs/arch/evidence/probe_stop_*.py`）输出与搬家前相同。
- CI 覆盖率门禁本地 82.37%（门槛 75%），其中 `stops.py` 90%。

**遗留**：`stops.py` 里仍有 Binance 自己的词汇——条件单状态 `NEW` / `TRIGGERING` / `TRIGGERED` / `FINISHED` 和"会立即触发"的错误码 `-2021`。按目标架构，这些应该由 Binance 适配器翻译成领域里的止损状态，应用层不该认识它们。本步只做搬移，不改这部分；我计划把翻译放到第 12 步（止损失败处置）一起做，因为那一步本来就要改止损状态的处理；翻译本身不改行为。

**行数**：手写约 330 行（新测试 281、`stops.py` 与原文件的差异 44 行里新增约 26、薄壳 22）；纯移动约 113 行。

**通用验收（原始输出尾部）**

```
3222 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 462.71s (0:07:42)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 33 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

3222 = 3199 + 23。第 1 步的特征化测试（含止损与成交相关的已知缺陷测试）、`tests/unit/binance/`、`tests/unit/lab/test_target_runtime.py` 全部不改断言通过。

## 第 9c 步：减仓定量与账户对账迁入 `trading/application`

**前置测试**：第 9b 步已补（`tests/characterization/test_execution_boundaries.py`），本步不新增。

**改了什么**

- `trading/application/execution/reduction.py`：`size_reduction(account, journal, *, owner, symbol, side, qty)`，从 `BinanceExecutionAdapter._send_order` 原样搬出：
  - 要求交易所上该品种恰好一个非零持仓；
  - 数量 0 表示全平，方向按交易所持仓定；
  - 方向必须是减仓方向；
  - 数量不超过交易所持仓，也不超过本策略确认的仓位；
  - 返回数量、方向和据以定量的检查点版本。
  
  唯一的写法差别：原来 `checkpoint` 为空时靠"自有仓位为 0"间接报错，现在在同一个条件里直接判断 `not checkpoint`，报错文字和时机相同。这样写是为了让严格 mypy 知道后面取 `revision` 时检查点一定存在。
- `trading/application/reconcile.py`：`reconcile_account(account, journal, *, owner, symbol)`，对账的五类检查原样搬出：
  - 多个持仓；
  - 交易所数量与账本不符；
  - 有结果不明的订单；
  - 有非本系统挂单；
  - 有未交付成交。
  
  对账出异常时记为"对账失败"。原来经执行器的 `unresolved()` 读结果不明的订单，现在直接读 `journal.orders(unresolved=True)`；前者内部就是这一句。
- `execution/adapters/binance/execution_adapter.py` 里这两段换成调用上面两个函数，删 40 行、加 6 行。以下几处留在原处，行为不变：
  - 先撤掉残留入场单；
  - 读适配器自身的 `execution_blockers()`，并放在问题列表最前面，与原来顺序相同；
  - 设置 `_ready`；
  - 写错误日志。
- 两个新模块只依赖 `ExchangeAccount`、`OrderJournal` 端口。导入契约、严格 mypy、"应用层参数不得是 `Any`"规则都已覆盖它们。

**已有测试的改动**：没有。

【实际运行结果】对两个新模块重新注入 15 个缺陷，15 个全部被 `test_execution_boundaries.py` 测出：
- 第 9b 步针对减仓与对账的 12 个；
- 新加 3 个：不看属主、不记录检查点版本、对账失败不记问题。

其中"不记录检查点版本"会让减仓单在日志预留时被判为基于过期检查点而拒绝，所以减仓单根本发不出去，测试因此失败。

**行数**：手写约 15 行（适配器里的 6 行调用、`reduction.py` 的数据类和函数签名、两个模块的说明）；纯移动约 70 行。

**通用验收（原始输出尾部）**

```
3222 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 447.74s (0:07:27)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 35 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

测试数量与第 9b 步相同（3222），没有新增或修改测试。

## 第 9d 步：执行决策迁入 `trading/application/execution/service.py`，适配器只做事件格式转换

这是第 9 步的最后一份。做完之后，Binance 执行适配器里不再有任何下单决策。

**改了什么**

- `trading/application/execution/service.py`：新的 `ExecutionService`，接收 `OrderIntent`（第 7 步的领域类型，不依赖 Rust）。原来 `BinanceExecutionAdapter` 里的决策原样搬来：
  - 重复的意图不再下单；
  - 入场需要对账干净并且熔断器放行，否则标记 `blocked`；
  - 挂单入场价 = 参考价 ×（1 ∓ 偏移）；
  - 减仓先撤残留入场单，再按自有仓位定量，并把检查点版本写进随订单保存的记录；
  - 任何失败都阻断入场、计入熔断器、不重试；
  - 收编交易所订单（触发的止损）、读成交事实、通知状态、轮询、对账。
- 时钟、单调时钟、锁、熔断器、成交成本日志（`EventLog` 端口）都由构造参数注入。应用层不再 import `monitoring`，也不读环境变量。
- `send()` 返回"是否已进入下单生命周期"和成交事实。随订单保存的记录由调用方提供的函数生成，只有通过全部检查的意图才会调用它，与原来生成记录的位置相同。
- `execution/adapters/binance/execution_adapter.py` 改为只做格式转换：把旧的订单事件转成 `OrderIntent` 和带事件头的记录，把成交事实转成 `FillEvent`。
  - 构造函数签名、`_ready`、`_status_callback`、`journal`、`scope` 和各个方法都保留，运行时、`ProtectiveStopManager` 和测试一行未改。
  - 成交成本日志通过一个小的端口实现写入同一个 `tca_binance.jsonl`。
- **去掉读环境变量（计划第 9 步已批准的改动）**：适配器不再从 `RISK_SCOPE`、`ORDER_JOURNAL_PATH` 取默认值，缺少作用域或日志路径时直接报错。`runner/binance_target.py` 和 `runner/binance_main.py` 本来就显式传入这两个值，不受影响。`runner/builders/alpha_builder.py` 改为自己读环境变量后显式传入，行为不变。
- 新增架构规则：`shared/confirmed_execution.py` 和 `execution/adapters/binance/` 下的模块不得读环境变量。

**没有做的（如实说明）**

- 运行时（`runner/binance_target.py`）仍然构造旧的 `TargetOrderEvent`，由适配器转成 `OrderIntent`。运行时本身改用 `OrderIntent` 要等第 10 步把会话迁入应用层时一起做，否则要同时改入场门和会话。
- 适配器里保留了一份与 `ExecutionService` 重复的参数校验，用来保持原来的报错文字和"参数不合法时不建日志文件"。第 15 步删薄壳时一起删。

**证据**

- 新增 `tests/unit/trading/test_execution_service.py`（21 个）：直接用 `OrderIntent` 驱动 `ExecutionService`，协作者全部注入，不经过旧事件类型。
- `tests/characterization/test_execution_boundaries.py` 新增 2 个：环境变量里即使设置了作用域和日志路径，适配器也不使用；数量无法解析的订单在有持仓可平的情况下也不会发出任何订单（不会被当成"数量为 0 的全平"）。
- 已有测试没有改动。第 1 步的特征化测试、第 9b 步的边界测试、`tests/unit/binance/`、`tests/unit/lab/test_target_runtime*.py` 全部通过。
- 【实际运行结果】新旧对比，在 master 和本步代码上分别运行：
  - 同一组订单（市价入场加平仓；挂单入场、部分成交、平仓时撤单）写出的成交成本记录，除时间和延迟外逐字段相同；
  - 4 个探针的 6 个场景输出相同。
- 【实际运行结果】对 `service.py` 和适配器注入 27 个缺陷，全部被测出。
- 对比时发现一个探针自身的问题：`probe_fill_truncation.py` 打印"最后一行订单"，而日志返回行的顺序取决于随机生成的订单号，在 master 上连跑 6 次就有 3 种输出。已改为按创建时间排序，改后 master 和本步各跑 4 次，8 次输出相同。这是探针的缺陷，不是产品行为的差别。

**可见的差别**

- 执行失败和入场被阻断的日志，记录器名字从 `execution.adapters.binance.execution_adapter` 变成 `trading.application.execution.service`，失败日志的文字从 "Binance order unresolved or blocked; no placement retry" 变成 "Order unresolved or blocked; no placement retry"。【实际运行结果】仓库里没有测试或配置按这些名字或文字过滤。
- 成交成本日志文件原来每批成交打开一次，现在每笔成交打开一次，写出的内容相同。
- 缺少字段的畸形订单事件（正常流程不会产生），失败的先后顺序可能不同，结果同样是不下单并阻断入场。

**行数**：分两次提交。第一次是实现（`service.py` 198 行，其中决策逻辑是从适配器搬来改写的；适配器净改约 90 行；规则、构建器、探针约 20 行）。第二次是测试（新测试 308 行，边界测试 25 行）。每次提交都少于 400 行手写。

**通用验收（原始输出尾部）**

```
3279 passed, 34 skipped, 24 deselected, 9 warnings, 313 subtests passed in 467.12s (0:07:47)
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
$ mypy trading/
Success: no issues found in 36 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

3279 = 3254（研究代码合入后）+ 1（已登记规格的完整性测试）+ 1（环境变量规则）+ 21 + 2。

## 流程调整：CI 分片并行跑全量测试（2026-11-14，技术负责人批准）

**起因**：技术负责人指出开发效率太慢、不必要的等待太多。【实际运行结果】约 2.6 小时里合并了 5 个 PR，每个 PR 本地全量测试约 8 分钟、CI 再约 8 分钟；第 9 步为守住每个改动 400 行拆成了 4 个 PR；有 4 次在等 CI 时停了下来。

**核对时发现的事实**：CI 的 `python-test` 任务此前只跑"核心清单"的 1,745 个测试，全量（第 9d 步合并后收集到 3,313 个）只在本地跑。我在向技术负责人说明方案时一度说成"CI 跑全量"，这是错的，已当场更正。

**批准的三项调整**（写入 `migration-plan.md` 0.4 和 `CLAUDE.md`）：

1. 全量测试只在 CI 运行；本地只跑相关测试和静态检查。
2. 一个步骤一个 PR，内含多次小提交，每次提交仍少于 400 行。
3. CI 测试并行化。

技术负责人没有选"审计期间先写下一步的测试"，所以审计结论处理完之前，仍然不做下一步的任何工作。

**改了什么**

- `scripts/ci_shard.py`：把 `tests/` 下除 `tests/performance` 以外的全部测试文件分成若干片，按文件大小从大到小发给当前最轻的一片。
- `.github/workflows/ci.yml`：新任务 `python-test-all`，4 片同时运行；`python-test` 不再等 `rust-test`。
- `tests/architecture/test_rules.py` 新增两条：4 片恰好是全部测试文件（不漏、不重复），每一片都在 CI 里运行，且不带筛选参数、不允许失败后继续、一片失败不取消其他片。对规则做了 4 种篡改，每种都会让它失败。

**试验结果**（PR #18 第一次运行，【实际运行结果】）

| 任务 | 耗时 |
|---|---|
| `python-test-all` 第 0–3 片 | 3 分 48 秒、3 分 3 秒、5 分 12 秒、4 分 7 秒 |
| `python-test`（核心清单、覆盖率门禁、离线研究验收） | 6 分 47 秒 |
| `rust-test` | 1 分 41 秒 |
| `container-startup` | 2 分 23 秒 |

- 全部 12 项检查通过。约一半的测试此前从未在 CI 环境运行过，这次没有出现环境差异或互相干扰造成的失败。
- 整个 CI 的等待时间约 7 分钟（由最慢的 `python-test` 决定），此前是 `rust-test` 加 `python-test` 串行约 8 分钟。
- 主要的节省来自不再本地跑全量：每个 PR 的机器等待从约 16 分钟降到约 7 分钟。
- 分片不够均匀（3 分到 5 分），文件大小只是运行时间的粗略替代；现在最慢的不是分片，先不调。

**另外做的**：注入缺陷自检工具改为在独立的工作区副本里并行运行，不再改动真实工作区。

## 提前执行：删除 4 个直接下单的研究脚本（2026-11-14，ADR-0008 子决策 5b）

技术负责人批准不等第 7–9 步的审计，单独删除。

- 删除 `research/strategies/` 下的 `run_hft_signal.py`、`run_liquidation_sniper.py`、`run_binary_signal.py`、`run_bybit_mm.py`，共约 2,600 行。目录里没有其他文件，随之消失。
- 删除前核对【实际运行结果】：仓库里没有任何代码 import 它们，没有测试覆盖它们，没有启动配置引用它们；对应的 4 个 systemd 单元已在 `infra/sync_systemd.sh` 的 `RETIRED_UNITS` 里。只有几份旧文档还提到其中一个脚本的旧路径，那些文档属于 `doc-vs-code.md` 里待清理的范围，本次不动。
- 本地：ruff 通过；架构测试和 `tests/unit/trading` 通过。全量测试由 CI 运行。
- 同一个 PR 的第二次提交把今天追加批准的三项写进了迁移计划（第 10b 步、第 10 步的对比测试、本次删除）。

## 第 9 步之后的自检审计（独立审计员，审计对象 `91f800f`）与处理

审计中途因用量额度用尽中断过一次，恢复后把范围收紧到第 9 步的四份改动；第 7、8 步只做了逐文件比对。审计员做了 16 个变异实验，9 个存活。

**结论：第 9d 步不是"行为不变"，我在记录里写错了。** 第 7、8、9a、9b、9c 步逐文件比对一致，没有发现行为差异；风控参数、3 倍上限、急停文件、状态转移表、策略函数都没有改动；测试里没有断言被改弱。

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| F1 | 高 | 第 9d 步的适配器薄壳在订单被拒或入场被阻断时，把失败处理多执行一遍：`blocked` 回调重复、熔断器多记一次失败、多一条错误日志；熔断器断开时就绪标志被误置为未就绪。记录里没有提，测试也测不出 | 已修：没有进入下单流程时薄壳直接返回。【实际运行结果】复现脚本在修复前后和第 9d 步之前的代码上各跑一次，修复后三种情况的回调次数、熔断器计数、就绪标志、错误日志条数与第 9d 步之前完全相同 |
| F2 | 高 | `service.py` 和薄壳上 4 个变异全部存活，其中一个关系到急停平仓（预留被拒后不回调 `blocked`，运行时的待决集合不会清空，平仓会一直等） | 已补 6 个测试：被拒入场的回调、熔断器计数和错误日志逐项精确断言（不再用"包含"）；重复意图在阻断期间不产生任何动作；被订单库拒绝的入场回调 `blocked` 且不计成功；成交写进成本日志文件 |
| F3 | 中 | 覆盖率门禁没有跟着第 8 步和第 9c 步搬家，量的是薄壳 | 已改：门禁加上 `trading.adapters.binance` 和 `trading.application.reconcile`，并让它同时运行应用层和特征化测试。【实际运行结果】83.68%（门槛 75%） |
| F4 | 中 | 架构规则是黑名单，换一个没列出的模块或经 numpy / pandas 读文件、读时钟、取随机数就能绕过 | 已改：domain 和 application 的 import 改为允许清单；新增一条规则拒绝经 numpy / pandas 的文件、时钟和随机数访问。两条规则都用审计员绕过旧规则的那几行做了自检 |
| F5 | 中 | 第 9d 步按 PR 计手写约 636 行，超过 400；测试提交排在实现提交之后 | 属实。第 9d 步是在技术负责人当天批准"一个步骤一个 PR、每次提交少于 400 行"之后做的，两次提交各自少于 400 行。测试排在实现之后是我的顺序错误：F1 正是先写测试就能发现的问题 |
| F6 | 中 | 重启后丢失"已被交易所拒绝"的标记不会被测出；已生效的止损在 90 秒内从交易所消失时的处理没有测试 | 已补 2 个测试 |
| F7 | 中 | 三个薄壳的接线没有测试，其中一个涉及测试网与主网密钥的选择 | 已补 4 个测试：执行器薄壳确实等待；两个薄壳的默认时钟是墙上时钟；`from_env` 按环境取对应的密钥，缺失时不回退到另一个环境 |
| F8 | 低 | 第 8 步记录把三处写成两处 | 已更正 |
| F9 | 低 | `service.py` 用 `getattr` 取 `execution_blockers`；薄壳给服务的时钟写死为 `time.time` | 记录。前者是为了与旧代码一样容忍没有这个方法的交易所对象；两处都在第 10 步组装时统一 |
| F10 | 低 | mypy 跳过 numpy / pandas，策略代码里的表达式类型都是 `Any` | 记录：严格类型检查对 `trading/domain/strategy` 基本不起作用，那部分靠因果性的性质测试约束 |
| F11 | 低 | "3 倍上限只定义一次"的测试按变量名匹配 | 记录 |

**对第 9d 步记录的更正**："可见的差别"一节漏了 F1。从 `19d84b1` 到本次修复之间，被拒或被阻断的订单会多触发一次失败处理。这段代码没有部署过，没有实际影响，但"行为不变"的说法是错的；我做的 27 个注入缺陷和新旧对比都没有覆盖"订单没有进入下单流程"这类路径，探针也不打印回调次数。

**审计员存活的 9 个变异的去向**：S1–S4、L2、P1、P2、C1、C2 现在各有测试约束（上表 F2、F6、F7）。

**本地验证**

```
$ python3 -m pytest tests/unit/trading tests/unit/binance tests/characterization tests/architecture -q
全部通过
$ ruff check --select E,W,F .
All checks passed!
$ mypy trading/
Success: no issues found in 36 source files
$ lint-imports
Contracts: 5 kept, 0 broken.
```

全量测试由 CI 的 4 个分片运行（见 PR）。

## 第 10 步（上半）：会话、主循环迁入 `trading/application`；启动时先校验、后连接账户

第 10 步按一个 PR、多次小提交做。因为这一步很大，做到一个可独立验证的位置先合并，剩下的列在本节末尾，下一个 PR 继续。

**前置测试（先在旧运行时上跑通）**

`tests/characterization/test_runtime_decisions.py`（6 个）：
- 回撤闩锁触发后，所有持仓被平掉，之后不再有入场单。
- 交易所侧的止损不在了、而价格已经越过止损价时，持仓被平掉。
- 挂单入场还没成交时，不会再发第二张入场单，再平衡计划本身也不会提出新的入场。
- K 线处理晚于收盘 10 分钟以上时不入场，原因记为"处理过晚"；下一根准时的 K 线恢复入场。
- 权益超过两分钟没更新、或不是正数时，按 0 定仓，不产生任何入场。
- 运行时实际组装出来的入场门：减仓不需要新鲜数据，也不受急停和回撤限制；只有字面上的 `True` 才算减仓；先问急停、再问回撤；超过 3 倍敞口的入场被拒。

写这些测试时发现一个之前没有记录的现状：**止损被击穿、持仓被平掉之后，下一根 K 线会按信号原样重新入场**，系统不记得刚发生过击穿。已如实固定在测试里。是否要改，留到第 12 步（止损失败处置）请技术负责人决定。

**改了什么（4 次提交）**

1. 运行时从旧的机器学习运行时借用的部件各归其位：
   - K 线类型和 `bar_from_kline` → `trading/domain/bars.py`（只用标准库，离线安全检查仍能导入）；
   - 权益来源 → `trading/application/equity.py`；
   - 入场就绪门 → `trading/application/risk/readiness.py`；
   - 事件日志 → `trading/adapters/files/event_log.py`。
   
   `runner/binance_main.py` 把它们原名导入回来，两个运行时和全部测试不用改。
2. 强平距离规则 → `trading/domain/risk/liquidation.py`；读杠杆和保证金模式环境变量的 `exchange_settings`、连接账户的 `build_adapters` → `trading/infrastructure/`。
3. 会话 → `trading/application/session.py`，主循环 → `trading/application/runtime.py`，成交消费者 → `trading/application/execution/fills.py`。
   - 协作者全部注入：模拟盘的撮合器、执行部件、下单事件的构造函数、事件日志、急停、回撤、模拟盘存储、状态输出。
   - 新增两个端口 `PaperStore`、`StatusSink`，文件实现写出的内容和写法与原来相同（原子写、落盘）。
   - `runner/binance_target.py` 从 535 行缩到 184 行，只负责验签、读环境变量、组装。构造函数和测试用到的名字都不变。
4. **启动顺序（本步唯一的行为变更，修 R11）**：规格验签、研究闸门、作用域、止损距离的校验全部通过后，才拿租约、建状态目录、连接账户。此前是先在交易所上设置杠杆，再因闸门不过而拒绝。

**已有测试的改动**

- `test_risk_state.py` 里固定 R11 的测试改了断言，这是计划写明的本步唯一一处：原断言"拒绝前账户杠杆已被设置"，现断言"拒绝时账户没有任何写入，状态目录也没有创建"。测试改名并去掉了"已知缺陷"标记。
- 同一文件里固定 R5 的测试，原来靠"还没读规格就先拿租约"才能跑到断言处；现在给它一个真实的规格文件，断言未改。
- 同一文件里 R11 测试打补丁的位置从旧模块改到 `trading.infrastructure.bootstrap`，原因同前几步。补丁失效的那一次运行，它用假密钥向 Binance 发了一次请求，被 401 拒绝，没有实际影响。

**证据**

- 【测试日志】模拟盘与研究逐笔一致的三个参数组合、断线恢复、全部特征化测试、Binance 执行测试通过。
- 【实际运行结果】两个运行时探针场景（急停平仓被拒后的重试、有挂单时的急停）输出与搬家前逐字相同。
- 【实际运行结果】注入缺陷自检：对搬过去的会话和主循环注入 15 个缺陷：11 个被测出（其中 1 个第一次配错了测试，换成全部运行时测试后被测出），4 个存活。存活的 4 个都是真缺口，已在第 10 步下半先补测试（见下一节）。
- 本地：ruff、`mypy trading/`（47 个文件）、lint-imports（5 条契约）、离线安全检查、架构测试全部通过。全量测试见 PR 的 CI。

**第 10 步还没做的（下一个 PR）**

- 模拟盘状态改存 SQLite，同时保留 JSON 文件。
- 运行时改用新的执行服务和 `OrderIntent`。现在组装处仍然构造旧的执行适配器和旧的订单事件，所以加载运行时仍会带入旧包。
- "加载运行时后没有旧包"的隔离测试，以及对整条运行时生效的 `no-legacy` 契约。这两条要等上一条做完。
- 实盘路径与研究的对比测试。

## 瘦身批次 A：删除 `research/` 下不可达的历史研究代码（2026-11-14，ADR-0008 子决策 2b）

技术负责人批准把删除提前，与主线穿插（`roadmap.md` 第 6 节）。

- 删除 46 个文件，14,533 行：`research/backtesting/`、`research/scripts/`、`research/` 下的 `alpha_factor.py`、`factor_backtest.py`、`factor_factory.py`、`orthogonalize.py`，以及只为它们服务的 `scripts/dz_sweep_4h.py`、`scripts/trend_bypass_backtest.py`。
- 删除前核对【实际运行结果】：仓库里没有任何代码、测试、CI、Makefile 或 systemd 单元引用这些文件；唯一出现它们名字的地方是 `features/dynamic_selector.py` 的一行文档字符串。
- 保留：`research/model_registry/`（`alpha/retrain/pipeline.py` 引用它，且有测试）、`research/datasets/`（CI 的离线验收在用）、`research/protocols/`、`research/results/`、`research/trials.jsonl`、`research/forward_looks.jsonl`、`research/rust/`（属于 Rust 库）。
- 本地：ruff 通过；`research.model_registry.registry` 和 `features.dynamic_selector` 仍可导入；架构测试和 `tests/unit/research` 通过。全量测试由 CI 运行。

## 第 10 步（下半，第一部分）：补测试缺口；模拟盘状态改存 SQLite

**先补的测试（针对上半自检存活的 4 个缺陷）**

- 资金费结算还没发布时不入场，原因记为"资金费未发布"。计划 0.6 要求过这一条，上半只测了"K 线过晚"那一半。
- 计算目标时只看到当根 K 线收盘前已结算的资金费。
- 权益刷新失败时入场被阻断，减仓不受影响；权益恢复后入场恢复。
- 模拟盘每个周期都把处理过的 K 线落盘。

【实际运行结果】补完后，这 4 个缺陷逐个注入都会被对应的测试抓到。

**模拟盘状态改存 SQLite（ADR-0007 子决策 3b）**

- `trading/adapters/sqlite/paper_store.py`：一份当前文档、一个版本号、一张归档表。
- `trading/infrastructure/paper_store.py`：先写数据库，再写一份内容相同的 `paper_state.json`，并记下该文件的摘要。
  - 已有的 JSON 文件在第一次启动时导入，原文件保留。
  - JSON 文件如果在上次保存之后被外部改过（回滚到旧版本运行过，或人工修改），以文件为准重新导入。
  - 只是比数据库落后一次保存的 JSON 不会覆盖数据库。两次写之间崩溃的两种情况都有测试。
- JSON 里十进制数的编码搬到 `trading/adapters/codec.py`，文件适配器和 SQLite 适配器互不依赖。
- 运行时入口改用这个组合存储。

**已有测试的改动**：没有。模拟盘与研究逐笔一致、重启续跑、断线恢复、"状态属于另一份规格时拒绝"、"手工改过的状态被拒绝直到重置"全部不改通过。

**本地**：ruff、`mypy trading/`（50 个文件）、lint-imports（5 条契约）、架构测试通过。全量测试见 PR 的 CI。

**第 10 步还剩**：运行时改用新的执行服务和 `OrderIntent`；导入隔离测试和 `no-legacy` 契约；实盘路径与研究的对比测试。

## 瘦身批次 B：删除 18 个没有任何引用的一次性脚本和旧下载器（2026-11-14，ADR-0008 子决策 2b）

- 标准比"不可达"更严：文件的模块名、路径在全部代码、测试、CI、Makefile、systemd 单元、配置里都没有出现（文档除外）。按这个标准 `scripts/`、`monitoring/`、`data/downloads/` 里共有 24 个文件。
- 删除其中 18 个，4,753 行：
  - `scripts/` 8 个一次性分析和回测脚本；
  - `monitoring/` 7 个分析脚本；
  - `data/downloads/` 3 个旧下载器。
- **没有删的 6 个**：`scripts/` 下的 `publish_research_model.py`、`restore_backup.py`、`setup_okx_credentials.py`、`daily_check.py`、`polymarket_health.py`、`dashboard.py`。它们是可能由人手工运行的运维工具，代码判断不了是否还在用。留给技术负责人确认。（更正，2026-11-14：这里原先写"其中发布规格的脚本是上线流程需要的"，说错了。上线流程用的是 `publish_strategy_spec.py`，它有引用、不在这 6 个里；`publish_research_model.py` 是已关闭的 ML 线的发布脚本。这 6 个已在批次 D 删除。）
- 本地：ruff 和架构测试通过。全量测试由 CI 运行。

## 瘦身批次 C：删除独立的 Rust 交易程序 `rust_bin/`（2026-11-14，ADR-0008 子决策 4b）

- 删除 `rust_bin/` 全部 14 个文件，3,161 行：`quant_trader` 的源码和 9 份配置。它自带行情、决策和下单，没有任何服务、脚本或 CI 步骤启动它。
- `Cargo.toml` 去掉对应的 `[[bin]]` 段；`Dockerfile` 去掉复制该目录的一行。依赖列表没有动：其中一部分仍被 Python 扩展用到，要不要精简另行处理。
- 【实际运行结果】`cargo metadata --locked --offline` 通过，说明清单和锁文件仍然一致。完整的 `cargo test` 和镜像构建由 CI 的 `rust-test`、`container-startup` 两项检查验证。

## 瘦身批次 D：删除 6 个运维脚本、旧的数据刷新脚本和它的 9 个下载器（2026-11-14，技术负责人逐项批准）

- 技术负责人 2026-11-14 答复："1 删除，2 删除"。对应两组文件，共 16 个，3,385 行（另有 1 个误提交的 `.pyc`）：
  - `scripts/` 下 6 个：`publish_research_model.py`、`restore_backup.py`、`setup_okx_credentials.py`、`daily_check.py`、`polymarket_health.py`、`dashboard.py`。
  - `data/downloads/data_refresh.py` 和它调用的 9 个 `download_*.py`。对应的定时任务此前已退役。
- 删除前用 `git grep` 搜索全部已跟踪文件：这 16 个文件只在文档里被提到，没有代码、测试、CI、服务单元引用。
- 提到它们的 4 处文档已改：`README.md`、`docs/binance_runbook.md`、`docs/operations.md`、`docs/add_new_symbol.md`；`data/downloads/__init__.py` 的说明改成现在实际存在的两个下载器。
- 顺带删除一个误提交的编译产物 `data/loaders/__pycache__/__init__.cpython-312.pyc`（它的源码早已不在）。

**引用扫描的一个漏洞与复查**

- 之前各批删除用的是命令行里的 `grep -r`。这个环境里的 `grep` 是一层包装，递归搜索时会跳过被 `.gitignore` 忽略的路径；而仓库里有 101 个文件"被忽略但已跟踪"，其中 `data/` 下有 21 个 Python 文件。所以此前"没有任何引用"的结论对这些路径没有覆盖到。
- 发现经过：准备本批时，9 个下载器被判成"没有引用"，实际上 `data_refresh.py` 导入了它们。提交前发现，当时撤销；后来技术负责人批准整组删除。
- 【实际运行结果】复查：用不经包装的 `grep` 在这 101 个文件里搜索今天删除的全部 68 个 Python 文件的模块名、路径和文件名，没有任何引用（只有 `__init__` 这个通用词的误匹配）。批次 A、B、C 和 4 个直接下单脚本的删除不受影响。每次删除后 CI 的全量测试也都通过。
- 之后的引用检查一律用 `git grep`（搜索全部已跟踪文件）。
- 本地：ruff 和架构测试通过。全量测试由 CI 运行。

## 第 10 步（下半，第二部分）：运行时改用新的执行服务；导入隔离；实盘路径与研究的对比测试

**改了什么**

- `trading/application/execution/executor.py`（新）：`OwnerExecutor`。订单意图进、带类型的成交出，不再经过旧系统的事件类型。
  - 随订单一起存进账本的记录只有 5 个普通字段（订单号、合约、方向、参考价、原因）。旧适配器写的记录多一个事件头，新代码读的时候忽略它，有测试。
  - **反方向不兼容**：新代码写的记录没有事件头。【实际运行结果】旧适配器在这样的账本上启动时报 `KeyError: 'header'` 并退出，不会下单。也就是回滚到这一步之前的代码时，不能沿用新代码写过的 `orders.sqlite`。目前没有任何执行模式的部署，所以没有受影响的数据；将来回滚要按这一条处理。
- `trading/infrastructure/target.py`（新）：组装一个合约的执行会话——订单账本、订单生命周期、执行服务、`OwnerExecutor`、成交消费、止损管理、入场闸门。
- `trading/infrastructure/settings.py`：回撤上限、单笔与每日下单上限、杠杆上限从环境变量读取，默认值与之前逐个相同（有测试钉住）。
- `runner/binance_target.py`：只负责加载验签后的规格，并把还没迁移的 4 样东西传进去：研究用的模拟撮合、下单限额器、急停文件、成交成本日志。不再导入 `runner.binance_main` 和 `lab.strategy`。
- 执行模式的运行时不再经过 `BinanceExecutionAdapter`。这个旧适配器还在，只给已关闭的 `runner.binance_main` 用。

**与计划不同的地方**

- 计划写的是把入口搬到 `trading/infrastructure/cli/target.py`，旧入口只做转发。没有这样做：入口必须调用验签加载器（项目规则要求只能经 `infra/model_signing.py`）和研究撮合，而新包按契约不能导入旧包。所以入口留在 `runner/`，用一份清单限定它还能从旧包拿什么（7 项，每项写明由哪一步去掉），清单只能缩短，有测试。
- 启动检查的顺序：止损距离检查在任何状态文件创建之前运行（有测试）。（更正，M2 审计 F7：这里原先写"仍然是执行模式下第一个运行的检查"，不准确。在它之前还有规格验签、主网闸门、scope 校验和限额读取；迁移前它排在 scope 校验之前，现在排在之后，只影响两项同时不合格时先报哪一个错。）

**导入隔离**

- 新增测试：在子进程里真实加载运行时主循环和组装模块，内存里没有任何旧包，也没有 Rust 扩展。这能发现函数内部的延迟导入，import-linter 发现不了。
- 【实际运行结果】加载整个入口 `runner.binance_target` 时，带入的旧包仍是 6 个（`_quant_hotpath`、`data`、`event`、`lab`、`runner`、`shared`），旧模块数从 27 个降到 25 个。（补充，M2 审计 F7：这是"导入入口时"的数字。【实际运行结果】以执行模式构建并预热运行时后，延迟导入还会带入 `execution`、`infra`、`monitoring` 三个旧包，合计 9 个；其中 `execution` 只是为了取成交成本日志的那个小类。）剩下的都来自研究用的模拟撮合 `lab.simulate`。路线图完成标准第 1 条还不成立，要等模拟撮合和限额器的去留确定。

**实盘路径与研究的对比测试**

- 做法：同一段行情（360 根 1 小时 K 线）同时喂给研究模拟和执行模式的运行时；每根 K 线之前把账户权益设成研究模拟的权益；假交易所对市价单立即全部成交。然后逐根比较：执行路径发出的订单（方向、数量、是否只减仓）必须等于研究发出的订单，确认后的持仓必须等于研究的持仓。
- 【实际运行结果】360 根 K 线、11 笔订单，订单和持仓全部一致，没有闸门拦截，没有对账问题。
- **没有比较、也不应该相等的部分**（按设计就不同）：
  - 成交价：研究按下一根开盘价加 3 个基点滑点，交易所按实际成交价。
  - 手续费和资金费的现金：它们通过权益进入决策，而权益在测试里被强制相等。
  - 部分成交：研究有成交量占比上限，假交易所一次成交完。
  - 风控限额：研究里没有。测试用 2,000 USDT 的规模，确认限额没有触发。
- 这个测试保证的是：只要成交和权益相同，执行路径的决策就和研究相同。差异只可能来自成交模型，不会来自决策代码。

**已有测试的改动**：没有改任何断言。

**变异自检**

【实际运行结果】向新代码逐个注入 22 个缺陷，每个只跑应当抓住它的那份测试：

- 第一轮抓住 20 个，漏掉 2 个：
  - "按开盘价而不是收盘价定量"：对比测试用的行情每根 K 线只动万分之几，换一个价格算出的数量落在同一个最小下单单位里，分辨不出来。
  - "去掉最小名义额的研究下限"：没有任何测试覆盖交易所下限低于研究下限的情况。
- 处理：对比测试改用价格在 1,000 附近、每次摆动约 1% 的行情；为手数规则补了一个直接的单元测试。两个缺陷重跑后都被抓住。
- 加强数据时的两个发现，都写进了测试的注释：
  - 行情再剧烈一些（每次摆动约 10%）时，默认 5% 的日回撤闩锁会触发，执行路径随之平仓，与研究分叉。这是按设计应有的差异：研究模拟里没有回撤闩锁。
  - 执行路径在滚动窗口上算目标仓位，研究在全部历史上算，两者可能在浮点数的最后几位不同（观察到 5e-17）。测试对目标仓位允许 1e-12 的相对误差，对订单和持仓仍要求完全相等。

**本地**：ruff、`mypy trading/`（52 个文件）、lint-imports（5 条契约）、架构测试、特征测试、运行时测试、`tests/unit/trading` 通过。全量测试见 PR 的 CI。

**没有做、留给后面的**

- 订单路径和回撤闩锁用的是系统时钟，不是运行时注入的时钟（迁移前就是这样）。生产中两者相同；测试里意味着挂单有效期和"按日重置"不会随模拟时间推进。留给第 10b 步之后的审计评估。
- 研究下限里 BTC 的最小名义额仍是 100 USDT（交易所实际约 66 USDT），这是一个待定的取舍，没有改。
- 模拟盘仍用旧包里的研究撮合；下单限额器、急停文件、单写者租约、成交成本日志仍从旧包传入（第 13、15 步处理）。

## 第 10b 步：交易所边界加类型；Binance 的状态词和错误码只留在适配器里（里程碑 M2）

**改了什么（不改变行为）**

- `trading/domain/venue.py` 新增不可变的数据类：`VenueOrder`（查询到的订单）、`OpenOrder`（挂单快照）、`StopOrder`（止损单）；以及下单结果、止损下单结果、报价三个带类型的字典。成交 `CanonicalFill` 和持仓 `VenuePosition` 原来就是数据类。
- `trading/domain/orders.py` 新增 `ConfirmedFill`：订单生命周期确认的一笔成交。之前是临时拼出来的无类型对象。
- Binance 适配器：
  - 查询订单、挂单、止损单时返回上面的数据类，属性名与之前相同，所以调用方不用改。
  - 止损单多了一个 `state` 字段，只有三种取值：生效中、已触发、已不存在。Binance 自己的状态词（`NEW`、`TRIGGERING`、`TRIGGERED`、`FINISHED` 等）在这里翻译；没见过的新状态词按"已不存在"处理，与之前的行为一致。
  - 止损下单被拒时多了一个 `breached` 标志：只有当交易所返回 -2021（价格已越过止损价）时为真。
- 应用层的止损管理 `trading/application/execution/stops.py` 只看 `state` 和 `breached`，不再认识任何 Binance 状态词和错误码。
- 端口 `trading/application/ports.py`：交易所相关的方法不再返回无类型的对象，改为数据类、只读协议或带类型的字典。
- `trading/infrastructure/bootstrap.py` 新增 `as_account`：由类型检查器证明 Binance 适配器实现了全部交易所端口，返回类型逐个对得上。

**保留下来的宽松处**

- 下单结果和报价仍是字典（现在带类型）。原因：这是 Binance、Bybit、OKX 三个适配器和所有假交易所共用的约定，冻结线的代码不动。
- 订单生命周期读取订单的合约、方向、状态时仍容忍缺字段（冻结线的适配器返回的对象不一定都有），协议里只声明了必有的字段。

**新增的架构规则**

- `trading/domain` 和 `trading/application` 的代码里不得出现交易所状态词和 Binance 错误码（-1000 到 -5999），文档字符串除外。
- 四个交易所端口的方法不得返回 `Any`、`Any` 的元组或无类型字典。
- 两条规则各带一个自检测试，确认规则本身能发现违规。

- 另补一条第 10 步验收里要求、当时漏掉的测试：加载运行时入口之后，内存里没有 `runner.binance_main`，也没有 `engine`、`decision`、`features`、`alpha` 这几个已关闭的 ML 包。

**已有测试的改动**：没有改任何断言。

**变异自检**

【实际运行结果】注入 13 个缺陷，每个只跑应当抓住它的测试，13 个全部被抓住。覆盖：两组状态词各少一个、未知状态词被当成生效中、`breached` 永远为假、错误码写错、已触发的止损不接管、已不存在的止损不补挂、任何拒绝都当成击穿、击穿从不上报、成交金额留成字符串、挂单方向不转小写、状态词和错误码重新写回应用层、端口重新返回 `Any`。

**一处自己的差错，以及它对上面结果的影响**

- `ConfirmedFill` 的测试是在本地回归跑起来之后才加的，没有单独运行就推送了。CI 发现其中一条断言写错（账本里存的数量是字符串 `"1.0"`，我写成了 `"1"`）。已改成"是字符串且数值等于 1"。
- 这条写错的断言和其它边界测试在同一个文件里，而变异自检遇到第一个失败就停。所以第一次的"13 个全部抓住"不可信：其中 9 个缺陷用的就是这个文件，不注入缺陷它也会失败。
- 处理：变异工具加了一步，先在不注入缺陷的代码上跑一遍所选测试，不通过就报"基线失败"、不给结论。修正断言后 13 个缺陷整批重跑。【实际运行结果】基线通过，13 个全部被抓住。上面的数字是重跑后的结果。

**本地**：ruff、`mypy trading/`（52 个文件）、lint-imports（5 条契约）通过；架构、`tests/unit/binance`、契约、特征、`tests/unit/trading`、运行时、`tests/offline`、Bybit 与执行层的测试通过。全量测试见 PR 的 CI。

**下一步**：按路线图，M2 完成后先做一次独立审计，审计结论处理完之前不开始第 11 步的代码。

## 决定执行：执行模式的最小下单额跟随交易所（2026-11-14，技术负责人答复"4 跟随交易所"）

**改了什么（行为变更，已批准）**

- 之前：执行模式的最小名义额取"交易所规则"和"研究常量"两者中较大的一个。研究常量是 BTC 100 USDT、ETH 20 USDT。
- 现在：执行模式的手数步长、最小数量、最小名义额完全按交易所公布的规则（`trading/infrastructure/target.py` 的 `exchange_lot_rules`）。
- 研究模拟和模拟盘不变，仍用研究常量。原因：已登记的研究和前向检验引用这些常量，登记之后不改；模拟盘按设计与研究逐笔一致。

**后果，需要知道**

- 名义额落在"交易所下限"和"研究常量"之间的订单，执行模式现在会发出，研究和模拟盘不会。有一条测试专门钉住这一点：300 USDT 权益、目标仓位 25% 时，执行模式买入 0.001 BTC（约 60 USDT），研究不下单。
- 这只影响小资金。按已登记的规模（每个品种 5,000 USDT、调仓带宽 10%），每笔订单至少 500 USDT，两边没有差别。
- 所以"实盘路径与研究逐根一致"的保证范围是：订单名义额不低于研究常量。对比测试里每笔订单都高于两个下限，仍然完全相等。

**改动的断言（1 处，我自己在第 10 步写的测试）**

- 原测试 `test_lot_rules_take_the_exchange_filters_but_never_a_minimum_below_the_research_floor` 断言"交易所下限 5、研究常量 100 时取 100"。现在改为 `test_execution_lot_rules_are_the_exchange_s_own_filters`，断言取 5。
- 新增 2 条：执行会话拿到的就是交易所的下限；上面"小订单执行发、研究不发"的那条。

**没有动的**：`lab/simulate.py` 的 `SYMBOL_RULES`；风控限额；模拟盘。

## M2 之后的独立审计（审计对象 `ddaa68c`，范围 `91f800f..ddaa68c`）与处理

审计员在自己的副本里工作，没有改动仓库。中途撞上账号用量上限停过一次，收窄范围后继续完成。

**没有被推翻的两条**

- **行为不变（执行模式接线）**：审计员用一个固定场景在新旧两个版本上各跑一遍并比较全部可观察结果。场景含重启、下单后超时、交易所拒单、止损被撤且价格已越过、止损触发并接管、K 线过晚、行情中断、持仓查询失败、急停（平仓与不平仓）、回撤闩锁、漏掉的 K 线。市价模式 1,429 条签名请求、241 行事件日志、订单库 4 张表、回撤库、状态文件、成交成本记录全部相同；挂单模式 1,724 条请求、500 行事件同样相同。差别只有已记录的那一处（订单记录不再带事件头）。
- **止损状态翻译等价**：14 种状态词 × 4 种账本行状态 × 4 个时间点 × 有无订单号 × 3 种持仓变化，加 9 种下单结果，共 1,362 种情况，新旧版本结果零差异。冻结线（Bybit、OKX）的测试通过。

**发现与处理**

| 编号 | 严重度 | 发现 | 处理 |
|---|---|---|---|
| F1 | 中 | "执行路径与研究逐根一致"只在单品种下成立。两个品种共用一个"每秒 1 单"的限额器（按真实时钟），第一个品种入场后要等 0.5 秒，第二个品种的入场检查落在同一秒内被拒，要到下一根 K 线才再试 | **我已复现**：【实际运行结果】双品种、120 根 K 线，ETH 的 10 次入场意图里 6 次被拒。不是这次改造引入的（迁移前配置相同、现象相同）。登记为已知缺陷 **R23**，加了一条钉住现状的测试；对比测试的说明改为写清适用范围。**怎么修需要技术负责人定**，见下 |
| F2 | 中 | 7 个风控相关的注入缺陷没有测试能抓到：低于 3 的杠杆上限被忽略、止损距离被写死、每日下单数不限、单笔上限取成每日上限、不要求平仓的急停也平仓、断线恢复后对每根漏掉的 K 线都做决策、目标未就绪时仍然下单 | 新增 7 个测试，逐条对应。7 个缺陷重跑后全部被抓住 |
| F3 | 中 | 没有测试保证"-2021 以外的拒绝不算击穿"。如果写错，止损下单遇到限频（429）会被当成击穿，运行时会市价平仓 | 新增测试：限频、5xx、网络错误三种情况都不得标记为击穿。现有代码本身是对的 |
| F4 | 低/中 | 启动顺序（R11）的测试只覆盖了主网闸门。去掉 scope 校验或止损距离校验不会被发现；启动时忽略执行阻断项、不设置杠杆也不会被发现 | 新增 3 条测试：scope 不符、止损超出强平距离时，在任何账户设置写入之前拒绝；连接账户时带上配置的杠杆和保证金模式，有阻断项时拒绝启动 |
| F5 | 低 | 执行器接受空白或带空格的 scope；把 `OSError` 从行情错误里去掉不会被发现 | 各补 1 条测试 |
| F6 | 低 | 模拟盘有一处没有列出的行为变化：手工删除 `paper_state.json` 不再重新开始记录（数据库里还有） | 这是 ADR-0007 子决策 3b 的预期结果（数据库是主存储），当时有测试但执行记录没写。现在补记，并写进 `CLAUDE.md`：重置要用 `--reset-paper-state` |
| F7 | 低 | 执行记录两处措辞不准：止损距离检查不是"第一个运行的检查"；"6 个旧包"是导入时的数字，执行模式下延迟导入还会多 3 个 | 已在原处加更正说明。【实际运行结果】执行模式下是 9 个旧包 |

**审计注入缺陷的统计**：28 个，当时抓住 15 个、漏 13 个。漏掉的 13 个里，1 个（手数规则）对应的代码已按技术负责人的决定改掉并另有测试；其余 12 个，加上我为同类情况多列的 2 个，共 14 个，补测试后用同一份缺陷清单重跑。【实际运行结果】基线通过，14 个全部被抓住。

**我自己的教训**：第 10 步我报告"22 个缺陷全部抓住"，这个数字只说明我想到的那 22 个被抓住了。审计员换一批缺陷，漏掉近一半，集中在"人设的限额有没有真的传到位"。以后接线类的改动，对每一个传进去的参数都要有一条"改了它、结果跟着变"的测试。

**审计员顺带提到、不在本次范围内的一点**：限额器把每一次通过检查的入场都计入"每日名义额"，包括后来没成交就撤掉的挂单。新旧版本相同。记入待办，第 13 步做限额落盘时一并处理。

**F1（R23）需要技术负责人决定怎么修**

- 现状的后果：两个品种同一根 K 线都要入场时，第二个品种可能晚一小时入场。实盘里第一个品种的下单流程如果超过 1 秒就不会触发，所以是间歇性的。
- 方案 A：限额不变。入场被"每秒 1 单"挡住时，在同一根 K 线内的下一个周期（5 秒后）重试，仍受"K 线过晚不入场"的 10 分钟限制。
- 方案 B：把"每秒 1 单"改成按品种计，账户层面等于每秒 2 单。这是改风控参数。
- 方案 C：保持现状，接受偶尔晚一小时。
- 我建议方案 A：不动任何限额，只是不把被限频挡住的入场直接丢掉。

**已有测试的改动**：没有改任何断言。新增 14 个测试函数（参数展开后 18 条用例），其中 1 个是钉住 R23 现状的已知缺陷测试。

**本地**：ruff、`mypy trading/`、lint-imports、架构测试、新测试文件、`test_runtime_decisions.py` 通过。全量测试见 PR 的 CI。

## 修复 R23：被"每秒 1 单"挡住的入场在同一根 K 线内重试（2026-11-14，技术负责人选定方案 A）

**技术负责人同日的三项答复**

1. R23 的修法：同一根 K 线内重试，限额不变（方案 A）。本节执行。
2. 止损被击穿并平仓后：该品种暂停入场，直到人工解除，并告警。第 12 步执行。
3. 研究和模拟盘的手数常量不改，只有执行模式跟随交易所。

**改了什么（行为变更，已批准）**

- 入场闸门的结果多了一个"可重试"标记：只有被"每秒下单数"这一条限额拒绝时为真。此时其它检查（急停、回撤、敞口、单笔与每日限额）都已通过，而这条限额过一秒自己就解除。
- 会话把这样的入场记下来，不再丢掉。运行时每个周期（5 秒）开头再问一次闸门：
  - 通过：用同一个订单号发出。
  - 仍被限频：继续等下一个周期。
  - 被其它原因拒绝（例如此时急停已激活）：放弃并记录原因。
  - 距离 K 线收盘超过 10 分钟：放弃，原因记为"K 线处理过晚"。这是原有的入场时限，没有放宽。
- 新的 K 线到来、或者触发平仓（急停、回撤、止损击穿）时，记下来的入场一律作废并写日志。
- 减仓单从不进入这个队列（减仓本来就不经过限额）。
- **没有改任何限额数值。** "每秒 1 单"仍然成立：重试的那一单同样要过限额器。

**改动的断言（1 处）**

- `tests/characterization/test_runtime_decisions.py` 里钉住 R23 现状的测试：去掉"已知缺陷"标记并改名；原有的两条断言（第二单被拒、原因是每秒下单数）保留，新增一条"该拒绝带可重试标记"。它原先的说明"下一根 K 线之前不会重试"已改写。

**新增测试**（`tests/unit/trading/test_deferred_entry.py`，15 条用例）

- 被限频的入场在同一根 K 线的下一个周期发出，订单号不变，只做一次决策。
- 连续被限频时一直等到通过；被其它原因拒绝时放弃；超过 10 分钟放弃；新 K 线作废旧的；平仓作废并且不被它拦住；反手时入场单保留自己的序号；减仓从不排队。
- 闸门只把"每秒下单数"标为可重试，其它限额和急停都不标。
- 用真实限额器、两个品种：第二个品种在同一根 K 线内晚一个周期入场（测试里真实等待 1.1 秒）。

**变异自检**：【实际运行结果】注入 12 个缺陷（基线通过），12 个全部被抓住：任何限额拒绝都重试、从不标记可重试、从不排队、过晚仍入场、仍被限频就丢掉、其它拒绝继续等、运行时不重试、平仓不作废、新 K 线不作废、排队的入场丢了序号、减仓也排队、重试时不问闸门直接发。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、`tests/offline` 通过。架构测试第一次有 1 条不过（我新写的一个参数类型用了 `Any`），改成协议类型后通过。ruff、`mypy trading/`、lint-imports 通过。全量测试见 PR 的 CI。

**修复后还剩的差别**：第二个品种的入场比第一个晚约 5 秒发出（同一根 K 线内）。研究模拟里两者同时成交，这 5 秒的价格差不在对比测试的范围内。

## 第 11 步：急停平仓可靠化（ADR-0005 子决策 1b、2b、3b、7b，2026-11-14 已批准）

**改了什么（行为变更，全部对应已批准的子决策）**

- 新增 `trading/application/execution/flatten.py`（`Flattener`），替代会话里原来的 `flatten`。急停（要求平仓时）、回撤闩锁、止损击穿三种情况都走它。运行时每个周期（约 5 秒）调用一次，直到仓位归零。
- **2b 先撤后平**：每次调用先撤销该品种全部入场挂单，并把撤单过程中确认的成交入账，再决定平多少。撤单结果不明时照样平仓，下一次调用再撤一次。
- **3b 急停和回撤按交易所持仓全平**：急停和回撤闩锁直接平掉交易所上该品种的全部持仓，不再受"本策略已确认的数量"限制。
  - 止损击穿后的平仓仍按账本定量（与其它减仓一样）。理由：子决策 3b 写明的范围只有急停和回撤。
  - 账本不会被这里改写。如果平仓前账本与交易所不一致，平仓后对账仍会报不一致，入场继续被阻断，直到人工处理（第 12b 步的工具）。这是"失败即阻断"，不是自动修正。
    （更正，2026-11-15：写这句时第 12b 步的范围只批准了"处置卡死订单"，不含修正账本。技术负责人 2026-11-15 另行批准了"人工把账本对齐到交易所"的命令，见第 12b 步。）
- **1b 有上限的重试**：
  - 每次尝试是一个新的订单意图，编号带序号（`<品种>:<K 线>:<原因>:0`、`:1`……）。第一次的编号与改造前相同。任何一个意图都不会重发。
  - 只有当该品种没有未结的减仓单、且距上一次尝试满 30 秒时，才发下一次。
  - 连续 10 次尝试都没有成交：停止自动重试，写一条告警，之后每根新 K 线再试一次；每次失败都再写告警。有成交后计数清零。
  - 状态文件里每个品种多了 `flatten_failures`。
- **7b 可观测性**：订单库拒绝预留时（已有未结减仓、有未消费成交、检查点版本不符、入场未决），原因写入事件日志（`blocked`，原因前缀 `journal:`）。急停平仓被拒计为一次失败，所以连续被拒 10 次（约 5 分钟）会告警。拒绝规则本身没有改。
- 执行服务和执行器新增两个方法：单独撤销入场挂单、读取订单库的拒绝原因。订单意图新增一个字段 `close_venue_position`（只允许用于只减仓、不带数量的意图）。

**告警目前是什么**：事件日志里的 `alert` 行、进程日志里的错误、状态文件里的失败次数。发到人手上的告警出口在第 12 步接入。

**改动的断言（2 个已知缺陷测试，`tests/characterization/test_exit_paths.py`）**

- R1 测试改名为 `test_a_rejected_flatten_is_retried_thirty_seconds_later_within_the_same_bar`，去掉已知缺陷标记：
  - 原来：同一根 K 线里 6 个周期只有 1 次尝试；仓位不变；下一根 K 线才平掉。
  - 现在：前 5 个周期（30 秒内）1 次尝试、仓位不变；第 6 个周期（满 30 秒）第 2 次尝试、仓位归零；两次的编号分别以 `:0`、`:1` 结尾。
- R2 测试改名为 `test_kill_flatten_cancels_a_resting_entry_first_then_closes_the_position`，去掉已知缺陷标记：
  - 原来：有入场挂单时不发平仓单；挂单保持"部分成交"；继续成交让敞口变大。
  - 现在：下一个周期挂单变为"已撤销"，部分成交的仓位被一张只减仓单平掉；之后 3 个周期不再发任何单。

**新增测试**（`tests/unit/trading/test_flatten.py`，15 条用例；更正，第 12b 步后审计：原先写 17 条）

- 重试参数就是批准的 30 秒、10 次。
- 连续 10 次失败：恰好 10 个意图，各发一次；1 条告警；同一根 K 线内不再尝试；新 K 线恰好 1 次；交易所恢复后平掉，失败计数清零。
- 结果不明的尝试（超时）之后，无论过多久都不发第二张（只查不重发）。
- 行情中断、新 K 线到不了时，重试照样按 30 秒进行。
- 撤单失败时照样平仓，下一个周期再撤成功。
- 急停时即使还没有任何成交、没有仓位，挂着的入场单也被撤销（不变量 K2）。
- 仓位归零后，下一次平仓从头计数，也不必等上一次留下的 30 秒。
- 急停按交易所持仓全平，包括账本为 0 的情况；之后账本与交易所的不一致继续被报告、入场继续被阻断。
- 止损击穿只平账本那一部分。
- 订单库拒绝减仓时，事件日志写明原因。
- 执行服务：带 `close_venue_position` 的意图不受"本策略已确认数量"限制；不带的仍受限制；该字段只能用于只减仓、不带数量的意图。

**同一意图至多发一次（不变量 O1）的已有测试**：一条断言都没有改。

**变异自检**

- 【实际运行结果】注入 19 个缺陷（基线通过），第一轮抓住 17 个，漏掉 2 个：
  - "去掉单独的撤单"：R2 测试仍然通过，因为执行服务在任何减仓之前也会撤入场挂单。单独撤单真正起作用的情形是"还没有仓位、只有挂单"，没有测试覆盖。
  - "仓位归零后不清零失败计数"：没有测试覆盖"新一轮平仓从头开始"。
- 处理：各补 1 条测试（见上面最后两条）。两个缺陷重跑后都被抓住。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试全部通过。ruff、`mypy trading/`（53 个文件）、lint-imports 通过。全量测试见 PR 的 CI。

**没有做、留给后面的**

- 告警还没有发到人手上（第 12 步接入告警出口）。
- 止损仍按账本定量；止损挂不上时仍只记一条问题，入场不受影响（R3，第 12 步）。
- 止损击穿后暂停入场直到人工解除（技术负责人已决定，第 12 步）。
- 账本与交易所不一致时只能阻断入场，修正账本要靠第 12b 步的人工工具。

## 第 12 步（第 1 部分）：止损挂不上时阻断入场（R3）；触发的止损跟踪到底（R8）

技术负责人 2026-11-14 批准 ADR-0005 子决策 4b。第 12 步分 3 个 PR：本部分；止损触发或击穿后暂停入场直到人工解除（技术负责人 2026-11-14、11-15 的决定）；告警出口。

**改了什么（行为变更）**

- **R3 / 4b**：每次对账后，如果本策略有已确认的持仓、却没有一张生效中的交易所止损，就阻断入场，并记一条问题"position without an active protective stop; entries blocked"。已有仓位不动。止损挂上之后，下一次对账自动恢复入场。
  - 这也覆盖"止损下单结果不明"和"仓位没有开仓价、挂不了止损"两种情况：在确认止损生效之前都不入场。
  - 这条检查放在"止损击穿后的平仓"之后：平仓已经把仓位平掉，就没有需要保护的仓位，也就不阻断。第一版放在平仓之前，回归测试里"击穿平仓后下一根 K 线重新入场"那条钉住现状的测试因此失败（阻断要等下一次对账才解除，而测试一个周期跨了一小时）。没有改那条测试，改的是检查的位置。
  - 检查本身出错（例如读不了账本）时按"没有止损"处理，阻断入场。
- **R8**：交易所报告止损已触发、但还没给出它变成的那张订单的编号时，不再把这张止损当作结束：
  - 状态记为"等待订单"，每次对账继续去查；查不到也继续等。
  - 等待期间不另挂新止损。理由：交易所上的仓位已经被平掉，此时挂的"全平"止损会留在交易所，可能在以后按过时的价格平掉新开的仓位。
  - 等待期间入场被阻断（按上面的 4b 规则：账本仍有仓位，但没有生效的止损）。
  - 拿到订单编号后，把它收进订单账本，成交照常入账。
- **修 R8 时发现的同类问题**：止损触发并被收进账本的那一次对账里，成交还没入账，账本仍显示有仓位，于是会立刻再挂一张新止损（改造前就是这样）。现在这一次对账不再挂新止损；下一次对账成交已入账、账本归零，就不会再挂。有一条新测试专门钉住。

**改动的断言（2 个已知缺陷测试，`tests/characterization/test_stops_and_fills.py`）**

- R3 测试改名为 `test_a_position_without_a_working_stop_blocks_entries_until_one_is_in_place`，去掉已知缺陷标记，改为在运行时层面验证（规则加在运行时的对账里）：
  - 保留：记录"止损被拒"的问题；没有生效的止损；仓位不变。（更正，第 12b 步后审计：原先写"每次同步都记"，不准确。旧测试逐次检查，新测试只检查所有事件里出现过。）
  - 改变：原来断言"入场仍然允许"（`_ready is True`）；现在断言"入场被阻断"，并且问题记录里有"没有生效的止损、入场被阻断"这一条。
  - 新增：止损恢复可挂之后，下一个周期止损生效、入场恢复。
- R8 测试改名为 `test_a_triggered_stop_is_followed_until_the_order_it_became_is_known`，去掉已知缺陷标记：
  - 原来断言：账本里没有止损订单的记录；没有成交入账；账本仍是 0.1；对账报告不一致。
  - 现在断言：等待期间报告"已触发、订单未知"、入场被阻断；查不到时继续等；拿到编号后收进账本、成交入账、账本归零、对账无问题；全程只有一张止损。

**新增测试**：止损被收进账本的那一次对账不再挂新止损；检查出错时阻断入场。

**变异自检**：【实际运行结果】注入 10 个缺陷（基线通过），10 个全部被抓住：触发即当作结束、等待中的止损不算"当前"、查不到就结束等待、收进账本后立刻挂新止损、规则不看持仓、规则不看止损、运行时不阻断、等待状态不算存活、检查出错时放行、检查完全不运行。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试通过。ruff、`mypy trading/`、lint-imports 通过。全量测试见 PR 的 CI。

**没有做、留给后面的**：止损触发或击穿后暂停入场（第 2 部分）；告警发到人手上（第 3 部分）；止损仍按账本定量（R4 的止损部分）。

## 第 12 步（第 2 部分）：止损触发或被击穿后暂停该品种入场，直到人工解除

技术负责人的决定：止损被击穿并平仓后暂停入场直到人工解除并告警（2026-11-14）；交易所止损正常触发后同样处理（2026-11-15）。

**改了什么（行为变更，已批准）**

- 每次对账时，如果某个品种的止损在这一次被确认触发、或者挂单时被交易所以"价格已越过"拒绝（击穿，此时运行时会市价平仓），运行时就暂停该品种的入场：
  - 暂停记录在状态目录的 `entry_pauses.sqlite`，重启后仍然有效。
  - 只暂停这一个品种；减仓和平仓不受影响。
  - 新暂停时写一条告警（`alert`），里面给出解除命令。状态文件里每个品种多了 `entry_pause`（暂停原因，或 `null`）。
- **只有人能解除**：`python -m trading.infrastructure.entry_pause --state-dir <目录> release --symbol BTCUSDT --reason "<为什么现在可以再交易>"`。
  - 必须写原因（与暂停记录一起保存，记录永不删除）。
  - 运行后要求把品种名再输入一遍才生效。没有跳过确认的参数，所以没有任何自动解除的路径。
  - `status` 子命令列出当前所有暂停。命令不会创建状态目录。
- **失败即阻断**：
  - 暂停记录读不出来时，入场被拒（"risk check failed"），状态显示 `unreadable`。
  - 暂停写不进去时，当次阻断入场并记一条问题，下次对账再写，直到写进去为止。
- 入场被暂停拒绝时，事件日志的原因是 `runner paused`。这是闸门里已有的那条检查（冻结的 Bybit 运行时也用它），这次只是接上了。

**改动的断言（1 处，按决定改变）**

- `tests/characterization/test_runtime_decisions.py` 的 `test_a_stop_that_would_trigger_at_once_flattens_the_position`：
  - 原来断言：击穿平仓后下一根 K 线原样重新入场（仓位回到原数量）。
  - 现在断言：下一根 K 线仍然做了决策，但入场被拒、原因是 `runner paused`，仓位保持为 0。

**新增测试**（`tests/unit/trading/test_entry_pause.py`，9 条用例）

- 存储：暂停一直保留到解除；重复暂停保留第一次的原因；重启后仍在；解除必须有原因；历史保留；文件损坏时报错而不是当作"没有暂停"。
- 命令：输错品种名不解除；原因为空不解除；输对才解除并显示原因；没有暂停时直接说明；`status` 列出暂停；不创建状态目录。
- 运行时：击穿平仓后暂停、告警、只影响这个品种、连续 3 根 K 线不入场；解除后恢复入场。正常触发后暂停，重启后仍然暂停，减仓不受影响。暂停写不进去时阻断入场并在之后写入。暂停记录读不出来时拒绝入场。
- 写测试时又碰到一次 R23 的情形：解除后的那根 K 线上两个品种同时入场，第二个要等一秒后的下一个周期。测试里按生产节奏多跑一个周期。

**变异自检**：【实际运行结果】注入 14 个缺陷（基线通过），第一轮抓住 13 个。漏掉的是"暂停写不进去时不阻断入场"：测试里那一刻入场本来就因为"账本仍有仓位、没有止损"被阻断了，分不出来。测试多跑一个周期（止损成交已入账，只剩暂停没写进去这一个原因）后，这个缺陷被抓住。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试通过。ruff、`mypy trading/`（55 个文件）、lint-imports 通过。全量测试见 PR 的 CI。

**没有做、留给后面的**：告警发到人手上（第 3 部分）。运营手册（M5）要写清解除前该核对什么（交易所持仓、订单账本、为什么止损会触发）。

## 第 12 步（第 3 部分）：告警发到人手上

**改了什么**

- 新增两种告警出口（`trading/adapters/notify/`）：
  - Telegram：设置了 `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID` 两个环境变量时使用（与旧系统已有的变量名相同）。机器人令牌只在发送地址里出现，任何日志都不写它；发送失败只记一条"未送达"和异常类型，交易照常进行。超过 Telegram 单条上限的内容截断。
  - 进程日志：两个变量都没设时使用。只设了一个是配置错误，启动时报错（报错信息里不带变量的值）。
  - 环境变量只在 `trading/infrastructure/settings.py` 里读。
- 新增 `trading/application/alerts.py`：
  - 事件日志里每一条 `alert` 立即发出（级别 critical）。目前写 `alert` 的地方有两个：平仓连续 10 次失败（第 11 步），入场被暂停（第 2 部分）。
  - 对账问题：持续满 2 分钟才发（一闪而过的不发），持续期间每 30 分钟重发一次，消失时发一条"已解决"。
  - 例外：挂单入场模式下挂单本来就会挂着，所以"有未完成订单"这条要持续超过挂单有效期再加 10 分钟才发。
  - 告警发送出错一律只记日志，不影响主循环和平仓。
- 执行模式下，标题带上账户范围（`RISK_SCOPE`），一眼能分清测试网和主网。状态文件多了 `alerts`：`telegram`、`log only`，模拟盘为 `null`。
- 新增参数：2 分钟、30 分钟、Telegram 发送超时 5 秒。都不是风控参数，只决定告警的节奏。

**没有做、留给后面的**

- 行情连续失败的告警在第 14 步。
- 主网执行是否应该在没有配置 Telegram 时拒绝启动：这是一个取舍，留到 M5 上测试网之前问技术负责人。现在是"没有配置就只写日志"。

**已有测试的改动**：没有改任何断言。新增 16 条用例（`tests/unit/trading/test_alerts.py`；更正，第 12b 步后审计：原先写 17 条）。

**变异自检**：【实际运行结果】注入 16 个缺陷（基线通过），16 个全部被抓住，包括：发送地址写错、发送异常外抛、日志里写出令牌、不截断、只设一个变量也接受、每条事件都当告警、通知出错打断事件日志、对账问题立即发出、不限重发频率、不发"已解决"、给没发过的问题发"已解决"、运行时不观察问题、漏掉"没有止损"这一条、挂单模式不放宽等待、执行模式不建出口、运行时不转发告警。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试通过。ruff、`mypy trading/`（59 个文件）、lint-imports（5 条契约，告警适配器加入"适配器互不依赖"）通过。全量测试见 PR 的 CI。

**第 12 步到此完成。** 按路线图，第 12b 步（卡死订单的人工处置工具）之后做一次独立审计。

## 第 12b 步：人工处置工具——卡死订单、账本对齐（只在人执行时才有动作）

技术负责人的批准：处置卡死订单的工具（ADR-0004 第 2 项，2026-11-14）；同一工具里加"人工把账本对齐到交易所"（2026-11-15）。

**改了什么**

- 单写者租约从 `shared/execution_lease.py` 搬到 `trading/adapters/files/lease.py`，旧位置留转发。运行时入口改从新位置导入，入口"还能从旧包拿什么"的清单少了一项（剩 6 项）。
- 订单库新增一张 `manual_actions` 表，以及两个只给人工工具用的操作：把一张未结订单手工结算；把某个策略的账本持仓手工设为给定值。两个操作都在一个事务里同时写一条审计记录（时间、动作、对象、改前、改后、交易所当时的情况、理由），审计记录不改不删。
- 新命令 `python -m trading.infrastructure.manual --state-dir <目录> ...`，三个子命令：
  - `orders`：列出所有未结订单，以及交易所对每一张的说法（还在挂、已成交多少、查无此单、查询失败）。
  - `settle --intent <编号> --reason "..."`：结算一张卡死的订单。
    - 交易所显示仍在挂着：拒绝，请先到交易所撤单。
    - 交易所查无此单：按"交易所不存在"结算；但如果订单库里已经记了这张单的成交，拒绝（前后矛盾，要人查）。
    - 交易所已是终态（成交、撤销、过期、拒绝）：按交易所的状态结算。如果交易所的成交量和订单库记的不一样（例如成交记录被截断的 R9），醒目提示差额没有进账本，并提示接着用 `position` 对齐。**不会补造任何成交。**
    - 查询交易所失败：什么都不改。
  - `position --symbol BTCUSDT --reason "..."`：把该品种策略的账本数量和开仓价设为交易所的值，已实现盈亏和手续费保留。
    - 该品种还有未结订单、或者有还没入账的成交时拒绝（先处理订单；成交要先让运行时跑一次把它入账）。
    - 这不是一笔成交，只是一条调整记录。检查点换一个新版本号，所以运行时下次启动时会按新的持仓重挂止损。
- **三道保护，所有子命令都有**：运行时还在跑（持有这个账户的租约）时拒绝，先停运行时；必须写理由；执行前把订单编号或品种名再输入一遍，没有跳过确认的参数。命令不会创建状态目录或订单库。
- 架构测试：只有这个命令文件调用那两个手工操作；运行时调用不到。
- 凭证、账户范围、测试网开关与运行时用同一组环境变量。读取放在 `trading/infrastructure/settings.py`（新增 `testnet_from_env`；`binance_config_from_env` 可传入环境）。

**已有测试的改动**

- `tests/characterization/test_risk_state.py` 的 R5 测试：替换租约的位置从旧模块改为运行时入口（入口现在从新位置导入）。断言不变。这和第 10 步搬 `build_adapters` 时的处理相同。
- `tests/architecture/test_rules.py`：入口允许导入的旧模块清单去掉了 `shared.execution_lease` 一项（更正，第 12b 步后审计：这一条原先只在上面"改了什么"里提到，漏写在这里）。

**新增测试**（`tests/unit/trading/test_manual_tools.py`，13 条用例）：查无此单的订单确认后结算并留痕；输错确认不改；仍在挂的不结算；交易所已成交但订单库漏记的结算并提示、不补成交；交易所不认识但订单库有成交的不结算；订单库拒绝非终态和空理由、已结算的不再结算；列表显示交易所的说法；账本对齐后对账一致并留痕；交易所空仓时账本归零、开仓价归零；有未结订单或未入账成交时不对齐；运行时持有租约时拒绝；理由为空拒绝；缺状态目录或订单库时拒绝且不创建；只有命令文件调用手工操作。

**变异自检**：【实际运行结果】注入 15 个缺陷（基线通过），15 个全部被抓住：结算不确认、结算仍在挂的订单、漏记成交不提示、交易所不认识却有成交也结算、不取租约、空理由放行、缺订单库时创建、对齐不确认、有未结订单也对齐、对齐丢掉已实现盈亏、空仓保留开仓价、订单库接受非终态、订单库忽略未入账成交、检查点版本不变、结算不留审计记录。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过。ruff、`mypy trading/`（61 个文件）、lint-imports 通过。全量测试见 PR 的 CI。

**一处偏离工作约定**：提交 `2241e9e` 把命令（约 190 行）和它的测试（约 240 行）放在了一起，共 430 行，超过"每次提交少于 400 行"。拆分需要改写已推送的分支，没有做；如实记在这里。

**下一步**：按路线图，第 12b 步之后做一次独立审计（范围：第 11、12、12b 步），审计结论处理完再开始第 13 步。

## 第 12b 步之后的独立审计（审计对象 `860e915`，范围 `ddaa68c..860e915`）与处理

审计员在自己的副本里工作，没有改动仓库。

**没有被推翻的**：同一意图不会发两次；平仓不会超过交易所持仓；排队的入场不会在 10 分钟之后、新 K 线之后或平仓之后发出；急停、回撤、暂停、无止损、未结减仓时不会入场；没有任何自动解除暂停、结算订单、改写账本的路径；人工命令确实要求理由、确认和租约，也不补造成交；Telegram 令牌不写进日志；我列出的断言改动与 git 一致。

**发现与处理**

| 编号 | 严重度 | 发现 | 我是否复现 | 处理 |
|---|---|---|---|---|
| 1 | 高（故障时） | 一笔始终无法入账的成交会让急停平仓被订单库拒绝（"有成交未入账"），平仓后的入账调用没被拦住，异常冒出主循环，进程反复退出重启，告警计数每次清零、永远发不出去。与 ADR-0005 子决策 7b"急停平仓不依赖账本检查点"的写法矛盾 | 【实际运行结果】12 个周期里 10 个抛异常，交易所持仓保持 -0.15，订单库拒绝 10 次 | PR 第 1 部分 |
| 2 | 中 | 止损正常触发后，如果暂停没写进去就重启，暂停丢失，入场放行 | 【实际运行结果】重启后暂停为空 | PR 第 2 部分 |
| 3 | 中 | 止损在运行时停机期间触发，人工把账本设为 0，重启后止损成交又入账，账本变成反向的 +0.15 | 【实际运行结果】账本 0.15、交易所 0 | PR 第 3 部分 |
| 4 | 中低 | 账本与交易所不一致时急停平仓，平仓后账本变成反向仓位，止损管理器还会给这个不存在的仓位挂一张全平止损 | 审计员运行确认 | PR 第 2 部分 |
| 5 | 中低 | "等待订单"的止损在交易所随后显示过期等结束状态时也一直等，持仓一直没有止损 | 审计员运行确认 | PR 第 2 部分 |
| 6 | 中（原有行为） | 急停不要求平仓时，挂着的入场单不撤，照样成交 | 审计员运行确认 | PR 第 1 部分 |
| 7 | 中低 | `RISK_SCOPE` 前后带空格时，运行时的租约用原值、人工工具用去空格的值，工具察觉不到运行时在跑 | 审计员运行确认 | PR 第 1 部分 |
| 8 | 中低 | 暂停库被删除或为空文件时读作"没有暂停" | 审计员运行确认 | PR 第 2 部分 |
| 9 | 低 | 告警在主循环里同步发送，Telegram 慢时会推迟同一周期里的急停平仓 | 只读 | PR 第 1 部分 |

审计员注入 22 个缺陷，15 个被抓住，7 个没有：紧急平仓读不到持仓时改按账本平；平仓器自己的异常保护被缩窄；结果不明的止损被当作有效保护；交易所给出订单号但查不到订单时当作已收编；查询交易所失败时人工结算仍按"不存在"结算；有多个持仓时不拒绝；订单列表遇到查询失败就崩溃。分别在对应部分补测试。

**记录里的不准确之处**（第 3 部分一并更正）：R3 测试的说明写"每次同步都记止损被拒"，新测试只检查了一次；`test_flatten.py` 实际 15 条用例（记录写 17），`test_alerts.py` 实际 16 条（记录写 17）；第 12b 步"已有测试的改动"漏写了架构测试里清单少了一项。

### 处理第 1 部分：退出路径（发现 1、6、7、9）

- **发现 1**：
  - 急停和回撤的平仓（按交易所持仓全平）在订单库预留时不再受"有成交未入账"和"检查点版本不符"两条规则限制，"已有未结减仓"仍然限制。这是 ADR-0005 子决策 7b 原本的规定，之前没有做到。
  - 平仓发出后的成交入账失败只记问题，不再抛出。
  - 运行时对每个品种的平仓（急停、回撤、止损击穿）都单独拦住异常：记一条问题，下一个周期继续；这条问题持续 2 分钟后会发告警。一个品种平仓出错不影响另一个品种。
- **发现 6**：急停（不论是否要求平仓）或品种被暂停时，每个周期撤掉该品种挂着的入场单，不平仓。理由：这两种状态都表示"不允许新增敞口"，挂单成交就是新增敞口。这是行为变更，但只朝更保守的方向。
- **发现 7**：运行时入口取租约时改用与订单库、人工工具相同的、校验并去掉空格的账户范围。
- **发现 9**：运行时用的 Telegram 出口改为后台发送：一个有界队列加一个发送线程，主循环从不等待；队列满时丢弃新告警并写日志。
- 补测试：平仓器自己的异常保护（撤单出错时照样平仓）；读不到交易所持仓时什么都不发。
- **已有测试的改动**：没有改任何断言。新增 `tests/unit/trading/test_exit_robustness.py`（11 条用例；更正：原先写 12 条，按 pytest 实际收集数）。
- **变异自检**：【实际运行结果】注入 13 个缺陷（基线通过），第一轮抓住 11 个。漏掉的两个都是我的断言太粗：外层保护也能拦住异常，所以只看"没有崩溃"分不出内层保护在不在；服务内部再读一次持仓也会失败，所以订单数不变。改为检查记下的问题文字、检查有没有写出平仓意图后，两个都被抓住。
- **本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过；ruff、`mypy trading/`、lint-imports 通过。

### 处理第 2 部分：止损与暂停的持久性（发现 2、4、5、8）

- **发现 2、8：暂停改为从持久记录推导。**
  - 订单库里的止损记录本身就是持久的。现在三种记录都算"止损被穿过"：已触发、等待订单、被击穿；另外新增一种"触发后没有形成订单"。被击穿以前记成普通的"被拒"，现在单独记为"被击穿"。
  - 暂停库新增一张表，记下每个暂停是为哪些止损记录设的。每次对账，凡是还没被任何暂停记下的止损记录，就为它设暂停。
  - 结果：暂停没写进去、写之前进程重启，下次对账都会补上；暂停库被删除或清空，所有出过止损的品种重新暂停，要人再解除一次。不会因为丢了文件而放行。
  - 原来放在内存里的"待写入暂停"和止损管理器的两个标志都去掉了。
- **发现 5**：止损处于"等待订单"时，如果交易所随后显示它已结束（撤销、过期、拒绝）且没有订单，就不再等，记为"触发后没有形成订单"（仍然算触发，仍然暂停），并为仍然存在的持仓补挂新止损。
- **发现 4**：挂新止损前先读交易所持仓。交易所没有持仓、或者方向与账本相反时，不挂止损，只记问题；读不到持仓时也不挂。这时入场照样被"没有止损"规则阻断。理由：全平止损挂在交易所没有的仓位上，会留在交易所，以后可能按过时的价格平掉新开的仓位。方向相同但数量不同时照常挂，因为全平止损平的是交易所的全部持仓。
- 补测试：结果不明的止损下单不算有效保护（审计员的缺陷 S24）；交易所给出了订单号但查不到订单时，不当作已收编（ST1）；不是"击穿"的拒绝不算触发。
- **已有测试的改动（都只改准备步骤，没有改任何断言）**：
  - `tests/unit/trading/test_venue_boundary.py`：只说领域词汇的假交易所补了一个读持仓的方法，返回与账本一致的多头 1。
  - `tests/unit/binance/test_binance_risk.py` 的两条止损测试（`test_stop_follows_position_lifecycle`、`test_short_stop_and_breach_reporting`）：原来只在账本里写持仓、交易所上是空仓，新规则下不会挂止损，两条测试失败。现在在准备步骤里让交易所持有与账本相同的仓位（0.1、0.2、0、-0.1）。断言（止损方向、触发价、换单、空仓撤单、击穿上报）一字未改。同一文件第三条测试本来就让交易所持有仓位，不受影响。
  - 迁移计划第 12 步写过"这个文件的止损测试不改断言通过"，这一条仍然成立；但准备步骤改了，这里如实说明：原来的准备步骤恰好描述了审计第 4 条指出的不安全情形（账本有仓、交易所没有）。
- **新增测试**：`tests/unit/trading/test_stop_durability.py`，9 条用例。
- **变异自检**：【实际运行结果】注入 12 个缺陷（基线通过），12 个全部被抓住。

### 处理第 3 部分：人工工具与记录（发现 3，以及审计员没被抓住的三个缺陷）

- **发现 3**：`position`（把账本对齐到交易所）在以下情况拒绝，什么都不改：
  - 该品种有一张止损处于"等待订单"；
  - 交易所显示某张存活的止损已经触发（运行时停着，还没收编它的成交）；
  - 查不到某张存活止损的状态。
  - 提示做法：先启动运行时一次，让它把止损的成交收进账本，停下后再看是否还需要对齐。
  - 交易所上仍在挂着的止损不构成拒绝理由：对齐会换一个新的检查点版本，运行时下次启动时撤掉旧止损、按新持仓重挂。
- 补测试：查询交易所失败时人工结算什么都不改（审计员缺陷 MA6）；交易所有多个持仓时拒绝对齐（MA7）；订单列表遇到查询失败时照常列出并注明（MA11）。这三处代码原本就是对的，只是没有测试。
- **记录更正**：R3 测试的说明、`test_flatten.py` 与 `test_alerts.py` 的用例数、第 12b 步漏写的一处测试改动、第 1 部分 `test_exit_robustness.py` 的用例数，都已在原处加更正说明。
- **新增测试**：`tests/unit/trading/test_manual_tools.py` 增加 4 个用例（共 17 个）。没有改任何已有断言。
- **变异自检**：【实际运行结果】注入 4 个缺陷（基线通过），4 个全部被抓住。

**审计处理到此完成。** 9 条发现全部处理；审计员没被抓住的 7 个缺陷都有了对应的测试。下一步：第 13 步。

## 第 13a 步：所有部件用同一个注入的时钟

**改了什么**：组装执行会话时，订单生命周期、执行服务、止损管理、回撤闩锁都改用运行时的时钟（之前它们读系统时钟，运行时读注入的时钟）。生产环境两者是同一个时钟，所以生产行为不变。下单后的等待和成交延迟的测量仍用真实时间。

**为什么**：M2 审计和我自己的测量都指出了这一点。时钟不一致时，"查无此单"的 90 秒宽限、挂单有效期、回撤的按日重置在测试里不会随模拟时间推进；将来做"模拟盘与回放对比"时也会不一致。

**已有测试的改动（不改断言）**：
- `tests/unit/trading/test_flatten.py` 的 `test_an_attempt_whose_outcome_is_unknown_is_never_followed_by_another`：模拟时间从 124 秒缩到 62 秒（超过 30 秒重试间隔，在 90 秒宽限之内）。原来走 124 秒仍然通过，只是因为宽限按真实时间算、测试实际只跑了几秒，恰好依赖了这个时钟缺陷。统一时钟后，按设计 90 秒后会结算为"查无此单"并再试一次，新测试专门验证了这一点。
- 其余运行时测试不受影响。

**新增测试**（`tests/unit/trading/test_one_clock.py`，2 条）：每个部件读到的都是运行时的时钟；交易所从未收到的订单在模拟时间满 90 秒后结算为"查无此单"，随后的下一次尝试平掉仓位。

**变异自检**：【实际运行结果】把 4 个部件逐个改回系统时钟，4 个都被抓住。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/unit/binance`、契约测试通过。

## 第 13b 步：限额器改用 Python 实现并按 UTC 日落盘；回撤阈值的硬上限

依据：ADR-0008 子决策 1（新包不依赖 Rust，已批准）；ADR-0007 子决策 2b（日限额落盘、按 UTC 自然日、读不出来按额度用完处理，已批准）；回撤硬上限（技术负责人 2026-11-15：日 10%、周 25%）。

**改了什么**

- 下单限额的规则搬进 `trading/domain/risk/order_limits.py`，是一个纯函数：检查顺序和拒绝原因的写法与原来的 Rust 限额器相同（单笔数量、单笔名义额、下单后持仓名义额、最近一秒笔数、当日笔数、当日名义额，第一条不满足的说了算）。
- `trading/application/risk/order_limiter.py`：限额器本身。当日笔数和名义额存进状态目录的 `order_limits.sqlite`，按账户范围和 UTC 日期分开计。
  - 重启不再清零（修 R12 的后一半）。
  - "日"从"进程启动后的滚动 24 小时"改为 UTC 自然日（已批准的定义变更）。
  - 计数读不出来、或者放行后写不进去：拒绝这笔下单，按"当日额度已用完"处理。
  - 最近一秒的窗口仍在内存里、用真实的单调时钟，与原来相同；重启最多忘掉一秒。
- 运行时不再用旧包里的 Rust 限额器。入口"还能从旧包拿什么"的清单少了一项（剩 5 项）。限额数值一个都没变。
- **回撤硬上限**：运行时读取回撤阈值时检查，日超过 10%、周超过 25%、或者不是正数，启动时报错。启动检查（`preflight`）里也读一次，所以在连接账户之前就拒绝。默认值仍是日 5%、周 12%。修 R12 的前一半。
  - 上限放在运行时读配置的地方，没有放进底层的回撤类，原因是两条现有测试用"重启时换成更宽的阈值（日 20%、周 30%）"来证明"放宽不会解除当前周期"，放进底层就得改它们的数据。冻结线（马上要删）也不受影响。

**改动的断言（2 个已知缺陷测试，`tests/characterization/test_risk_state.py`）**

- R12 前一半：原来断言回撤类接受日 99.9%、周 99.9%；现在断言读取配置时拒绝 99.9%，接受正好 10% 和 25%。测试名改为 `test_a_drawdown_limit_above_the_hard_ceiling_refuses_to_start`。
- R12 后一半：原来断言新建的限额器从零开始（第三笔被拒，重启后又放行）；现在断言重启后仍然拒绝。测试名改为 `test_the_daily_order_count_survives_a_restart`。

**其它测试改动**：`tests/unit/trading/test_runtime_wiring_limits.py` 里的配置类改从新模块导入（字段和数值相同，断言不变）；`tests/architecture/test_rules.py` 的入口旧包清单去掉一项。

**新增测试**（`tests/unit/trading/test_order_limits.py`）：
- 与 Rust 限额器对拍：3 组、每组 80 笔随机订单，逐笔比较是否放行和拒绝的规则，覆盖每一条规则。
- 每秒笔数的窗口边界与 Rust 相同（恰好 1 秒时移出窗口）；只有放行的订单计数。
- 重启保留当日计数；UTC 零点切换；按账户范围分开；读不出来或写不进去时拒绝。
- 回撤上限的边界；启动检查在连接账户前拒绝；运行时把计数写进自己状态目录、自己账户范围下。

**变异自检**：【实际运行结果】注入 12 个缺陷（基线通过），第一轮抓住 11 个。漏掉的"当日名义额等于上限也拒绝"其实有测试（第二笔 250 恰好凑满 500 仍放行），是我给它配错了测试（配成了随机对拍，随机数碰不到恰好相等）；换成边界测试后被抓住。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过；ruff、`mypy trading/`（64 个文件）、lint-imports 通过。

## 第 13c 步：受管状态目录；急停文件路径单一来源；运行时不写急停文件

依据：ADR-0007 子决策 1c；ADR-0005 子决策 5b、6b（都已批准）。

**改了什么**

- **受管状态目录**（`trading/infrastructure/state_dir.py`，修 R6）：
  - 由人运行一次 `python -m trading.infrastructure.state_dir --state-dir <目录> init`：建好订单库、回撤库、限额库、暂停库，写一个未激活的急停文件，最后写清单 `manifest.json`（记账户范围、是否测试网、建立时间）。已初始化的目录不会被覆盖。
  - 执行模式的启动检查（`preflight`）在连接账户之前核对：清单存在、账户范围与清单一致、每个库都在且能以只读方式打开、急停文件在且能读。任何一项不满足就拒绝启动，并给出初始化命令；整个过程不创建任何文件。
  - 模拟盘不受影响。
- **急停文件**（修 R5）：
  - 路径只由一个函数从状态目录给出，运行时和命令共用。
  - 新命令 `python -m trading.infrastructure.kill --state-dir <目录> activate|clear|status`，必须指定状态目录、目录必须已初始化；没有默认路径。
  - 新的读取类 `trading/adapters/files/kill_switch.py`：在受管目录里，急停文件运行中消失按"激活、不平仓"处理；损坏仍按激活处理。
  - 运行时只读急停文件。架构测试保证：新包和运行时入口里，只有急停命令和初始化命令会写它。
  - 旧的 `shared/kill_file.py` 留给已关闭的 ML 运行时，冻结线删除时一起删。
- 入口"还能从旧包拿什么"的清单少了一项（剩 4 项：验签、研究撮合 2 项、成交成本日志）。

**为什么测试用的临时目录不受影响**：运行时只在受管目录（有清单）里按"缺失即激活"读急停文件。测试在临时目录里建运行时，没有清单，保持原语义，所以 10 处建运行时的测试都不用改。生产入口则必须先通过启动检查，目录一定是受管的。

**改动的断言（2 个已知缺陷测试，`tests/characterization/test_risk_state.py`）**

- R6：原来断言改了账户范围或删掉回撤库后，回撤类静默新建一个空库、闩锁消失。现在断言启动检查拒绝：账户范围与清单不符时报"属于另一个范围"，回撤库被删时报"缺少"，并且不创建文件。回撤类本身仍会新建缺失的库，拒绝发生在启动时、它被创建之前。测试名改为 `test_execution_refuses_to_start_when_a_store_is_missing_or_the_scope_changed`。
- R5：原来断言命令的默认路径是另一个运行时的目录。现在断言命令必须指定状态目录、运行时读的路径就是命令写的路径、受管目录里文件消失按激活处理。测试名改为 `test_the_kill_command_writes_exactly_where_the_runtime_reads`。

**其它测试改动（只改准备步骤）**：审计第 1 部分加的入口测试（`test_exit_robustness.py` 里取租约用同一账户范围那条）现在先初始化状态目录，因为执行模式要求受管目录。断言不变。

**新增测试**（`tests/unit/trading/test_state_dir.py`）：初始化的内容；不覆盖；缺账户范围时不建任何东西；5 种损坏各自被拒；账户范围不符被拒；启动检查不创建文件；受管目录里急停文件消失按激活；命令写、运行时读的是同一个文件；命令要求已初始化的目录；只有命令写急停文件；生产入口从未初始化的目录启动时拒绝且什么都不创建。

**变异自检**：【实际运行结果】注入 10 个缺陷（基线通过），第一轮里"入口不把状态目录交给启动检查"一项原本没有测试能抓（唯一走入口的测试用的是已初始化的目录），写之前补了"从未初始化目录启动被拒"的测试，之后 10 个全部被抓住。

**本地回归**：特征测试、两份运行时测试、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过；ruff、`mypy trading/`（68 个文件）、lint-imports 通过。

**部署影响**：目前没有以执行模式运行的部署。将来第一次以执行模式启动前，要先运行一次初始化命令（写进 M5 的运行手册）。

## 第 13d 步：权益未知时只做"整仓平掉"这一种减仓（R13）

技术负责人 2026-11-15 决定：权益未知时允许减仓（按持仓定量，不依赖权益），不允许入场。

**改了什么**

- 执行模式的会话里，权益未知（读取失败或已过期，按 0 处理）时，不再走正常的调仓计划，改为：
  - 目标仓位为 0，或者与现有持仓方向相反：把已确认的持仓整个平掉（只减仓单）。
  - 目标仓位与现有持仓同方向（包括缩小）：算不出该减多少，不下单。
  - 无论如何都不入场。
- 规则写在领域层 `trading/domain/rebalance.py` 的 `close_without_equity`。研究用的调仓计划 `plan` 没有改，已登记研究和模拟盘的结果不受影响（`tests/unit/lab` 全部通过）。
- 背景：权益读不到时，回撤闩锁只阻断入场、不触发平仓；原来策略自己的调仓计划因为权益为 0 也什么都不下，所以目标已经要求平仓时仓位还挂着。

**改动的断言（1 个已知缺陷测试，`tests/characterization/test_exit_paths.py`）**

- R13：原来断言"权益为 0 时调仓计划不产生任何减仓"。现在研究计划的两条断言原样保留（研究计划确实没变），新增一条：执行会话在权益未知、目标为 0 时整仓平掉。测试名改为 `test_with_equity_unknown_the_execution_session_still_closes_a_position_the_target_no_longer_wants`，去掉已知缺陷标记。

**新增测试**（`tests/unit/trading/test_unknown_equity.py`，9 条）：6 种目标与持仓组合；运行时在权益过期后，目标为 0 或反向时平掉仓位、同方向缩小时不下单、全程没有任何入场单。

**变异自检**：【实际运行结果】注入 5 个缺陷（基线通过），5 个全部被抓住。

**本地回归**：特征测试、`tests/unit/lab`（整个目录，含模拟盘与研究逐笔一致）、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过；ruff、`mypy trading/`、lint-imports 通过。

**第 13 步到此完成**（13a–13d）。剩下的已知缺陷：R9（一张订单超过 100 笔成交时结算不了，第 14 步一并修）和 R20（同一账户可以在两个目录各起一个运行时，ADR-0007 写明"后续单独处理"）。

## 第 14a 步：行情失败告警；交易所判定为终局的行情错误；预热阶段的行情失败

依据：迁移计划第 14 步；ADR-0007 子决策 4b（已批准）。

**改了什么**

- **连续失败告警**：行情请求连续失败 5 次（按退避约 2.5 分钟）时写一条告警（经告警出口发出）；同一次中断只告警一次；成功一次计数清零，下一次中断再告警。新增参数：5 次。
- **终局错误**：交易所以"不可重试"拒绝（例如品种下线、参数错误）时：
  - 立即告警；
  - 每次对账阻断入场并记问题，直到行情再次成功；
  - 仍然按最长间隔（300 秒）重试，以便恢复。
- **预热阶段**：执行模式启动时，如果行情接口不可用，不再直接退出进程。运行时先刷新权益、对账、维护止损、执行风险动作（急停平仓、回撤平仓），然后按退避重试预热。每次重试前清空上一次没做完的预热留下的 K 线。模拟盘仍按原来的方式报错退出。
  - 此时还没有任何 K 线，平仓意图的编号用当前小时命名，参考价记为 0（只减仓单不需要参考价）。
- 入口的主循环改为反复调用"预热或保持"，直到预热完成。

**已有测试的改动**：没有。`test_target_runtime.py`、`test_target_runtime_resilience.py` 不改断言通过（迁移计划的验收要求）。

**新增测试**（`tests/unit/trading/test_market_failures.py`，5 条）：连续 5 次失败恰好一次告警、恢复后清零、再次中断再告警；终局错误立即告警、两次对账都阻断入场、恢复后放行；预热时行情不可用，急停平仓照样执行，行情恢复后预热完成；预热只做了一半时，重试从头开始；模拟盘预热失败仍报错。写测试时我自己犯了一处错：恢复后没有让下一根 K 线到期，运行时根本不请求行情，第二次中断自然不会失败，已改正。

**变异自检**：【实际运行结果】注入 10 个缺陷（基线通过），10 个全部被抓住。入口主循环改用"预热或保持"这一行没有自动测试（要真实启动进程），只做了代码阅读核对。

**本地回归**：特征测试、`tests/unit/lab`、`tests/unit/trading`、架构测试、`tests/offline`、`tests/unit/binance`、契约测试、`tests/unit/bybit`、`tests/unit/execution` 通过。

## 第 14b 步：一张订单的全部成交分页读完（R9）

**改了什么**：Binance 适配器查询某一张订单的成交时，每页取 1000 笔（交易所的上限）；取满一页就从这一页最大的成交编号加一处接着取，直到某页不满为止。超过 100 页（10 万笔）就报错，订单保持未定、继续阻断入场，不会按一部分成交结算。不指定订单的查询仍是一次请求，与原来相同。订单生命周期没有改，所以冻结的 Bybit/OKX 不受影响。

**改动的断言（1 个已知缺陷测试，`tests/characterization/test_stops_and_fills.py`）**

- R9：原来断言 150 笔成交的订单永远结算不了、账本停在 0.15、下一张减仓在下单前就被拒。现在断言订单结算、账本与交易所都是 0.2、随后 200 笔成交的平仓单把仓位平到 0、对账无问题。测试名改为 `test_an_order_with_more_than_100_trades_settles_and_reductions_go_through`。

**测试辅助的改动**：假交易所的成交查询支持"从某个编号开始"（`fromId`），并记下每次请求，以便测试检查分页。

**新增测试**（`tests/unit/trading/test_trade_pages.py`，3 条）：2500 笔成交的订单分 3 页读完并结算、每页 1000 笔、适配器返回的成交恰好 2500 笔且不重复；超过页数上限时订单保持未定并报错；不指定订单的查询仍是一次请求。

**变异自检**：【实际运行结果】注入 4 个缺陷（基线通过），第一轮漏 1 个："翻页时从最后一笔本身开始"。订单生命周期会按成交编号去重，所以结算结果没变，但适配器会返回重复成交；补上"恰好 2500 笔、不重复"的断言后被抓住。

**第 14 步到此完成。** 剩下的已知缺陷只有 R20（同一账户可以在两个目录各起一个运行时；ADR-0007 写明后续单独处理）。

