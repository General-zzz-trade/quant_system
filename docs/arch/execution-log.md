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
- 【实际运行结果】注入缺陷自检：对搬过去的会话和主循环注入 15 个缺陷，运行中，结果在本 PR 的后续提交里补上。
- 本地：ruff、`mypy trading/`（47 个文件）、lint-imports（5 条契约）、离线安全检查、架构测试全部通过。全量测试见 PR 的 CI。

**第 10 步还没做的（下一个 PR）**

- 模拟盘状态改存 SQLite，同时保留 JSON 文件。
- 运行时改用新的执行服务和 `OrderIntent`。现在组装处仍然构造旧的执行适配器和旧的订单事件，所以加载运行时仍会带入旧包。
- "加载运行时后没有旧包"的隔离测试，以及对整条运行时生效的 `no-legacy` 契约。这两条要等上一条做完。
- 实盘路径与研究的对比测试。
