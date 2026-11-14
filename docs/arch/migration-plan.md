# 迁移计划（阶段 3）

- 依据：`docs/arch/current-state.md`、`docs/arch/target.md`、`docs/adr/0001`–`0008`。
- 状态：**已批准执行**（2026-11-14）。各子决策的批准范围以对应 ADR 的状态行为准，未答的子决策对应的工作不执行。
- 共 15 步。每一步只做一件事，做完系统可运行，全量测试必须全绿。

## 0. 约定

### 0.1 顺序

特征化测试（第 1 步）→ 门禁与抽接口（第 2–3 步）→ 迁移实现（第 4–10 步）→ 修复退出侧（第 11–14 步）→ 删除旧代码（第 15 步）。

迁移实现阶段先做资金风险相关模块：风控（4）、下单（5、6）、对账与止损（9），然后才是策略函数和入口。

### 0.2 行数口径（需要您确认）

"每个改动少于 400 行"在本计划里按**手写的新增与修改行数**计，包含测试代码。两类改动单独计数、单独列出：

- **纯移动**：文件内容不变地换位置，旧位置留一个只做重导出的薄壳。
- **纯删除**：只删文件，不改其他代码的行为。

如果您不接受这个口径，第 7–10 步和第 15 步需要拆得更细，总步数会超过 15。

### 0.3 行为变更

- 第 1–9 步不改变任何运行时行为。验收标准里都包含"现有测试不改一行断言地通过"。
- 第 10 步有一处启动顺序的调整。第 11–14 步是行为变更，各自对应 ADR 里的子决策，**只有对应子决策被批准才执行**；未批准的步骤跳过，不影响其他步骤。
- 特征化测试固定的是现状，包括缺陷。它的断言只允许在已批准的行为变更步骤里修改，并且在该步骤的汇报里逐条列出改了哪个断言、对应哪个子决策。其他任何时候都不改测试断言。

### 0.4 每一步的固定流程

1. 先补或确认特征化测试，运行并贴出原始输出。
2. 实施改动。
3. 运行全量测试、lint、架构规则，贴出原始输出。
4. 汇报：改了什么、没改什么、遗留风险、下一步建议。

每一步的通用验收（下面不再重复）：

```bash
python3 -m pytest tests/ --ignore=tests/performance -q      # 不少于基线的 2920 通过，0 失败
ruff check --select E,W,F .
python scripts/check_local_safety.py
cargo test                                                   # 仅当改动涉及 Rust 时
lint-imports                                                 # 第 2 步之后
```

通用回滚：每一步是一个提交，`git revert` 即可。有状态迁移或部署动作的步骤另外写明。

**2026-11-14 修订（技术负责人批准，原因是同一套测试每个 PR 在本地和 CI 各跑一遍，等待过长）：**

- **全量测试只在 CI 运行。** CI 的 `python-test-all` 任务把 `tests/` 下除 `tests/performance` 以外的全部测试分成 4 片同时运行。本地只跑与改动相关的测试，加上 ruff、`mypy trading/`、`lint-imports` 和架构测试。执行记录里贴 CI 的结果。
- **一个步骤一个 PR。** PR 里分多次小提交，每次提交仍然少于 400 行手写、只做一件事。
- 不变的规则：合并前 CI 全部检查必须通过；先写测试再搬代码；资金路径的改动做注入缺陷自检和新旧行为对比。

### 0.5 自检审计

每 3 步做一次（第 3、6、9、12、15 步之后），按 `target.md` 第 7 节的不变量清单逐条核对，不修改代码。

2026-11-14 修订：第 3 步之后的审计与第 4–6 步并行进行，没能起到放行作用（审计报告 H2）。从第 6 步之后的审计起，审计结论出来并处理完之前，不开始下一步代码改动。

### 0.6 补查后追加的测试要求（2026-11-14）

测试质量审查的变异实验里存活的缺陷，在搬动对应代码的那一步之前补上能杀死它的测试（`current-state.md` 5.5）：

| 步 | 必须先补的测试 |
|---|---|
| 4 | `latched()` 出错时按已闩锁处理；更宽的配置不能放松当前周期；敞口计入未结入场意图；kill 文件 `active` 非布尔值按激活处理；入场门实际组合（就绪门 + 组合门 + 敞口门 + 限额器）的拒绝原因 |
| 5、6 | 订单身份不符时不入账；无效成交事实（空 trade_id、价格 0、数量 NaN）逐条被拒；成交查询的 `limit` 被遵守（假交易所需要支持截断） |
| 9 | 减仓数量不超过自有仓位；对账不一致、存在非本系统挂单时阻断入场；`TargetFillConsumer` 拒绝与落库事实不符的事件；空头、平到零、翻转的持仓记账；止损下单结果不明与撤销中状态的处理 |
| 10 | 回撤闩锁触发平仓；`stop_breached` 触发平仓；不叠加入场；K 线处理过晚、资金费未发布时不入场；权益过期时按 0 处理 |

另外，第 2 步修正 `tests/unit/lab/test_target_runtime.py` 的路由夹具（不带 `symbol` 的请求要发到正确的品种），并补 ETH 侧断言。这是修测试夹具，不是改断言；修正后测试更严格。
第 7 步把因果测试换成随机切点、覆盖全部小时相位、随机替换未来数据的性质测试，并附变异自检。

## 1. 步骤一览

| 步 | 内容 | 类别 | 手写行数 | 纯移动 / 删除 | 行为变化 | 依赖的决策 |
|---|---|---|---|---|---|---|
| 1 | 特征化测试：退出路径与订单生命周期 | 测试 | 约 380 | — | 无 | — |
| 1b | 纸面运行不因行情失败退出；纸面状态落盘（2026-11-14 批准提前） | 修复 | 约 140 + 审计后补约 30 | — | **有** | ADR-0007 子决策 3、4 |
| 2 | 门禁与架构规则骨架 | CI | 约 150 | — | 无 | ADR-0001、0002 |
| 3 | 端口定义与契约测试 | 抽接口 | 约 320 | — | 无 | ADR-0003 |
| 4 | 风控规则与存储分离 | 迁移 | 约 370 | 约 260 | 无 | — |
| 5 | 订单状态机显式化 | 迁移 | 约 260 | — | 无 | ADR-0004 |
| 6 | 订单库单一所有者 | 迁移 | 约 380 | 约 110 | 无 | ADR-0004 |
| 7 | 领域层：策略、再平衡、持仓、时间语义 | 迁移 | 约 90 | 约 420 | 无 | — |
| 8 | Binance 适配器迁入 `trading/adapters` | 迁移 | 约 260 | 约 800 | 无 | ADR-0008 子决策 1 |
| 9 | 下单生命周期、对账与止损迁入 `trading/application` | 迁移 | 约 300 | 约 780 | 无 | — |
| 10 | 会话、主循环与组装根迁入；闸门先于账户写操作 | 迁移 | 约 350 | 约 400 | 启动顺序 | — |
| 11 | 急停平仓可靠化 | 修复 | 约 390 | — | **有** | ADR-0005 子决策 1、2、3、7 |
| 12 | 止损失败处置与告警出口 | 修复 | 约 300 | — | **有** | ADR-0005 子决策 4 |
| 12b | 卡死订单的人工处置工具（2026-11-14 批准追加） | 新增 | 约 150 | — | 无（只有人执行时才有动作） | ADR-0004 "需要您确认"第 2 项 |
| 13 | 受管状态目录与急停路径单一来源；下单日限额落盘（2026-11-14 批准追加，约 120 行） | 修复 | 约 330 + 120 | — | **有** | ADR-0007 子决策 1、2；ADR-0005 子决策 5、6 |
| 14 | 行情连续失败告警；预热阶段的行情失败处理 | 修复 | 约 150 | — | **有** | ADR-0007 子决策 4 |
| 15 | 删除被取代的旧代码与薄壳（第一批）；之后按批删除不可达模块、`rust_bin/`、4 个直接下单的研究脚本 | 删除 | 约 60 | 删除约 2,900，后续各批另计 | 无 | ADR-0004；ADR-0008 子决策 2、4、5（2026-11-14 批准）。Bybit / OKX / Polymarket 代码保留 |

行数是估计值，实施时如果某一步超过 400 行手写改动，我会停下来报告并提出拆分方式，不会硬做。

**2026-11-14 追加批准的三项（技术负责人）：**

- **卡死订单的人工处置工具**（第 12b 步）：命令行工具，显示交易所实际情况，操作员确认并写明理由后才把订单标为已处置，留痕。程序不自动执行。
- **下单日限额落盘**（并入第 13 步）：计数存进 SQLite，键是 scope 加 UTC 日期；读不出来时按额度已用完处理。这改变了"日"的定义（从进程启动后的滚动 24 小时改为 UTC 自然日），限额数值不变。
- **旧代码**（第 15 步及之后各批）：分批删除不可达模块、`rust_bin/` 和 4 个直接下单的研究脚本；每批删除前列清单，删除后 CI 全量测试必须全绿。Bybit / OKX / Polymarket 代码继续冻结保留，只做安全修改。4 个直接下单的脚本风险性质不同，放在第一批。

## 2. 逐步说明

### 第 1 步：特征化测试——退出路径与订单生命周期

- **改动范围**：新增 `tests/characterization/`。不动任何生产代码。
  - 把 `docs/arch/evidence/` 的 4 个探针改写成 pytest：平仓失败后同一根 K 线不重试（R1）；挂单阻塞 kill 平仓且入场单继续成交（R2）；止损被拒后仍允许入场（R3）；止损触发无订单号时成交不入账（R8）；超过 100 笔成交不结算（R9）；同 scope 不同目录可同时取得租约（R20）。
  - 订单状态转移：对现有测试覆盖的所有流程，记录实际发生的"从 → 到"状态对，固定为一张表。第 5 步的转移表以它为依据。
  - 风控现状：回撤库缺失或 scope 改名后闩锁消失（R6）；kill 命令行默认路径（R5）；权益为 0 时 `plan` 返回空（R13）；`OrderLimiter` 新实例从 0 计数（R12）；主网闸门不过时账户杠杆已被设置（R11）。
  - 每个固定缺陷的测试命名为 `test_current_behaviour_*`，带 `known_defect` 标记和对应的 R 编号。
- **预计行数**：约 380（全部是测试）。
- **前置条件**：无。
- **回滚**：删除目录。
- **验收**：新增测试在当前 HEAD 上全部通过；通用验收全绿；`git diff --stat` 只有 `tests/characterization/` 和 `pytest.ini` 的标记注册。
- **新增 CI 规则**：无（第 2 步纳入门禁）。

### 第 2 步：门禁与架构规则骨架

- **改动范围**：
  - 新建空包 `trading/`，含 `domain`、`application`、`adapters`、`infrastructure` 四个子包，只有 `__init__.py`。
  - 新增 `.importlinter`，含 `target.md` 2.3 的四条契约。
  - 新增 `tests/architecture/`：环境变量只在 `trading/infrastructure` 读取；`trading/domain` 不 import `logging` 和 `os`；门禁清单自检（断言 CI 的 pytest 列表包含 `tests/architecture`、`tests/characterization`、`tests/offline`、`tests/contract`、`tests/unit/lab`、`tests/unit/binance`）。
  - `Makefile` 的 `CORE_TESTS` 和 `.github/workflows/ci.yml` 的 pytest 列表加入上述目录；CI 的 lint 任务加 `lint-imports`，并把 ruff 的手写文件清单补上 `shared/kill_file.py`、`shared/exposure_gate.py` 和 `trading/`。
  - `pyproject.toml` 的打包清单加入 `/trading/**`。
- **预计行数**：约 150。
- **前置条件**：批准 ADR-0001、0002。开发环境安装 import-linter。【实际运行结果】本机 pip 的默认镜像取不到它，用 `-i https://pypi.org/simple` 可以取到 2.16 版。
- **回滚**：`git revert`。
- **验收**：`lint-imports` 输出 4 条契约全部 KEPT；`make test` 全绿；`tests/contract`（17 个）和 `tests/offline`（347 个）首次进入 pytest 门禁后全部通过。
- **新增 CI 规则**：4 条 import-linter 契约；3 个架构测试。
- **风险**：契约写法我没有在本仓库实际运行过。如果 import-linter 对"遗留顶层包作为外部包"的处理与预期不同，`no-legacy` 契约改用 AST 测试实现，效果相同。

### 第 3 步：端口定义与契约测试

- **改动范围**：
  - 新增 `trading/application/ports.py`：`MarketData`、`ExchangeAccount`、`OrderGateway`、`StopOrderGateway`、`OrderJournal`、`DrawdownStore`、`KillSwitch`、`SpecSource`、`Clock`、`Sleeper`、`Notifier`、`EventLog`、`StatusSink`。方法名沿用现有命名。
  - 新增 `tests/contract/ports/`：同一组测试分别对现有的 `BinanceAdapter`（接假 HTTP）和 `FakeBinance` 运行，检查方法签名、返回类型、出错时抛异常而不是返回空。
  - 把 `shared/entry_risk.py`、`shared/confirmed_execution.py`、`shared/exposure_gate.py` 构造参数里的 `Any` 换成端口类型（只在类型检查时导入，不影响运行）。
- **预计行数**：约 320。
- **前置条件**：第 2 步；批准 ADR-0003。
- **回滚**：`git revert`。
- **验收**：端口契约测试通过；对 `trading/` 运行 mypy 无错误（需要安装 mypy，仓库里已有配置）；现有测试不改断言通过。
- **新增 CI 规则**：架构测试——`trading/application` 下的函数签名不出现 `Any`。
- **预期会暴露的差异**：`FakeBinance` 的成交查询忽略 `limit` 参数；`BinanceAdapter.get_balances` 出错返回空快照。这两处先记录为已知差异，不在本步修改。

### 第 4 步：风控规则与存储分离

- **改动范围**：
  - `trading/domain/risk/exposure.py`：常量 `MAX_TOTAL_LEVERAGE` 和纯判定函数。
  - `trading/domain/risk/drawdown.py`：周期键、高水位、触发判定（取自 `shared/drawdown_guard.py:53-56,84-86`）。
  - `trading/domain/risk/kill.py`：`KillState`。
  - `trading/adapters/sqlite/drawdown_store.py`：回撤库的建表与读写。
  - `trading/application/risk/entry_gate.py`：入场组合门，协作者用端口类型。新增工厂方法 `for_execution(...)`，四个组件缺一抛错；原构造函数签名不变。
  - `shared/exposure_gate.py`、`shared/drawdown_guard.py`、`shared/entry_risk.py` 改为薄壳：公开的类名、方法、默认值全部不变，内部委托给新代码。旧运行时和离线回放继续从旧路径导入。
- **预计行数**：手写约 370；等价移动约 260。
- **前置条件**：第 3 步。
- **回滚**：`git revert`。数据库表结构不变，不需要数据回滚。
- **验收**：
  - `tests/offline/test_risk_safety.py`（21 个）、`tests/unit/binance/test_binance_risk.py` 不改断言通过。
  - 第 1 步的风控特征化测试不改断言通过（缺陷仍在，本步不修）。
  - 新增：3 倍上限的边界值测试（不变量 L1）；`for_execution` 缺组件抛错（L3）。
- **新增 CI 规则**：架构测试——字面量 `MAX_TOTAL_LEVERAGE = ` 在仓库里只出现一次。
- **不做**：不改任何阈值和默认值；`OrderLimiter` 不动。

### 第 5 步：订单状态机显式化

- **改动范围**：
  - `trading/domain/orders.py`：`OrderStatus` 枚举（值等于现有字符串）、合法转移表、终态集合、`check_transition`。
  - `shared/confirmed_execution.py` 改状态的各处调用 `check_transition`。**观察模式**：生产运行时遇到表外转移只记 ERROR 日志并照常继续；测试运行时抛错。这样即使转移表漏了一条合法转移，也不会影响线上下单。
  - `shared/confirmed_execution.py:33`、`runner/binance_target.py:131`、`runner/binance_main.py:52` 的终态字面量改为引用同一个定义。
- **预计行数**：约 260。
- **前置条件**：第 1 步的状态转移表；批准 ADR-0004。
- **回滚**：`git revert`。数据库里的状态值不变。
- **验收**：
  - 新增：枚举全部状态对的测试，合法的通过、非法的抛错（不变量 O2）。
  - `tests/offline/test_execution_safety.py`、`test_market_execution.py`、`test_execution_recovery.py`、`tests/unit/binance/`、`tests/unit/bybit/test_durable_execution.py` 在测试模式（表外转移抛错）下不改断言通过。
- **新增 CI 规则**：架构测试——Binance 路径的文件里不再出现终态集合的字面量。
- **待定**：观察模式何时改为生产也抛错，在第 6 步后的自检审计里根据日志决定。

### 第 6 步：订单库单一所有者

- **改动范围**：
  - `trading/adapters/sqlite/order_journal.py`：接管 `shared/execution_journal.py` 的全部内容，并新增方法承接散落在别处的 SQL——意图预留（现在在 `shared/confirmed_execution.py:107-155`）、改状态、存订单号、结算、成交入库、止损订单收编（`execution_adapter.py:225-231`）、止损表的建表与读写（`protective_stops.py:48-71,121-124`）。
  - `shared/confirmed_execution.py`、`execution/adapters/binance/execution_adapter.py`、`protective_stops.py` 改为调用这些方法。
  - `shared/execution_journal.py` 改为薄壳。冻结线（Bybit / OKX 适配器、`engine/durable_fills.py`）不改，它们经薄壳拿到同一个类。
- **预计行数**：手写约 380；等价移动约 110。
- **前置条件**：第 5 步。
- **回滚**：`git revert`。表结构不变；新旧代码读写同一个库文件，回滚不需要处理数据。
- **验收**：
  - 订单相关的全部现有测试不改断言通过：`tests/offline/test_execution_safety.py`（17）、`test_market_execution.py`（10）、`test_execution_recovery.py`、`test_execution_state.py`、`tests/unit/binance/`（35）、`tests/unit/lab/test_target_runtime.py`。其中包含子进程强杀的崩溃恢复测试。
  - 第 1 步的特征化测试不改断言通过。
  - 新增：用旧代码写出的库文件，新代码能读并恢复未决订单（兼容性测试）。
- **新增 CI 规则**：架构测试——`shared/confirmed_execution.py` 和 `execution/adapters/binance/` 下不再出现 `.execute(` 和 SQL 关键字。
- **风险**：本步是整个计划里逻辑改动最密的一步。预留事务的边界（`BEGIN IMMEDIATE`）必须原样保留，由并发预留测试（`test_parallel_coordinators_cannot_reserve_two_entries`、`test_two_coordinators_cannot_place_the_same_intent`）守护。

### 第 7 步：领域层——策略、再平衡、持仓、时间语义

- **改动范围**：
  - 移动：`lab/strategy.py` → `trading/domain/strategy/overlay.py`；`lab/rebalance.py` → `trading/domain/rebalance.py`；`lab/data.py` 的 `frame_from_bars`、`aggregate`、`funding_series` → `trading/domain/market.py`；`shared/strategy_position.py` → `trading/domain/position.py`。
  - 旧位置留薄壳。`lab/data.py` 的 `load_symbol`（读文件）留在原处。
  - `trading/domain/orders.py` 增加纯 Python 的 `OrderIntent`，供后续步骤替换依赖 Rust 事件类型的 `lab/orders.py`。
- **预计行数**：手写约 90；纯移动约 420。
- **前置条件**：第 2 步。
- **回滚**：`git revert`。
- **验收**：
  - `tests/unit/lab/`（63 个）不改断言通过，其中包括截尾窗口等于全历史（F2）和纸面等于研究（P1）。
  - 新增：性质测试——改动时刻 T 之后的任意 K 线，T 及之前的目标权重不变（F1）。
  - `lint-imports` 的 `core-no-io` 契约此时开始对真实代码生效。
- **新增 CI 规则**：无新规则；第 2 步的契约开始有实际约束对象。

### 第 8 步：Binance 适配器迁入 `trading/adapters`

- **改动范围**：
  - 移动 `execution/adapters/binance/` 的 `rest.py`、`adapter.py`、`config.py`、`urls.py` 到 `trading/adapters/binance/`。一个类实现行情、账户、下单、止损四个端口，不拆类。
  - 适配器返回的数据类（成交、持仓、订单状态）在 `trading` 里定义，不再依赖 `execution/models`。
  - REST 限流策略用纯 Python 实现，替换对 Rust `RustRateLimitPolicy` 的依赖；先写与 Rust 实现的对拍测试。
  - `BinanceConfig.from_env` 的读环境变量部分移到 `trading/infrastructure/settings.py`；测试网判定只保留一处。
  - `execution/adapters/binance/` 旧文件改为薄壳，监控脚本和旧运行时继续可用。`ws_kline_client.py`（死代码）不迁。
- **预计行数**：手写约 260；纯移动约 800。
- **前置条件**：第 3 步；批准 ADR-0008 子决策 1。
- **回滚**：`git revert`。
- **验收**：
  - `tests/unit/binance/test_binance_adapter.py`、`test_exchange_settings.py`、`tests/unit/execution/test_binance_urls.py` 不改断言通过。
  - 第 3 步的端口契约测试改为对 `trading.adapters.binance` 运行并通过。
  - 新增：`import trading.adapters.binance` 之后，`sys.modules` 里没有任何 `execution.*`、`shared.*`、`strategy.*`、`monitoring.*`、`_quant_hotpath`。
- **新增 CI 规则**：上面这条导入隔离测试；`adapters-independent` 契约开始生效。

### 第 9 步：下单生命周期、对账与止损迁入 `trading/application`

- **执行时拆分（2026-11-14）**：为守住每个改动 400 行，本步分 4 个 PR 执行：9a 下单生命周期；9b 止损管理器和其余前置测试；9c 减仓定量与对账；9d TCA 改走 `EventLog`、删除读环境变量的代码、订单事件改用 `OrderIntent`。各自的记录见 `execution-log.md`。
- **改动范围**：
  - `shared/confirmed_execution.py` → `trading/application/execution/lifecycle.py`。`time.sleep` 和 `time.time` 换成 `Sleeper`、`Clock` 端口。
  - `execution/adapters/binance/execution_adapter.py` 里的减仓定量、意图去重、对账 → `trading/application/execution/` 和 `trading/application/reconcile.py`。写 TCA 日志改为经 `EventLog` 端口，不再 import `monitoring`。读 `RISK_SCOPE`、`ORDER_JOURNAL_PATH` 的代码删除，改为由组装根传入。
  - `protective_stops.py` → `trading/application/execution/stops.py`。
  - 订单事件从依赖 Rust 的 `TargetOrderEvent` 换成第 7 步的 `OrderIntent`。
  - 旧位置留薄壳，Bybit / OKX 适配器继续从 `shared.confirmed_execution` 导入。
- **预计行数**：手写约 300；纯移动约 780。
- **前置条件**：第 6、7、8 步。
- **回滚**：`git revert`。
- **验收**：
  - 第 6 步列出的全部订单测试不改断言通过。
  - 第 1 步的特征化测试不改断言通过。
  - 新增：用假时钟驱动的测试替换原来依赖真实等待的路径，断言"下单后等待再查询"的顺序不变。
- **新增 CI 规则**：`core-no-io` 契约对 `lifecycle` 生效（不得 import `time`、`sqlite3`）。

### 第 10 步：会话、主循环与组装根迁入；闸门先于账户写操作

- **改动范围**：
  - `runner/binance_target.py` 的 `TargetSession`、`TargetRuntime` → `trading/application/session.py`、`runtime.py`。
  - `trading/infrastructure/settings.py`（读环境变量并校验）、`bootstrap.py`（组装根）、`cli/target.py`（入口）、`lease.py`。
  - 从 `runner/binance_main.py` 取的 6 个符号各归其位，`trading` 不再 import `runner.binance_main`。
  - `runner/binance_target.py` 改为薄壳，转发到 `trading.infrastructure.cli.target`。systemd 单元不用改。
  - 纸面模式仍走现有的研究撮合路径，只是代码位置迁入（ADR-0006 排法 1）。
  - **启动顺序调整**：spec 验签、研究闸门、scope 校验全部通过之后，才连接账户并设置杠杆（修 R11）。这只改变启动失败时的行为：以前先改账户设置再报错，现在直接报错。
- **预计行数**：手写约 350；纯移动约 400。
- **前置条件**：第 4、9 步。
- **纸面存储**：按 ADR-0007 子决策 3b，把第 1b 步的 `paper_state.json` 改为 `PaperStore` 端口的 SQLite 实现；首次启动时读取已有的 JSON 并迁入，迁入成功后保留原文件作备份。存档格式加入版本号。
- **回滚**：`git revert`。如果已经部署，回滚后重启 `binance-target.service`。回滚前的代码只认 `paper_state.json`，所以第 10 步迁入时保留 JSON 文件并在每次保存时同步写一份，直到第 15 步之后的审计确认可以停写。
- **验收**：
  - `tests/unit/lab/test_target_runtime.py` 不改断言通过（纸面等于研究；执行生命周期、重启、平仓）。
  - `python3 -m runner.binance_target --help` 正常；用 `--max-cycles` 对假交易所跑通一轮。
  - 第 1 步里固定 R11 的特征化测试在本步改断言：闸门不过时账户 POST 次数为 0（不变量 L6）。这是本步唯一改动的断言。
  - 新增：`import runner.binance_target` 之后，`sys.modules` 里没有 `runner.binance_main`、`engine.*`、`decision.*`、`features.*`。
- **新增 CI 规则**：`no-legacy` 契约对整条候选运行时生效；上面的导入隔离测试。

### 第 11 步：急停平仓可靠化

- **改动范围**：新增 `trading/application/execution/flatten.py`，替换 `session.flatten`。按 ADR-0005 已批准的子决策实现：
  - 子决策 2：先撤销该品种全部入场挂单。
  - 子决策 3：按交易所实际持仓或按账本定量。
  - 子决策 1：平仓意图 id 带尝试序号；上一次结算为未成交终态后按间隔重试，到上限后告警。
  - 子决策 7：减仓被拒绝预留时写事件日志。
- **预计行数**：约 390（实现约 200，测试约 190）。
- **前置条件**：第 10 步；ADR-0005 子决策 1、2、3、7 已逐项批准。
- **回滚**：`git revert`。订单库里带尝试序号的意图行对旧代码无害（旧代码把它们当作普通意图）。
- **验收**：
  - 第 1 步里固定 R1、R2 的特征化测试改断言为新行为：注入一次拒绝后，在同一根 K 线内仓位被平掉；有入场挂单时 kill 先撤单再平仓，挂单不再成交（不变量 K2）。汇报里逐条列出改动的断言。
  - "同一意图至多发一次"的全部现有测试不改断言通过（O1）。
  - 新增：重试上限到达后停止并产生告警；撤单结果不明时仍然平仓。
  - 新增（第 3 步后审计 M2）：平仓被拒后行情接口仍然不可用、K 线不前进时，按 30 秒间隔重试，不等下一根 K 线；预热失败期间的情形由第 14 步覆盖。
- **新增 CI 规则**：无。

### 第 12 步：止损失败处置与告警出口

- **改动范围**：
  - `trading/adapters/notify/`：`Notifier` 的 Telegram 实现和空实现。
  - `application/execution/stops.py`：有持仓但没有有效止损时，入场就绪标志为假并告警（ADR-0005 子决策 4b）。
  - 止损已触发但拿不到实际订单号的行不再标为终态，后续同步继续回看（修 R8）。
  - 对账问题、止损失败、重试耗尽统一经 `Notifier` 发出。
- **预计行数**：约 300。
- **前置条件**：第 10 步；ADR-0005 子决策 4 已批准。
- **回滚**：`git revert`。
- **验收**：
  - 第 1 步里固定 R3、R8 的特征化测试改断言为新行为（不变量 K5）。
  - `tests/unit/binance/test_binance_risk.py` 的止损测试不改断言通过。
  - 新增：通知发送失败不影响主循环。
- **新增 CI 规则**：无。
- **需要外部确认**：R8 的触发前提（交易所是否会先报已触发、稍后才给订单号）要在测试网验证。本步的修复对两种情况都安全。

### 第 13 步：受管状态目录与急停路径单一来源

- **改动范围**：
  - 状态目录清单（`manifest.json`）与显式初始化命令。执行模式启动时，清单或任一状态库缺失、scope 不符即拒绝启动（ADR-0007 子决策 1c，修 R6）。
  - kill 文件路径由一个函数从状态目录解析，命令行与运行时共用；命令行必须给出状态目录（ADR-0005 子决策 5，修 R5）。
  - `KillSwitch` 端口只有读方法；写方法只在 `trading/infrastructure/cli/kill.py`（子决策 6）。
- **预计行数**：约 330。
- **前置条件**：第 10 步；对应子决策已批准。
- **回滚**：`git revert`。回滚后清单文件被旧代码忽略，无害。
- **部署影响**：已有的执行模式状态目录需要运行一次初始化命令补上清单。当前没有以执行模式运行的部署，所以没有存量迁移。
- **验收**：
  - 第 1 步里固定 R5、R6 的特征化测试改断言为新行为（不变量 L5、K3、K4）。
  - 新增：纸面模式下目录不存在时仍可自动初始化并启动。
- **新增 CI 规则**：架构测试——`KillSwitch` 的写方法只在 `cli/kill.py` 被调用。

### 第 14 步：行情连续失败告警；预热阶段的行情失败处理

行情失败不退出进程、kill 平仓照常执行、执行模式只交易新鲜 K 线，已经在第 1b 步及其审计后补丁里完成。本步只做剩下的部分。

- **改动范围**：`application/runtime.py`。行情连续失败超过阈值时经 `Notifier` 告警；不可重试的错误（`BinanceNonRetryableError`，例如品种下线、参数错误）不再无限退避而是告警并停止入场；预热阶段行情失败时，执行模式先完成对账、止损维护和风险动作，再重试预热，而不是直接退出。
- **预计行数**：约 150。
- **前置条件**：第 12 步（需要 `Notifier`）。
- **回滚**：`git revert`。
- **验收**：
  - 新增：连续失败达到阈值时恰好告警一次，恢复后计数清零。
  - 新增：预热阶段行情不可用时，kill 平仓仍被执行。
  - 新增：不可重试的错误触发告警，入场保持阻断。
  - `tests/unit/lab/test_target_runtime.py`、`test_target_runtime_resilience.py` 不改断言通过。
- **新增 CI 规则**：无。

### 第 15 步：删除被取代的旧代码与薄壳（第一批）

- **改动范围**（纯删除，除非另注）：
  - （ADR-0008 子决策 5 未答，暂不删除）4 个默认真实下单的研究脚本。
  - 未接线的订单实现：`execution/bridge/`、`execution/store/`，`execution/models/` 里只被测试引用的 9 个文件，以及只覆盖它们的 5 个测试文件。
  - `execution/__init__.py` 和 `execution/adapters/__init__.py` 的急切重导出（改为不导入任何东西）。
  - 已无引用的薄壳：`lab/strategy.py`、`lab/rebalance.py` 等，前提是全仓库 grep 无导入者。
  - `execution/adapters/binance/ws_kline_client.py`、`execution/tests/`、`runner/control_plane.py`。
- **预计行数**：手写约 60（改 `__init__` 和少量导入）；删除约 2,900（`execution/bridge` 914、`execution/store` 321、`execution/models` 的 9 个文件 955、Rust 订单状态机 `execution/rust/order_state_machine.rs` 及其注册约 300、其余约 360，另加只覆盖它们的测试文件）。
- **前置条件**：第 10 步；ADR-0004（删除未接线的订单实现）已批准。`execution/adapters/common/`、`execution/sim/`、Python `risk/` 包等其他死代码属于 ADR-0008 子决策 2，未答之前不删。
- **回滚**：`git revert`。
- **验收**：
  - 删除前运行可达性扫描并贴出结果：被删文件除了经包 `__init__` 的重导出被加载之外，没有任何启动面或在用代码调用。
  - 全量测试全绿；测试总数的减少量等于被删测试文件里的测试数，逐个列出。
  - `python3 -c "import execution.safety.limits"` 加载的仓库内模块数从 40 降到个位数。
- **新增 CI 规则**：架构测试——`execution/__init__.py` 和 `execution/adapters/__init__.py` 不含 import 语句。

## 3. 15 步之后仍未完成的事项

15 步结束后，下列事项仍然存在。它们不在计划内是因为步数上限，不是因为不重要。按建议的先后排列：

| 事项 | 对应 | 说明 |
|---|---|---|
| 冻结交易所的执行硬开关 | R18，ADR-0008 子决策 3 | 约 40 行的安全修改。如果您确认冻结线保留，建议紧接第 15 步做 |
| 下单日限额落盘，纯 Python 限额器 | R12，ADR-0007 子决策 2 | 改变"日"的定义，需批准 |
| 纸面与实盘同一份会话代码 | 不变量 P2，ADR-0006 | 15 步结束时 P2 仍不成立 |
| 成交查询分页 | R9 | 约 60 行 |
| 权益未知时仍生成减仓意图 | R13 | 改 `plan` 的入口条件，属于策略执行行为变更 |
| 监控脚本与运行时统一账户判定 | R15 | 改冻结线旁的运维脚本 |
| 4xx 响应的分类 | R16 | 先要在测试网确认交易所行为 |
| 账户级互斥 | R20 | 需要主机侧约定 |
| 研究数据加载加日期上界；试验登记进各 study | R21 | 研究纪律 |
| 其余死代码、`rust_bin/`、三套旧回测 | ADR-0008 子决策 2、4 | 约 2.5 万行纯删除，分批 |
| ML 线与冻结交易所的去留 | ADR-0008 子决策 3 | 取决于您的回答 |
| 旧文档处置 | `doc-vs-code.md` | 删除 46 个、更新 7 个；不是代码改动，可与任何一步并行 |
| 人工处置卡死订单的工具 | ADR-0004 | 现在只能手改 SQLite |

## 4. 不变量在各步之后的状态

| 不变量 | 现在 | 第 10 步后 | 第 15 步后 |
|---|---|---|---|
| L1 3 倍上限 | ✓ | ✓ | ✓ |
| L2 入场必经入场门 | ✗ | ✓（分层契约） | ✓ |
| L3 执行模式组件齐全 | ✗ | ✓ | ✓ |
| L5 闩锁与限额跨重启 | ✗ | ✗ | 闩锁 ✓（第 13 步）；限额 ✗ |
| L6 闸门先于账户写 | ✗ | ✓ | ✓ |
| K2 急停平仓可靠 | ✗ | ✗ | ✓（第 11 步） |
| K3–K4 kill 路径与只读 | ✗ | ✗ | ✓（第 13 步） |
| K5 无止损不入场 | ✗ | ✗ | ✓（第 12 步） |
| F1–F4 无前视 | ✓ | ✓ | ✓ |
| O1、O4 至多发一次、成交恰好一次 | ✓ | ✓ | ✓ |
| O2 转移表 | ✗ | ✓（观察模式） | ✓ |
| O3 订单库单一写入者 | ✗ | ✓（新包内） | ✓ |
| P1 纸面等于研究 | ✓ | ✓ | ✓ |
| P2 纸面与执行同一份代码 | ✗ | ✗ | ✗ |
| P3 纸面状态连续 | ✓（第 1b 步） | ✓ | ✓ |
| S1 分层规则 | ✗ | ✓ | ✓ |
| S2 资金测试全在门禁 | ✗ | ✓（第 2 步） | ✓ |

"第 15 步后"一列以第 11–14 步对应的子决策全部获批为前提。任何一个未获批，对应的不变量保持 ✗。
