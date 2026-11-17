# 架构说明（2026-11-16 晚重写，R9；2026-11-17 更新）

- 这是仓库里唯一的架构说明。`trading/__init__.py`、`trading/application/ports.py` 和 `tests/architecture/test_rules.py`
  指向本文；ADR 里引用的"2.1""2.3""第 3 节"仍是这几节。
- 依据是代码（基线：分支 `r7d-paper-compare`，即运行时扩展计划 R1–R7 之后）。本文与代码不一致时以代码为准，并改正本文。
- 为什么这样设计见 `docs/adr/`；每一步改了什么见 `docs/arch/execution-log.md`；运行时扩展的计划见
  `docs/arch/runtime-lab-strategies-plan.md`。2026-11-14 的现状诊断、迁移计划、路线图和 M5 设计已在 R9 删除（迁移和 M5 之前的步骤都已完成），原文在 git 历史里。

## 1. 范围

- 一个交易系统：币安 USDT 保证金永续合约上的目标仓位运行时，入口 `runner/binance_target.py`，代码在 `trading/`。
- 一个研究侧：`lab/`（研究、实验室的评审员、模拟盘对照），依赖 `trading.domain` 和 `trading.adapters.simulated`，不属于 `trading/`。
- 运行时交易一份签名规格里列出的品种（最多 10 个），用规格里写明的策略。策略在 `trading/domain/strategy/` 里，研究和实盘调用同一份代码。
- 已删除、不在范围内：机器学习运行时、Bybit / OKX / Polymarket、Rust 扩展、Docker 部署（2026-11-15，M4；最后一版在 git 标签
  `archive/ml-line-2026-11-15`）。

## 2. 分层与依赖规则

### 2.1 四层

依赖只能向内：`infrastructure → adapters → application → domain`。

| 层 | 放什么 | 不许做的 |
|---|---|---|
| `trading/domain` | 策略、规格解析、再平衡、订单与状态、持仓、交易所返回的类型化记录、问题种类及其效果表、风控规则。纯计算 | 任何 I/O；`logging`、`sys` |
| `trading/application` | 用例：每个品种的会话、行情侧、退出侧、状态、下单生命周期、止损、平仓、对账、入场门、亏损停止线；端口定义 | 任何 I/O；可以用 `logging`；时间只经 `Clock` 端口 |
| `trading/adapters` | 端口的实现：币安 REST、SQLite、文件、模拟交易所、告警出口和心跳 | 读环境变量；适配器之间互相 import |
| `trading/infrastructure` | 配置（唯一读环境变量的地方）、组装、命令行入口、人用的工具（状态目录、急停、暂停、手工工具、备份、单元告警） | — |

### 2.2 包结构

```
trading/
  domain/
    bars.py, market.py         已收盘 K 线、窗口框架、资金费窗口
    strategy/base.py           Strategy 接口、Window、PerSymbol、OverlayStrategy、STRATEGIES、strategy_from
    strategy/overlay.py        BTC/ETH 多周期策略（StrategySpec、compute_targets）
    strategy/ideas.py          实验室的 29 个想法（IdeaStrategy）
    strategy/spec.py           签名规格的解析：TargetSpec、parse_spec（第 1、2 版）
    rebalance.py               目标权重 → 意图（plan）
    orders.py, position.py     订单、状态、成交；策略持仓
    venue.py                   交易所返回的类型化记录
    issues.py, symbol_state.py 问题种类和它们的效果表；品种状态
    events.py                  事件种类
    risk/                      敞口上限、回撤、限额、急停、逐仓强平距离
  application/
    ports.py                   全部端口（Protocol）
    runtime.py                 主循环：预热、三个侧（行情、退出、状态）
    market_poller.py           行情侧：新 K 线、资金费间隔、按品种或一起决定
    exit_supervisor.py         退出侧：权益、对账、止损维护、暂停、急停 / 回撤 / 击穿平仓、亏损停止线、账户检查
    session.py                 一个品种：窗口、目标、再平衡、下单
    reset_wait.py              止损暂停后等策略重置（规格的 after_stop，入场门和退出侧共用）
    status.py, alerts.py, issues.py, event_guard.py, equity.py, account_ownership.py, mode_policy.py
    execution/                 lifecycle、service、executor、stops、flatten、reduction、fills、recorded_journal
    risk/                      entry_gate、exposure_gate、order_limiter、readiness、loss_stop
  adapters/
    binance/                   adapter（行情、账户、下单、止损）、rest、rate_limit、urls
    sqlite/                    schema、order_journal、drawdown_store、limit_store、entry_pause、paper_state、backup
    files/                     event_log、kill_switch、lease、host_lock、state_files
    simulated/                 bar_broker（研究撮合器）、exchange（模拟交易所）
    notify/                    file（告警文件）、telegram、log、heartbeat
  infrastructure/
    config.py                  环境变量 → RuntimeConfig 和 Secrets
    cli/target.py              运行时入口：启动检查、两种模式的组装、循环
    target.py, paper.py, modes.py, bootstrap.py
    state_dir.py, state_backup.py, kill.py, entry_pause.py, manual.py, unit_alert.py
```

`runner/binance_target.py` 是一层薄壳：它提供 `trading` 不能 import 的两样东西（签名规格的加载器 `infra.model_signing`、成交分析
日志 `monitoring.tca`），其余都在 `trading/infrastructure/cli/target.py`。

### 2.3 规则的自动化检查

- `.importlinter` 的 5 条契约（`lint-imports`）：分层只能向内；`trading` 不 import 遗留包、`lab`、`tests`；`domain` 和 `application`
  不 import 做 I/O 或读时钟的模块；`domain` 不 import `logging`、`sys`；适配器之间互不依赖。契约以文件为准。
- import-linter 查不了的用测试补（`tests/architecture/`）：`test_rules.py`（`domain`、`application` 里不经 import 的 I/O；只有
  组装层读环境变量；SQLite 只在 SQLite 适配器里；3 倍上限只定义一次；资金路径测试都在 CI 里）、`test_config_rules.py`、
  `test_issue_rules.py`（应用层不在问题文字里找字）、`test_event_kind_rules.py`、`test_artifact_loading.py`（规格只经签名加载）、
  `test_paper_path.py`、`test_research_fills.py`、`test_shared_parameters.py`、`test_no_ai_sdk.py`。

## 3. 端口

全部是 `typing.Protocol`，在 `trading/application/ports.py`。端口方法答不出时抛错，从不返回会被当成"什么都没有"的空结果。

| 端口 | 方法 | 实现 |
|---|---|---|
| `MarketData` | `get_klines`、`get_funding_history`、`get_funding_intervals`（R4） | `adapters.binance` |
| `ExchangeAccount` | `get_account_equity`、`get_positions`、`get_position_snapshot`（R5）、`get_open_orders`、`filters`、`execution_blockers`、`get_ticker` | `adapters.binance`、`adapters.simulated` |
| `FundingIncome` | `get_funding_income` | `adapters.binance` |
| `OrderGateway` | `send_market_order`、`send_limit_order`、`get_order`、`cancel_order`、`get_recent_fills`、`get_ticker` | `adapters.binance`、`adapters.simulated` |
| `StopOrderGateway` | `place_stop_market`、`get_algo_order`、`cancel_algo_order` | 同上 |
| `AccountSnapshot` | 整个账户的持仓、挂单、条件单、品种设置（ADR-0010 的启动检查和运行中每分钟的检查） | `adapters.binance` |
| `OrderJournal` | 意图、订单、成交、检查点、止损行 | `adapters.sqlite.order_journal`（唯一写入者） |
| `DrawdownGuard`、`LimitStore`、`EntryPauses`、`PaperStore` | 回撤闩锁、当日计数、入场暂停、模拟盘状态 | `adapters.sqlite` |
| `KillSwitch`、`KillLatch` | `state`、`allow_order`；`trip`（只给亏损停止线用） | `adapters.files.kill_switch` |
| `EventLog`、`StatusSink` | `write` | `adapters.files` |
| `Notifier` | `alert` | `adapters.notify`（告警文件、Telegram、只写日志） |
| `BarListener`、`PaperVenue`、`GateClock` | 模拟盘：撮合已收盘 K 线、存取快照、按 K 线时间走的入场门时钟 | `adapters.simulated`、`infrastructure.paper` |
| `Clock`、`Sleeper` | 秒数；睡眠 | 组装层传入；测试用假时钟 |

只读的交易所记录用 `...View` 协议描述（`PositionView`、`OrderView`、`FillView` 等），币安适配器返回 `trading.domain.venue` 里的冻结记录。

## 4. 进程、状态和部署

- 单进程、单线程轮询，每 5 秒一轮（`POLL_SECONDS`）。模拟盘和执行是同一个进程的两种组装（`ModePolicy`，
  `trading/infrastructure/modes.py` 是两者差别的唯一清单）。
- 配置在入口读一次（`trading/infrastructure/config.py`，ADR-0017），成为不可变的 `RuntimeConfig`，密钥另放在 `Secrets`；启动时记一条
  不含密钥的 `config` 事件，带规格的策略、参数、品种和是否做空。
- 执行模式的状态目录由人用 `state_dir init` 建一次：清单 `manifest.json`、唯一的状态库 `state.sqlite`（订单、止损、限额、
  暂停、回撤）和急停文件 `kill_switch.json`；运行时在里面写事件日志 `target_events.jsonl` 和状态文件 `target_status.json`，
  从不新建状态库或它的表。主机上执行模式的目录在
  `/var/lib/quant/state/<网络>`（服务单元里写死），备份在 `/var/lib/quant/backups/<网络>`，主机锁在 `/var/lib/quant/locks`。
- 模拟盘的目录默认是 `data/runtime/paper`，它自己建 `state.sqlite`，拒绝带清单的执行目录。
- 服务单元（`infra/systemd/`）：`binance-target`（模拟盘）、`binance-target-testnet`、`binance-target-mainnet`（不自动重启、
  不能开机启动）、`quant-alert@`、`state-backup@<网络>.timer`、`health-watchdog`、`log-archive`。一台主机只跑一个网络的执行进程。
- 执行模式启动前要有告警出口（`ALERT_FILE`，或 Telegram 两项；二选一）和 https 的 `HEARTBEAT_URL`，告警出口确认
  "execution starting" 之后才发任何账户请求；然后只读检查整个账户（ADR-0010），账户里有不属于策略账本的东西就拒绝启动。

## 5. 一个周期

`trading/application/runtime.py` 每轮跑三个侧：行情侧、退出侧、状态。执行模式下三个侧互相隔离：一个侧出错，阻断全部入场直到它
恢复，其他侧照常运行；模拟盘里任何错误都直接报错停下（模拟盘不许悄悄和研究分叉）。

### 5.1 行情侧（`market_poller.py`）

- **资金费间隔**（R4）：预热时和每小时第一轮读一次交易所公布的每个品种的结算间隔（`get_funding_intervals`）。读失败每分钟重试，
  最后读到的值最多用 2 小时；之后，以及交易所没有列出的品种，间隔未知：这个品种的问题 `FUNDING_INTERVAL_UNKNOWN`，不入场，
  减仓照常。等待"收盘时或之前的那次结算"和入场前的检查都按品种自己的间隔。
- **新 K 线**：执行模式只按最新一根决定（停机期间错过的只延长窗口）；模拟盘按时间顺序逐根补放，每根先让模拟交易所撮合、
  退出侧结算，再决定。
- **按品种决定或一起决定**（R3、R7d）：
  - 策略按品种算、也不读别的品种（`cross_sectional` 为假、`references()` 为空，例如 BTC/ETH 多周期策略）：每个品种的 K 线一到就
    决定，一个品种晚到或出错不影响别的品种。
  - 横截面策略，或者要读别的品种的策略（例如读 BTCUSDT）：所有品种都拿到同一个收盘的 K 线后，**一次调用**
    `strategy.latest(...)` 算出所有品种的目标，再逐个品种下单。有品种还没到时谁都不决定；那个收盘已经晚到不能开仓
    （`MAX_ENTRY_DELAY_MS`，10 分钟）而某个品种仍没到时，这个品种记 `DECISION_FAILED`、禁止开仓。策略本身出错时所有品种都
    记 `DECISION_FAILED`。所有品种必须共用同一个策略对象。
  - 已知的限制（R1–R4 审计第 9 条，留给 R8）：一起决定时一个品种一直卡住，所有品种的决定都停，包括策略想要的减仓。
    止损、急停和亏损停止线在退出侧，不受影响。

### 5.2 一个品种（`session.py`）

- 窗口（`Window`）：已收盘的小时线（OHLCV 加 R7a 的成交额、笔数、主动买入量和额；没有的字段是 NaN）和收盘时已公布的资金费。
  窗口长度由策略决定（`window_hours`、`funding_window`）。
- 决定：目标权重（占这个品种可用资金的有符号比例）和是否 ready；规格不允许做空时负目标按 0 交易（R6），已有空头平掉。
  不 ready 不下单。资金：执行模式按账户份额（1/N 个品种），有策略资金时按 `CappedEquity`；未知时为 0，只能整仓平掉。
- 入场拒绝的条件：K 线处理太晚、资金费间隔未知、那次结算还没公布。然后 `rebalance.plan` 生成意图，入场经入场门
  （`risk/entry_gate.py`：数量、品种、价格、急停、回撤、暂停、钱包、敞口、限额，任何异常都拒绝），减仓不经入场门，
  都经同一个下单生命周期（`execution/lifecycle.py`：持久的意图，至多发一次，只认交易所成交）。

### 5.3 退出侧（`exit_supervisor.py`）

- 每分钟最多一次整账户检查（ADR-0011）；读亏损停止线（`risk/loss_stop.py`）；每 5 秒对账。
- 对账（R5）：先把每个品种的成交消化完、刷新订单状态，再**一次**读整个账户的持仓（`get_position_snapshot`），然后每个品种用
  这份快照对账和维护止损。快照读不出，或快照里没有某个品种的行：那个品种照旧自己读。某个品种消化成交出错，只有它记
  `RECONCILE_FAILED`。平仓、手工工具和入场时的敞口检查仍读最新的持仓。
- 问题都是类型化的（ADR-0014）；每种问题对交易的效果（阻断一个品种或全部入场、平仓、暂停、只告警）只在
  `trading/domain/issues.py` 的表里。每个品种的状态（warming、ready、blocked、paused、flattening、halted）每轮由退出侧设定，
  只有 ready 能入场。
- 止损：每个确认的仓位一张交易所 `STOP_MARKET closePosition`，按交易所持仓定方向和价格（ADR-0012）；多头卖出止损在开仓价
  下方，空头买入止损在开仓价上方。急停、回撤和止损击穿的平仓按交易所持仓整仓平掉。止损触发或击穿后这个品种暂停入场 4 小时，
  到时自动解除（`STOP_PAUSE_SECONDS`）。
- 等策略重置（技术负责人 2026-11-17，只会更严）：规格写 `"after_stop": "pause_then_reset"` 时，暂停到点结束后，空仓的品种
  还要等策略做过一次准备好、目标正好为 0 的决定才开仓（`trading/application/reset_wait.py`）。入场门在下单数计数之前就按这条
  规则拒绝，退出侧把它当暂停显示和处理（撤掉挂着的入场单）。止损后有一笔开仓成交（以订单记录为准，重启后也查得到）或由人用
  `entry_pause release` 解除，等待就结束；重启前见过的 0 不记得，要等下一次。读不出来就挡住开仓。
- 亏损停止线：有策略资金时，已实现盈亏扣手续费 + 资金费 + 持仓浮动盈亏低于策略资金的 15%（`STRATEGY_LOSS_STOP_PCT`，只能调紧）
  就打开急停并平仓；这是运行时唯一写急停文件的地方，它从不关闭急停。

## 6. 策略接口和规格

### 6.1 接口（`trading/domain/strategy/base.py`，R1）

- `Strategy`：`name`；`cross_sectional`（某个品种的目标是否用到别的品种的数据）；`window_hours()`、`funding_window()`；
  `references()`（除了交易的品种还要读哪些品种，R7a）；`history(windows)`（每个收盘的目标和 ready，研究回放用）；
  `latest(windows)`（最后一个收盘，实盘用，必须等于 `history` 的最后一行）。
- `PerSymbol`：只读自己品种窗口的策略的基类。`OverlayStrategy` 是现有的 BTC/ETH 多周期策略（`overlay.py`），是第一个实现。
- `STRATEGIES`：名字 → 构造函数，包括 overlay 和实验室的 29 个想法；`strategy_from` 拒绝不认识的名字。
- 防未来函数：`tests/unit/lab/test_lookahead.py` 对每个特征模块和每个想法检查，切点之后的数据不改变切点之前的输出。

### 6.2 实验室的想法（`trading/domain/strategy/ideas.py`，R7b）

- 实验室第一轮的 29 个想法，函数体照抄，输入改为从窗口构造（`lab_frame`）。`IdeaStrategy` 是接口实现；参数必须正好是想法
  参数表里的名字，都是正数。5 个是横截面的；读 BTCUSDT 的想法要求规格列出 BTCUSDT（运行时只读它交易的品种）。
- 和实验室原函数逐位相同（`tests/unit/trading/test_lab_ideas.py`）；实盘用的截尾窗口和完整历史在同一收盘给出相同的值
  （允许 1e-9 的相对误差，来自 pandas 滚动标准差的最后一位小数）。

### 6.3 签名规格（`trading/domain/strategy/spec.py`，R2）

- 规格只经签名加载（`infra/model_signing.py`），然后由 `parse_spec` 解析成 `TargetSpec`：策略名称和参数、执行设置
  （`band`、`execution`、`ttl_bars`）、品种、`allow_short`。
- 第 1 版（`btc_eth_multitimeframe_target`）：BTC/ETH 多周期策略，试运行用的就是它；是否做空由它的 `long_only` 决定。
- 第 2 版（`target_strategy`）：`strategy` {`name`, `params`}、`execution`、`symbols`、`allow_short`、`passed`，可选的
  `after_stop`（`"pause"` 默认，或 `"pause_then_reset"`，2026-11-17）。严格解析：不认识的字段、类型不对的值、超过 10 个品种、
  不是 USDT 永续的名字、重复的品种、不认识的策略、读了规格没列出的品种，都按名字拒绝；第 1 版不能带 `after_stop`。
- 清单由实验室算、人来发布（技术负责人 2026-11-16）：运行时只交易规格里列出的品种，从不自己开始交易一个新品种。每个品种的
  交易所杠杆和保证金模式由人先在币安设好，否则启动检查拒绝。
- 主网要求规格的 `passed` 为真；代码从不修改它。
- 发布：`python3 -m scripts.publish_strategy_spec --study <目录> --out models_binance/strategy`（用同一个解析器检查后签名）。
- 草稿：`python3 -m scripts.draft_strategy_spec --run <实验室一轮> --snapshot <快照>` 从冠军卡写出第 2 版草稿（`passed` 永远是
  false，不签名），并列出实盘还缺的止损规则（有缺口时退出码 2）。

## 7. 研究侧

- `trading/adapters/simulated/bar_broker.py`：研究撮合器 `BarBroker`，`lab.simulate`、实验室回放和模拟盘共用；研究用的下单规则
  常量只有 BTCUSDT、ETHUSDT，所以模拟盘只能跑这两个品种（其他品种按名字拒绝）。
- `lab/um_judge.py`：实验室的评审员（时段、每段的品种清单、配置 = 想法的参数 + 止损方案、滚动前推、打分、最终检查；验证段只在
  `final` 里、只为过了淘汰线的想法回放）。R7c 起它通过策略接口回测：从数据快照构造每个合约的窗口，横截面策略拿到这一段的清单，
  按品种的策略拿到自己一个品种，再加上它声明要读的品种，然后回放 `history` 的目标。`lab.um_lab.lab_ideas()` 给出登记的想法
  （实盘会跑的那份代码）；还是研究函数的想法用 `lab.um_lab.legacy(...)` 包上也能回测，但不能上实盘。
- `lab/paper_compare.py`：模拟盘记录和研究在同一批 K 线上逐笔比较成交（退出码 0 一致、1 不一致、2 无法比较）。R7d 起按记录里的
  策略身份重建研究，所有品种的目标在模拟盘看到的 K 线上一次算出，横截面策略也能对照。
- 2026-11-17 起的实验室工具：
  - `lab/um_portfolio.py`：持有 BTC 加仓位控制的基准，以及每一折只用之前的数据挑成员的组合；
  - `lab/um_monthly.py`：每月一次在新快照上重选，时段随快照往后移，窗口文件先提交再回测；
  - `lab/um_forward.py`：冠军冻结后每周一次登记的前向查看，只为已登记、已到期的查看下载数据；
  - `lab.um_lab.vol_stop_pct` 和 `StopVariant(vol_k=...)`：按波动率收紧的止损；`lab.simulate` 的 `reenter_after_stop`
    有 `pause`、`signal_reset`、`pause_then_reset` 三种，最后一种就是规格 `after_stop` 的同一条规则。
  - 研究撮合器可以给每个仓位单独的止损距离（`set_next_stop_pct`，只能比配置的更近），只在研究里用，模拟盘从不设。
- 研究的纪律（试验登记、前向检验、数据快照不可变）见 `CLAUDE.md` 和 `research/protocols/`。

## 8. 不变量和检查它们的测试

| 不变量 | 测试 |
|---|---|
| 3 倍敞口上限只定义一次，只在入场时检查 | `tests/architecture/test_rules.py`、`tests/unit/trading/test_domain_risk.py` |
| 每张入场单都经过入场门；执行模式入场门的组件缺一不可 | `tests/characterization/test_runtime_decisions.py`、`test_domain_risk.py` |
| 权益、价格、持仓或状态库未知时不入场，仓位按 0 算 | `tests/unit/trading/test_unknown_equity.py`、`test_lost_state_files.py` |
| 回撤闩锁和当日计数跨重启保持；检查通过之前不写账户 | `tests/characterization/test_risk_state.py` |
| 急停：先撤后平；命令和运行时用同一个文件；文件丢失按激活；只有急停命令和亏损停止线写它 | `tests/characterization/test_exit_paths.py`、`tests/unit/trading/test_state_dir.py` |
| 有仓位而没有有效止损时阻断入场 | `tests/characterization/test_stops_and_fills.py` |
| 同一意图至多提交一次；只有交易所成交才入账、恰好一次；崩溃重启不重复下单 | `tests/offline/test_execution_safety.py`、`test_execution_recovery.py`、`tests/characterization/test_order_facts.py` |
| 无前视 | `tests/unit/lab/test_lookahead.py`、`tests/unit/trading/test_strategy_causality.py`、`tests/unit/binance/test_binance_adapter.py` |
| 模拟盘与研究逐笔相同；模拟盘不构造带凭据的适配器 | `tests/unit/lab/test_target_runtime.py`、`tests/unit/lab/test_paper_safety.py`、`tests/architecture/test_paper_path.py` |
| 账户里有不属于策略的东西时拒绝启动，运行中阻断入场 | `tests/characterization/test_runtime_account_check.py`、`test_foreign_fills.py` |
| 执行模式没有告警出口或心跳不能启动；心跳失败不影响交易 | `tests/characterization/test_operator_startup.py`、`tests/unit/trading/test_execution_prerequisites.py` |
| 一个账户身份在本机只有一个执行进程 | `tests/characterization/test_host_lock_startup.py`、`tests/unit/trading/test_host_lock.py` |
| 恢复只收紧风险状态 | `tests/characterization/test_restore_hardening.py`、`test_state_restore.py` |
| 主网执行单元不自动重启 | `tests/unit/infra/test_systemd_units.py` |
| 配置只读一次，配置记录不含密钥 | `tests/unit/trading/test_runtime_config.py`、`tests/architecture/test_config_rules.py` |
| 亏损停止线到线就急停平仓，算不出时阻断入场 | `tests/unit/trading/test_strategy_loss_stop.py`、`tests/characterization/test_strategy_loss_stop_runtime.py` |
| 策略接口：`latest` 等于 `history` 的最后一行；BTC/ETH 策略行为不变 | `tests/unit/trading/test_strategy_interface.py` |
| 规格严格解析，第 1 版照旧加载 | `tests/unit/trading/test_spec_v2.py`、`test_runtime_extension_audit.py`、`tests/characterization/test_spec_v2_runtime.py` |
| 一起决定：一次调用、缺数据时谁都不决定 | `tests/unit/trading/test_decide_together.py`、`tests/characterization/test_decide_together_runtime.py` |
| 资金费间隔按品种，未知时不入场、减仓照常 | `tests/unit/trading/test_funding_interval.py`、`tests/characterization/test_funding_interval_runtime.py` |
| 整账户快照；10 个品种时请求权重留有余量 | `tests/unit/trading/test_account_snapshot.py`、`tests/characterization/test_account_weight.py` |
| 规格允许才开空；空头的止损、敞口、平仓、浮动盈亏的符号 | `tests/unit/trading/test_allow_short.py`、`tests/characterization/test_shorts_runtime.py` |
| 实验室的想法和原函数逐位相同；评审员通过接口回测；模拟盘对照登记的策略 | `tests/unit/trading/test_lab_ideas.py`、`tests/unit/lab/test_um_judge_interface.py`、`test_paper_compare_strategies.py` |
| 测试从不联网 | `tests/conftest.py`、`tests/unit/tooling/test_network_guard.py` |

## 9. 决策记录

`docs/adr/` 下的 0001–0017。已被取代的在各自开头标明，不删除。

## 10. 不在这一版里的

- 程序自己换品种清单（技术负责人选的是人来发布）；超过 10 个品种；不同策略同时运行、共用一个账户。
- 只读不交易的品种：策略要读的品种必须在规格里列出，也就会被交易。
- 模拟盘只有 BTCUSDT、ETHUSDT 的研究用下单规则；对照工具的资金费完整性检查仍按每 8 小时一次。
- 币安是否接受同方向的第二张整仓止损还没有验证，在小资金试运行里验证（见 `CLAUDE.md`）。
