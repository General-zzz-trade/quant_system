# 币安 USDⓈ-M 运行手册（2026-11-15 重写，2026-11-16 按 S6–S8 更新，2026-11-16 晚按 R1–R7 和取消测试网阶段更新）

本手册只覆盖目标仓位运行时 `runner/binance_target.py`。**本手册不构成实盘批准。** 命令和环境变量都以代码为准；与代码不一致时，以代码为准并改正本文。小资金实盘试运行的准备、每日检查和记录见 `docs/m5_checklist.md`；架构见 `docs/arch/target.md`。

## 0. 现状与硬约束

- 只交易签名规格里列出的币安 USDⓈ-M 永续（最多 10 个，R2）；现有的规格是 BTCUSDT、ETHUSDT 的多周期策略。
- 默认是模拟盘（`--dry-run`），默认用测试网（`BINANCE_TESTNET` 不设或为 `1`）。
- 主网执行要求签名规格里的 `passed` 为真。现在的规格是 `passed: false`，所以主网一定被拒绝。
- 敞口上限 3 倍权益（`AccountExposureGate`），由代码常量保证，配置只能调低。
- 回撤限额：默认日 5%、周 12%；硬上限日 10%、周 25%，超过上限拒绝启动。
- 任何代码路径都不会自动提高限额或重启实盘服务。技术负责人 2026-11-16 定的例外只有这些：仓位基数随策略自己的盈亏和资金费变化；止损后的入场暂停 4 小时后自动解除；策略累计亏损到停止线时，程序会打开急停并平仓（只会打开，从不关闭）。

## 1. 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `RISK_SCOPE` | 账户身份。执行模式必填；状态目录、租约、限额计数都按它区分 | 无 |
| `BINANCE_TESTNET` | `1` 测试网；`0` 主网（主网还要过规格门槛） | `1` |
| `BINANCE_TESTNET_API_KEY` / `_SECRET` | 测试网凭据（不会退回去用主网凭据） | 无 |
| `BINANCE_API_KEY` / `_SECRET` | 主网凭据 | 无 |
| `MAX_DAILY_DRAWDOWN_PCT` / `MAX_WEEKLY_DRAWDOWN_PCT` | 回撤限额（百分比） | 5 / 12 |
| `BINANCE_MAX_ORDER_NOTIONAL` / `BINANCE_MAX_DAILY_NOTIONAL` / `BINANCE_MAX_DAILY_ORDERS` | 单笔、当日名义额和当日笔数上限 | 5000 / 50000 / 50 |
| `BINANCE_MAX_LEVERAGE` | 账户敞口上限（倍），不能超过 3 | 3 |
| `BINANCE_EXCHANGE_LEVERAGE` / `BINANCE_MARGIN_TYPE` | 规格里每个品种应有的交易所杠杆和保证金模式（试运行：10 倍逐仓）。**程序只核对、不修改**，由人在币安上设好，不符就拒绝启动（ADR-0010）。不设保证金模式表示全仓。逐仓时止损必须在强平距离的一半以内 | 3 / 全仓 |
| `ALERT_FILE` | 告警文件的绝对路径（技术负责人 2026-11-16：不用 Telegram）。每条告警追加一行 JSON，启动告警以"这一行写进去了"为确认。程序自己不通知任何人，要有东西来读这个文件：现在是 Claude Code 会话，大约每小时读一次，它只能自己做降低风险的动作（打开急停、停止服务），重启、解除急停和暂停、改配置都要先问人。目录由主机安装创建（`/var/lib/quant/alerts`），写不进去就拒绝启动。和 Telegram 二选一，同时设会拒绝启动 | 无 |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | 另一种告警出口，可以不用（试运行用 `ALERT_FILE`）。执行模式要有一个告警出口，`ALERT_FILE` 或这一对，都没有就拒绝启动（S7）；只设一个一律拒绝启动；模拟盘不设时只写日志 | 无 |
| `HEARTBEAT_URL` | 外部心跳监控的 https 地址。**执行模式必填**；按密钥处理，不写进日志和配置记录（S7） | 无 |
| `STRATEGY_CAPITAL_USDT` | 策略资金（USDT）。**主网执行必填**。策略和手动交易共用主账户，仓位按 min(账户权益, 这个数 + 策略自己的已实现盈亏扣手续费 + 策略各品种自首笔成交起的 USDT 资金费收入) 算；同一 `RISK_SCOPE` 账本里所有 `target.*` 的品种都算，包括当前规格已经不交易的；资金费最多每 10 分钟向币安查一次，读到的每一行存进状态库（第 2 版，2026-11-17），以后只读最近一段；从未查到、超过 2 小时未更新、或程序停了超过三个月（币安收入历史只保留三个月，中间断档）时仓位基数为 0（基数为 0 时不做部分减仓，只做整仓平仓）；用 BNB 付的手续费不计入盈亏（账户已关闭 BNB 抵扣，2026-11-16 核对）；回撤锁定和 3 倍敞口仍按整个账户（技术负责人 2026-11-16） | 无 |
| `STRATEGY_LOSS_STOP_PCT` | 策略累计亏损的停止线，占策略资金的百分比。亏损 = 已实现盈亏扣手续费 + 资金费 + 持仓浮动盈亏。浮动盈亏只算策略账本里的仓位：账本数量 ×（币安标记价 − 账本开仓价），账户上同品种的手动仓位不算。到线就打开急停并平掉全部仓位，由人处理（技术负责人 2026-11-16，取代"连亏 3 笔就停"）。到线之后，这个进程不再开仓，直到进程结束。只能调紧：大于 0、不超过 15；没设策略资金时不能设 | 15 |
| `STRATEGY_LEVERAGE` | 满仓时仓位是策略资金的几倍（技术负责人 2026-11-16：试运行 2 到 3 倍）。不设就是 1；大于 1 时必须设策略资金；不超过 3（账户敞口的硬上限），也不超过 `BINANCE_EXCHANGE_LEVERAGE`。由人写，程序从不改它。回测见 `research/results/overlay_pilot_leverage_2026-11-16/`：2 倍最好，3 倍收益更低、回撤翻倍 | 1 |
| `STOP_PCT` | 交易所止损的距离，命令行没给 `--stop-pct` 时用它（主网服务单元不给）。不设就是 0.08；0.005 到 0.15 之间。逐仓时必须在强平距离的一半以内：10 倍逐仓最多 0.0475，试运行用 0.045 | 0.08 |
| `QUANT_MODEL_SIGN_KEY` | 规格签名密钥。实盘环境下不允许加载未签名的规格 | 无 |

凭据只放在主机的环境变量文件里，不提交、不打印。全部变量和规则见 `.env.example`。

## 2. 发布策略规格

```bash
QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_strategy_spec \
    --study research/results/overlay_2026-11-13 --out models_binance/strategy
```

规格里的门槛结果会原样复制：门槛未通过时主网保持拒绝，模拟盘和测试网可以运行。发布前用运行时同一个解析器检查规格（第 1 版：BTC/ETH 多周期策略；第 2 版：策略、参数、最多 10 个品种、是否做空，以及可选的 `after_stop`，见 `docs/arch/target.md` 6.3）。

- 第 2 版规格写 `"after_stop": "pause_then_reset"` 时，止损的 4 小时暂停结束后，空仓的品种还要等策略要过一次 0 才开仓（技术负责人 2026-11-17，只会更严）；不写就和以前一样。
- 实验室冠军的规格草稿：`python3 -m scripts.draft_strategy_spec --run research/lab/runs/<日期> --snapshot <快照>`。草稿的 `passed` 永远是 false、不签名；签名发布和标记通过只由技术负责人做。

## 3. 模拟盘（systemd，默认）

```bash
sudo systemctl start binance-target.service      # 单元里写的是 --dry-run
cat /quant_system/data/runtime/paper/target_status.json
tail -f /quant_system/logs/binance_target.log
```

- 模拟盘的状态目录默认是 `data/runtime/paper/`（执行模式在主机上用 `/var/lib/quant/state/<网络>`，服务单元里写死，两者不共用；模拟盘拒绝带 `manifest.json` 的执行目录，`--reset-paper-state` 也不例外），里面有：
  - `target_status.json`：每个周期重写；
  - `target_events.jsonl`：意图、阻断、问题、告警；
  - `state.sqlite`：订单库、止损、限额、暂停、回撤，以及模拟盘状态（模拟交易所的快照和游标），在同一事务里保存（S10）。模拟盘第一次启动时自己建这个库，不需要 `state_dir init`；
  - `paper_state.json`：每次保存后同步写的一份副本，只写不读。
- 删除或修改 `paper_state.json`**不会**改变模拟盘。要重新开始，用 `--reset-paper-state`：旧的 `state.sqlite` 改名放到一边（`state.<毫秒>.sqlite`），JSON 副本归档。
- 2026-11-16 之前的模拟盘目录（只有 `paper_state.sqlite` 和 JSON，没有 `state.sqlite`）会被拒绝启动，用 `--reset-paper-state` 开一个新记录。
- 模拟盘走执行路径（入场闸门、订单库、止损、对账）在模拟交易所上运行，逐根补放错过的 K 线，结果与研究模拟相同（`tests/unit/lab/test_target_runtime.py`）；有意的分叉（入场被拒、止损、暂停、急停和回撤的风险退出）各写一条 `paper_divergence` 事件。
- **核对模拟盘与研究**（S11，设计 G6）：模拟盘不记 K 线，所以先下载一个覆盖模拟盘记录的新快照，再逐笔比较成交：
  ```bash
  python3 -m data.downloads.binance_public --start 2020-01-01 --end <今天>      # -> data_files/binance_public_<今天>/
  python3 -m lab.paper_compare --paper-dir data/runtime/paper --data-dir data_files/binance_public_<今天>
  ```
  - 第一行是结论：`一致`（退出码 0）、`不一致`（1，列出第一处不同的两笔）、`无法比较`（2，说明缺什么或哪里与模拟盘看到的不同）。
  - 有意的分叉单独列出，不算不一致；但要有对应的 `paper_divergence` 事件。
  - 只读，只输出成交，不输出权益和收益。

## 4. 执行模式（小资金实盘试运行在主网）

技术负责人 2026-11-16 取消了测试网阶段：试运行直接用主网单元 `binance-target-mainnet`，准备工作逐项见 `docs/m5_checklist.md`。
测试网单元 `binance-target-testnet` 仍在，只在需要时由人用。一台主机只跑一个网络的执行进程；状态、备份、告警文件和主机锁都在
`/var/lib/quant` 下（技术负责人 2026-11-16）。下面的 `<网络>` 是 `mainnet` 或 `testnet`。

1. **安装主机**（root，幂等，不读密钥，不启动任何服务）：`sudo bash /quant_system/infra/install_host.sh`。按哈希安装依赖，创建 `/var/lib/quant/{state,backups,alerts}` 和日志目录，同步服务单元，部署日志轮转，最后跑离线安全检查。
2. **写 `.env`**：照 `.env.example`，权限 600，属主 ubuntu。执行模式要有一个告警出口（试运行用 `ALERT_FILE`）和 `HEARTBEAT_URL`；主网还要有 `STRATEGY_CAPITAL_USDT`。
3. **初始化状态目录**，每个账户身份一次，由人来做（以 ubuntu 身份）：
   ```bash
   python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/<网络> init
   python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/<网络> check
   ```
   - `init` 先创建主机锁目录 `/var/lib/quant/locks`（S6），再创建清单、唯一的状态数据库 `state.sqlite`（带版本号、全部表）和一个未激活的急停文件；已初始化的目录不会被覆盖。
   - 运行时的各存储只打开这个数据库，从不新建文件或表。数据库丢失或被清空时启动会被拒绝；从备份恢复见第 7 节。
   - 换目录或换 `RISK_SCOPE` 就是换状态；但同一把 API Key 在本机只能有一个执行进程（主机锁），与目录和 `RISK_SCOPE` 无关。
   - 数据库升版本时（2026-11-17 起是第 2 版：资金费存进状态库），已经初始化的目录要在运行时停着的时候迁移一次：
     `python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/<网络> migrate`，再 `check`。
     迁移只加表，旧文件在旁边留一份（`state.sqlite.v1-<时间>`），并记进 `manual_actions`；版本不对时启动会拒绝并提示这条命令。
     从旧版本的备份恢复时，恢复会把副本一起迁移；完好的旧版本库不能用恢复替换，只能迁移。
     迁移之前，每小时的备份会拒绝旧版本的库（每次失败都告警），所以升级代码后要尽快迁移。
4. **开启备份**：`sudo systemctl enable --now state-backup@<网络>.timer`（每小时一份快照，保留 14 天，失败时告警）。
5. **启动前一键检查**（只读，2026-11-17）：
   ```bash
   python3 -m scripts.account_check --pilot --spec models_binance/strategy/final_spec.json --state-dir /var/lib/quant/state/mainnet
   ```
   逐项打 ✓ 或 ✗：启动时的配置检查、主网、日亏限额 3%、签名密钥、告警文件夹、策略资金、规格签名和"已通过"、状态目录、账户上有没有任何仓位或挂单、单向持仓、每个品种的杠杆和保证金模式、余额。最后写"可以启动"或"还缺 N 项"。不显示任何密钥，只发 GET 请求。
6. **启动**：
   - 主网：`sudo systemctl start binance-target-mainnet.service`。单元里固定了主网（`/usr/bin/env BINANCE_TESTNET=0 …`），不自动重启，没有 `[Install]`，不能开机启动；失败后由人看过再手动启动。命令行不带 `--stop-pct`，止损距离来自 `.env`。
   - 测试网：`sudo systemctl start binance-target-testnet.service`。单元里固定了测试网（`/usr/bin/env BINANCE_TESTNET=1 …`），即使 `.env` 写错也连不到主网；失败后 60 秒自动重启，每小时最多 3 次（人手动启动也算）。
   - 两个单元每次失败都经 `quant-alert@` 走配置的告警出口发一条。
   - 启动检查依次核对：规格签名和门槛（主网要求 `passed` 为真）、账户身份、回撤限额上限、止损距离（逐仓时必须在强平距离的一半以内）、状态目录；主网还要有策略资金；然后要求告警出口和心跳地址（S7）、读凭据、拿主机锁（S6）；然后发 "MAINNET execution starting"（测试网是 "testnet execution starting"），**告警出口没确认（告警文件那一行没写进去，或 Telegram 15 秒内没回应）就不启动、不发任何账户请求**。
   - 连接账户之后、运行时建好之前，只读检查整个账户（ADR-0010）。下面任何一项成立就拒绝启动，逐项列出，并发一条 "execution start refused"：
     - 任何品种上有不属于策略账本的持仓；
     - 有不属于策略的普通挂单或条件单；
     - 规格里某个品种的杠杆或保证金模式与配置不同；
     - 账户读不出来。
   - 策略和手动交易共用主账户（2026-11-15 的决定），所以启动执行模式之前，要先在币安上把手动仓位和挂单处理掉。
   - 运行中每分钟最多一次心跳（等行情时也发）；心跳监控失联说明进程或主机停了。
6. **前台试跑**（不经 systemd，排查时用）：
   ```bash
   python -m runner.binance_target --spec models_binance/strategy/final_spec.json \
       --state-dir /var/lib/quant/state/<网络> --enable-execution [--stop-pct <距离>]
   ```
   systemd 单元在跑时，这条命令会被主机锁拒绝。网络由 `.env` 的 `BINANCE_TESTNET` 决定（不设就是测试网）。

## 5. 风险操作（都由人执行）

| 目的 | 命令 |
|---|---|
| 急停（阻断新敞口） | `python -m trading.infrastructure.kill --state-dir <dir> activate --reason "..."` |
| 急停并平仓 | 同上，加 `--flatten`：撤掉挂单，按交易所持仓整仓平掉 |
| 解除 / 查看急停 | `... kill --state-dir <dir> clear` / `status` |
| 查看入场暂停 | `python -m trading.infrastructure.entry_pause --state-dir <dir> status` |
| 解除某品种的暂停 | `... entry_pause --state-dir <dir> release --symbol BTCUSDT --reason "..."`（要键入确认） |
| 查看未结订单与交易所视图 | `python -m trading.infrastructure.manual --state-dir <dir> orders` |
| 手工结算一张卡住的订单 | `... manual --state-dir <dir> settle --intent <id> --reason "..."`（运行时必须已停止） |
| 账本对齐到交易所持仓 | `... manual --state-dir <dir> position --symbol BTCUSDT --reason "..."`（运行时必须已停止） |

- 在初始化过的目录里，急停文件**丢失时按"激活"处理**。运行时从不关闭急停；它只在策略累计亏损到停止线时打开急停并带平仓（见下）。
- 止损在交易所侧触发或被击穿后，该品种暂停入场 4 小时，到时自动解除（技术负责人 2026-11-16），同时告警，告警里写明恢复时间；人可以用 `entry_pause release` 提前解除。从备份恢复时设的暂停仍要人解除。
- 规格写了 `after_stop: pause_then_reset` 时，4 小时之后空仓的品种还在"等策略目标回到 0"（状态文件里 `entry_pause` 显示这句话，`entry_pause status` 不列出）。止损后有一笔开仓成交就永久结束；人也可以用同一条 `entry_pause release` 结束这个等待（要键入确认，留一行由人解除的记录）。暂停记录丢失时，会根据订单库里的止损记录重新暂停，不会因为丢了文件而放行。
- 手工工具要求运行时已经停止：同一账户身份的运行时在本机任何目录里运行时都拒绝（主机锁），同一目录的租约也要拿得到；还要写明理由、键入确认，并留下审计记录。它们从不凭空生成成交。

## 6. 告警与心跳

- 执行模式必须配置告警出口：`ALERT_FILE`（技术负责人 2026-11-16：试运行用告警文件，不用 Telegram），或者 Telegram 两项；两者同时设拒绝启动；启动时 "execution starting" 要这个出口确认才继续（S7）。
- 用告警文件时要知道：程序自己不通知任何人，要有东西来读（现在是 Claude Code 会话，约每小时一次）。读的一方轮换文件时只把旧文件移走，**不要自己新建**，否则程序以服务用户身份写不进去。文件里的文字有一部分来自交易所返回的错误信息，读的一方只把它当数据。磁盘写满时告警写不进去，而心跳照常；程序日志在同一块盘上时，那条 "Alert not written to the alert file" 也可能写不进去。所以每小时的检查要自己看磁盘用量（`df`），并把健康检查的状态文件（`data/runtime/health_status.json`）过期或读不出来当成告警；健康检查在磁盘用到 85% 和 95% 时会提前往告警文件里写一条。读文件的规则：`notice` 级别的行（启动、启动被拒、服务失败）必须读；解析不了的行（写了一半的残行、空行）跳过，但要记下来；告警文件要对服务用户可读可写，别人建的只写文件写不进去；`ALERT_FILE` 不要指到状态目录里。模拟盘没有配置时只写进程日志（状态里 `alerts: "log only"`）。
- `alert` 类事件立即发出。持续的问题（例如没有止损、平仓失败、行情不可用）超过 2 分钟后发出，每 30 分钟重复，恢复时发"已解决"。
- 服务失败（包括随后被自动重启的那次）和备份失败由 `quant-alert@<单元>` 发一条消息；出口没确认时（Telegram 没回应，或告警文件写不进去），告警单元自己也显示失败。
- 外部心跳：每分钟最多一次，在单独线程里发、带超时，失败只记日志，不影响交易和平仓。心跳监控在另一处，主机宕机时由它通知人。

## 7. 主机维护、备份与恢复

```bash
sudo bash /quant_system/infra/install_host.sh   # 安装或更新：依赖、目录、单元、日志轮转、安全检查；从不启动服务
sudo bash infra/sync_systemd.sh                 # 只同步单元：停用并删除退役单元，复制有变化的单元，重载；从不重启服务
sudo systemctl restart binance-target.service   # 看过改动后由人重启（模拟盘）
```

- 服务单元：`binance-target`（模拟盘）、`binance-target-testnet`（测试网执行）、`binance-target-mainnet`（主网执行，不自动重启、不能开机启动，只能人手动 `systemctl start`）、`quant-alert@`、`state-backup@<网络>.timer`、`health-watchdog`、`log-archive`。
- 2026-11-15 退役了一批单元（见 `docs/deploy_truth.md`）。主机上要运行一次同步脚本才会停掉它们。
- 看门狗（`health-watchdog`）每 5 分钟检查模拟盘状态文件和最新 K 线的延迟，只告警、从不重启。
- **备份**：`state-backup@<网络>.timer` 每小时用 SQLite 在线备份给 `state.sqlite` 拍快照，写到 `/var/lib/quant/backups/<网络>/state-<UTC 时间>.sqlite`，检查通过才保留；成功后删除 14 天前的快照。急停文件不备份。
- **恢复**（只在状态库丢失或损坏、启动检查拒绝它时；运行时必须已停止）：
  ```bash
  sudo systemctl stop binance-target-mainnet.service
  python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/mainnet restore \
      --snapshot /var/lib/quant/backups/mainnet/state-<时间>.sqlite --reason "<为什么>"
  ```
  - 要键入 scope 确认；库完好时、运行时在跑时、快照是模拟盘的库时都拒绝。
  - 恢复只会收紧：当天和本周的回撤锁定，当天下单计数视为用完，所有品种暂停入场；急停文件不动；损坏的库连同它的 -wal、-shm、-journal 一起移进 `state.sqlite.broken-<时间>/`；留一条审计记录。
  - **回撤锁定的后果：恢复后下次启动会按市价平掉所有仓位，并且空仓到本 ISO 周结束**（技术负责人 2026-11-16 决定保持）。备份恢复演练会真的平仓。
  - 恢复后先 `state_dir check`，再启动；启动时的账户检查（ADR-0010）会拒绝任何与交易所不一致的状态。然后由人核对交易所持仓和账本，逐个品种 `entry_pause release`。

## 8. 出问题时

| 现象 | 运行时的行为 | 人要做的 |
|---|---|---|
| 行情接口失败 | 退避重试（5/10/20/40 秒…最长 300 秒），连续 5 次后告警；平仓、止损、急停照常执行 | 看日志；网络恢复后自动继续 |
| 交易所说"不可重试"（如品种下线） | 立即告警，阻断入场，按 300 秒重试 | 查原因 |
| 启动时行情不可用 | 不退出；先对账、维护止损、执行急停和回撤平仓，再重试预热 | 同上 |
| 有持仓但没有有效止损 | 阻断入场，2 分钟后告警 | 查止损下单失败的原因 |
| 策略累计亏损到停止线（默认策略资金的 15%） | 打开急停并带平仓：撤掉挂单，按交易所持仓整仓平掉（同品种的手动仓位也会被平掉），停止开仓，告警。重启后等行情、预热期间也会检查 | 这次试运行结束。只解除急停会马上再次触发（下一次对账，几秒内；重启后在预热时），因为亏损还在账本里；解除到再次触发之间入场仍然被拒绝。要继续就开一次新的试运行：新建状态目录（`state_dir init`），重新设 `STRATEGY_CAPITAL_USDT`；旧目录留作记录 |
| 策略亏损算不出来（账本读不出；本进程从未读到资金费；账本有仓位而币安没有返回该品种或标记价） | 不平仓，阻断全部入场，撤掉未成交的入场挂单，记一条问题（`strategy_loss_stop`），持续 2 分钟后告警 | 查原因；读得到之后自动恢复。资金费一时刷新不了不算：停止线用状态库里存着的总数，重启后也一样（仓位基数仍按 2 小时规则归零）。存着的资金费够不到首笔成交，或者最后一行早于 88 天（程序停了超过三个月，中间的资金费再也读不到）时，算未知 |
| 止损被击穿 | 按交易所持仓整仓平掉（S5b），暂停该品种入场 4 小时（到时自动解除），告警 | 查原因；需要时提前用 `entry_pause release` 解除，或在 4 小时内启用急停 |
| 启动告警没被告警出口确认 | 拒绝启动，不发任何账户请求 | 用告警文件时查目录是否存在、服务用户能否写、磁盘是否已满；用 Telegram 时查配置和网络 |
| 主机锁被占（"already has an execution process"） | 拒绝启动 | 同一把 API Key 已有执行进程在跑；先停掉它（`systemctl status`），不要换目录或换 `RISK_SCOPE` 绕过 |
| 心跳监控报失联 | — | 进程或主机停了：看 `systemctl status` 和日志；主机宕机时交易所侧止损仍在 |
| 外来成交（`foreign_fill`） | 只记账本自己那部分，多出的记为外来成交，告警 | 核对是否有手动仓位被止损或急停一起平掉 |
| 回撤到达限额 | 锁定到下一个 UTC 日或周，按交易所持仓整仓平掉 | 不用自动解除；查原因 |
| 状态库损坏或丢失（启动时） | 拒绝启动并说明缺什么 | 用 `state_dir restore` 从最新快照恢复（第 7 节）；恢复后核对再逐个解除暂停 |
| 启动时账户里有不属于策略的持仓、挂单或条件单，或者杠杆、保证金模式与配置不符 | 拒绝启动，逐项列出；不改账户里的任何东西 | 在币安上平掉、撤掉手动的仓位和挂单，或者按配置设好杠杆和保证金模式，再启动。**不要**用 `manual position` 收编手动仓位：收编后它归策略的止损、急停和回撤管。**被拒绝期间运行时完全不运行**：策略自己仓位的止损不补挂，急停、回撤平仓不执行，限价入场单不按时撤，所以要同时在币安上核对策略自己的仓位、止损和挂单 |
| 限额计数读不出 | 拒单（按"已用完"处理） | 查状态目录 |

## 9. 主网

- 主网执行要求签名规格的 `passed` 为真。代码从不修改它，只有技术负责人亲自把它标为通过（做法见 `docs/m5_checklist.md`）。
  现在的规格是 `passed: false`，所以主网被拒绝。
- 上实盘的路径（技术负责人 2026-11-16，正式记录在 `CLAUDE.md` 的 Current status）：没有模拟盘和测试网阶段。实验室的幸存者先冻结
  规则，在新数据上跟踪 4 周，四次都成立才进小资金试运行；另外，已有的 BTC/ETH 策略可以做校准试运行，收集真实成交。
  真钱要等技术负责人说"go live"。
- 退出侧还没验证的一件事（同方向第二张整仓止损，见 `CLAUDE.md`）在试运行里验证。
