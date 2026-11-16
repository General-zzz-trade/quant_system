# 小资金实盘试运行：开始前的核对清单、每日检查和记录（2026-11-16 晚重写，R9）

> 文件名沿用 M5（别的文档引用它）。技术负责人 2026-11-16 取消了模拟盘和测试网阶段：通过实验室检查的策略直接上主网的
> 小资金试运行，由技术负责人决定是否加资金。原来的测试网运行不再做；它的 4 周计时和演练改在试运行期间进行
> （ADR-0016 第 7 条的通过标准：连续 4 周对账无差异，没有未解释的告警）。
>
> 依据是 `CLAUDE.md` 的 Current status（Path to live、Calibration pilot、Shared account）。**这不是实盘批准**：真钱要等技术负责人
> 说"go live"。命令和环境变量以代码为准，详细的行为见 `docs/binance_runbook.md`。

## 1. 两种试运行

- **校准试运行**：已有的 BTC/ETH 多周期策略（第 1 版规格，只做多，市价单）在主网上收集真实成交（成交价、滑点、止损成交、被拒），
  用来核对回测的假设。它没有被证明有优势，也不指望有；亏损停止线限定了成本。
- **实验室的幸存者**：先冻结规则，在新数据上跟踪 4 周（每周登记一次查看，见 `research/forward_looks.jsonl`），四次都成立才进
  试运行。规格由技术负责人发布；多品种、做空的第 2 版规格要等运行时扩展（PR #119–#130）合并后才能用。

两种都用下面同一份清单。

## 2. 硬限制（程序自己执行）

| 限制 | 设置 | 到线时 |
|---|---|---|
| 日亏 3% | `MAX_DAILY_DRAWDOWN_PCT=3`（代码默认是 5，要由人设成 3；按整个账户的权益算） | 锁定到下一个 UTC 日，按交易所持仓整仓平掉 |
| 累计亏掉策略资金的 15% | `STRATEGY_LOSS_STOP_PCT`（不设就是 15，只能调小） | 打开急停并平仓，停止开仓；这次试运行结束 |
| 急停 | `python3 -m trading.infrastructure.kill ...`（见第 6 节） | 撤掉挂单，带 `--flatten` 时整仓平掉 |
| 交易所止损 | `STOP_PCT` | 交易所侧成交；这个品种暂停入场 4 小时，到时自动解除 |

## 3. 开始之前（由人完成，逐项打勾）

规格
- [ ] 技术负责人亲自把规格标为通过：把研究结果里的 `final_spec.json` 复制到一个新目录，把 `passed` 改为 `true`，再用
  `QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_strategy_spec --study <那个目录> --out models_binance/strategy`
  签名发布。代码从不改 `passed`；`passed` 不是真时主网拒绝启动。
- [ ] 规格里的品种、是否做空和执行方式是这次要跑的（`parse_spec` 拒绝它不认识的东西；启动时的 `config` 事件会记下它们）。

账户
- [ ] 主网 API Key 只开读取和合约交易，不开提现，绑定主机 IP。
- [ ] 账户里没有策略账本以外的东西：任何品种上的手动仓位、普通挂单、条件单（启动检查会拒绝；策略品种上的手动仓位会和策略的
  仓位合在一起）。用只读的 `python3 -m scripts.account_check` 核对。
- [ ] 规格里每个品种在币安上由人设成 10 倍逐仓（程序只核对、从不修改；不符就拒绝启动）。

`.env`（照 `.env.example`；权限 600，属主 ubuntu；只放在主机上）
- [ ] `RISK_SCOPE`：含 `binance`，不含 `testnet`。网络由主网单元固定（`BINANCE_TESTNET=0`）。
- [ ] `BINANCE_API_KEY` / `BINANCE_API_SECRET`；验签用的 `QUANT_MODEL_SIGN_KEY`（主网上未签名的规格一律拒绝）。
- [ ] `BINANCE_EXCHANGE_LEVERAGE=10`、`BINANCE_MARGIN_TYPE=ISOLATED`，与币安上的设置相同。
- [ ] `STRATEGY_CAPITAL_USDT`：技术负责人定的策略资金。主网没有它拒绝启动。
- [ ] `STRATEGY_LEVERAGE`：满仓是策略资金的几倍，1 到 3，不超过 `BINANCE_EXCHANGE_LEVERAGE`，由技术负责人写。
- [ ] `STOP_PCT`：交易所止损的距离（主网单元不带 `--stop-pct`，所以用它）。逐仓时必须在强平距离的一半以内：10 倍逐仓最多
  0.0475，超过就拒绝启动。
- [ ] `MAX_DAILY_DRAWDOWN_PCT=3`；`STRATEGY_LOSS_STOP_PCT` 不设或更小。
- [ ] 下单限额 `BINANCE_MAX_ORDER_NOTIONAL`、`BINANCE_MAX_DAILY_NOTIONAL`、`BINANCE_MAX_DAILY_ORDERS`：按策略的需要由人设
  （实验室的报告写明需要多大）。
- [ ] 告警出口 `ALERT_FILE`（绝对路径，例如 `/var/lib/quant/alerts/mainnet.jsonl`）。Telegram 是另一种出口，可以不用；两者同时设
  会拒绝启动。
- [ ] `HEARTBEAT_URL`：外部心跳监控的 https 地址（每分钟最多一次），失联时它通知到人。心跳监控在另一处，不在这台主机上。

主机
- [ ] 检出目录在 `/quant_system`，是 master 上已合并的提交。
- [ ] `sudo bash /quant_system/infra/install_host.sh` 全部步骤通过（建 `/var/lib/quant` 下的 state、backups、alerts 目录，同步服务
  单元，最后跑离线安全检查；从不启动服务）。
- [ ] 以 ubuntu 身份、`RISK_SCOPE` 与 `.env` 相同，初始化状态目录：
  `python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/mainnet init`，再 `check`。
- [ ] 开启备份：`sudo systemctl enable --now state-backup@mainnet.timer`；一小时后 `/var/lib/quant/backups/mainnet/` 里有快照。
- [ ] 有人读告警文件：现在是 Claude Code 会话，大约每小时一次；它只能自己做降低风险的动作（打开急停、停止服务），重启、解除急停
  和暂停、改配置都要先问技术负责人。

还没做完的工程项（`CLAUDE.md` 的 "still required first"）
- [ ] 累计资金费存进状态库，取代"首笔成交早于 88 天时仓位基数为 0"的保护（技术负责人 2026-11-16：在那之前保护保留）。

## 4. 启动

- [ ] `sudo systemctl start binance-target-mainnet.service`（不自动重启，不能开机启动，只能人手动启动）。
- [ ] 告警文件里有一行 "MAINNET execution starting"（带 scope、账户身份前缀和状态目录）。没有这一行就没有任何账户请求。
- [ ] `/var/lib/quant/state/mainnet/target_status.json` 里各品种的状态正常，`config_digest` 与启动日志一致。
- [ ] 心跳监控显示在线。
- [ ] 记下开始时间（UTC）和提交号。

试运行里要验证的事（原来只能在测试网做）
- [ ] 同方向已有一张整仓止损时，币安是否接受第二张。持仓后让止损的修订号变化（例如加一点仓位），看新止损是否挂上、旧止损是否在
  新止损挂上之后被撤掉。结果交技术负责人；被拒时由技术负责人决定是否改成先撤后挂。

## 5. 每日检查

- [ ] 过去 24 小时的告警都已解释（写在运行记录里）。
- [ ] 状态文件：没有 blocked / halted / paused 的品种，或者原因已知。
- [ ] 事件日志里没有 `ledger_mismatch`、`foreign_holding`、`foreign_fill`、`unresolved_order` 持续超过一根 K 线的情况。
- [ ] 最近一小时有新的备份快照。
- [ ] 心跳监控没有失联记录；磁盘用量（`df`）正常。
- [ ] 真实成交和回测的假设对照：成交价、滑点、止损成交、被拒（校准试运行的目的）。不因此改参数。

任何一项不符：记录、查明原因；对账差异或未解释的告警让 4 周计时从头开始。

## 6. 停下来和收尾

| 情况 | 做什么 |
|---|---|
| 要立即停止开仓 | `python3 -m trading.infrastructure.kill --state-dir /var/lib/quant/state/mainnet activate --reason "..."` |
| 要立即平掉全部仓位 | 同上，加 `--flatten`。只停服务（`sudo systemctl stop binance-target-mainnet.service`）不会平仓，交易所止损仍在 |
| 亏损停止线触发 | 这次试运行结束。只解除急停会马上再次触发（亏损还在账本里）。要继续由技术负责人决定开一次新的试运行：新的状态目录（`state_dir init`），重新设 `STRATEGY_CAPITAL_USDT`；旧目录留作记录 |
| 日亏 3% 锁定 | 程序已整仓平掉；锁定到下一个 UTC 日自己解除；查原因 |
| 急停的解除 | 只由人：`python3 -m trading.infrastructure.kill --state-dir <目录> clear`；运行时从不关闭急停 |

## 7. 演练

实盘上的演练会花真钱（平仓的手续费），做哪几项、什么时候做由技术负责人定；做了就写记录。

| 演练 | 怎么做 | 预期 |
|---|---|---|
| 告警送达 | `sudo systemctl start quant-alert@drill.service` | 告警文件里多一行 "drill failed on …" |
| 心跳失联 | 停服务超过心跳监控的宽限时间 | 心跳监控通知到人 |
| 重启 | 人停服务再启动 | 启动检查全部重新通过；订单库和交易所一致 |
| 急停平仓 | `kill ... activate --reason ... --flatten` | 撤掉挂单，按交易所持仓整仓平掉；告警；之后由人 `kill clear` |
| 备份恢复 | 只在状态库真的丢失或损坏时：停服务，`state_dir restore --snapshot <最新快照> --reason …` | 回撤锁定、当天计数用完、所有品种暂停；**下次启动按市价平掉所有仓位，本周空仓**；人逐个解除暂停 |

### 记录模板

```
事件或演练：
日期（UTC）与提交号：
步骤（实际执行的命令）：
预期：
实际（事件日志行、告警文件的行、状态文件片段）：
结果：
后续：
```
