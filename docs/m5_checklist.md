# M5 测试网连续运行：开始前的核对清单、每日检查与演练记录（S11）

依据：ADR-0016 第 7 条（通过标准）、`docs/arch/design-2026-11-m5.md` 第 6 节和第 10 节 S11。**这不是实盘批准。** 只在测试网运行；主网在当前规格下一定被拒绝。

## 1. 通过标准（ADR-0016 第 7 条）

- 连续 4 周对账无差异，没有未解释的告警。
- 7 类演练各至少一次，并有记录（见第 4 节）：急停平仓、回撤锁定、止损触发、重启恢复、备份恢复、告警送达、心跳失联。
- 出现任何差异都从头计时；往测试网主机部署资金路径的改动也从头计时。

## 2. 开始计时之前（由人完成，逐项打勾）

账户与密钥
- [ ] 测试网 API Key：只开读取和合约交易，不开提现，绑定主机 IP。
- [ ] 测试网账户里 BTCUSDT、ETHUSDT 的杠杆和保证金模式按 `.env` 设好（程序只核对，不修改）。
- [ ] 账户里没有手动仓位、挂单或条件单（启动检查会拒绝）。

告警与心跳
- [ ] Telegram 机器人和聊天：`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`。
- [ ] 外部心跳监控（死信开关类服务）：`HEARTBEAT_URL`（https），宽限 5 分钟以内，失联时通知到人。心跳在另一处，不在这台主机上。

主机
- [ ] 检出目录在 `/quant_system`，是 master 上已合并的提交。
- [ ] 先在一台干净的主机或容器里跑一次 `sudo bash infra/install_host.sh`（S8 的验收项，开发机上没做）。
- [ ] 在目标主机上 `sudo bash /quant_system/infra/install_host.sh`，全部步骤通过。
- [ ] 写好 `/quant_system/.env`（照 `.env.example`，权限 600，属主 ubuntu）。
- [ ] 以 ubuntu 身份初始化：`python3 -m trading.infrastructure.state_dir --state-dir /var/lib/quant/state/testnet init`，再 `check`。
- [ ] 开启备份：`sudo systemctl enable --now state-backup@testnet.timer`；一小时后 `/var/lib/quant/backups/testnet/` 里有快照。

只能在测试网验证的事（ADR-0012 实施更正，S4–S5 审计第 3 条）
- [ ] 同方向已有一张整仓止损时，币安是否接受第二张。方法：在测试网持仓后让修订号变化（加一点仓位），看新止损是否挂上、旧止损是否在新止损挂上后被撤掉。结果交技术负责人；如果被拒，由技术负责人决定是否改成先撤后挂。

启动
- [ ] `sudo systemctl start binance-target-testnet.service`。
- [ ] Telegram 收到 "testnet execution starting"（含 scope 和账户身份前缀）。
- [ ] `target_status.json` 里各品种状态正常，`config_digest` 与启动日志一致。
- [ ] 心跳监控显示在线。
- [ ] 记下计时开始的时间（UTC）和提交号。

## 3. 每日检查

- [ ] 过去 24 小时的告警都已解释（写在运行记录里）。
- [ ] 状态文件：没有 blocked / halted / paused 的品种，或者原因已知。
- [ ] 事件日志里没有 `ledger_mismatch`、`foreign_holding`、`foreign_fill`、`unresolved_order` 持续超过一根 K 线的情况。
- [ ] 最近一小时有新的备份快照。
- [ ] 心跳监控没有失联记录。
- [ ] 成本核对（设计第 10 节）：只核对延迟和流程；滑点用主网公开逐笔成交另估，不改参数。

任何一项不符：记录、查明原因；如果是对账差异或未解释的告警，计时从头开始。

## 4. 演练（每类至少一次，各写一份记录）

| 演练 | 怎么做 | 预期 |
|---|---|---|
| 急停平仓 | `kill --state-dir … activate --reason … --flatten` | 撤掉挂单，按交易所持仓整仓平掉；告警；之后 `kill clear` |
| 回撤锁定 | 在测试网把日回撤限额临时调低（改 `.env`，人来改，重启服务）到当前回撤以下 | 锁定到下一个 UTC 日，整仓平掉；告警；演练后恢复限额 |
| 止损触发 | 在测试网持仓时人为让价格穿过止损（或用很近的 `--stop-pct` 单独跑一次） | 交易所止损成交；品种暂停入场；告警；人用 `entry_pause release` 解除 |
| 重启恢复 | `systemctl kill binance-target-testnet`（模拟崩溃） | 60 秒后自动重启；每次失败都有告警；启动检查全部重新通过；订单库和交易所一致 |
| 备份恢复 | 停服务；把 `state.sqlite` 移走；`state_dir restore --snapshot <最新快照> --reason …` | 恢复后回撤锁定、当天计数用完、所有品种暂停；急停文件没变；启动检查通过；**启动后所有仓位按市价平掉，本周空仓**；人逐个解除暂停 |
| 告警送达 | 停服务，`sudo systemctl start quant-alert@drill.service` | Telegram 收到 "drill failed on …" |
| 心跳失联 | 停服务超过心跳宽限 | 心跳监控通知到人；恢复后在线 |

演练前后都执行 `sudo systemctl reset-failed binance-target-testnet.service`（启动次数限制把人手动启动也算进去）。

### 演练记录模板

```
演练：
日期（UTC）与提交号：
步骤（实际执行的命令）：
预期：
实际（事件日志行、Telegram 消息、状态文件片段）：
结果：通过 / 不通过
后续：
```
