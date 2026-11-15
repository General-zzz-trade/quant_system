# 币安 USDⓈ-M 运行手册（2026-11-15 重写）

本手册只覆盖目标仓位运行时 `runner/binance_target.py`。**本手册不构成实盘批准。** 命令和环境变量都以代码为准；与代码不一致时，以代码为准并改正本文。

## 0. 现状与硬约束

- 只做币安 USDⓈ-M 永续的 BTCUSDT、ETHUSDT。其他交易所的代码已删除。
- 默认是模拟盘（`--dry-run`），默认用测试网（`BINANCE_TESTNET` 不设或为 `1`）。
- 主网执行要求签名规格里的 `passed` 为真。现在的规格是 `passed: false`，所以主网一定被拒绝。
- 敞口上限 3 倍权益（`AccountExposureGate`），由代码常量保证，配置只能调低。
- 回撤限额：默认日 5%、周 12%；硬上限日 10%、周 25%，超过上限拒绝启动。
- 任何代码路径都不会自动提高限额、解除暂停、写急停文件或重启实盘服务。

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
| `BINANCE_EXCHANGE_LEVERAGE` / `BINANCE_MARGIN_TYPE` | 交易所杠杆与保证金模式（逐仓时止损必须在强平距离的一半以内） | 3 / 不改 |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | 告警出口。两个都设才发 Telegram；都不设只写日志；只设一个拒绝启动 | 无 |
| `QUANT_MODEL_SIGN_KEY` | 规格签名密钥。实盘环境下不允许加载未签名的规格 | 无 |

凭据只放在主机的环境变量文件里，不提交、不打印。

## 2. 发布策略规格

```bash
QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_strategy_spec \
    --study research/results/overlay_2026-11-13 --out models_binance/strategy
```

规格里的门槛结果会原样复制：门槛未通过时主网保持拒绝，模拟盘和测试网可以运行。

## 3. 模拟盘（systemd，默认）

```bash
sudo systemctl start binance-target.service      # 单元里写的是 --dry-run
cat /quant_system/data/runtime/binance_target/target_status.json
tail -f /quant_system/logs/binance_target.log
```

- 状态目录默认是 `data/runtime/binance_target/`，里面有：
  - `target_status.json`：每个周期重写；
  - `target_events.jsonl`：意图、阻断、问题、告警；
  - `paper_state.sqlite`：模拟账户，旁边有一份 JSON 镜像。
- 删除 JSON 镜像**不会**重置模拟账户。要重新开始，用 `--reset-paper-state`；旧记录会归档。
- 模拟盘逐根补放错过的 K 线，结果与研究模拟相同（`tests/unit/lab/test_target_runtime.py`）。

## 4. 测试网执行（M5）

1. **初始化状态目录**，每个账户身份一次，由人来做：
   ```bash
   RISK_SCOPE=<scope> BINANCE_TESTNET=1 python -m trading.infrastructure.state_dir --state-dir <dir> init
   RISK_SCOPE=<scope> BINANCE_TESTNET=1 python -m trading.infrastructure.state_dir --state-dir <dir> check
   ```
   - `init` 创建清单、各个库和一个未激活的急停文件；已初始化的目录不会被覆盖。
   - 换目录或换 `RISK_SCOPE` 就是换账户身份，要重新初始化一个新目录。
2. **启动**（手动在前台运行，不经 systemd）：
   ```bash
   python -m runner.binance_target --spec models_binance/strategy/final_spec.json \
       --state-dir <dir> --enable-execution [--stop-pct 0.08]
   ```
   - 在连接账户之前，启动检查依次核对：规格签名和门槛、主网门槛、账户身份、回撤限额上限、止损距离、状态目录。
   - 状态目录要有清单，账户身份要一致，每个库的表要齐，急停文件要能读。任何一项不满足就拒绝启动，并且不创建库、急停文件或清单。
3. **自动重启**：执行模式下要不要自动重启，项目规则规定由人决定。现有单元是模拟盘用的，不能直接改成执行模式；执行模式的单元是 M5 的待定事项。

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

- 在初始化过的目录里，急停文件**丢失时按"激活"处理**。运行时从不写急停文件。
- 止损在交易所侧触发或被击穿后，该品种暂停入场，直到人解除，同时告警。暂停记录丢失时，会根据订单库里的止损记录重新暂停，不会因为丢了文件而放行。
- 手工工具要求运行时已经停止（持有同一把租约）、写明理由、键入确认，并留下审计记录。它们从不凭空生成成交。

## 6. 告警

- `alert` 类事件立即发出。持续的问题（例如没有止损、平仓失败、行情不可用）超过 2 分钟后发出，每 30 分钟重复，恢复时发"已解决"。
- 没有配置 Telegram 时，状态里显示 `alerts: "log only"`，告警只写进程日志。主网是否必须配置告警出口，是 M5 开始前要问的事。

## 7. 主机维护

```bash
sudo bash infra/sync_systemd.sh     # 停用并删除退役单元，复制有变化的单元，重载 systemd；从不重启服务
sudo systemctl restart binance-target.service      # 看过改动后由人重启
sudo bash infra/deploy_logrotate.sh                 # 日志轮转（每天，保留 14 天）
```

- 2026-11-15 退役了一批单元（见 `docs/deploy_truth.md`）。主机上要运行一次同步脚本才会停掉它们。
- 如果主机上还在跑旧的 Docker 容器（影子进程、Bybit），在旧的检出目录里执行 `docker compose down`。
- 看门狗（`health-watchdog`）每 5 分钟检查 binance-target 的状态文件和最新 K 线的延迟，只告警、从不重启。

## 8. 出问题时

| 现象 | 运行时的行为 | 人要做的 |
|---|---|---|
| 行情接口失败 | 退避重试（5/10/20/40 秒…最长 300 秒），连续 5 次后告警；平仓、止损、急停照常执行 | 看日志；网络恢复后自动继续 |
| 交易所说"不可重试"（如品种下线） | 立即告警，阻断入场，按 300 秒重试 | 查原因 |
| 启动时行情不可用 | 不退出；先对账、维护止损、执行急停和回撤平仓，再重试预热 | 同上 |
| 有持仓但没有有效止损 | 阻断入场，2 分钟后告警 | 查止损下单失败的原因 |
| 止损被击穿 | 按账本平仓，暂停该品种入场，告警 | 核对后用 `entry_pause release` 解除 |
| 回撤到达限额 | 锁定到下一个 UTC 日或周，按交易所持仓整仓平掉 | 不用自动解除；查原因 |
| 状态文件损坏或丢失（启动时） | 拒绝启动并说明缺什么 | 从备份恢复，或初始化新目录（等于换账户身份） |
| 限额计数读不出 | 拒单（按"已用完"处理） | 查状态目录 |

## 9. 主网

主网目前一定被拒绝。在考虑主网之前，必须全部满足：
- 前向检验按登记的条件判定为"继续合格"；
- 退出侧缺口已经修好（见 `CLAUDE.md`）；
- 纸面运行日志与回放核对一致；
- 账户所有人本人决定。

`final_spec.json` 的 `passed` 不由代码修改。
