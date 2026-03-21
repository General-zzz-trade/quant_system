# Production Runbook

> 更新时间: 2026-03-20
> 作用: 固定当前主机上活跃交易服务与 framework recovery 路径的排障顺序，避免把不同运行时混成同一套口径
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 适用范围

当前仓库有三类运行时，排障口径不同：

| 路径 | 入口 | 当前状态 | 本文适用内容 |
|---|---|---|---|
| 方向性 alpha | [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) | 当前活跃 host service | 适用第 2-4 节 |
| 高频做市 | [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) | 当前活跃 host service | 适用第 2 节、第 5 节 |
| Framework runtime | [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | 候选 / 收敛路径 | 适用第 6-10 节 |

最重要的边界：

- `bybit-alpha.service` 当前不是 `LiveRunner`
- `bybit-mm.service` 当前不是 `LiveRunner`
- `checkpoint / startup reconcile / user stream reconnect / ops-audit / POST /control` 这类语义只对 framework path 成立

---

## 2. 当前活跃 host services

### 2.1 方向性 alpha

- 服务名: `bybit-alpha.service`
- 入口命令: `python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m --ws`
- 日志: [`logs/bybit_alpha.log`](/quant_system/logs/bybit_alpha.log)
- 事实来源:
  - `systemctl status bybit-alpha.service`
  - `journalctl -u bybit-alpha.service`
  - `tail -f /quant_system/logs/bybit_alpha.log`
  - Bybit demo 账户真实余额 / 持仓 / 挂单 / 成交

### 2.2 高频做市

- 服务名: `bybit-mm.service`
- 入口命令: `python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20 ...`
- 日志: [`logs/bybit_mm.log`](/quant_system/logs/bybit_mm.log)
- 事实来源:
  - `systemctl status bybit-mm.service`
  - `journalctl -u bybit-mm.service`
  - `tail -f /quant_system/logs/bybit_mm.log`
  - Bybit demo 账户真实余额 / 持仓 / 挂单 / 成交

当前 host 互斥规则：

- `bybit-alpha.service` 与 `bybit-mm.service` 不能在同一 Bybit 账户 / 同一 symbol 上同时启动
- 启动时会按 `base_url + account_type + category + api_key` 归一到账户作用域，再对 symbol 加主机级文件锁
- 冲突时进程应非零退出，而不是带着脏共享账户状态继续运行
- 启动 guard 当前统一使用退出码 `73`，并由 systemd unit 的 `RestartPreventExitStatus=73` 阻止无限重启风暴

当前持久 kill 规则：

- `bybit-alpha.service` 的 drawdown / WS stale kill 会写入 `data/runtime/kills/`
- `bybit-mm.service` 的 daily-loss hard kill 会写入 `data/runtime/kills/`
- 只要对应 kill latch 还在，服务下次启动就必须失败，不允许用重启绕过风控
- 人工复核后才允许 clear：
  - `python3 -m scripts.ops.runtime_kill_latch --service alpha --json`
  - `python3 -m scripts.ops.runtime_kill_latch --service alpha --clear --json`
  - `python3 -m scripts.ops.runtime_kill_latch --service mm --symbol ETHUSDT --json`
  - `python3 -m scripts.ops.runtime_kill_latch --service mm --symbol ETHUSDT --clear --json`

---

## 3. 方向性 alpha 排障顺序

`bybit-alpha.service` 当前没有 `LiveRunner` 的 health / control / checkpoint 恢复面，因此排障必须按下面顺序：

1. 先看服务是否真的在跑
   - `sudo systemctl status bybit-alpha.service`
2. 再看业务日志是否继续前进
   - `tail -f /quant_system/logs/bybit_alpha.log`
   - 心跳日志关键字：`WS HEARTBEAT`
3. 再看交易所端真实状态
   - 余额
   - 当前持仓
   - 当前挂单
   - 最近成交
4. 只有进程活着还不算“交易活着”
   - 如果日志时间戳不再前进，或账户长期 `0 持仓 / 0 挂单 / 0 新成交`，不能仅凭 systemd `active` 判断服务健康
5. 如果触发 drawdown kill，必须把它当成“禁止新开仓且应尽快收平”的状态
   - 当前 `bybit-alpha` 会压平 runner 本地信号，避免 heartbeat 长期显示假 `sig=1`
   - 对 COMBO symbol，会在后续 bar 拿到该 symbol 最新价后强制平掉组合仓位，并同步 `PortfolioManager`
   - 同样的 `PortfolioManager / PortfolioCombiner / kill enforcement` 语义现在也适用于 `--once` 和 REST poll fallback，不再只有 `--ws` 路径是完整行为
   - `python3 -m scripts.ops.runtime_health_check --service alpha` 现在会把 `pm.killed=True` 直接判为失败，不再把这种状态误报成健康
   - 如果 kill 已经持久化到 `data/runtime/kills/`，直接 `systemctl restart` 不会恢复交易；必须先查明原因，再手工 clear

常用命令：

```bash
sudo systemctl restart bybit-alpha.service
sudo systemctl status bybit-alpha.service --no-pager -l
journalctl -u bybit-alpha.service -n 100 --no-pager
tail -f /quant_system/logs/bybit_alpha.log
python3 -m scripts.ops.runtime_health_check --service alpha
python3 -m scripts.ops.runtime_kill_latch --service alpha --json
```

当前限制：

- 没有 framework health server
- 没有 framework `/control` / `/ops-audit`
- 没有 `LiveRunner` checkpoint / startup reconcile 恢复链

---

## 4. 方向性 alpha 的“成功启动”标准

对 `bybit-alpha.service`，只有以下条件同时满足，才算成功启动：

1. `systemd` 显示 `active (running)`
2. 日志出现新的当前时间戳，而不是停在旧时间
3. WebSocket 已连接
4. 至少满足以下之一：
   - 出现新的 heartbeat
   - 出现新的挂单
   - 出现新的持仓
   - 出现新的成交

不允许把下面情况误判为成功：

- 只有 PID 存在
- 只有 `systemd active`
- 只有程序启动 banner，但账户真相没有任何变化

---

## 5. 高频做市排障顺序

`bybit-mm.service` 的健康判断标准比方向性 alpha 更严格：

1. `systemd` 状态
2. `logs/bybit_mm.log` 是否持续有最新时间戳
3. 是否出现新的 quote / fill / metrics 日志
4. 账户侧是否存在挂单、仓位或新成交

典型假活场景：

- `systemd active`
- 进程仍在
- 但 `logs/bybit_mm.log` 停在旧时间
- 账户侧 `0 持仓 / 0 挂单`

这种情况必须按“交易没有真的在跑”处理，而不是按“服务正常”处理。

当前 [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) 已增加 market-data stale fail-fast：

- 若长时间没有新的 WS 消息或 orderbook depth，进程会显式报错退出
- `bybit-mm.service` 依赖 `Restart=on-failure` 重新拉起
- 这比“进程继续存活但失去行情驱动”更安全
- 若触发 daily-loss hard kill，会落盘为持久 kill latch；后续启动会直接被拒绝，直到人工 clear

常用命令：

```bash
sudo systemctl restart bybit-mm.service
sudo systemctl status bybit-mm.service --no-pager -l
journalctl -u bybit-mm.service -n 100 --no-pager
tail -f /quant_system/logs/bybit_mm.log
python3 -m scripts.ops.runtime_health_check --service mm
python3 -m scripts.ops.runtime_kill_latch --service mm --symbol ETHUSDT --json
```

如果要一次检查两个当前活跃 host services：

```bash
python3 -m scripts.ops.runtime_health_check
```

---

## 6. Framework 路径的恢复链路

以下内容只对 `runner/live_runner.py` / `quant-framework` / `quant-runner.service` 成立。

`LiveRunner.build()` 的当前恢复链路是：

1. 装配 coordinator / execution / risk / monitoring
2. 如启用持久化，打开 SQLite stores
3. 从 state store 读取最新 checkpoint
4. 调用 `coordinator.restore_from_snapshot()`
5. 如启用 `reconcile_on_startup`，拉取 venue state 做启动对账
6. 启动 runtime、user stream、reconcile scheduler、health、shutdown handler

处理原则：

- checkpoint 是本地状态恢复来源
- venue state 是事实校验来源
- startup reconcile 发现 mismatch 时当前只告警，不自动修复

---

## 7. Framework 路径的 user stream / timeout / reconcile 语义

### 7.1 User stream

当前 `LiveRunner.start()` 的 user stream 策略：

1. 启动时执行 `user_stream.connect()`
2. 后台线程循环调用 `user_stream.step()`
3. `step()` 抛异常时记录告警并重连
4. 停机时调用 `user_stream.close()` 并等待线程退出

### 7.2 Timeout

- `timeout_tracker.check_timeouts()` 在主循环中持续运行
- timeout 表示本地在阈值内未观察到终态
- timeout 不等于 venue 必定未成交
- timeout 后若触发 cancel，状态机会进入 `pending_cancel`
- 之后若收到晚到 fill，应收敛到 `filled`

### 7.3 Reconcile

当前 reconcile 覆盖：

- positions
- balances
- fills
- orders

处理原则：

- `warning` drift：告警但继续运行
- `critical` drift：可触发 halt callback
- 当前仍以“先发现、再人工决策”为主

---

## 8. Framework 路径的 checkpoint / restart 真相

当前已锁住的恢复能力：

- coordinator snapshot 可保存并恢复
- inference bridge / tick processor 支持 checkpoint / restore
- timeout cancel -> checkpoint / restart -> late fill 组合恢复已有测试覆盖
- restart 后 venue 缺失 late fill 时，可通过 reconcile 明确报 drift

关键原则：

- restart 后先恢复本地状态，再做 startup reconcile
- 不应跳过 reconcile 直接信任旧 checkpoint
- 若 checkpoint 可恢复但 venue 已漂移，必须人工确认仓位与余额

---

## 9. Framework 路径的验证测试

关键测试入口：

- [`tests/integration/test_crash_recovery.py`](/quant_system/tests/integration/test_crash_recovery.py)
- [`tests/integration/test_execution_recovery_e2e.py`](/quant_system/tests/integration/test_execution_recovery_e2e.py)
- [`tests/integration/test_execution_timeout_restart_recovery.py`](/quant_system/tests/integration/test_execution_timeout_restart_recovery.py)
- [`tests/integration/test_operator_control_recovery_flow.py`](/quant_system/tests/integration/test_operator_control_recovery_flow.py)
- [`tests/persistence/test_checkpoint_restore.py`](/quant_system/tests/persistence/test_checkpoint_restore.py)
- [`tests/unit/runner/test_live_runner.py`](/quant_system/tests/unit/runner/test_live_runner.py)
- [`tests/unit/monitoring/test_alert_manager.py`](/quant_system/tests/unit/monitoring/test_alert_manager.py)
- [`tests/unit/monitoring/test_health_monitor_extended.py`](/quant_system/tests/unit/monitoring/test_health_monitor_extended.py)

---

## 10. 当前仍未完全收口

- `bybit-alpha.service` 与 `bybit-mm.service` 目前没有统一纳入 framework recovery / control plane
- startup reconcile 目前只报告 mismatch，不自动修复
- user stream 断连后的恢复仍主要依赖 reconnect + reconcile
- active host services 与 compose / CI deploy 路径仍然分叉
