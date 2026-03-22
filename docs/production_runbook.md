# Production Runbook

> 更新时间: 2026-03-22
> 作用: Strategy H 生产配置 + 活跃服务排障 + framework recovery 路径
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. Strategy H Production Configuration

### 1.1 Runner 架构 (6 runners)

| Runner | Timeframe | Role | Cap (BTC) | Cap (ETH) |
|--------|-----------|------|-----------|-----------|
| BTC 4h | 4h | Primary direction | 15% | — |
| BTC 1h | 1h | Position scaler (MultiTFConfluenceGate) | 8% | — |
| BTC 15m | 15m | COMBO AGREE with 1h | 5% | — |
| ETH 4h | 4h | Primary direction | — | 10% |
| ETH 1h | 1h | Position scaler (MultiTFConfluenceGate) | — | 6% |
| ETH 15m | 15m | COMBO AGREE with 1h | — | 5% |

Key rules:
- **4h = primary direction**, trades independently (not in COMBO)
- **1h = position scaler** via `MultiTFConfluenceGate`, scaled by 4h alignment:
  - 4h agrees: **1.3x** boost
  - 4h opposes: **0.3x** reduce
  - 4h neutral: **0.7x**
- **15m = COMBO AGREE with 1h** — both must agree direction; cap 5%

### 1.2 Risk Management

**Dynamic leverage (drawdown-based)**:
- DD ≥ 10% → 0.75x leverage
- DD ≥ 20% → 0.5x leverage
- DD ≥ 35% → 0.25x leverage

**BB Entry Scaler** (Bollinger Band position sizing):
- Oversold → 1.2x size
- Overbought → 0.3x size

**Vol-adaptive deadzone**:
- Formula: `dz × (realized_vol / vol_median)`
- Clamped to `[0.5x, 2.0x]` of base deadzone

**ATR trailing stop**:
- Initial: 1.2 × ATR
- Trail step: 0.2 × ATR

**4h z-score stop**:
- 1h/15m runners exit when 4h signal reverses

### 1.3 Maker Order Optimization

- **PostOnly limit orders**, 45s timeout for tight spreads
- **1-tick spread**: place at bid/ask (no tick improvement to avoid PostOnly reject)
- **Fill rate tracking**: `_limit_fills / _market_fallbacks` logged every 10 trades

### 1.4 Recommended Production Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Leverage | **3x** | MaxDD ~18%, CAGR ~115% |
| Initial capital | $500–2000 | |
| Hard stop | **-$200** | Micro-live phase loss limit |
| MAX_ORDER_NOTIONAL | $5,000 | Hard limit in config.py |

---

## 2. 适用范围

当前仓库有三类运行时，排障口径不同：

| 路径 | 入口 | 当前状态 | 本文适用内容 |
|---|---|---|---|
| 方向性 alpha | [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) | 当前活跃 host service | 适用第 3-5 节 |
| Framework runtime | [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | 候选 / 收敛路径 | 适用第 6-10 节 |

最重要的边界：

- `bybit-alpha.service` 当前不是 `LiveRunner`
- `checkpoint / startup reconcile / user stream reconnect / ops-audit / POST /control` 这类语义只对 framework path 成立

---

## 3. 当前活跃 host services

### 3.1 方向性 alpha

- 服务名: `bybit-alpha.service`
- 入口命令: `python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m --ws`
- 日志: [`logs/bybit_alpha.log`](/quant_system/logs/bybit_alpha.log)
- 事实来源:
  - `systemctl status bybit-alpha.service`
  - `journalctl -u bybit-alpha.service`
  - `tail -f /quant_system/logs/bybit_alpha.log`
  - Bybit demo 账户真实余额 / 持仓 / 挂单 / 成交

当前持久 kill 规则：

- `bybit-alpha.service` 的 drawdown / WS stale kill 会写入 `data/runtime/kills/`
- 只要对应 kill latch 还在，服务下次启动就必须失败，不允许用重启绕过风控
- 人工复核后才允许 clear：
  - `python3 -m scripts.ops.runtime_kill_latch --service alpha --json`
  - `python3 -m scripts.ops.runtime_kill_latch --service alpha --clear --json`

常用命令：

```bash
sudo systemctl restart bybit-alpha.service
sudo systemctl status bybit-alpha.service --no-pager -l
journalctl -u bybit-alpha.service -n 100 --no-pager
tail -f /quant_system/logs/bybit_alpha.log
python3 -m scripts.ops.runtime_health_check --service alpha
python3 -m scripts.ops.runtime_kill_latch --service alpha --json
```

---

## 4. 方向性 alpha 排障顺序

1. 先看服务是否真的在跑
   - `sudo systemctl status bybit-alpha.service`
2. 再看业务日志是否继续前进
   - `tail -f /quant_system/logs/bybit_alpha.log`
   - 心跳日志关键字：`WS HEARTBEAT`
3. 再看交易所端真实状态（余额、持仓、挂单、成交）
4. 只有进程活着还不算"交易活着"
   - 如果日志时间戳不再前进，或账户长期 `0 持仓 / 0 挂单 / 0 新成交`，不能仅凭 systemd `active` 判断服务健康
5. 如果触发 drawdown kill，必须把它当成"禁止新开仓且应尽快收平"的状态
   - 如果 kill 已经持久化到 `data/runtime/kills/`，直接 `systemctl restart` 不会恢复交易；必须先查明原因，再手工 clear

---

## 5. 方向性 alpha 的"成功启动"标准

对 `bybit-alpha.service`，只有以下条件同时满足，才算成功启动：

1. `systemd` 显示 `active (running)`
2. 日志出现新的当前时间戳，而不是停在旧时间
3. WebSocket 已连接
4. 至少满足以下之一：出现新的 heartbeat / 挂单 / 持仓 / 成交

不允许把下面情况误判为成功：仅有 PID / 仅有 `systemd active` / 仅有启动 banner 但账户无变化

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

1. 启动时执行 `user_stream.connect()`
2. 后台线程循环调用 `user_stream.step()`
3. `step()` 抛异常时记录告警并重连
4. 停机时调用 `user_stream.close()` 并等待线程退出

### 7.2 Timeout

- `timeout_tracker.check_timeouts()` 在主循环中持续运行
- timeout 不等于 venue 必定未成交
- timeout 后若触发 cancel，状态机进入 `pending_cancel`；之后若收到晚到 fill，收敛到 `filled`

### 7.3 Reconcile

覆盖: positions, balances, fills, orders

- `warning` drift：告警但继续运行
- `critical` drift：可触发 halt callback
- 当前以"先发现、再人工决策"为主

---

## 8. Framework 路径的 checkpoint / restart 真相

- coordinator snapshot 可保存并恢复
- inference bridge / tick processor 支持 checkpoint / restore
- timeout cancel -> checkpoint / restart -> late fill 组合恢复已有测试覆盖
- restart 后先恢复本地状态，再做 startup reconcile
- 不应跳过 reconcile 直接信任旧 checkpoint
- 若 checkpoint 可恢复但 venue 已漂移，必须人工确认仓位与余额

---

## 9. Framework 路径的验证测试

- [`tests/integration/test_crash_recovery.py`](/quant_system/tests/integration/test_crash_recovery.py)
- [`tests/integration/test_execution_recovery_e2e.py`](/quant_system/tests/integration/test_execution_recovery_e2e.py)
- [`tests/integration/test_execution_timeout_restart_recovery.py`](/quant_system/tests/integration/test_execution_timeout_restart_recovery.py)
- [`tests/integration/test_operator_control_recovery_flow.py`](/quant_system/tests/integration/test_operator_control_recovery_flow.py)
- [`tests/persistence/test_checkpoint_restore.py`](/quant_system/tests/persistence/test_checkpoint_restore.py)
- [`tests/unit/runner/test_live_runner.py`](/quant_system/tests/unit/runner/test_live_runner.py)

---

## 10. 当前仍未完全收口

- `bybit-alpha.service` 目前没有统一纳入 framework recovery / control plane
- startup reconcile 目前只报告 mismatch，不自动修复
- user stream 断连后的恢复仍主要依赖 reconnect + reconcile
- active host services 与 compose / CI deploy 路径仍然分叉
