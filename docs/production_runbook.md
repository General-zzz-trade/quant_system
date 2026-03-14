# Production Runbook

> 更新时间: 2026-03-15
> 作用: 固定当前生产路径上的恢复与排障顺序，避免 live runtime、checkpoint、user stream、reconcile 出现多套口径
> 适用范围: 当前 Python-default 生产 runtime
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前生产入口

- 默认生产入口: `runner/live_runner.py` (legacy) 或 `runner/run_trading.py` (新分解入口)
- 新分解模块: `TradingEngine`, `RiskManager`, `OrderManager`, `BinanceExecutor`, `RecoveryManager`, `LifecycleManager`, `RunnerLoop` (各 <200 LOC, 独立可测试)
- 默认发布路径: repo-root `docker-compose.yml` + `.github/workflows/ci.yml` + `.github/workflows/deploy.yml` + `scripts/deploy.sh`
- 默认 compose 服务名: `paper-multi`
- `deploy/` 下其他 systemd / k8s / argocd / docker 示例工件都视为 candidate/experimental，不是当前默认发布真相源
- 当前运行时形态: Python 编排 + Rust 热路径内核
- 当文档与其他说明冲突时，以 `docs/runtime_truth.md` 为准
- 当前最小 operator controls 已通过 `LiveRunner.halt()/reduce_only()/resume()/flush()/shutdown()/apply_control()` 暴露
- 当前外部 tooling / API 建议通过 [`runner/control_plane.py`](/quant_system/runner/control_plane.py) 的 `OperatorControlPlane` 统一进入 runtime
- 如启用 `health_port`，当前 health server 也暴露 `GET /operator`、`GET /control-history`、`GET /execution-alerts`、`GET /ops-audit` 与 `POST /control`
- 如配置 `health_auth_token_env`，上述 health/control 端点统一要求 `Authorization: Bearer <token>`
- 当前 `LiveRunner.operator_status()` / `LiveRunner.control_history` 可用于查看最后一次控制动作与最近 reconcile / kill-switch 状态
- 当前 operator / ops 视图已补稳定 incident 字段：`stream_status`、`incident_state`、`last_incident_category`、`last_incident_ts`、`recommended_action`
- 当前 operator control 动作也会进入统一 alert 链路，作为结构化 ops alert 发出

---

## 2. 启动恢复顺序

`LiveRunner.build()` (legacy) 或 `run_trading.build_runner()` (new) 的恢复链路是：

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

## 3. User Stream 断连

当前 `LiveRunner.start()` 的用户流策略：

1. 启动时先执行一次 `user_stream.connect()`
2. 后台线程循环调用 `user_stream.step()`
3. 若 `step()` 抛异常，记录告警
4. 等待 1 秒后再次执行 `connect()`
5. 主进程停止时调用 `user_stream.close()` 并等待线程退出

运维判断：

- 若日志持续出现 `User stream reconnect failed`，说明私有流不可恢复
- 此时不应仅依赖本地订单状态，应立刻查看 reconcile 报告与 venue 真实订单
- 若 user stream 长时间不可用，应考虑人工降级到 reduce-only 或停机

---

## 4. Timeout / Pending Order

当前主循环每轮都会执行：

- `timeout_tracker.check_timeouts()`

行为：

- 超过 `pending_order_timeout_sec` 的订单会被标记为 timed out
- 若配置了 cancel 回调，会触发取消
- timed out 只说明本地未观察到终态，不代表 venue 一定未成交

处理顺序：

1. 先看 timeout 日志中的 order id
2. 再查 venue 真实订单状态
3. 再看下一轮 reconcile 是否出现 drift
4. 若出现 fills/orders mismatch，以 venue 事实为准做人工判断

特殊情况：

- timeout 后如果先进入 `pending_cancel`，随后又收到晚到 fill，不应简单视为异常脏数据
- 当前订单状态机允许 `pending_cancel -> filled`
- 运维上应把这类事件视为“撤单请求到达过晚，订单已成交”

---

## 5. Reconcile Drift

当前 reconcile 覆盖：

- positions
- balances
- fills
- orders

处理原则：

- `warning` drift: 告警但继续运行
- `critical` drift: 可触发 halt callback
- startup reconcile 与 periodic reconcile 目前都以“先发现、再人工决策”为主

建议排障顺序：

1. 确认 drift 类型: position / balance / fill / order
2. 对比本地 snapshot 与 venue state
3. 检查是否由 user stream 断连、重复 fill、乱序 fill、timeout 引起
4. 若 drift 已不可解释，触发 kill / 停机，避免继续放大风险

cancel-replace 注意事项：

- 原单进入 `canceled` 终态后，后续若又收到该原单的 fill，应视为状态冲突并优先人工核查
- 替换单与原单必须被视为两笔独立订单，不能把原单的晚到回报错误归到替换单上

---

## 6. Checkpoint / Restart

当前已验证的恢复能力：

- coordinator snapshot 可保存并恢复
- inference bridge / tick processor 支持 checkpoint / restore
- checkpoint/restart 一致性已有测试覆盖
- timeout cancel -> checkpoint/restart -> late fill 的组合恢复已有测试覆盖

恢复原则：

- restart 后先恢复本地状态，再做 startup reconcile
- 不应跳过 reconcile 直接信任旧 checkpoint
- 若 checkpoint 可恢复但 venue 已漂移，必须人工确认仓位与余额

---

## 7. 当前已锁住的恢复测试

- `tests/integration/test_crash_recovery.py`
- `tests/integration/test_execution_recovery_e2e.py`
  - 乱序 + 重复 fill + checkpoint/restart 后最终状态与 one-shot 一致
  - restart 后若 venue 缺失晚到 fill，可通过 reconcile 明确报 drift
- `tests/integration/test_execution_timeout_restart_recovery.py`
  - timeout 触发 `pending_cancel` 后，checkpoint/restart 仍能正确接住晚到 fill
- `tests/persistence/test_checkpoint_restore.py`
- `tests/unit/execution/test_reconcile_scheduler.py`
- `tests/unit/runner/test_live_runner.py`
  - user stream step 异常后重连
  - main loop timeout 检查
  - startup reconcile 的 position / balance mismatch
- `tests/integration/test_operator_control_execution_flow.py`
  - `halt -> resume -> success fill -> reconcile`
  - `reduce_only -> blocked opening order / allowed flagged order`
- `tests/integration/test_operator_control_recovery_flow.py`
  - `flush -> drift -> manual halt`
  - `startup mismatch -> reduce_only -> flush -> ops audit`
  - `health /ops-audit` 可同时暴露 operator/control/execution/model ops 视图
  - `restart + reconnect + late fill + reconcile + ops audit`
  - `user stream` 持续失效时，可通过 `reduce_only -> flush -> manual halt` 进入人工降级链，且 `ops audit` 可见
  - `checkpoint restore -> reduce_only -> reconcile overlap` 时，incident 仍保持一致，late fill 收敛后维持 `reduce_only` 运行态
  - `model promote -> autoload pending + reduce_only` 时，`ops audit` 可同时暴露 runtime 降级态和模型未 reload 状态
  - `model reload -> reloaded` 后，`ops audit` 会把 `autoload_pending` 收回，并留下最近一次 hot-reload 结果
  - `model reload -> failed` 时，`autoload_pending` 保持为真，`model_reload=failed` 与当前 `reduce_only` 降级态可同时观察
  - 上述模型 reload 场景也会进入 `model_alerts`，用于和 execution incidents 分开观察
  - `ops audit.timeline` 会把 control、execution、model 三类记录按时间统一串起来，便于复盘单次 incident
  - 当前 `ops audit.timeline` 已优先使用持久化 `event_log` 的 `operator_control / execution_incident / model_reload` 记录，并与 registry `model_action` 合并
  - runtime 重启后，`ops audit.timeline` 仍应能够从同一 `event_log + registry` 重建近期 control / execution incident / model reload / model action 复盘链
  - timeline 的默认语义是“先聚合，再按时间倒序排序，最后按查询 `limit` 裁剪”
  - `LiveRunner.build()` 通过 state checkpoint 恢复后，外部 `health /ops-audit` 观察口也应继续给出同一条持久化复盘链
  - incident 聚合字段会在上述场景中稳定反映 `degraded/critical` 与建议动作
- `tests/integration/test_execution_rejection_contract.py`
  - retryable rejection 后的成功重试仍会进入 synthetic fill / ingress
  - 该链路当前也会进入统一 execution alert / ops audit 观察面
- `tests/execution_safety/test_duplicate_events.py`
- `tests/execution_safety/test_out_of_order_fills.py`

---

## 8. 仍未完全收口

- startup reconcile 目前只报告 mismatch，不自动修复
- user stream 断连后的恢复仍主要依赖 reconnect + reconcile，而非更强的 healer 流程
- timeout、late fill、reconcile 之间还缺少更强的端到端集成测试
- live runtime 的恢复动作语义已开始统一到 operator / ops incident 视图，但还未完全覆盖所有 recovery 子系统
- replay 当前只承担“事实序列一致性”和“incident category 映射一致性”验证，不承担新的告警运行时职责

## 8.5 模型重训练与 Alpha 监控

### 自动重训练

当前 cron 配置 `0 2 * * 0` (每周日 2am UTC) 自动检查并重训练:

```bash
# 手动触发
python3 -m scripts.auto_retrain --symbol ETHUSDT --horizons 12,24

# 查看历史
cat logs/retrain_history.jsonl | python3 -m json.tool
```

重训练门控:
- IC gate: 新模型 avg IC > 0.02
- Sharpe gate: 新模型 Sharpe > 1.0
- Comparison gate: 新模型在 >70% 的 OOS 指标上优于旧模型
- Bootstrap gate: bootstrap p5 > 0

失败时自动恢复旧模型，不影响生产。

### Alpha 健康监控

`monitoring/alpha_health.py` 提供实时 IC 追踪:

| 状态 | 触发条件 | 仓位缩放 | 运维动作 |
|---|---|---|---|
| normal | IC 正常 | 1.0 | 无 |
| warning | IC 负值持续 7 天 | 1.0 | 关注日志 |
| reduce | IC 负值持续 14 天 | 0.5 | 减仓运行 |
| halt | IC < -0.02 持续 14 天 | 0.0 | 停止交易，触发重训练 |

排障顺序:
1. 检查 `alpha_ic_{horizon}` Prometheus 指标
2. 查看 `alpha_health.status()` 各 horizon 状态
3. 若进入 reduce/halt，检查 `logs/retrain_history.jsonl` 是否已触发重训练
4. 若重训练失败，手动检查数据质量和特征工程

### IC 加权集成

当前生产模型使用 `ic_weighted` 集成:
- 自动降权 IC 差的 horizon，无需手动调整
- `ic_ema_span: 720` (30 天滚动)
- 若所有 horizon IC 均为负，有效信号为 flat

### 模型性能基线

| 品种 | Sharpe | IC | Walk-Forward | 说明 |
|---|---|---|---|---|
| ETHUSDT | 2.32 | 0.061 | STRONG (100%) | 最稳定，固定配置最优 |
| BTCUSDT | 1.60 | 0.021 | GOOD (95%) | 参数敏感，可考虑自适应 |

---

## 9. Incident Matrix

| 场景 | 现场判断 | 默认动作 | 何时升级 |
|---|---|---|---|
| user stream reconnect 失败 | 私有流失效，本地订单状态可能落后 | 立即查看 venue 与 reconcile 报告 | 持续失败或 drift 增大时降级/停机 |
| timeout，无后续回报 | 本地未见终态，不代表未成交 | 发起 cancel，等待 reconcile | 下一轮 reconcile 出现 order/fill drift |
| `pending_cancel -> filled` | 撤单请求发出太晚 | 视为正常晚到成交，按 `filled` 处理 | 成交数量/价格与本地预期严重不符 |
| 原单 `canceled` 后又收到 fill | 终态后冲突回报 | 不自动恢复原单，人工核查 | 立即告警，必要时 halt |
| duplicate fill，同 payload | 幂等重复 | 忽略 | 无 |
| duplicate fill，不同 payload | 数据损坏或映射错误 | fail fast / 人工核查 | 立即告警 |
| restart 后 reconcile 报缺失 late fill | checkpoint 落后于 venue | 以 venue 事实为准，人工确认 | position/balance/fill drift 为 critical |
| startup reconcile mismatch | 本地恢复态不可信 | 告警，先人工决策 | mismatch 不可解释或持续扩大 |

当前制度：

- venue fill / venue state 高于 timeout、cancel intent、checkpoint
- `pending_cancel` 仍是中间态
- `canceled` / `filled` / `rejected` / `expired` 是终态
- `reduce_only` 是人工降级运行态，应在 operator / ops 视图中表现为 `incident_state=degraded`
