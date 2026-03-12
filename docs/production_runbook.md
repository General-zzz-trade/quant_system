# Production Runbook

> 更新时间: 2026-03-12
> 作用: 固定当前生产路径上的恢复与排障顺序，避免 live runtime、checkpoint、user stream、reconcile 出现多套口径
> 适用范围: 当前 Python-default 生产 runtime
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前生产入口

- 默认生产入口: `runner/live_runner.py`
- 当前运行时形态: Python 编排 + Rust 热路径内核
- 当文档与其他说明冲突时，以 `docs/runtime_truth.md` 为准

---

## 2. 启动恢复顺序

`LiveRunner.build()` 当前的恢复链路是：

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
- `tests/execution_safety/test_duplicate_events.py`
- `tests/execution_safety/test_out_of_order_fills.py`

---

## 8. 仍未完全收口

- startup reconcile 目前只报告 mismatch，不自动修复
- user stream 断连后的恢复仍主要依赖 reconnect + reconcile，而非更强的 healer 流程
- timeout、late fill、reconcile 之间还缺少更强的端到端集成测试
- live runtime 的恢复动作语义还没有统一成单一 canonical incident policy

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
