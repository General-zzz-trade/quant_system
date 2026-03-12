# Execution Contracts

> 更新时间: 2026-03-12
> 目标: 固定当前执行层关于 order status、late fill、timeout cancel、canonical fill 的制度边界
> 适用范围: 当前默认 production runtime 的 execution / reconcile / recovery 语义
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前真相源

当前执行层的状态与事实语义以以下文件为准：

- [`execution/state_machine/transitions.py`](/quant_system/execution/state_machine/transitions.py)
- [`execution/state_machine/machine.py`](/quant_system/execution/state_machine/machine.py)
- [`execution/models/fills.py`](/quant_system/execution/models/fills.py)
- [`tests/execution_safety/test_late_execution_report.py`](/quant_system/tests/execution_safety/test_late_execution_report.py)
- [`tests/execution_safety/test_cancel_replace_flow.py`](/quant_system/tests/execution_safety/test_cancel_replace_flow.py)

---

## 2. Order Status 基线

当前合法订单状态：

- `pending_new`
- `new`
- `partially_filled`
- `filled`
- `pending_cancel`
- `canceled`
- `rejected`
- `expired`

终态：

- `filled`
- `canceled`
- `rejected`
- `expired`

一旦进入终态，不应再发生后续转换。

---

## 3. 关键转换规则

### 3.1 提交与成交

- `pending_new -> new`
- `pending_new -> partially_filled`
- `pending_new -> filled`
- `pending_new -> rejected`

说明：

- 当前明确允许“还未观察到 NEW，就先观察到部分成交或完全成交”的极端路径

### 3.2 正常执行

- `new -> partially_filled`
- `new -> filled`
- `new -> pending_cancel`
- `new -> canceled`
- `new -> expired`

### 3.3 撤单中的晚到成交

- `pending_cancel -> canceled`
- `pending_cancel -> filled`
- `pending_cancel -> partially_filled`
- `pending_cancel -> rejected`

说明：

- `pending_cancel -> filled` 当前是明确允许的
- 其含义不是状态损坏，而是“撤单请求发出过晚，venue 已完成成交”

---

## 4. Timeout 语义

`OrderTimeoutTracker` 的当前制度是：

- timeout 表示“本地在阈值内未观察到终态”
- timeout 不等价于“venue 一定未成交”
- timeout 后如果触发取消，状态机会进入 `pending_cancel`
- 若此后收到晚到 fill，应收敛到 `filled`

因此：

- timeout 是本地观测超时语义
- fill 是 venue 事实语义
- 两者冲突时，以 fill 事实为准

---

## 5. Cancel-Replace 语义

当前制度要求：

- 原单与替换单是两笔独立订单
- 原单进入终态后，不应被替换单的回报污染
- 原单进入 `canceled` 终态后，若又收到该原单的 fill，应视为高风险冲突并优先人工核查

说明：

- 这类 case 与 `pending_cancel -> filled` 不同
- 前者是“撤单申请中仍成交”
- 后者是“原单已终态后又出现新成交事实”

---

## 6. CanonicalFill 基线

[`CanonicalFill`](/quant_system/execution/models/fills.py) 当前是执行层的标准成交事实对象。

关键字段：

- `venue`
- `symbol`
- `order_id`
- `trade_id`
- `fill_id`
- `side`
- `qty`
- `price`
- `fee`
- `fee_asset`
- `liquidity`
- `ts_ms`
- `payload_digest`

制度要求：

- `fill_id` 必须稳定
- 同一真实成交重复到达时，应复用同一 `fill_id`
- `payload_digest` 用于识别“同 fill_id 但 payload 不同”的数据损坏

---

## 7. 当前仍未完全收口

- 通用事件层 `FillEvent` 与执行层 `CanonicalFill` 还未完全统一到单一事实模型
- `ack / reject / fill` 的跨模块 contract 还未形成单一 schema 文档
- reconcile / state machine / venue adapter 之间的事件映射还需继续收口

---

## 8. Incident Matrix

| 场景 | 本地状态解释 | 事实优先级 | 默认动作 | 升级条件 |
|---|---|---|---|---|
| `timeout` 但无后续回报 | 本地观测超时 | venue 事实高于本地 timeout | 发起 cancel，等待 reconcile | reconcile 出现 fill/order drift |
| `pending_cancel -> filled` | 撤单发起过晚，订单已成交 | fill 高于 cancel 意图 | 收敛为 `filled` | 成交数量/价格异常 |
| `pending_cancel -> partially_filled` | 撤单过程中又发生部分成交 | fill 高于 cancel 意图 | 收敛为 `partially_filled`，继续 reconcile | 长时间无法终态收敛 |
| `canceled -> filled` | 原单已终态后又出现成交 | 高风险冲突 | 不自动复活，人工核查 | 立即告警，必要时 halt |
| duplicate fill，同 payload | 幂等重复 | 原成交事实有效 | 忽略重复事件 | 无 |
| duplicate fill，不同 payload | 数据损坏或 venue/order 映射异常 | 无法直接信任本地 | fail fast / 告警 | 立即人工核查 |
| restart 后缺失 late fill | checkpoint 落后于 venue | venue facts 高于 checkpoint | reconcile 报 drift，人工确认 | position/fill drift 为 critical |
| startup reconcile mismatch | 本地恢复态与 venue 不一致 | venue 高于 checkpoint | 告警，不自动修复 | critical drift 或不可解释 mismatch |

解释：

- “事实优先级”当前默认是 `venue fills / venue state > checkpoint > timeout / cancel intent`
- `pending_cancel` 不是终态
- `canceled` 是终态，因此 `canceled -> filled` 与 `pending_cancel -> filled` 不应混为一谈

---

## 9. 下一步

后续改造应围绕以下目标推进：

1. 为 `ack / reject / fill` 增加统一 contract 文档和测试
2. 增强 restart + reconnect + reconcile 的组合恢复测试
3. 明确 `CanonicalFill -> event.types.FillEvent` 的映射边界
