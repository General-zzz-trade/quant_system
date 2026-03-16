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

## 3. Ack 基线

当前 execution bridge 的最小公共 ack 视图已固定为：

- [`execution/models/acks.py`](/quant_system/execution/models/acks.py)
- [`execution/models/rejections.py`](/quant_system/execution/models/rejections.py)

字段基线：

- `status`
- `ok`
- `command_id`
- `venue`
- `symbol`
- `attempts`
- `deduped`
- `result`
- `error`

当前 `normalize_ack()` 的作用是：

- 把真实 [`Ack`](/quant_system/execution/bridge/execution_bridge.py) 和测试/适配层 ack-like 对象收口到同一最小视图
- 让上层 bridge 不再依赖某个具体 ack 对象的私有字段形状

当前 `ack_to_rejection()` 的作用是：

- 把 `REJECTED / FAILED` 收口成统一的 `CanonicalRejection`
- 区分“明确拒绝”和“失败但可能可重试”两类结果
- 为后续 reject event 公共模型预留稳定边界

当前 [`execution/models/rejection_events.py`](/quant_system/execution/models/rejection_events.py) 已建立 `CanonicalRejectionEvent` 这一 event-like 公共观察对象。

当前 [`execution/bridge/live_execution_bridge.py`](/quant_system/execution/bridge/live_execution_bridge.py) 已支持：

- `on_reject`: 暴露标准化 `CanonicalRejection`
- `on_reject_event`: 暴露 `CanonicalRejectionEvent`
- `alert_manager`: 将 `CanonicalRejectionEvent` 映射为结构化 alert，进入统一 ops 观察链路

注意：

- `CanonicalRejectionEvent` 目前是 execution 层的公共观察对象
- 它还没有并入 `event.types` 或 Rust route matcher
- 它通过 [`execution/observability/rejections.py`](/quant_system/execution/observability/rejections.py) 进入 unified alert/ops 观察链路
- 也就是说，它现在不是主事件总线的一部分，但已经是统一运维观察面的一部分
- 当前 rejection alert 还会统一带上 `category=execution_rejection`、`reason_family` 和稳定 `routing_key`
- 当前 rejection `routing_key` 语义是 `venue:symbol:status:reason_family`
- 当前 execution 侧统一 alert taxonomy helper 位于 [`execution/observability/alerts.py`](/quant_system/execution/observability/alerts.py)
- 当前 execution 侧 incident taxonomy helper 位于 [`execution/observability/incidents.py`](/quant_system/execution/observability/incidents.py)
- 当前已统一的 execution incident category 包括：
  - `execution_rejection`
  - `execution_timeout`
  - `execution_reconcile`
  - `execution_fill`（当前用于 synthetic fill receipt）
- 当前 execution incident payload 的最小稳定字段要求：
  - `category`
  - `routing_key`
  - timeout: `venue / symbol / order_id / timeout_sec`
  - reconcile: `venue / drift_count / should_halt / symbols / severity_scope`
  - synthetic fill: `venue / symbol / fill_id / order_id / qty / side / synthetic`
  - rejection: `event_type / status / symbol / venue / reason / reason_family / command_id / retryable / deduped`

---

## 4. 关键转换规则

### 4.1 提交与成交

- `pending_new -> new`
- `pending_new -> partially_filled`
- `pending_new -> filled`
- `pending_new -> rejected`

说明：

- 当前明确允许“还未观察到 NEW，就先观察到部分成交或完全成交”的极端路径

### 4.2 正常执行

- `new -> partially_filled`
- `new -> filled`
- `new -> pending_cancel`
- `new -> canceled`
- `new -> expired`

### 4.3 撤单中的晚到成交

- `pending_cancel -> canceled`
- `pending_cancel -> filled`
- `pending_cancel -> partially_filled`
- `pending_cancel -> rejected`

说明：

- `pending_cancel -> filled` 当前是明确允许的
- 其含义不是状态损坏，而是“撤单请求发出过晚，venue 已完成成交”

---

## 5. Timeout 语义

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

## 6. Cancel-Replace 语义

当前制度要求：

- 原单与替换单是两笔独立订单
- 原单进入终态后，不应被替换单的回报污染
- 原单进入 `canceled` 终态后，若又收到该原单的 fill，应视为高风险冲突并优先人工核查

说明：

- 这类 case 与 `pending_cancel -> filled` 不同
- 前者是“撤单申请中仍成交”
- 后者是“原单已终态后又出现新成交事实”

---

## 7. CanonicalFill 基线

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

### 7.1 公共事件映射边界

当前已建立统一映射 helper：

- [`execution/models/fill_events.py`](/quant_system/execution/models/fill_events.py)
- [`execution/bridge/live_execution_bridge.py`](/quant_system/execution/bridge/live_execution_bridge.py) 与 [`execution/algo_adapter.py`](/quant_system/execution/algo_adapter.py) 现在也复用该 helper 生成 ingress-side fill 事件
- [`execution/models/fill_events.py`](/quant_system/execution/models/fill_events.py) 当前也承载 ingress dedup identity 计算
- [`execution/ingress/router.py`](/quant_system/execution/ingress/router.py) 当前通过 `FillDeduplicator` + Rust `RustPayloadDedupGuard` 执行 ingress 去重与 payload mismatch fail-fast

当前边界是：

- `CanonicalFill -> FillEvent`：
  - 生成公共最小事实视图
  - 只保证 `fill_id/order_id/symbol/qty/price`
  - 不强行把 `side/trade_id/venue/payload_digest` 塞进公共事件层
- `CanonicalFill -> ingress fill event`：
  - 保留 execution/state 推进需要的 richer 字段
  - 供 [`execution/ingress/router.py`](/quant_system/execution/ingress/router.py)、[`execution/bridge/live_execution_bridge.py`](/quant_system/execution/bridge/live_execution_bridge.py)、[`execution/algo_adapter.py`](/quant_system/execution/algo_adapter.py) 驱动 pipeline / state reducers

这意味着：

- `FillEvent` 是公共最小事实模型（6字段: fill_id, order_id, symbol, qty, price, side[optional]）
- `CanonicalFillIngressEvent` 是 pipeline 输入的富化事实（含 side, fee, venue, pnl）
- `CanonicalFill` 是 execution 内部 richer fact model（Decimal 精度, raw 保留）
- 三者通过 `fill_events.py` 中的映射函数连接，round-trip 一致性由 `tests/unit/execution/test_fill_roundtrip.py` 锁定
- `side` 现在在三层中一致流转（FillEvent.side 为 Optional，向后兼容）
- `fill_to_record()` 与 `CanonicalFill.to_record()` 产出完全一致的 13 字段 dict
- digest 计算统一由 `execution/models/digest.py` 的 `stable_hash()` 驱动，`fill_events.py._stable_hash()` 和 `message_integrity.compute_payload_digest()` 均委托到同一实现

标准 fill 流转路径（唯一正确路径）：

```
VenueAdapter.mapper → CanonicalFill (Decimal, 全字段)
  → FillDeduplicator.accept_or_raise() (幂等去重)
  → canonical_fill_to_ingress_event() → CanonicalFillIngressEvent (float, pipeline用)
  → coordinator.emit(ingress_event) → StatePipeline → RustStateStore
  同时:
  → canonical_fill_to_public_event() → FillEvent (最小公共契约)
  → emit_handler._handle_fill() → OSM transition + event_recorder
```

当前 synthetic fill 边界：

- bridge / algo 生成的 synthetic ingress fill 现在统一走 `build_synthetic_ingress_fill_event()`
- synthetic fill 必须自带稳定 `fill_id`
- synthetic fill 必须自带 `payload_digest`
- bridge 侧 identity 默认以 `command_id / order_id` 为种子
- algo 侧 identity 默认以 `intent_id / order_id + fill_seq` 为种子
- [`execution/bridge/live_execution_bridge.py`](/quant_system/execution/bridge/live_execution_bridge.py) 与 [`execution/algo_adapter.py`](/quant_system/execution/algo_adapter.py) 现在都支持可选 `incident_logger`
- 当 runtime 传入统一 incident sink 时，bridge/algo synthetic fill 与 rejection 可以沿同一 execution incident 持久化链进入 `event_log` / `ops_timeline`

当前跨层护栏：

- [`tests/integration/test_execution_synthetic_fill_contract.py`](/quant_system/tests/integration/test_execution_synthetic_fill_contract.py)
  - direct bridge synthetic fill 经过 ingress 后必须幂等
  - algo synthetic fills 必须保持“同 seq 同身份、不同 seq 不同身份”
  - synthetic fill 身份必须可直接用于 reconcile
  - bridge/algo synthetic fill 现在也会在注入 runtime incident sink 时进入持久化 `execution_incident` 时间线
- [`tests/integration/test_execution_rejection_contract.py`](/quant_system/tests/integration/test_execution_rejection_contract.py)
  - rejection event 必须可观察
  - rejection 必须可进入 unified alert/ops 观察链路
  - rejection 不能误发 fill 到 ingress
  - rejection 不能推进 position / event_index
  - retryable `FAILED` 之后的成功重试仍必须能正常进入 synthetic fill / ingress
  - rejection 现在也会在注入 runtime incident sink 时进入持久化 `execution_incident` 时间线

### 7.2 Ingress Dedup 基线

当前 ingress fill 去重制度：

- key 基线是 `(venue, symbol, fill_key)`
- `fill_key` 优先使用 `fill_id`，其次 `trade_id`
- 若两者都缺失，退化为 `(order_id, ts, price, qty, side)` 的稳定哈希
- digest 优先使用 `payload_digest`
- 若无 `payload_digest`，退化为关键成交字段的稳定哈希

当前要求：

- canonical fill 路径与 legacy fill-like 路径在等价字段输入下，必须产出相同 dedup key/digest
- duplicate with same digest 必须被幂等丢弃
- duplicate with different digest 必须 fail fast，不允许静默覆盖
- dedup key/digest 的真相源当前位于 `ingress_fill_dedup_identity()`
- 所有 `_stable_hash` / `payload_digest` 计算统一委托给 `execution/models/digest.py`（单一真相源，7个旧实现已收口）

当前护栏：

- [`tests/unit/execution/test_fill_ingress_router_dedup.py`](/quant_system/tests/unit/execution/test_fill_ingress_router_dedup.py)
- [`tests/unit/execution/test_fill_deduplicator.py`](/quant_system/tests/unit/execution/test_fill_deduplicator.py)
- [`tests/unit/test_rust_parity_v2.py`](/quant_system/tests/unit/test_rust_parity_v2.py)

### 7.3 Ingress Sequencing 基线

当前 execution ingress 的序列重排真相源是 Rust `RustSequenceBuffer`，Python wrapper 位于：

- [`execution/ingress/sequence_buffer.py`](/quant_system/execution/ingress/sequence_buffer.py)

当前要求：

- wrapper 必须完整暴露 `push / expected_seq / buffered_count / flush / pending_count / reset`
- wrapper 语义必须继续与 [`execution/safety/out_of_order_guard.py`](/quant_system/execution/safety/out_of_order_guard.py) 保持一致

当前护栏：

- [`tests/unit/execution/test_sequence_buffer_wrapper.py`](/quant_system/tests/unit/execution/test_sequence_buffer_wrapper.py)
- [`tests/unit/execution/test_out_of_order_guard_parity.py`](/quant_system/tests/unit/execution/test_out_of_order_guard_parity.py)
- [`tests/unit/test_rust_parity_v2.py`](/quant_system/tests/unit/test_rust_parity_v2.py)

### 7.4 Order Ingress Dedup 基线

当前 `ORDER_UPDATE` ingress 的 key/digest 真相源已固定为：

- [`execution/models/orders.py`](/quant_system/execution/models/orders.py) 中的 `ingress_order_dedup_identity()`
- [`execution/ingress/order_router.py`](/quant_system/execution/ingress/order_router.py) 中的 `_dedup_key_and_digest()` 只是对该 helper 的显式 wrapper，用于固定迁移边界

当前边界：

- [`execution/adapters/binance/mapper_order.py`](/quant_system/execution/adapters/binance/mapper_order.py) 生成 `CanonicalOrder` 时复用同一 helper
- [`execution/ingress/order_router.py`](/quant_system/execution/ingress/order_router.py) 进入 Rust `RustPayloadDedupGuard` 前也复用同一 helper

当前要求：

- canonical order 与 order ingress 在等价字段输入下，必须产出相同 `order_key/payload_digest`
- duplicate with same digest 必须幂等丢弃
- duplicate with different digest 必须 fail fast，不允许静默覆盖

当前护栏：

- [`tests/unit/execution/test_order_ingress_router_dedup.py`](/quant_system/tests/unit/execution/test_order_ingress_router_dedup.py)
- [`execution/tests/ingress/test_order_ingress_advances_event_index.py`](/quant_system/execution/tests/ingress/test_order_ingress_advances_event_index.py)
- [`tests/unit/execution/test_order_projection_rust_parity.py`](/quant_system/tests/unit/execution/test_order_projection_rust_parity.py)

---

## 8. Reconcile / Healer 动作基线

当前 controller / policy / healer 的动作边界：

- `ReconcileController` 负责聚合 drift，并生成 `PolicyDecision`
- `ReconcilePolicy` 负责把 severity 映射到 `ACCEPT / ALERT / HALT / MANUAL_REVIEW`
- `ReconcileHealer` 只对 `ACCEPT` 且可自动修复的 drift 生效

当前要求：

- unsupported drift 类型不能消耗 `max_auto_heal_per_cycle` 配额
- healer 只会对 position / balance drift 生成修复动作
- `HALT` 与 `ALERT` drift 不会被 healer 自动修复

当前护栏：

- [`tests/unit/execution/test_healer.py`](/quant_system/tests/unit/execution/test_healer.py)
- [`tests/unit/execution/test_reconcile_scheduler.py`](/quant_system/tests/unit/execution/test_reconcile_scheduler.py)
- [`tests/unit/execution/test_reconcile_controller_report.py`](/quant_system/tests/unit/execution/test_reconcile_controller_report.py)

当前 `ReconcileReport` 语义：

- `all_drifts` 按 `positions -> balances -> fills -> orders` 的顺序聚合
- `decisions` 与 `all_drifts` 一一对应，controller 不会重排 drift
- `ok` 反映的是各子对账结果是否通过，不直接等价于 policy 是否选择 `ACCEPT`
- `should_halt` 只反映 `decisions` 中是否存在 `HALT`

---

## 9. 当前仍未完全收口

- 通用事件层 `FillEvent` 与执行层 `CanonicalFill` 仍未完全统一到单一事实模型
- `ack / reject / fill` 的跨模块 contract 已开始收口，但 reject 仍停留在 execution 观察层，而非主事件总线
- reconcile / state machine / venue adapter 之间的事件映射还需继续收口

---

## 10. Incident Matrix

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
| local order 存在但 venue 不存在 | 本地订单视图可能滞后或丢单 | venue state 高于本地订单缓存 | 报 order drift，优先 reconcile/人工核查 | 本地仍视为 active order 时按 critical 处理 |

解释：

- “事实优先级”当前默认是 `venue fills / venue state > checkpoint > timeout / cancel intent`
- `pending_cancel` 不是终态
- `canceled` 是终态，因此 `canceled -> filled` 与 `pending_cancel -> filled` 不应混为一谈

---

## 11. State Ownership

| State | Primary Owner | Purpose |
|-------|--------------|---------|
| Position qty/avg_price | RustStateStore (pipeline) | Trading decisions, P&L |
| Order lifecycle status | OrderStateMachine | Timeout, reconcile, audit |
| Account balance | RustStateStore (pipeline) | Risk limits, sizing |
| Open order count | OrderStateMachine | RiskGate.max_open_orders (execution safety) |

**Rule**: No trading decision path (signal generation, position sizing, alpha inference)
should read from OrderStateMachine. OSM is write-only from the decision perspective —
it receives events but does not inform signal generation or position sizing.

**Exception**: `RiskGate` reads `active_orders()` from OSM to enforce `max_open_orders`.
This is an execution safety gate (pre-order risk check), not a trading signal decision.
The position truth used by RiskGate for notional limits comes from RustStateStore
via `coordinator.get_state_view()`.

---

## 12. 下一步

后续改造应围绕以下目标推进：

1. 增强 restart + reconnect + reconcile 的组合恢复测试
2. 明确 `CanonicalFill -> event.types.FillEvent` 的映射边界
3. 为 order projection / reconcile kernel 建立更完整的迁移前 contract tests
4. 继续收口 rejection reason family 与更细的 routing policy
