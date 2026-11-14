# ADR-0004：订单状态机与订单库所有权

- 状态：**已批准**（2026-11-14）：方案 B，以在用实现为准，删除另外 4 套。"需要您确认"第 2 项（人工处置卡死订单的工具）**未答**。
- 日期：2026-11-14

## 背景

【代码】在用的订单状态机是 `shared/confirmed_execution.py` 加 `shared/execution_journal.py`。它的正向性质成立并有测试：先落库后发单、同一意图至多发一次、只有交易所成交记录才算成交、每笔成交恰好入账一次。

问题有三个：

1. **状态是自由字符串。** `reserved`、`submitted`、`unknown`、`rejected_unconfirmed`、`filled`、`canceled`、`expired`、`rejected`、`not_found` 分散在 3 个文件里，`_set_status` 不校验转移是否合法（`shared/confirmed_execution.py:218-221`）。终态集合以字面量在 6 处各写一份，内容不一致。
2. **订单库没有唯一写入者。** 适配器直接 `INSERT INTO submissions`（`execution/adapters/binance/execution_adapter.py:225-231`），止损管理器在同一个库里自建表（`protective_stops.py:48-53`），`engine/durable_fills.py:64,71` 直接 `UPDATE execution_consumers`。
3. **另有 4 套未接线的实现**：Rust 的 `RustOrderStateMachine`、`execution/models/intents.py`、`execution/bridge` 加 saga、`execution/models/commands.py` 加 `ack_store`。其中 `execution/bridge` 带"超时重发 3 次"的语义，与在用实现相反。

## 备选方案

### A. 保持字符串，只补文档和测试

- 优点：零代码改动。
- 缺点：非法转移仍然不会被拒绝；状态名拼错不会被发现。

### B. 状态枚举加显式转移表，SQLite 适配器是唯一写入者（建议）

- `trading/domain/orders.py` 定义 `OrderStatus` 枚举和合法转移表。枚举值就是现有字符串，所以已有的数据库行不用迁移。
- `OrderJournal` 端口提供 `transition(intent_id, to)`，实现里查转移表，非法转移抛错。
- 三张订单表和止损表的全部 SQL 收进 `trading/adapters/sqlite/order_journal.py`。止损收编、成交消费都通过端口方法完成。
- 用一条契约保证：`sqlite3` 只允许在 `trading.adapters.sqlite` 被 import。
- 转移表从现有行为归纳，先用特征化测试把现有转移固定下来再改。
- 优点：状态机变成可以穷举测试的数据；库结构只有一个地方知道。
- 缺点：`lifecycle` 里所有改状态的地方都要过一遍；归纳转移表时可能发现现有的非法转移，需要逐个决定是修代码还是承认它合法。

### C. 改用 Rust 的 `RustOrderStateMachine`

- 它已经有转移表（`execution/rust/order_state_machine.rs:272-284` 的 `is_transition_allowed`）。
- 缺点：它只在内存里，没有持久化；状态名与在用实现不同（`pending_new / new / partially_filled / ...`），没有 `unknown`、`rejected_unconfirmed`、`not_found` 这几个处理"结果不明"的状态，而这正是在用实现的核心；会让资金路径依赖 Rust 扩展（与 ADR-0008 相反）。

### D. 事件溯源

- 订单的每次变化追加一条不可变事件，当前状态由事件序列推导。
- 优点：审计线索最完整，天然有状态历史。
- 缺点：要改表结构并迁移现有数据；改动量远超 400 行；现有 45 个以上测试都要重写。当前规模下收益不抵成本。

## 建议

B。其余 4 套实现按 ADR-0008 删除。

## 已知需要在转移表里明确的点

这些是诊断中发现的现有行为，归纳转移表时要逐条决定：

- `unknown` 是粘性的：之后查到非终态订单不会改回 `submitted`。
- `_observe` 里任何瞬时异常都会把 `submitted` 或 `rejected_unconfirmed` 覆盖成 `unknown`（`shared/confirmed_execution.py:304-306`）。重启后"已确认拒绝"的标记随之丢失。
- 止损触发后直接以 `submitted` 插入一行，不经过预留。
- `settled` 从不回退，仓库里没有人工处置卡死订单的入口。

## 后果

- 状态相关的 6 处终态字面量收敛到一个定义。
- 订单库的表结构不变。
- 是否增加状态历史表（每次转移留一行）是后续可选项，不在本决策内。

## 需要您确认

1. 是否同意以在用实现为准，删除另外 4 套。
2. 是否需要一个人工处置工具（把卡死的订单标为已结算，需要操作员给出理由并留痕）。现在只能手改 SQLite。

## 实施记录（2026-11-14，第 5、6 步，第 6 步后审计 M8、M5）

与方案 B 的文字有三处不同，如实记录：

- 端口没有 `transition(intent_id, to)`。适配器提供 `update_status(intent_id, status, settle=...)`，返回写入前的状态；转移表的检查在执行器 `shared/confirmed_execution.py` 里做（第 9 步执行器迁入 `trading/application` 后归位）。
- 非法转移在生产中不抛错，只记 ERROR 日志（观察模式）；测试中任何非法转移都会让测试失败。审计指出，纸面模式不经过执行器，所以"等生产日志再决定何时改为拒绝"这个条件实际上不会满足。第 9 步之前要么补齐转移表的测试证据，要么把表收窄到有证据的转移，再决定是否改为拒绝。
- "`sqlite3` 只允许在 `trading.adapters.sqlite` 被 import"用 AST 测试实现（`tests/architecture/test_rules.py`），不是 import-linter 契约。

单一写入者只在 Binance 目标仓位运行时的路径上成立。ML 运行时的 `engine/durable_fills.py` 和冻结的 Bybit/OKX 适配器仍通过 `connect()` 直接写这几张表；它们随 ADR-0008 子决策 3 的去留一起处理。

