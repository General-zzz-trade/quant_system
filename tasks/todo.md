# Python 内核全面 Rust 化迁移计划

## 已完成 (P1-P4 历史)

- [x] P1: Pipeline 持有 Rust 类型、消除 market_from_rust 转换
- [x] P2: RustMarketEvent/FillEvent/FundingEvent 热路径事件 (23.9x)
- [x] P3: RustRiskEvaluator 6 规则 Rust 化
- [x] P4: RustFeatureEngine 105 特征增量计算
- [x] Python 死代码清理 (~425 LOC: StateProjector + MLDecisionModule)

---

## 现状

| 层 | Python LOC | Rust LOC | 说明 |
|---|---|---|---|
| state/ | 1,540 | 2,123 | Reducer 数学全 Rust，Python 保留类型+适配层 |
| engine/ | 4,383 | 557 | pipeline apply 走 Rust，编排/snapshot/derive 仍 Python |
| decision/ | ~2,500 | 631 | 热路径数学 Rust，编排/信号/分配纯 Python |
| execution/ | ~1,340 | 584 | 状态机/存储有 Rust，路由/桥接纯 Python |
| event/ | ~4,109 | 65 | 几乎全 Python |
| features/ | ~4,214 | 4,495 | RustFeatureEngine 已覆盖生产 105 特征 |

**核心瓶颈**: 每 bar ~12 次 Python↔Rust 边界跨越（Decimal↔i64 转换占 pipeline 30%+ 时间）

---

## Phase 5: 统一 RustStateStore 路径 — 消除适配层 (~800 LOC)

**目标**: pipeline 默认走 RustStateStore，消灭 rust_adapters.py 中 `*_to_rust()` / `*_from_rust()` 转换。

- [ ] 5.1 `StatePipeline.__init__` 默认创建 RustStateStore（当前需显式传入）
- [ ] 5.2 `EngineCoordinator` 默认用 store path，不再维护 Python-side state dicts
- [ ] 5.3 `RustStateStore` 增加 `export_snapshot()` — Rust 侧 i64→f64 转换，直接返回 Python dict
- [ ] 5.4 `derive_portfolio_and_risk()` 迁入 Rust — `process_event()` 内部直接算
- [ ] 5.5 `_build_snapshot()` 移入 Rust — RustStateStore 直接输出 snapshot
- [ ] 5.6 删除 rust_adapters.py 中不再使用的转换函数
- [ ] 5.7 Python reducers 移到 test-only 路径

**删除 ~400 LOC Python / 新增 ~300 LOC Rust**
**验证**: 全量测试 + pipeline throughput benchmark

---

## Phase 6: Pipeline 全 Rust 化 (~650 LOC)

**目标**: `StatePipeline.apply()` 变成单一 Rust 调用。

- [ ] 6.1 `rust_pipeline_apply_full()` — event → classify → normalize → reduce → derive → snapshot → 返回 RustPipelineOutput
- [ ] 6.2 `RustPipelineOutput` PyO3 类 (markets/account/positions/portfolio/risk/features/snapshot/advanced)
- [ ] 6.3 Python `StatePipeline.apply()` 简化为 ~50 LOC：调 Rust → 包装 → 返回
- [ ] 6.4 删除 `_apply_rust_fast()` / `_apply_store_path()` / slow path 三路径

**删除 ~400 LOC Python / 新增 ~500 LOC Rust**
**验证**: pipeline parity 测试 + e2e backtest 对比

---

## Phase 7: 执行层整合 (~400 LOC)

**目标**: 删除 OrderStateMachine / AckStore / DedupStore 的 Python wrapper。

- [ ] 7.1 `OrderStateMachine` 直接暴露 RustOrderStateMachine（删除 ~200 LOC wrapper）
- [ ] 7.2 `InMemoryAckStore` 替换为 RustAckStore（保留 SQLite 版）
- [ ] 7.3 `InMemoryDedupStore` 替换为 RustDedupStore（保留 SQLite 版）
- [ ] 7.4 ingress router dedup 逻辑直接调 Rust guard

**删除 ~250 LOC Python**
**验证**: execution 单元测试 + order lifecycle 测试

---

## Phase 8: Coordinator 精简 (~455 LOC)

**目标**: EngineCoordinator 从 "状态管家" 变成 "事件路由器"。

**依赖**: Phase 5+6

- [ ] 8.1 Coordinator 不再持有 Python-side state dicts — 全部委托 RustStateStore
- [ ] 8.2 `on_event()` 简化为：dispatch → pipeline.apply() → decision_bridge → execution_bridge
- [ ] 8.3 删除所有 `*_from_rust()` / `*_to_rust()` 调用
- [ ] 8.4 feature_hook 直接从 RustStateStore 读 market state

**删除 ~200 LOC Python**
**验证**: e2e live paper trading 测试

---

## Phase 9: 决策引擎热路径 (~500 LOC)

**目标**: DecisionEngine.run() 核心循环在 Rust 内执行。

- [ ] 9.1 `RustDecisionPipeline` — Rust 侧决策管道：(snapshot, signals) → OrderSpec[]
  - 整合已有: `rust_fixed_fraction_qty`, `rust_build_delta_order_fields`, `rust_validate_order_constraints`, `rust_limit_price`
- [ ] 9.2 Python `DecisionEngine.run()` 简化为：compute signals → call Rust pipeline → unpack
- [ ] 9.3 `RebalanceModule` drift 计算迁入 Rust
- [ ] 9.4 信号层保留 Python（可扩展性 > 性能）

**删除 ~300 LOC Python / 新增 ~600 LOC Rust**
**验证**: decision determinism 测试 + parity 对比

---

## Phase 10: Event 层精简 (~300 LOC)

**目标**: 热路径 event 分发用 Rust，schema 定义保留 Python。

- [ ] 10.1 `RustEventDispatcher` — Rust 事件路由表 (type → handler)
- [ ] 10.2 `InMemoryEventStore` 默认用 RustInMemoryEventStore
- [ ] 10.3 `InMemoryCheckpointStore` 默认用 RustCheckpointStore
- [ ] 10.4 event/types.py 保留 Python（不迁移）

**删除 ~150 LOC Python / 新增 ~200 LOC Rust**

---

## 不迁移清单

| 模块 | 原因 |
|---|---|
| features/enriched_computer.py (1,462) | 研究/训练引用多，RustFeatureEngine 已覆盖生产 |
| decision/signals/** (~600) | 冷路径，可扩展性优先 |
| decision/ensemble/** (~300) | Meta-learning，低频 |
| engine/loop.py / scheduler.py / saga.py (~1,000) | I/O 绑定 |
| event/types.py (477) | Schema 定义，变动频繁 |
| execution/adapters/** (~1,000) | 交易所 REST/WS，I/O 绑定 |
| state/ Python 类型 (market.py 等, ~450) | snapshot/backtest API 契约 |
| SQLite 存储 | SQLite 本身是瓶颈 |

---

## 预期成果

| 指标 | 当前 | Phase 5-6 后 | Phase 5-10 全完成 |
|---|---|---|---|
| Python 内核 LOC | ~18,086 | ~16,800 | ~15,186 |
| Rust LOC | ~8,455 | ~9,255 | ~10,555 |
| 每 bar 边界跨越 | ~12 次 | ~3 次 | ~1 次 |
| Pipeline 加速 | 5.11x | ~8x | ~10x |

## 执行顺序

**Phase 5 → 6 → 7 → 8 → 9 → 10**

- 5+6 核心收益（消除转换层+pipeline 全 Rust），先做
- 7 独立性强，可与 8 并行
- 8 依赖 5+6
- 9 最复杂，放后面
- 10 收益最小，最后做
