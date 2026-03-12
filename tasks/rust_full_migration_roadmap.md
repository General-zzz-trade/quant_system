# Python → Rust 全面迁移路线图

> 状态: 历史迁移路线图快照（2026-03-07）
> 当前 Rust 迁移边界请优先参考:
> [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md),
> [`docs/rust_replacement_matrix.md`](/quant_system/docs/rust_replacement_matrix.md),
> [`tasks/refactor_master_plan.md`](/quant_system/tasks/refactor_master_plan.md)

## 当前状态 (2026-03-07)

**Rust**: 41 .rs 模块, ~12,300 LOC, 33 classes + 69 functions
**已迁移热路径**: pipeline, reducers, detect/normalize, sizing, constraints, ML decision
**已删除 Python fallback**: P2.5 完成 (-730 LOC)

### 剩余 Python 代码分布

| 层 | Python LOC | Rust 覆盖 | 性质 |
|----|-----------|----------|------|
| 热路径 (per-event) | ~1,500 | 95% | 主要剩余: Decimal↔i64 转换 |
| 温路径 (per-bar) | ~2,500 | 5% | 编排逻辑, 信号集成 |
| 冷路径 (init/IO) | ~25,000 | 3% | 配置, 适配器, 监控 |

### 热路径唯一瓶颈

`state/rust_adapters.py` 的 Decimal↔i64 转换:
- 每事件 44 次 Decimal 乘除 = ~6.3 µs/event
- 占热路径 Python 开销的 ~70%

---

## Phase 3: 消除 Decimal 边界 (P3)

**目标**: State 全部留在 Rust 堆上, Python 不再持有 Decimal state
**预计收益**: 5-7% 吞吐量提升, 每事件节省 ~5 µs
**工作量**: ~800 LOC Rust + ~200 LOC Python 修改

### 核心改动

1. **扩展 RustStateStore** — 持有全部 state (market/position/account/portfolio/risk)
   - `store.get_market(symbol)` 返回 Python dict (按需导出, 非每事件)
   - `store.process_event(event)` 一次 Rust 调用完成 detect→normalize→reduce→snapshot
   - 消除 `market_to_rust()` / `market_from_rust()` 44 次 Decimal 转换

2. **Pipeline 直连 RustStateStore**
   - `StatePipeline.__init__` 接受 `RustStateStore` 实例
   - `apply()` 直接传 event → Rust, Rust 内部更新 state
   - 返回 `RustProcessResult` (advanced, event_id, snapshot_dict)

3. **Coordinator 持有 RustStateStore**
   - 替换 `self._markets: Dict[str, MarketState]` 等 Python dict
   - 只在需要时 (decision, snapshot, feature hook) 才导出 state

### 文件改动

| 文件 | 改动 |
|------|------|
| `ext/rust/src/state_store.rs` | 扩展 `process_event()` 方法, 内部调用 reducers |
| `engine/pipeline.py` | 新增 RustStateStore fast path (绕过所有 Python state) |
| `engine/coordinator.py` | 持有 RustStateStore, 按需导出 |
| `state/rust_adapters.py` | 保留 adapters (给 slow path / backtest), 但热路径不再使用 |

### 保留 Python 的部分

- `state/market.py`, `position.py`, `account.py` — **保留**: backtest/replay 需要, 且是纯数据类
- `state/reducers/*.py` — **保留**: 自定义 reducer (策略回测覆盖用)
- `state/snapshot.py` — **保留**: 仍需 MappingProxyType 冻结 (Rust 导出后)

---

## Phase 4: 信号层 Rust 加速 (P4)

**目标**: 信号计算全链路在 Rust 内完成
**预计收益**: Decision 路径 ~100µs → ~10µs
**工作量**: ~1,500 LOC Rust
**前置**: P3 完成

### 核心改动

1. **RustDecisionEngine** — 信号→权重→目标仓位 一次 Rust 调用
   - 输入: RustStateStore snapshot + features dict
   - 输出: Vec<RustOrderSpec>
   - 消除 Python 侧 ensemble/allocation 循环

2. **信号模型迁移**
   - `signals/adaptive_ensemble.py` (273 LOC) → Rust weighted avg
   - `allocators/constraints.py` 已在 Rust, 但调用仍从 Python
   - `sizing/*.py` 已在 Rust, 同上

3. **Python 侧简化**
   - `decision/engine.py` → 薄 wrapper: `rust_engine.decide(store, features)`
   - 保留 Python 信号接口 (研究用), 但生产路径纯 Rust

### 文件改动

| 文件 | 改动 |
|------|------|
| `ext/rust/src/decision_engine.rs` | 新建: 完整决策管道 |
| `decision/engine.py` | 简化为 Rust wrapper |
| `decision/signals/adaptive_ensemble.py` | 保留 Python, 新增 Rust 等价 |

---

## Phase 5: Event 层 Rust 化 (P5)

**目标**: Event 创建、验证、路由全在 Rust
**预计收益**: 每事件省 ~2 µs (验证 + 路由)
**工作量**: ~600 LOC Rust
**优先级**: 低 (当前开销小)

### 核心改动

1. **RustEventHeader** — 替代 Python EventHeader dataclass
   - 已有 `rust_event_id()`, `rust_now_ns()`
   - 扩展: `rust_new_root_header()` 直接返回完整 header

2. **RustEventValidator** — 已有 11 个 `rust_validate_*` 函数
   - 组合为 `rust_validate_event(event)` 一次调用

3. **事件类型保留 Python** — EventType enum, MarketEvent 等数据类
   - 这些是接口类型, 迁移无收益

---

## 不迁移的部分 (~25,000 LOC)

以下代码**永远保留 Python**, 迁移无收益:

| 类别 | LOC | 原因 |
|------|-----|------|
| `execution/` 交易所适配器 | ~13,700 | IO 密集, 非 CPU 瓶颈 |
| `engine/` 编排层 | ~2,000 | 纯 control flow, dict 操作 |
| `features/` 特征管道 | ~3,000 | 编排层 (计算已在 Rust) |
| `core/` 框架代码 | ~1,900 | 初始化, 配置, 插件 |
| `event/` 事件框架 | ~3,500 | 类型定义, 序列化, 回放 |
| `state/reducers/` | ~460 | 策略回测自定义路径 |
| `decision/signals/` | ~2,000 | 研究层信号逻辑 |

### 为什么不迁移

1. **IO 密集** — 交易所适配器的瓶颈在网络延迟, 不在 CPU
2. **编排逻辑** — dict 拷贝、锁、回调分发在 Python 和 Rust 速度差异 <10%
3. **研究兼容** — 信号策略需要快速迭代, Python 更合适
4. **类型定义** — dataclass / enum 在两侧没有性能差异

---

## 迁移优先级总结

| 优先级 | Phase | 收益 | 工作量 | 推荐 |
|--------|-------|------|--------|------|
| **P3** | 消除 Decimal 边界 | 5-7% 吞吐量 | 1 周 | **立即做** |
| **P4** | 信号层 Rust | Decision ~10x | 2 周 | 中期 |
| **P5** | Event 层 Rust | ~2 µs/event | 3 天 | 低优先级 |
| — | execution/features/core | 无 | — | **不做** |

## "全部删掉 Python 内核" 可行性

**不可行也不必要。** 原因:

1. 热路径已经 95% Rust — 剩余 5% 是 Decimal 转换 (P3 解决)
2. 编排层 (coordinator, loop, scheduler) 迁移收益 <1%
3. Python 保留研究灵活性 — 策略迭代速度 > 运行时微秒级优化
4. 交易所适配器 ~14,000 LOC 迁移 ROI 为零 (IO bound)

**正确目标**: P3 完成后, 热路径 100% Rust, Python 只做编排 + 研究。这就是最优架构。
