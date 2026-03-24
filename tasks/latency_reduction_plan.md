# 延迟压缩计划 — 基于实测基线

> **Status**: COMPLETED (2026-03-24) — 性能优化计划，历史基线参考.
> 更新时间: 2026-03-12. 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).
> 性能里程碑请参考 [`tasks/perf_milestones.md`](/quant_system/tasks/perf_milestones.md).

## 实测基线（lowlatency 6.11 + isolcpus=1）

### 单品种 BTCUSDT (p50, us)
| 阶段 | p50 | 占比 | 根因 |
|------|-----|------|------|
| Features hook | 798 | 60% | push_external_data kwargs解包 + get_features→dict + None过滤 + getattr链 |
| Pipeline state | 312 | 24% | _export_store_state: get_markets/positions→dict + portfolio_from_rust + risk_from_rust |
| Decision | 196 | 15% | regime detect + decide + Decimal数学 |
| **TOTAL** | **1321** | 100% | |

### 4品种 (p50, us)
| 阶段 | p50 |
|------|-----|
| Features | 887 |
| Pipeline | 384 |
| Decision | 565 |
| **TOTAL** | **1845** |

---

## Phase A: Feature Hook 优化 — 已完成

### 实施结果

1. **启用 Unified Predictor 路径**（最大收益）
   - `runner/testnet_validation.py`: 单品种配置自动尝试 `RustUnifiedPredictor`
   - 消除了独立的 `_rust_push` + `bridge.enrich()` 链路
   - 单次 Rust 调用完成: push_bar + predict + get_features
   - Features: 798 → ~530 μs (benchmark 高方差，2vCPU 虚拟化环境)

2. **事件类型快速检查**
   - `engine/feature_hook.py`: `kind is not _EventType.MARKET` 直接枚举比较
   - 消除了每 tick 的 `str(kind_val).lower()` 字符串构造+转换

3. **消除 import-on-use**
   - `engine/feature_hook.py`: `import time` 移到模块级

### 未实施 & 原因

- **A.1 (PyDict)**: Rust PyDict 逐项 set_item 比 HashMap 自动转换更慢（测试确认）
- **A.3/A.4**: 微基准显示 Python kwargs 开销仅 ~7μs，ROI 不够
- **_LazyConvert proxy**: `__getattr__` 拦截开销比 eager 转换更高（测试确认回归）

### 深度分析发现

| 组件 | 隔离测试 (p50 us) | 全管道 (p50 us) | 差距原因 |
|------|-------------------|-----------------|----------|
| _unified_push 各部分之和 | ~35 | ~530 | Python 方法调度、帧分配、缓存效应 |
| pipeline.apply 各部分之和 | ~52 | ~298 | 同上 |
| bridge.enrich 隔离 | ~52 | ~201 | 上下文切换开销 |

**结论**: 单个组件已接近最优 (~30-50μs)。~1000μs 总延迟中 ~800μs 来自 Python 解释器开销（方法调度、帧分配、dict 操作、GC 压力）。进一步优化需要 Phase D（全 Rust 热路径）。

- [x] A.unified: 启用 Unified Predictor 路径
- [x] A.kind: 事件类型快速检查
- [x] A.import: 消除 per-tick import
- [x] A.dict_copy: 消除 _export_store_state dict() 拷贝

---

## Phase B: Pipeline State Export 优化 — 部分完成

### B.2 已完成: 消除 dict() 拷贝
`_export_store_state` 不再做 `dict(store.get_markets())` — 直接传递 Rust 返回的 dict。

### B.1 LazyProxy — 放弃
`_LazyConvert.__getattr__` 拦截比 eager `portfolio_from_rust` 更慢。
portfolio_from_rust 仅 ~14μs, risk_from_rust 仅 ~7μs，不值得复杂化。

- [x] B.2: 消除 dict() 拷贝
- [ ] ~~B.1: LazyProxy~~ (放弃 — 开销更高)
- [ ] B.3: Rust timestamp 改为 epoch float (低优先)

---

## Phase C: Decision 优化（目标: 196→80us, -59%）— 待定

低优先。Decision 隔离开销小 (~30μs)，全管道 ~190μs 主要是 Python 开销。

- [ ] C.1: 批量 decide
- [ ] C.2: features 直接传递

---

## Phase D: 全路径 Rust 化（终极目标: <100us）

唯一能突破 Python 解释器开销墙的方案。

把 coordinator._handle_pipeline_event 整体移入 Rust：

```
event(PyObject) → Rust:
  1. 提取 OHLCV 字段
  2. push_bar → features (已有)
  3. push_external_data → ML predict (已有)
  4. state store process_event (已有)
  5. decision logic (新增)
  → 返回: (features_dict, decision_events, snapshot)
```

单次 GIL 获取，单次 Python↔Rust 边界跨越。

- [ ] D.1: Rust 侧 `process_tick()` 一体化 API
- [ ] D.2: Decision logic Rust 化
- [ ] D.3: 替换 coordinator._handle_pipeline_event

---

## 当前效果

| 阶段 | 原始基线 | 当前 p50 | 改进 |
|------|---------|---------|------|
| Features | 798 us | ~530 us | -34% |
| Pipeline | 312 us | ~298 us | -4% |
| Decision | 196 us | ~188 us | -4% |
| **TOTAL** | **1321 us** | **~1020 us** | **-23%** |

注：2vCPU 虚拟机 benchmark 方差 ±30%，绝对值仅供趋势参考。
