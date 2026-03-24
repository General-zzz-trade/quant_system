# Remediation Plan — 2026-03-14

> Prepared by: Claude Code (首席治理工程师)
> Scope: P0-P4 整改，不扩展新功能，不替换生产入口

---

## 1. 当前架构真相

| 维度 | 真相 | 证据 |
|------|------|------|
| 生产主入口 | `runner/live_runner.py` | runtime_truth.md:26, systemd ExecStart |
| 架构模式 | Python 主编排 + Rust 热路径 | runtime_truth.md:189 |
| 状态真相源 | RustStateStore (pipeline 唯一写通道) | execution_contracts.md §11 |
| 约束真相源 | `rust/src/constraint_pipeline.rs` | runtime_contracts.md, parity tests |
| 配置真相源 | `runner/config.py` LiveRunnerConfig (93字段) | 4 factory methods: lite/paper/testnet/prod |
| 脚本真相源 | `scripts/catalog.py` (99 脚本, 6 分组) | render_catalog() CLI |
| 模型治理 | registry → shadow_compare → promote → SIGHUP → reload | model_governance.md |
| CI 门禁 | pytest + execution/tests + cargo test + ruff + 57% coverage | ci.yml |
| 恢复链 | 8 组件 atomic bundle | recovery.py, runtime_contracts.md |
| 重构主计划 | 5 工作流 × 4 阶段，全部标记 DONE | refactor_master_plan.md |

---

## 2. 最高风险可维护性热点

### 已解决（本会话前序工作）

| # | 热点 | 状态 |
|---|------|------|
| 1 | tick_processor HashMap.unwrap() 热路径 panic | ✅ →expect |
| 2 | RiskGate 多 symbol notional 用错价格 | ✅ 已修 |
| 3 | Timeout monotonic→wall-clock 转换漂移 | ✅ →elapsed_sec |
| 4 | auto_retrain parity 在 SIGHUP 后 | ✅ 移到前 |
| 5 | RiskAggregator KILL 不触发 KillSwitch | ✅ gate_chain 自动触发 |
| 6 | _apply_scale Decimal→float 类型漂移 | ✅ Decimal 保持 |
| 7 | fill 硬编码 FILLED 丢失 PARTIALLY_FILLED | ✅ 支持 is_partial |
| 8 | DrawdownBreaker HWM 未持久化 | ✅ checkpoint |
| 9 | Circuit breaker cooldown 循环 | ✅ max_trips=5 |
| 10 | 被拒订单进入 attribution_tracker | ✅ 仅追踪已接受 |

### 剩余热点（本轮整改目标）

| # | 热点 | 严重度 | 影响 | 难度 |
|---|------|--------|------|------|
| H1 | LiveRunner.build() 仍有 ~1200 LOC 线性构建 | medium | runtime | M |
| H2 | infra/config/schema.py 标记 deprecated 但仍在代码库 | low | config | S |
| H3 | deploy.sh 引用不存在的 compose services | medium | ops | S |
| H4 | Dockerfile COPY _quant_hotpath/ 路径可能不存在 | low | ops | S |
| H5 | CI 57% coverage 门禁偏低 | low | ci | S |

---

## 3. 具体整改工作流

### P1: LiveRunner 复杂度降低

**目标**: 不改外部行为，将 build() 的子模块提取为更清晰的层次。

**当前状态**: build() 已拆为 14 个 `_build_*` 静态方法 + 1 个 build() 编排入口。每个 `_build_*` < 120 LOC。这已经是第一轮重构的结果。

**本轮工作**:
1. 将 `_build_alert_rules()` 和 `_build_health_server()` 从模块级函数移入 LiveRunner 或 runner/builders/
2. 将 `_build_multi_tf_ensemble()` 从模块级函数移入 runner/builders/inference.py
3. 为 build() 编排入口加类型注解和阶段注释
4. 补充 build() 阶段依赖 DAG 的文档注释

**不做**: 不改 build/from_config/start/stop 的公共 API。

### P2: 配置真相源统一

**目标**: 减少 legacy config schema 与 LiveRunnerConfig 的漂移。

**当前状态**: infra/config/schema.py 已标记 deprecated (本会话)。LiveRunnerConfig 有 4 个 factory methods。

**本轮工作**:
1. 在 infra/config/schema.py 顶部加 `import warnings; warnings.warn(...)` 运行时警告
2. 确认 from_config() 读取的字段与 LiveRunnerConfig 一致
3. 为 from_config() YAML 解析补测试（flat + nested 格式）

**不做**: 不删除 schema.py（可能有外部消费者）。

### P3: Scripts 治理

**目标**: 提高可发现性，减少叙事干扰。

**当前状态**: catalog.py 覆盖 99 脚本，6 分组。cli.py 提供 `quant catalog` 入口。

**本轮工作**:
1. 确认 catalog.py 覆盖全部 scripts/*.py（补充遗漏）
2. 为 ARCHIVE_CANDIDATE 脚本加文件级注释
3. 在 scripts/README.md 加快速导航表

**不做**: 不物理移动脚本到 archive/ 目录（避免破坏已有引用）。

### P4: CI 和验证边界

**目标**: 确保每项重构有测试保护。

**当前状态**: CI 包含 pytest tests/ + execution/tests/ + cargo test + ruff。57% 覆盖率门禁。

**本轮工作**:
1. 确认本轮改动的测试覆盖
2. 运行全量验证集

**不做**: 不调 coverage 门禁（需团队讨论）。

---

## 4. 需要修改的精确文件

| PR | 文件 | 改动类型 |
|----|------|---------|
| P1a | `runner/live_runner.py` | 移动 3 个模块级 helper 到 builders/ |
| P1b | `runner/builders/monitoring.py` (新建) | 从 live_runner 提取 alert rules + health server |
| P1c | `runner/builders/inference.py` | 接收 multi_tf_ensemble builder |
| P2a | `infra/config/schema.py` | 加运行时 deprecation warning |
| P2b | `runner/live_runner.py` | from_config() 字段对齐注释 |
| P3a | `scripts/catalog.py` | 补充遗漏脚本 |
| P3b | `scripts/README.md` | 加快速导航表 |

---

## 5. 风险控制

| 风险 | 控制措施 |
|------|---------|
| build() 行为漂移 | 提取前后运行全量 runner 测试 |
| Config 解析回退 | 保留 schema.py 兼容路径，仅加 warning |
| 脚本层误删 | 不删文件，仅改 catalog 和注释 |
| CI 破坏 | 每步改动后运行 `make test` |

---

## 6. 验证计划

```bash
# 每步改动后
pytest tests/unit/runner/ -x -q -k "not test_control_plane_flush"

# P1 完成后
pytest tests/unit/runner/ tests/unit/engine/ -x -q

# P2 完成后
pytest tests/unit/runner/test_config.py -x -q

# 全量验证
make test-py && make test-exec && make test-rust
```

---

## 7. 明确不会改动的内容

1. **不替换生产入口** — live_runner.py 保持为 Python 主编排
2. **不重写事件模型** — event/types.py 的 frozen dataclass 保持不变
3. **不改执行契约** — timeout/late fill/dedup/reconcile 语义不变
4. **不改约束管线** — constraint_pipeline.rs 保持为唯一真相源
5. **不做 Rust-only 重写** — Python+Rust 混合架构保持
6. **不增加新策略** — 不扩展交易逻辑
7. **不删研究资产** — scripts/ 脚本保留在原位
8. **不改模型治理链** — register→shadow→promote→reload→rollback 不变
