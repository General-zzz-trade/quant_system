# 下一步开发计划

> **Status**: COMPLETED (2026-03-24) — 已被后续计划完全取代.
> 更新时间: 2026-03-12. 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).
> 生产入口已从 `runner/live_runner.py` 迁移至 `runner/alpha_main.py` (EngineCoordinator + AlphaDecisionModule).
> 本文保留为阶段性计划快照，不再作为当前执行主计划.

> 基准日期: 2026-03-12
> 基准来源: 当前源码，而非历史 README / 迁移路线图
> 当前判断: 系统已形成核心闭环，但尚未完全收口；下一阶段目标是收敛生产主路径、统一契约、强化恢复能力

---

## 当前阶段判断

| 维度 | 当前状态 | 结论 |
|---|---|---|
| 功能闭环 | 已具备行情、状态、特征、决策、风控、执行、监控、回测、实盘 | 核心闭环已形成 |
| 生产真相源 | 已收口至 `runner/alpha_main.py` + `CLAUDE.md` (原分散在 `runner/live_runner.py`、`rust/`、若干文档中) | 已完成 |
| Python / Rust 边界 | 已混合运行，Rust 已覆盖大量热路径，Python 仍主编排 | 迁移未最终完成 |
| live/backtest/replay 一致性 | 大方向一致，但约束与边界仍需明确 | 需要统一契约 |
| 故障恢复 | 已有 reconcile / healer / timeout / checkpoint / replay | 需要制度化演练与补强 |

---

## P0：运行时收口（当前最高优先级）

### 1. 定义唯一生产主路径

**目标**
- 明确当前唯一生产入口
- 明确 Python / Rust ownership
- 明确 standalone Rust trader 的定位

**要做的事**
- [x] 新建 `docs/runtime_truth.md`
- [x] 明确 `runner/alpha_main.py` 为当前 Python 生产主入口 (原 `runner/live_runner.py`，已迁移)
- [ ] 明确 `rust/src/bin/main.rs` 为候选/演进路径，而非当前默认真相源
- [ ] 梳理核心子系统 owner：`engine` / `state` / `features` / `risk` / `execution`
- [ ] 清理 README / operations / api 中与现状冲突的说法

**交付物**
- `docs/runtime_truth.md`
- 一份 Python vs Rust ownership matrix
- 一份运行路径说明：live / backtest / replay / rust binary

### 2. 统一 live / backtest / replay 契约

**目标**
- 统一事件语义
- 统一状态快照最小字段
- 统一线上与回测约束

**要做的事**
- [x] 在 `docs/runtime_truth.md` 中定义契约基线
- [x] 列出 `Event` / `Snapshot` / `OrderSpec` / `Fill` 最小字段集
- [ ] 对齐 live 与 backtest 的 `deadzone` / `min_hold` / trend hold / monthly gate / vol target
- [x] 补一组契约对齐测试

**交付物**
- `docs/runtime_truth.md` 中的契约基线章节
- `docs/runtime_contracts.md`
- 新增契约测试：`tests/contract/test_runtime_contracts.py`
- 新增 replay 基线测试：`tests/replay/test_event_ordering.py`、`tests/replay/test_replay_determinism.py`、`tests/replay/test_replay_vs_live_equivalence.py`
- 回测约束对齐第一轮：`decision/backtest_module.py`、`tests/unit/decision/test_backtest_module_constraints.py`
- 回测入口透传：`scripts/backtest_engine.py`、`tests/unit/scripts/test_backtest_engine_constraints.py`
- 配置层透传对齐：`runner/testnet_validation.py`、`tests/unit/runner/test_build_ml_stack_v8.py`

### 3. 强化故障恢复闭环

**目标**
- 让系统不仅能跑，而且在异常时能恢复

**要做的事**
- [x] 明确 crash recovery 标准流程
- [ ] 明确 checkpoint restore 与 replay 恢复边界
- [x] 强化 user stream 断连后的 reconcile 策略
- [x] 梳理 timeout tracker / order state machine / reconcile scheduler 的联动
- [ ] 为断连、重复 fill、乱序 fill、重启恢复补集成测试

**交付物**
- `docs/production_runbook.md`
- 恢复类集成测试

---

## P1：生产行为收口

### 4. 执行层行为统一

**目标**
- 收敛 `ExecutionBridge`、`LiveExecutionBridge`、algo adapter 的边界

**要做的事**
- [ ] 明确 direct execution 与 algo execution 的路由标准
- [ ] 统一 ack / reject / fill 的 canonical result
- [ ] 收敛 order state machine 与 user stream 的一致性语义
- [ ] 强化 latency / TCA / reject reason 可观测性

### 5. 风控体系分层固化

**目标**
- 把现在多层 gate 的职责说清楚，并落成统一动作语义

**要做的事**
- [ ] 明确 decision overlay、execution boundary、portfolio/account hard risk 的分层
- [ ] 统一 `allow / reduce / reject / kill`
- [ ] 梳理 margin / stale data / kill switch / portfolio risk 的联动
- [ ] 补风控回放测试

### 6. 模型上线闭环固定

**目标**
- 把 research -> registry -> live autoload 变成稳定制度

**要做的事**
- [ ] 统一训练特征 schema 与线上特征 schema
- [ ] 固定 model registry / artifact store / autoload 规则
- [ ] 把 walk-forward / significance / overfit detection 结果变成准入门槛
- [ ] 定义模型晋升、回滚、灰度流程

---

## P2：Rust 迁移第二阶段

### 7. 删除无效双栈维护成本

**目标**
- 继续把高频纯逻辑向 Rust 收口，但不做无意义重写

**要做的事**
- [ ] 识别仍在 Python 热路径中的高频逻辑
- [ ] 删除已无意义的 fallback / 兼容层
- [ ] 明确哪些 Python 模块长期保留
- [ ] 扩充 parity tests，保证替换安全

**优先迁移候选**
- execution ingress / sequencing
- replay / backtest kernel 的统一纯逻辑
- dispatcher 周边高频纯逻辑
- order lifecycle 的纯状态变换

---

## P3：文档与运维真相源统一

### 8. 文档体系收口

**目标**
- 让新开发者不再依赖过时文档猜现状

**要做的事**
- [ ] 更新 `README.md` 的规模、能力边界、运行方式
- [ ] 更新 `docs/operations.md` 的生产路径说明
- [ ] 更新 `docs/api.md` 的契约与 owner 说明
- [ ] 区分“当前现状文档”与“迁移规划文档”

**建议新增文档**
- [x] `docs/runtime_truth.md`
- [ ] `docs/testing_matrix.md`
- [ ] `docs/production_runbook.md`

---

## 执行顺序

```text
Phase 1:
  P0-1 生产主路径真相源
  P0-2 live/backtest/replay 契约基线

Phase 2:
  P0-3 故障恢复闭环
  P1-4 执行层行为统一
  P1-5 风控分层固化

Phase 3:
  P1-6 模型上线闭环
  P2-7 Rust 迁移第二阶段
  P3-8 文档与运维统一
```

---

## 本轮已开始执行

- [x] 重写 `research.md` (现已整合至 `CLAUDE.md`)
- [x] 新建 `docs/runtime_truth.md`
- [x] 新建 `docs/runtime_contracts.md`
- [x] 更新 `tasks/next_development_plan.md`
- [x] 修正文档中与真实生产路径冲突的内容（README / operations / api 第一轮）
- [x] 补最小契约测试与 replay 真测试
- [x] 回测约束对齐第一轮：补 `monthly_gate` / `trend_follow` / `vol_target`
- [x] `trend_follow` 语义第一轮收口：从入场门禁改为更接近 live 的持仓延续逻辑
- [x] `monthly_gate` 语义第一轮收口：从仅限制开仓改为更接近 live 的门限失效平仓逻辑
- [x] 配置层透传对齐：补 strategy-level `trend_follow` / `vol_target` 覆盖
- [x] 新增 cross-path parity tests：直接对比 live `LiveInferenceBridge` 与 backtest module 的 trend hold / monthly gate / min_hold / vol_target 行为
- [x] 恢复路径第一轮加固：补 `LiveRunner` 的 user stream reconnect / timeout loop / startup reconcile 定向测试
- [x] 修复 `LiveRunner.stop()` 在 `_running` 已清零时漏清理资源的问题
- [x] 修复 `LiveRunner._apply_perf_tuning()` 对 `nohz_full=(null)` 的健壮性问题
- [x] 修复 startup reconcile 读取本地余额字段错误的问题
- [x] 补晚到执行回报测试：验证 timeout cancel 后的 `pending_cancel -> filled` 恢复路径
- [x] 补 cancel-replace 安全测试：锁定“原单终态后不可被晚到 fill 复活”的边界
- [x] 补恢复集成测试：覆盖乱序/重复 fill + checkpoint/restart + reconcile 联动
- [x] 研究脚本收口第一轮：提取 `scripts/signal_postprocess.py`，统一 monthly gate / trend hold / vol_target / min_hold / rolling z-score / discrete exit rule helper，并接入 `walkforward_validate.py` / `walkforward_validate_1m.py` / `backtest_alpha_v8.py` / `backtest_portfolio.py` / `train_short_production.py` / `train_multi_horizon.py` / `backtest_small_cap.py` / `backtest_honest.py` / `train_btc_v9b.py` / `train_eth_v9.py`
- [x] 研究脚本收口第二轮：`ic_analysis_short_features.py` 改为复用共享 bear mask helper，并保留原有 warmup 全 False 语义
- [x] `scripts/` 架构整理第一轮：新增 `scripts/catalog.py` 和 `scripts/README.md`，明确 train / validate / research / data / ops / shared 六类分层与当前主力入口
- [ ] 下一步：继续把 backtest 约束状态机向 live `LiveInferenceBridge` 靠拢，并补更深的恢复类集成测试
