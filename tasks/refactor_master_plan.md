# 代码库改造总计划

> 状态: 当前这轮代码库收口工作的主计划，核心 checklist 已完成
> 更新时间: 2026-03-12
> 当前系统现状请配合 [`research.md`](/quant_system/research.md) 一起阅读
> 下一阶段执行主线请参考: [`next_phase_plan.md`](/quant_system/tasks/next_phase_plan.md)

> 更新时间: 2026-03-12
> 基准: 当前源码、`docs/runtime_truth.md`、`docs/runtime_contracts.md`、`docs/production_runbook.md`
> 目标: 把当前“功能闭环已形成、但仍在分叉演进”的系统，推进到单一真相源、统一契约、可恢复、可维护的生产系统

---

## 1. 当前诊断

当前代码库的真实状态：

- 生产主路径已经明确，默认入口是 `runner/live_runner.py`
- Python 编排 + Rust 热路径的混合架构已经成立
- 恢复、回放、执行安全、契约测试已经有基础
- 最大问题不是缺少功能，而是语义分叉、脚本层历史沉积、执行语义未完全收口、模型治理制度不足

核心改造原则：

1. 先加契约和测试，再改语义
2. 先收口共享 helper，再删除重复实现
3. 先做非破坏性索引和标记，再做移动/归档
4. 先制度化 incident policy，再深化执行层重构

---

## 2. Workstream A: 约束状态机统一

**目标**

- 让 live / backtest / replay / research 在关键约束上尽可能共享同一真相源

**当前问题**

- `alpha/inference/bridge.py` 是 live 语义来源
- `decision/backtest_module.py` 是回测对齐实现
- `scripts/signal_postprocess.py` 是研究脚本层共享 helper
- 三者已经更接近，但还不是单一实现

**任务**

- [x] 扩展 `deadzone` 边界 parity tests
- 持续把脚本层重复的 `monthly_gate / min_hold / trend / exit` 收进共享 helper
- 为剩余自定义研究脚本明确“复用共享语义”还是“保留局部差异”

**完成标准**

- 同一输入序列下，live/backtest 在核心约束上的行为一致
- 新研究脚本不再复制关键约束逻辑

---

## 3. Workstream B: 恢复与 Incident Policy

**目标**

- 把“有恢复组件”升级为“有统一恢复制度”

**当前问题**

- checkpoint / reconcile / timeout / late fill / drift 已有能力
- 但动作语义和运维决策边界仍不够统一

**任务**

- 收口 `ack / reject / fill / timeout / pending_cancel / drift` 的 canonical 语义
- 扩充 restart + reconcile + reconnect 的组合测试
- 将 incident matrix 固化到 runbook

**完成标准**

- 关键恢复案例都能映射到固定排障步骤
- execute/reconcile 恢复行为有明确 contract

---

## 4. Workstream C: Scripts 治理

**目标**

- 把 `scripts/` 从“平铺工具箱”变成“受治理的研究与运维工作层”

**当前问题**

- 文件多、历史沉积深、当前主力入口和实验脚本混在一起

**任务**

- 维护 `scripts/catalog.py` 作为分类真相源
- 用 `current / experimental / archive-adjacent` 状态标记主力入口
- 在 `scripts/README.md` 中固定推荐入口
- 后续对明显过时脚本打标并逐步归档

**完成标准**

- 新开发者可以明确知道哪些脚本当前应该使用
- 继续重构时不需要靠人工记忆判断脚本地位

---

## 5. Workstream D: 模型治理

**目标**

- 把“能训练、能加载模型”升级为“模型生命周期可治理”

**任务**

- 固定 registry / loader / live autoload 的职责边界
- 固定 promotion / rollback / shadow compare 流程
- 增强 schema mismatch / model mismatch 检查

**完成标准**

- 生产模型变更有明确准入、回滚、审计规则

---

## 6. Workstream E: Rust 第二阶段收口

**目标**

- 只迁高频、纯逻辑、边界清晰的部分，减少双栈维护成本

**任务**

- 识别 Python 热路径中仍可安全迁移的纯逻辑
- 删除无意义 fallback
- 补 parity tests，确保替换安全

**非目标**

- 不激进重写 `runner/live_runner.py`
- 不重写 `engine/` 主因果链
- 不打散当前单通道状态推进结构

---

## 7. 执行顺序

### Phase 1

- Workstream A: 约束状态机统一
- Workstream C: Scripts 治理

#### Phase 1 Checklist

- [x] 为 live/backtest 增加 `deadzone` 边界 parity tests
- [x] 修复 `decision/backtest_module.py` 的 deadzone flatten 语义
- [x] 继续把高频训练/回测脚本接到 `scripts/signal_postprocess.py`
- [x] `train_multi_horizon.py`
- [x] `backtest_small_cap.py`
- [x] `backtest_honest.py`
- [x] `train_btc_v9b.py`
- [x] `train_eth_v9.py`
- [x] `train_4h_production.py`
- [x] `backtest_multi_tf.py`
- [x] `train_btc_production.py`
- [x] `train_eth_production.py`
- [x] `train_btc_v9.py`
- [x] `train_15m.py`
- [x] `walk_forward.py`
- [x] `backtest_adaptive.py`
- [x] `backtest_hybrid_15m.py`
- [x] `research_15m_alpha.py`
- [x] `train_30m_production.py`
- [x] 给剩余高频 scripts 标记 `current / experimental / archive-adjacent`
- [x] 为 scripts 共享 helper 增加更系统的单测覆盖

验证方式:
- `pytest -q tests/unit/decision/test_backtest_live_parity.py`
- `pytest -q tests/unit/decision/test_backtest_module_constraints.py tests/unit/alpha/test_inference_bridge.py tests/unit/scripts/test_backtest_engine_constraints.py tests/unit/runner/test_backtest_core.py tests/replay/test_replay_vs_live_equivalence.py tests/contract/test_runtime_contracts.py`
- `python3 -m py_compile` 针对改动的 scripts 文件

完成标准:
- 高频研究/训练/回测脚本不再各自维护同构的 `rolling_zscore` / `should_exit_position` 副本
- live/backtest 在核心离散约束上没有已知未锁住的行为差异

### Phase 2

- Workstream B: 恢复与 Incident Policy

#### Phase 2 Checklist

- [x] 盘点 `execution/` 中 `ack / reject / fill / timeout / pending_cancel` 的当前语义
- [x] 新增 execution contract 文档
- [x] 新增 restart + reconnect + reconcile 组合恢复测试
- [x] 将 incident matrix 固化到 runbook

完成标准:
- 恢复动作可以映射到统一 incident policy
- execution 终态语义不再依赖多处解释

### Phase 3

- Workstream D: 模型治理

#### Phase 3 Checklist

- [x] 固定 `registry / loader / live autoload` 职责边界
- [x] 形成 production promote / rollback / shadow compare 流程
- [x] 增加 schema mismatch / model mismatch 测试

完成标准:
- 生产模型变更有明确准入与回滚制度

### Phase 4

- Workstream E: Rust 第二阶段收口

#### Phase 4 Checklist

- [x] 识别 Python 热路径纯逻辑候选
- [x] 为候选逻辑补 parity tests
- [x] 删除已无意义 fallback

完成标准:
- 双栈维护成本明显下降，且 parity tests 保持通过

---

## 8. 本轮已开始落地

- `scripts/catalog.py` 已建立 scripts 分类真相源
- `scripts/README.md` 已建立 scripts 分层说明
- `scripts/cli.py` 已支持 catalog 输出
- `scripts/signal_postprocess.py` 已开始承接研究脚本共享后处理逻辑
- `decision/backtest_module.py` 已进一步收口 deadzone flatten 语义，并新增 live/backtest parity tests 覆盖 deadzone entry blocking / fade exit
- `train_btc_production.py`、`train_eth_production.py`、`train_btc_v9.py` 已切换到共享 `rolling_zscore` / `should_exit_position`
- `train_btc_v9b.py` 与 `train_eth_v9.py` 的 bootstrap / OOS 验证路径中残留的离散退出副本已切换到共享 `should_exit_position`
- `train_15m.py`、`walk_forward.py`、`backtest_adaptive.py` 已切换到共享 helper；`tests/unit/scripts/test_signal_postprocess.py` 已补共享 helper 单测
- `backtest_hybrid_15m.py`、`research_15m_alpha.py`、`train_30m_production.py` 已切换到共享 helper，并在 `scripts/catalog.py` 中补充了更多 current / experimental / archive-adjacent 状态标记
- `scripts/catalog.py` 与 `scripts/README.md` 已补齐 `train_multi_horizon.py`、`train_btc_production.py`、`train_eth_production.py`、`train_short_production.py`、`train_4h_production.py`、`backtest_portfolio.py`、`backtest_multi_tf.py`、`walkforward_short.py` 等高频入口的状态标记
- `docs/runtime_truth.md`、`docs/runtime_contracts.md`、`docs/production_runbook.md` 已形成运行时和恢复真相源基础
- `docs/execution_contracts.md` 已建立 execution 层状态机 / timeout / late fill / canonical fill 的真相源基础
- `tests/integration/test_execution_timeout_restart_recovery.py` 已补 timeout cancel -> restart -> late fill 的组合恢复测试
- `docs/production_runbook.md` 与 `docs/execution_contracts.md` 已补 incident matrix，固定 timeout / late fill / duplicate fill / startup drift 的默认动作
- `docs/model_governance.md` 已建立 registry / loader / shadow compare / promote / rollback 的模型治理真相源
- `alpha/model_loader.py` 已新增 feature schema mismatch 拒绝加载逻辑，并补充 registry rollback / loader rollback / schema mismatch 测试
- `docs/rust_replacement_matrix.md` 已补 Phase 4 当前候选：timeout/sequencing、dispatcher routing、order projection/reconcile kernel、backtest signal constraints
- `tests/unit/engine/test_dispatcher.py` 已补 dispatcher routing 的 Rust/Python parity tests；`tests/unit/execution/test_out_of_order_guard_parity.py` 已补 `OutOfOrderGuard` 对 Rust `RustSequenceBuffer` 的重排 / flush / 容量 / reset parity tests
- `engine/dispatcher.py` 已删除不再使用的 `EventType` 导入 fallback，dispatcher routing 现完全以 Rust route matcher 为真相源
