# 下一阶段开发计划

> 状态: 当前主计划已从第一轮”收口”进入第三轮”alpha 自动化 + regime-adaptive 参数 + 执行优化”
> 更新时间: 2026-03-13
> 上位现状文档: [`research.md`](/quant_system/research.md)
> 当前收口成果: [`refactor_master_plan.md`](/quant_system/tasks/refactor_master_plan.md)

---

## 1. 目标

下一阶段不再以“补模块”为主，而是继续减少系统里的多套语义来源。

优先顺序：

1. 统一 execution 公共事实模型
2. 统一 `ack / reject` 的公共回执语义
3. 推进 Rust 第二阶段的 execution 纯逻辑迁移
3. 继续治理 `scripts/`
4. 补模型治理的操作闭环
5. 继续加深恢复类集成测试

---

## 2. Workstream A: Execution 公共事实模型统一

### 目标

- 固定 `CanonicalFill` 与公共 `FillEvent` 的映射边界
- 明确哪些字段是公共最小事实，哪些字段只属于 execution 私有语义

### 当前问题

- execution 内部以 `CanonicalFill` 为标准成交事实
- 公共事件层以 `FillEvent` 为最小事实
- `live_execution_bridge`、`algo_adapter`、`ingress/router` 仍各自构造 fill-like 对象

### 任务

- [x] 新增统一 fill 映射 helper
- [x] 让 ingress 路径复用统一映射逻辑
- [x] 让 `live_execution_bridge` / `algo_adapter` 复用同一 ingress fill helper
- [x] 为 `CanonicalFill -> FillEvent` 增加单测
- [x] 为 `Ack -> CanonicalRejection` 建立统一 helper
- [x] 让 `LiveExecutionBridge` 通过 `on_reject` 回调暴露统一 `CanonicalRejection`
- [x] 建立 `CanonicalRejectionEvent` 作为 event-like 公共观察对象
- [x] 明确 `CanonicalRejectionEvent` 不进入主事件总线，而进入统一 alert/ops 观察链路
- 文档中明确“公共事件是最小视图，execution 保留 richer fact model”

### 完成标准

- 新增 fill-like 事件时不再各自复制映射逻辑
- `CanonicalFill` 与 `FillEvent` 的边界可测试、可解释

---

## 3. Workstream B: Rust 第二阶段迁移

### 目标

- 在不动总装配的前提下，继续把 execution 的高频纯逻辑向 Rust 上收

### 优先候选

- timeout / sequencing helpers
- order projection / reconcile kernel
- execution ingress 纯状态逻辑

### 完成标准

- 每个候选都先有 parity tests，再迁移实现
- [x] ingress dedup 的 Rust payload guard 语义已补 parity / contract tests
- [x] ingress dedup key/digest 已收到统一 fill 映射层，不再由 router 独占维护
- [x] order ingress key/digest 已收到统一 order 映射层，mapper 与 router 复用同一 helper
- [x] order ingress `_dedup_key_and_digest()` 已固定为迁移边界 wrapper
- [x] order projection 已补一层 Python projection vs Rust state machine 的 parity 护栏
- [x] ingress sequencing wrapper 已补齐 `flush / pending_count`，并加 wrapper parity tests
- [x] order projection / reconcile kernel 已补第一批 contract tests，并修复 `missing_venue` 漂移缺失
- [x] reconcile healer 配额语义已收口，unsupported drift 不再吞掉 auto-heal budget
- [x] reconcile report 聚合语义已补 contract tests，锁住 `all_drifts / decisions / ok / should_halt`
- [x] bridge / algo synthetic fill 已统一稳定 `fill_id + payload_digest` 规则
- [x] synthetic fill 已补 bridge/algo -> ingress -> reconcile 的跨层集成测试
- [x] rejection 已补跨层 contract tests，锁住“可观察但不入 ingress / 不改状态”
- [x] rejection/retry -> success fill 的组合恢复 contract 已补跨层测试
- [x] execution alert taxonomy 已扩展到 timeout / reconcile drift / synthetic fill

## 7. Workstream F: Operator Control Plane 基础层

### 目标

- 先把当前 runtime 的最小人工控制动作收口成统一接口

### 任务

- [x] `LiveRunner` 暴露 `halt / reduce_only / resume / flush / shutdown`
- [x] `ControlEvent` 可直接通过 `LiveRunner.apply_control()` 执行
- [x] 为 operator control 增加 runner 级单测
- [x] 增加 `operator_status()` 和 `control_history`，让控制动作可查询、可审计
- [x] operator control 已接入统一 alert 链路
- [x] operator control 已补 execution/ingress/reconcile 跨层集成测试
- [x] 外部 tooling / API 已有统一入口 `OperatorControlPlane`
- [x] `OperatorControlPlane` 已提供稳定请求/结果对象
- [x] 已补 `flush -> drift -> manual halt` 组合控制/恢复测试
- [x] health server 已暴露 `GET /operator` / `POST /control`
- [x] health server 已暴露 `GET /control-history` 审计历史出口
- [x] AlertManager 已提供最近 alert 历史，health server 已暴露 `GET /execution-alerts`
- [x] health server 已暴露 `GET /ops-audit`，汇总 control / execution / model ops 审计视图
- [x] `operator_status()` / `ops_audit_snapshot()` 已补 incident 聚合字段：`stream_status / incident_state / last_incident_category / recommended_action`

---

## 4. Workstream C: Scripts 第二轮治理

### 目标

- 从“高频脚本已收口”推进到“次级脚本也有清晰地位”

### 任务

- 继续给剩余次级入口加状态标记
- 继续减少局部后处理副本
- 逐步把 archive-adjacent 路径和 current 路径分得更清楚

---

## 5. Workstream D: 模型治理操作闭环

### 目标

- 从“有制度和测试”推进到“有 runbook 和可执行上线流程”

### 任务

- promotion checklist
- rollback runbook
- live autoload 边界说明
- artifact / schema 完整性校验继续加强
- [x] loader 已提供 production artifact / autoload inspection 回执
- [x] `quant model-inspect` 已提供最小外部 inspection 入口
- [x] `quant model-promote` 已提供最小外部 promotion 入口
- [x] `quant model-rollback` 已提供最小外部 rollback 入口
- [x] registry rollback 已支持按上一个稳定版本 / 指定 version / 指定 model_id 执行
- [x] model promote / rollback 已支持 `reason / actor` 审计字段
- [x] `quant model-history` 已提供最小外部审计历史入口
- [x] `quant ops-audit` 已提供 runtime control + model ops 的统一外部审计视图

---

## 6. Workstream E: 更深恢复测试

### 目标

- 继续把恢复链路从组件级验证推进到更重的组合场景

### 任务

- reconnect + timeout + drift overlap
- startup reconcile 后接 late fill
- [x] user stream 持续失效 -> reconcile 驱动降级 / 人工 halt 已补 integration coverage
- [x] startup mismatch -> reduce_only -> flush -> ops audit 已补 integration coverage
- [x] retryable reject -> success fill -> ops audit 已补 integration coverage
- [x] health `/ops-audit` 已补真实 runner + model actions integration coverage
- [x] restart + reconnect + late fill + reconcile + ops audit 已补 integration coverage
- [x] recovery 场景已断言 incident 聚合字段，不再只验证 control / alert 是否存在
- [x] checkpoint restore + reduce_only + reconcile overlap 已补 integration coverage
- [x] model promote + autoload pending + reduce_only 已补 ops-audit integration coverage
- [x] model reload -> reloaded / noop / failed 已有统一 `model_reload` 观察字段，且已补 reloaded 集成覆盖
- [x] model reload -> failed 已补 integration coverage，并验证 pending 与 reduce_only 可同时观察
- [x] model reload 已补 `model_alerts` 观察链，避免与 execution incidents 混在同一分类里
- [x] `ops_audit.timeline` 已补统一时间线视图，串起 control / execution / model 三类事件
- [x] `ops_audit.timeline` 已接入持久化 `event_log + registry action`，可在重启后重建近期 control / model 时间线
- [x] `execution_timeout / execution_reconcile` 已补 `execution_incident` 持久化写入，并进入 runtime / CLI timeline
- [x] bridge / algo 已补 `incident_logger` 回调路径，synthetic fill / rejection 可进入 runtime 持久化 `execution_incident` timeline
- [x] runtime 重启后的 `ops_audit.timeline` 已补 integration coverage，可从同一 `event_log + registry` 重建 control / execution incident / model reload / model action 复盘链
- [x] health `/ops-audit` 已补重启后端到端复盘覆盖，外部观察口与 runtime `ops_audit_snapshot()` 在持久化 timeline 上保持一致
- [x] runtime `ops_timeline(limit=...)` 与 CLI `quant ops-audit --limit ...` 已补排序/裁剪合同测试，锁住“先聚合、按 `ts` 倒序、再裁剪”的稳定语义
- [x] `LiveRunner.build()` 的真实 checkpoint/restore 已补 ops-audit 连续性覆盖，state 恢复与持久化 incident timeline 同时验证
- [x] health `/control` 与 `/ops-audit` 已补 auth 合同测试，`/control-history` 空历史 shape 已锁住
- [x] replay 已补最小 incident category 对照测试，验证事实序列与 `execution_fill / execution_reconcile` 映射一致
- [x] execution incident payload 最小字段已补合同测试，锁住 timeout / reconcile / fill / rejection 的稳定 taxonomy/meta
- [x] health `/control-history` 与 `/execution-alerts` 已补 auth / 空 shape 合同测试
- [x] model rollback 已补 `ops_audit / timeline` 可见性覆盖，当前制度按带 rollback metadata 的 `model_action` 暴露

---

## 8. Workstream G: Alpha 自动化与模型演进 (2026-03-13 新增)

### 目标

- 从"手动训练 + 固定参数"进入"自动重训练 + 自适应参数 + IC 监控"阶段

### 已完成

- [x] IC 加权集成 (`ensemble_method: "ic_weighted"`) 替换固定权重 `mean_zscore`
- [x] h48 horizon 剔除，保留 [12, 24]，ETH IC 从 0.054 提升到 0.061
- [x] Alpha 健康监控 (`monitoring/alpha_health.py`)，三级自动响应 + Prometheus 导出
- [x] 自动重训练管线 (`scripts/auto_retrain.py`)，cron 每周日 2am UTC
- [x] 重训练验证门控 (IC>0.02, Sharpe>1.0, comparison>70%, bootstrap p5>0)
- [x] 训练后自动恢复 `ic_weighted` 集成 (修复 train_v11 默认输出 mean_zscore 的问题)
- [x] Walk-forward 验证框架 (`scripts/walk_forward.py`)，21 folds，ETH=STRONG，BTC=GOOD
- [x] 自适应参数选择器 (`alpha/adaptive_config.py`)，含 `select_robust()` 多窗口选择
- [x] 自适应参数回测验证 (`scripts/backtest_adaptive.py`)
- [x] 15m alpha 研究完成 (BTC h64 有 alpha IC=0.053, ETH 无)
- [x] BTC 15m 模型训练完成 (`models_v8/BTCUSDT_15m/`)
- [x] 混合 15m 执行策略验证完成 (已证明不一致优于 1h baseline)

### 待做

- [ ] BTC 自适应参数集成到 live runner (仅 confidence=high 时覆盖固定配置)
- [ ] ETH 保持固定配置，不启用自适应
- [ ] Alpha 健康监控集成到 live runner order sizing
- [ ] BTC 15m 模型作为辅助信号集成 (非替代 1h 主模型)
- [ ] 监控 auto_retrain cron 实际运行情况与日志
- [ ] Walk-forward 定期重跑验证 alpha 持续性

---

## 9. Workstream H: 下一步方向候选

### 方向 1: Regime-Adaptive 参数优化

当前核心发现: 最优参数随 regime 显著漂移 (ETH deadzone 0.3-2.5, long_only 反复切换)。
- 研究更好的 regime 检测方法 (volatility clustering, trend detection)
- 基于 regime 切换参数，而不是基于 lookback period 选择
- Oracle gap 巨大 (Sharpe 4.63→7.48)，说明参数空间有大量未捕获 alpha

### 方向 2: 执行优化

- WS-API 订单路径 (~4ms) 已就绪，集成到 live runner
- 订单执行质量分析 (滑点、填充率)
- TWAP/VWAP 算法订单在大仓位时使用

### 方向 3: 多资产与因子组合

- SOL、XRP 等 altcoin alpha 研究
- 跨资产因子组合 (momentum/vol/carry)
- portfolio-level rebalancing 集成
