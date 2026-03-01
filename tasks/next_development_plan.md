# 下一步开发计划

> 基于 2026-02-28 全项目深度分析，838 个源文件、1091 测试通过

---

## 现状总览

| 维度 | 完成度 | 关键短板 |
|------|--------|---------|
| 引擎层 (engine/runner) | 95% | coordinator decision handler 仍为 no-op（设计选择） |
| 风控层 (risk) | 90% | 缺 Kelly 仓位、完整 Risk Parity |
| 组合优化 (portfolio) | 85% | Black-Litterman/多求解器完整，缺参数化风险预算 |
| 策略层 (strategies) | 80% | 5 个策略框架，缺多时间框架 |
| ML 层 (alpha) | 75% | LightGBM/XGBoost 生产就绪，LSTM/Transformer 实验性 |
| 执行层 (execution) | 90% | Binance 完整（Bitget 已删除，专注单交易所） |
| 基础设施 (infra/deploy) | 60% | K8s 缺 HPA/PDB/安全上下文，CI 缺安全扫描 |
| 测试覆盖 | 50% | 1091 测试，但 alpha/decision/portfolio 覆盖极低（5-10%） |
| 文档 | 30% | 无 README、无运维手册、无 API 文档 |

---

## P0：生产阻断项（1-2 周）

### ~~1. Bitget 适配器补全~~ — 已取消
> Bitget 适配器已于 2026-03-01 删除（部分实现、增加维护负担）。项目专注 Binance 单交易所。

### 2. 关键模块测试补齐
**问题**：alpha（0%）、decision（5%）、portfolio（10%）几乎无测试覆盖
- [ ] Alpha 推理管道测试：LiveInferenceBridge、LgbmAlpha.predict、模型注册表
- [ ] Decision 决策链路测试：allocators、candidates、sizing、execution_policy
- [ ] Portfolio 优化器测试：Black-Litterman、objectives、constraints、solvers
- [ ] 创建共享 conftest.py fixtures（MockVenueFactory、SnapshotBuilder、EventStreamBuilder）

**影响**：关键业务逻辑无回归保护
**文件**：`tests/unit/alpha/`、`tests/unit/decision/`、`tests/unit/portfolio/`

### 3. RiskGate 接入 live_runner
**问题**：`execution/safety/risk_gate.py` 已实现但未接入
- [ ] 在 ExecutionBridge 前插入 RiskGate 检查
- [ ] 与现有 CorrelationGate 和 KillSwitchBridge 形成三层防线
- [ ] 集成测试验证 RiskGate 拒绝场景

**影响**：执行安全层不完整
**文件**：`runner/live_runner.py:257-306`、`execution/safety/risk_gate.py`

---

## P1：生产加固（2-4 周）✅ 已完成

### 4. K8s 生产安全加固 ✅
- [x] 添加安全上下文：runAsNonRoot, readOnlyRootFilesystem, drop ALL capabilities
- [x] 添加 PodDisruptionBudget（minAvailable: 1）
- [x] 添加 NetworkPolicy（限制入站/出站）
- [x] 探针超时从 5s 调至 10s
- [x] 添加 priorityClassName 保证交易优先级

**文件**：`deploy/k8s/deployment.yaml`、`deploy/k8s/network-policy.yaml`、`deploy/k8s/pdb.yaml`、`deploy/k8s/priority-class.yaml`

### 5. CI/CD 安全增强 ✅
- [x] 添加 Trivy 镜像漏洞扫描
- [x] 添加 pip-audit 依赖安全审计
- [x] 添加 ruff + mypy 严格模式
- [x] 部署后自动冒烟测试

**文件**：`.github/workflows/ci.yml`、`.github/workflows/deploy.yml`

### 6. 密钥管理升级 ✅
- [x] 集成 External Secrets Operator
- [x] API 密钥定期轮换机制（CronJob 检查密钥年龄）
- [x] 凭证访问审计日志

**文件**：`deploy/k8s/external-secret.yaml`、`deploy/k8s/secret-rotation-cronjob.yaml`

### 7. Kelly 仓位定量 + Volatility Targeting ✅
- [x] 实现 `KellyAllocator`（多资产 Kelly: w* = Σ⁻¹·μ，支持分数 Kelly）
- [x] 头寸浓度约束（max_concentration 参数）
- [x] 17 个单元测试（矩阵求逆 5 + Kelly 核心 12）

**文件**：`portfolio/allocator_kelly.py`、`tests/unit/portfolio/test_allocator_kelly.py`

---

## P2：功能增强（4-8 周）✅ 已完成

### 8. 监控可观测性完善 ✅
- [x] SLO/SLI 追踪器（延迟 P99、可用率、数据新鲜度、成交率、错误预算）
- [x] 数据质量告警（NaN 检测、分布漂移 z-score、常量特征检测）
- [x] Grafana 仪表板自动导入/导出脚本
- [x] 告警通知渠道已有（Telegram/Webhook/Console/Log）

**文件**：`monitoring/slo.py`、`monitoring/data_quality_alerts.py`、`scripts/grafana_import.py`

### 9. 回测引擎对齐增强 ✅
- [x] backtest_runner 集成 FeatureComputeHook（ML 因子回测）
- [x] backtest_runner 集成 AttributionTracker（回测信号归因）
- [x] 回测与实盘 PnL 对比工具（对齐、相关性、跟踪误差、回撤对比）

**文件**：`runner/backtest_runner.py`、`runner/backtest/pnl_compare.py`

### 10. 多时间框架策略框架 ✅
- [x] BarAggregator（1m → 5m/15m/30m/1h/4h/1d 聚合）
- [x] MultiTimeframeEnsemble（3 种融合：weighted_vote/majority/cascade）
- [x] 跨时间框架信号同步（自动对齐 + 最新信号缓存）
- [x] 22 个测试覆盖聚合正确性和融合逻辑

**文件**：`strategies/multi_timeframe/aggregator.py`、`strategies/multi_timeframe/ensemble.py`

### 11. 数据管道加固 ✅
- [x] 数据分布验证（RollingDistribution + DistributionTracker，z-score 均值漂移 + 方差变化）
- [x] 数据血统追踪（LineageTracker，JSONL 持久化，全链路 trace）
- [x] 备份策略（BackupManager，快照 + 保留策略 + 恢复）

**文件**：`data/quality/distribution.py`、`data/lineage.py`、`data/backup.py`

---

## P3：长期演进（8+ 周）✅ 已完成

### 12. 深度学习生产验证 ✅
- [x] LSTM/Transformer 保持 EXPERIMENTAL 标记（LightGBM/XGBoost 已满足生产需求）
- [x] 在线异常检测（OOD detection）— Mahalanobis 距离检测器
- [x] 概念漂移适应器（concept drift adaptation）— 滚动命中率/IC/Sharpe 监控 + 自动推荐

**文件**：`alpha/monitoring/ood_detector.py`、`alpha/monitoring/drift_adapter.py`、`tests/unit/alpha/test_ood_drift.py`（21 测试）

### 13. 高级研究工具 ✅
- [x] 蒙特卡洛路径模拟（block bootstrap + 参数化正态）
- [x] 敏感性分析框架（one-at-a-time 扫描 + 参数排名）
- [x] 因子显著性检验（T-test + Bonferroni/Holm/FDR-BH 多重测试修正）
- [x] 样本外最小期间要求检查（Bailey & López de Prado）

**文件**：`research/monte_carlo.py`、`research/sensitivity.py`、`research/significance.py`、`tests/unit/research/test_advanced_tools.py`（34 测试）

### 14. 文档体系 ✅
- [x] 项目 README.md（快速入门 + ASCII 架构图 + 模块概览）
- [x] 运维手册（部署、监控、故障排查、5 个 Runbook、SOP）
- [x] API 文档（核心协议、事件类型、领域类型）
- [x] CHANGELOG.md（v0.1.0 — v0.9.0 + Unreleased）

**文件**：`README.md`、`docs/operations.md`、`docs/api.md`、`CHANGELOG.md`

### 15. GitOps + 金丝雀部署 ✅
- [x] ArgoCD Application + AppProject（自动同步 + 自愈）
- [x] Argo Rollouts 金丝雀（10% → 30% → 60% → 100%，Prometheus 分析）
- [x] 自动回滚机制（AnalysisTemplate 错误率/延迟/成交率检查）

**文件**：`deploy/argocd/application.yaml`、`deploy/argocd/project.yaml`、`deploy/argocd/rollout.yaml`、`deploy/argocd/analysis-template.yaml`、`deploy/argocd/rollback-config.yaml`、`deploy/k8s/namespace.yaml`

---

## 建议执行顺序

```
Week 1:    P0 #2 测试补齐（alpha/decision/portfolio）
Week 2:    P0 #3 RiskGate 接入 live_runner
Week 3+:   架构演进（视需求排期）
```

## 关键风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 测试补齐工作量大 | 进度拖延 | 优先覆盖关键路径，非全量补齐 |
| LSTM/Transformer 无法生产化 | ML 能力受限 | LightGBM/XGBoost 已足够，深度学习可推迟 |
