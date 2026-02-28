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
| 执行层 (execution) | 70% | Binance 完整，**Bitget 残缺**（无 WS、无 MarketDataRuntime） |
| 基础设施 (infra/deploy) | 60% | K8s 缺 HPA/PDB/安全上下文，CI 缺安全扫描 |
| 测试覆盖 | 50% | 1091 测试，但 alpha/decision/portfolio 覆盖极低（5-10%） |
| 文档 | 30% | 无 README、无运维手册、无 API 文档 |

---

## P0：生产阻断项（1-2 周）

### 1. Bitget 适配器补全
**问题**：14 个文件但缺核心能力，无法实盘使用
- [ ] 实现 `BitgetMarketDataRuntime`（参照 BinanceMarketDataRuntime）
- [ ] 实现 `BitgetWsMarketStreamClient`（WebSocket 行情流）
- [ ] 补全 `BitgetFuturesVenueClient`（get_balance, get_positions, get_instruments）
- [ ] `live_runner.py` 增加 `if exchange == "bitget"` 分支
- [ ] Bitget 适配器集成测试

**影响**：多交易所部署被阻断
**文件**：`execution/adapters/bitget/`

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

## P1：生产加固（2-4 周）

### 4. K8s 生产安全加固
- [ ] 添加安全上下文：runAsNonRoot, readOnlyRootFilesystem, drop ALL capabilities
- [ ] 添加 PodDisruptionBudget（minAvailable: 1）
- [ ] 添加 NetworkPolicy（限制入站/出站）
- [ ] 探针超时从 5s 调至 10s
- [ ] 添加 priorityClassName 保证交易优先级

**文件**：`deploy/k8s/deployment.yaml`、新增 `deploy/k8s/network-policy.yaml`、`deploy/k8s/pdb.yaml`

### 5. CI/CD 安全增强
- [ ] 添加 Trivy 镜像漏洞扫描
- [ ] 添加 pip-audit 依赖安全审计
- [ ] 添加 ruff + mypy 严格模式
- [ ] 部署后自动冒烟测试

**文件**：`.github/workflows/ci.yml`、`.github/workflows/deploy.yml`

### 6. 密钥管理升级
- [ ] 集成 Kubernetes Sealed Secrets 或 External Secrets Operator
- [ ] API 密钥定期轮换机制
- [ ] 凭证访问审计日志

**文件**：`infra/auth/`、`deploy/k8s/`

### 7. Kelly 仓位定量 + Volatility Targeting
- [ ] 实现 `KellyAllocator`（基于预期 Sharpe 的最优头寸）
- [ ] 扩展 `VolTargetAllocator` 支持动态 leverage 调整
- [ ] 头寸浓度约束（单品种 < 组合 X%）
- [ ] 单元测试 + 回测验证

**文件**：新增 `portfolio/allocator_kelly.py`、修改 `portfolio/allocator.py`

---

## P2：功能增强（4-8 周）

### 8. 监控可观测性完善
- [ ] 定义 SLO/SLI（延迟 P99 < 5s、可用率 > 99.9%）
- [ ] OpenTelemetry 完整集成到关键路径（market → decision → order → fill）
- [ ] 数据质量告警（NaN、缺失值、分布异常）
- [ ] Grafana 仪表板自动导入脚本
- [ ] 告警通知渠道扩展（Slack/Telegram webhook）

**文件**：`monitoring/`、`infra/tracing/`、`deploy/grafana/`

### 9. 回测引擎对齐增强
- [ ] backtest_runner 集成 FeatureComputeHook（ML 因子回测）
- [ ] backtest_runner 集成 AttributionTracker（回测信号归因）
- [ ] 回测与实盘 PnL 对比工具

**文件**：`runner/backtest_runner.py`

### 10. 多时间框架策略框架
- [ ] 设计 MultiTimeframeEnsemble 架构
- [ ] 实现跨时间框架信号同步机制
- [ ] 1m/5m/15m/1h 多周期信号融合
- [ ] 回测验证

**文件**：新增 `strategies/multi_timeframe/`

### 11. 数据管道加固
- [ ] 增量更新机制（避免重复下载）
- [ ] 数据分布验证（mean shift、outlier detection）
- [ ] 数据血统追踪（source/version/timestamp 元数据）
- [ ] 备份策略（定时 parquet 快照 → S3/OSS）

**文件**：`data/`、`scripts/`

---

## P3：长期演进（8+ 周）

### 12. 深度学习生产验证
- [ ] LSTM/Transformer 生产验证或正式废弃
- [ ] 在线异常检测（OOD detection）
- [ ] 概念漂移适应器（concept drift adaptation）

### 13. 高级研究工具
- [ ] 蒙特卡洛路径模拟
- [ ] 敏感性分析框架
- [ ] 因子显著性检验（T-test + 多重测试修正）
- [ ] 样本外最小期间要求检查

### 14. 文档体系
- [ ] 项目 README.md（快速入门 + 架构概览）
- [ ] 运维手册（部署、故障排查、告警响应 SOP）
- [ ] API 文档（核心模块接口自动生成）
- [ ] CHANGELOG.md

### 15. GitOps + 金丝雀部署
- [ ] ArgoCD/FluxCD 替代手动 workflow_dispatch
- [ ] 金丝雀部署（shadow → canary → full rollout）
- [ ] 自动回滚机制

---

## 建议执行顺序

```
Week 1-2:  P0 #1 Bitget 补全 + P0 #2 测试补齐（并行）
Week 2-3:  P0 #3 RiskGate 接入 + P1 #4 K8s 加固
Week 3-4:  P1 #5 CI 安全 + P1 #7 Kelly 仓位
Week 4-6:  P2 #8 监控 + P2 #9 回测对齐
Week 6-8:  P2 #10 多时间框架 + P2 #11 数据管道
Week 8+:   P3 长期演进
```

## 关键风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Bitget API 不稳定 | 多所部署延迟 | 先 mock 测试，后接真实 API |
| 测试补齐工作量大 | 进度拖延 | 优先覆盖关键路径，非全量补齐 |
| LSTM/Transformer 无法生产化 | ML 能力受限 | LightGBM/XGBoost 已足够，深度学习可推迟 |
| K8s 安全加固影响现有部署 | 服务中断 | staging 环境先行验证 |
