# 运行监控与发布验收

本阶段实现并验证工程门控，没有启动真实服务、连接账户、发送通知或下单。离线检查不代表在线 paper 或实盘验收通过。

## 已修复的问题

旧 Compose 健康检查只导入 Python 模块。在没有任何运行状态、没有主循环心跳时仍退出 0，不能证明服务就绪。复现证据为 `stage5-health-before.log`。

第四项提交的 GitHub 容器检查还发现 `.dockerignore` 遗漏运维入口：`scripts.check_execution_runtime` 和 `scripts.migrate_runtime_state` 不在镜像中。源码挂载测试没有覆盖这个构建差异；原 CI 失败保留在 `stage4-github-container.log`。本阶段补齐白名单，并让离线测试逐个执行镜像内运维 CLI 的 `--help`。

## 运行状态与执行边界

- `shared/runtime_status.py` 原子写入 `data/runtime/runtime-status.json`，包含启动标识、进程身份、阶段、心跳和脱敏后的数据/对账证据。使用临时文件、fsync 和替换，拒绝重复 JSON 键、异常数值、错配身份和符号链接。
- `runner/runtime_monitor.py` 在现有主循环同步采集；没有后台线程替挂起的主循环续期。正常每 5 次循环发布，心跳超过 30 秒即失效。执行对账还必须在 120 秒内成功；组合和钱包保留实际配置的更严格时效。
- 新增风险先检查心跳及执行对账，再进入原风控与额度预留。减仓继续经过原有验证，不改策略数学、仓位公式或减仓规则。读取监控数据库使用只读模式，缺库不会被创建为空库。
- `scripts/check_runtime_health.py` 分别报告 `live`、`ready`、`execution_eligible`，从证据重算而非相信一个就绪标志。Linux 进程启动 tick、系统 boot ID 和 PID namespace 防止新容器复用 PID 后误用旧状态。paper 永远不报告执行资格。
- Compose 只修改 paper 健康检查；现有 live 配置未启用、未修改。

只读检查示例：

```bash
python -m scripts.check_runtime_health \
  --state-dir /persistent/paper-state --scope PAPER_SCOPE --require ready
```

检查失败退出 1；不会连接交易所、启动进程或提交订单。

## 发布证据

`shared/release_acceptance.py` 与 `scripts/check_release_acceptance.py` 校验：

1. 精确的 40 位提交 SHA、master、push 触发的 `.github/workflows/ci.yml`，以及该提交最新匹配 CI 的成功结果。CLI 从 GitHub 实际读取证据。
2. Docker 本地不可变 image ID 与镜像中的 OCI revision 标签。这里记录的是本地镜像 ID，不冒称 registry digest。
3. 当前 paper 命令的全部 runner、实际只读挂载的模型目录、模型引用文件与行情/OI/LS CSV。所有必需文件必须存在且非空，生成确定的 SHA256 清单；禁止源代码覆盖挂载和输入遮蔽。
4. 持久状态 scope/state_id/schema 与启动后新的 paper 进程状态。启动后再次计算输入哈希，核对全部 runner、精确镜像/提交/输入摘要及新鲜健康证据。

`preflight` 只表示发布前证据通过，`paper_accepted` 仍为 false。只有启动后检查通过才产生 paper 验收收据。`execution_eligible`、`live_acceptance` 始终为 false；`--require-execution` 明确失败。

部署工作流仍仅手动触发、仅支持 `quant-paper`。`scripts/deploy.sh --check quant-paper` 只做本地存储前置检查；实际部署入口必须有明确提交及仓库，先完成发布预检，再以已检查的 image ID 启动，等待健康并产生收据。只有开始变更服务后失败才尝试回滚，预检失败不会重建旧服务。回滚必须有上次通过的收据、备份镜像及相同输入字节；缺失证据或输入变化会明确失败。

本轮没有调用实际部署入口或手动触发 Deploy 工作流。CI push 只触发测试。

## 尚未通过的上线条件

- 现有 paper 命令含 `SUIUSDT`、`AXSUSDT`，不在当前 `SYMBOL_CONFIG` 中，发布预检会阻断。本轮不擅自替换交易标的或真实配置。
- 公共 OI 接口此前在本环境返回 HTTP 451；真实数据、模型质量、完整在线 paper、账户基线和交易所结算均未验收。
- 保留原有数据门控的 bar 时效（interval + 5 分钟）。只收闭合 bar 时可能在 bar 间变为未就绪；持续在线健康及部署等待时间必须在目标运行条件下验收。不会为使发布变绿而放宽门控。
- 输入哈希证明验收时点的字节，不证明模型质量，也不能证明持续数据更新或模型热重载后的输入。当前工具不归档旧模型/数据，输入变化可能阻断自动回滚。
- 本地文件锁、SQLite、Linux `/proc` 与本地 Docker 是当前工具边界；跨主机容灾、真实状态迁移和实盘启用仍未验收。

具体回归数量、故障注入与提交记录见 [engineering-sequence.md](engineering-sequence.md)。
