# Runtime Direction

> 更新时间: 2026-03-13
> 状态: 当前阶段方向冻结
> 作用: 明确未来一个阶段内默认 runtime 的单一叙事，避免再次出现”双默认入口”
> 补充: 当前阶段新增 alpha 自动化基础设施 (auto_retrain, alpha_health, ic_weighted)，不改变 runtime 默认路径

## 决策

当前默认运行时继续固定为 Python-default 路径：

- 默认编排入口: [`runner/live_runner.py`](/quant_system/runner/live_runner.py) (legacy, maintained for compatibility) 或 [`runner/run_trading.py`](/quant_system/runner/run_trading.py) (新分解入口)
- 默认发布路径: repo-root `docker-compose.yml` + `.github/workflows/ci.yml` + `.github/workflows/deploy.yml`
- 默认运维脚本: [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)

Rust 继续承担以下角色：

- 状态推进、事件热路径、特征与部分执行/风控原语的核心内核
- standalone binary / 独立运行时候选路径
- 性能与内核能力演进的主要载体

## 本阶段不做的事

- 不把 `ext/rust/src/bin/main.rs` 提升为默认生产入口
- 不在默认 runbook、默认 compose、默认 workflow 中引入第二套等价主路径
- 不在文档里把“Rust candidate runtime”描述成“当前默认事实”

## 对后续开发的约束

- 新增 deploy、CI、runbook、smoke test 时，默认答案必须先对齐 Python-default 主路径
- 若某个候选工件仍保留在 `deploy/`，必须明确标成 `candidate-production` 或 `experimental`
- 若未来决定让 Rust binary 成为默认入口，必须先更新 [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md) 与本文档，再调整默认 workflow / compose / runbook

## 退出条件

只有在以下条件都满足后，才重新评估默认 runtime 切换：

1. 默认发布路径已经收口成单一真相源
2. 默认 CI 已覆盖 Python 主路径、`execution/tests/`、Rust tests 与默认镜像构建
3. operator / recovery / deploy / rollback 的默认流程不再依赖候选路径补洞
