# Runtime Direction

> 更新时间: 2026-03-19
> 状态: 方向文档，不是当前 active deploy 真相源
> 作用: 约束 framework 收敛方向，避免把“未来默认路径”与“当前主机上正在交易的服务”写成一回事

## 1. 当前事实

当前 active host services 仍然是：

- [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) → `bybit-alpha.service`
- [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) → `bybit-mm.service`

当前 framework path 仍然是：

- [`runner/live_runner.py`](/quant_system/runner/live_runner.py)

这三者当前并没有统一成单一运行时。

## 2. 本文约束的对象

本文只约束 framework 收敛方向：

- framework 默认编排入口继续固定为 [`runner/live_runner.py`](/quant_system/runner/live_runner.py)
- framework 配置示例继续以 [`infra/config/examples/`](/quant_system/infra/config/examples) 为准
- framework 候选部署路径继续是 `quant-framework` / `infra/systemd/quant-runner.service`

它不声称：

- `runner/live_runner.py` 已经是当前主机上的默认交易服务
- compose / GitHub Actions 已经等于当前 host trading deployment

## 3. 本阶段不做的事

- 不把 `rust/src/bin/main.rs` 提升为默认生产入口
- 不把 `bybit-alpha.service` 文档误写成已经切到 `LiveRunner`
- 不把 `bybit-mm.service` 包装成 framework 子路径

## 4. 对后续开发的约束

- 新增 framework deploy / runbook / smoke test 时，默认答案先对齐 `runner/live_runner.py`
- 新增 active host service 文档时，必须明确写清它是否属于 framework path
- 如果未来真的让 `LiveRunner` 取代 `run_bybit_alpha` 成为默认 host trading service，必须先更新 [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md) 与 [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md)

## 5. 退出条件

只有以下条件都满足后，才重新评估“默认 runtime 已收口”：

1. host 上的默认交易服务切到 framework path
2. deploy workflow 也切到同一路径
3. framework path 覆盖当前 directional alpha 与 market maker 所需的核心运维能力
