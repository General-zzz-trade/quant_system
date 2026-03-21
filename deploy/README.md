# Deploy Status

这个目录只保存候选或辅助部署工件，不是当前仓库的默认部署真相源。

## 1. 当前默认真相源

优先以这些文件为准：

- [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md)
- [`docker-compose.yml`](/quant_system/docker-compose.yml)
- [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)
- [`infra/systemd/bybit-alpha.service`](/quant_system/infra/systemd/bybit-alpha.service)
- [`infra/systemd/bybit-mm.service`](/quant_system/infra/systemd/bybit-mm.service)

## 2. 本目录当前存在什么

| 工件 | 当前定位 |
|---|---|
| [`deploy/systemd/quant-trader.service`](/quant_system/deploy/systemd/quant-trader.service) | framework / candidate systemd 示例，不是当前活跃 host service |
| [`deploy/systemd/logrotate-quant.conf`](/quant_system/deploy/systemd/logrotate-quant.conf) | 候选辅助文件 |
| [`deploy/trading-tune.service`](/quant_system/deploy/trading-tune.service) | host tuning 辅助文件 |
| [`deploy/tune-os.sh`](/quant_system/deploy/tune-os.sh) | host tuning 辅助脚本 |

## 3. 当前不存在的东西

本目录当前没有以下已成型默认工件：

- `deploy/docker/docker-compose.yml`
- `deploy/k8s/`
- `deploy/argocd/`

如果其他文档还把这些路径写成“当前候选清单”，那是旧说法。

## 4. 规则

如果本目录里的工件与：

- host 上实际 systemd 服务
- repo-root compose
- GitHub Actions workflow

发生冲突，以 [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md) 为准。
