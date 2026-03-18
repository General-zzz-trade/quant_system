# Deployment Truth

> 更新时间: 2026-03-18
> 目标: 集中写清当前默认发布真相源，消除 deploy/compose/systemd/CI 之间的认知漂移
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前默认生产部署

| 项目 | 值 |
|------|-----|
| **入口脚本** | `scripts/ops/run_bybit_alpha.py` |
| **systemd 服务** | `bybit-alpha.service` |
| **命令** | `python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws` |
| **Compose 服务名** | `quant-paper` (paper) / `quant-live` (live, 需 `--profile live`) |
| **交易所** | Bybit Demo (api-demo.bybit.com) |
| **品种** | BTC (h=96), ETH (1h+15m AGREE), SUI (1h), AXS (1h) |
| **模型** | 以各模型目录 `config.json` 为准；当前实盘主线支持 `ridge_primary` / `ic_weighted` 等配置驱动装配 |
| **杠杆** | 动态：$0-20K 默认 1.5x，上 20K 后降到 1.0x；再叠加 z-score conviction scale |

---

## 2. 部署方式优先级

| 优先级 | 方式 | 适用场景 |
|--------|------|---------|
| **1** | systemd `bybit-alpha.service` | 当前默认生产（直接在服务器上运行） |
| 2 | `docker compose up quant-paper` / `docker compose --profile live up quant-live` | Docker 部署 |
| 3 | `scripts/deploy.sh` | CI/CD 滚动部署 |

---

## 3. 部署工件清单

### 默认路径（当前活跃）

| 工件 | 位置 | 说明 |
|------|------|------|
| systemd 服务 | `/etc/systemd/system/bybit-alpha.service` | 当前运行中 |
| systemd 模板 | `infra/systemd/bybit-alpha.service` | 仓库版本（必须与 /etc/ 同步） |
| Compose 服务 | `docker-compose.yml` → `quant-paper` / `quant-live` | Docker 部署 (multi-stage) |
| 部署脚本 | `scripts/deploy.sh` | 滚动部署 |
| CI | `.github/workflows/ci.yml` | 构建 + 测试 |
| CD | `.github/workflows/deploy.yml` | CI 成功后自动部署 |

### 候选/实验性（非默认，不自动部署）

| 工件 | 位置 | 说明 |
|------|------|------|
| LiveRunner 框架 systemd | `deploy/systemd/quant-trader.service` | 使用 `runner.live_runner`，非活跃生产 |
| LiveRunner 框架 compose | `docker-compose.yml` → `quant-framework` | 需 `--profile framework` |
| Rust standalone | 候选，无 compose 配置 | 演进方向，非当前默认 |
| Kubernetes | `deploy/k8s/` | 未使用 |
| ArgoCD | `deploy/argocd/` | 占位，未配置 |
| 替代 Docker | `deploy/docker/` | 未使用 |

---

## 4. 同步规则

- `infra/systemd/bybit-alpha.service` 是仓库真相源
- 修改后必须同步到 `/etc/systemd/system/`：`sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/ && sudo systemctl daemon-reload`
- `docker-compose.yml` 的 `quant-paper`/`quant-live` 命令必须与 systemd `ExecStart` 一致
- `docker-compose.yml` 中默认发布服务 `quant-paper` / `quant-live` / `quant-framework` 都必须带显式 `image:` 与 healthcheck
- `CLAUDE.md` 的部署命令必须与上述一致

---

## 5. 验证命令

```bash
# 验证 systemd 服务运行
sudo systemctl status bybit-alpha.service

# 验证 compose 配置有效
python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('OK')"

# 验证 deploy.sh 语法
bash -n scripts/deploy.sh && echo "OK"

# 验证 systemd 模板与实际一致
diff infra/systemd/bybit-alpha.service /etc/systemd/system/bybit-alpha.service

# 验证 compose 命令与 systemd 一致
grep "ExecStart" /etc/systemd/system/bybit-alpha.service
grep "command:" docker-compose.yml | grep run_bybit_alpha
```

---

## 6. 部署操作手册

### 重启服务
```bash
sudo systemctl restart bybit-alpha.service
```

### Docker 部署（替代 systemd）
```bash
docker compose up -d quant-paper              # Paper trading
docker compose --profile live up -d quant-live  # Live trading
```

### 滚动部署（CI/CD）
```bash
./scripts/deploy.sh              # 默认: quant-paper
./scripts/deploy.sh quant-live   # 指定 live 服务
```

### 查看日志
```bash
tail -f /quant_system/logs/bybit_alpha.log       # 实时日志
journalctl -u bybit-alpha.service -f             # systemd 日志
docker compose logs -f quant-paper               # Docker 日志
```
