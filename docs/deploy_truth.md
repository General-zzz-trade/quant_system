# Deployment Truth

> 更新时间: 2026-03-16
> 目标: 集中写清当前默认发布真相源，消除 deploy/compose/systemd/CI 之间的认知漂移
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前默认生产部署

| 项目 | 值 |
|------|-----|
| **入口脚本** | `scripts/ops/run_bybit_alpha.py` |
| **systemd 服务** | `bybit-alpha.service` |
| **命令** | `python3 -m scripts.run_bybit_alpha --symbols ETHUSDT ETHUSDT_15m SUIUSDT AXSUSDT --ws` |
| **Compose 服务名** | `alpha-runner` |
| **交易所** | Bybit Demo (api-demo.bybit.com) |
| **品种** | ETH (1h+15m AGREE), SUI (1h), AXS (1h) |
| **模型** | Ridge 60% + LightGBM 40% |
| **杠杆** | 1.5x (Kelly optimal 1.4x) |

---

## 2. 部署方式优先级

| 优先级 | 方式 | 适用场景 |
|--------|------|---------|
| **1** | systemd `bybit-alpha.service` | 当前默认生产（直接在服务器上运行） |
| 2 | `docker compose up alpha-runner` | Docker 部署（等效于 systemd） |
| 3 | `scripts/deploy.sh` | CI/CD 滚动部署（默认只部署 alpha-runner） |

---

## 3. 部署工件清单

### 默认路径（当前活跃）

| 工件 | 位置 | 说明 |
|------|------|------|
| systemd 服务 | `/etc/systemd/system/bybit-alpha.service` | 当前运行中 |
| systemd 模板 | `infra/systemd/bybit-alpha.service` | 仓库版本（必须与 /etc/ 同步） |
| Compose 服务 | `docker-compose.yml` → `alpha-runner` | Docker 部署等效 |
| 部署脚本 | `scripts/deploy.sh` | 滚动部署（默认 alpha-runner） |
| CI | `.github/workflows/ci.yml` | 构建 + 测试 |
| CD | `.github/workflows/deploy.yml` | CI 成功后自动部署 |

### 候选/实验性（非默认，不自动部署）

| 工件 | 位置 | 说明 |
|------|------|------|
| Rust standalone | `docker-compose.yml` → `trader-rust` | 演进方向，非当前默认 |
| Paper trading | `docker-compose.yml` → `paper-multi` | 已停用 |
| Kubernetes | `deploy/k8s/` | 未使用 |
| ArgoCD | `deploy/argocd/` | 占位，未配置 |
| 替代 Docker | `deploy/docker/` | 未使用 |
| systemd 模板 | `deploy/systemd/` | 候选模板 |

---

## 4. 同步规则

- `infra/systemd/bybit-alpha.service` 是仓库真相源
- 修改后必须同步到 `/etc/systemd/system/`：`sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/ && sudo systemctl daemon-reload`
- `docker-compose.yml` 的 `alpha-runner` 命令必须与 systemd `ExecStart` 一致
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
docker compose up -d alpha-runner
```

### 滚动部署（CI/CD）
```bash
./scripts/deploy.sh              # 默认: alpha-runner only
./scripts/deploy.sh paper-multi  # 指定其他服务
```

### 查看日志
```bash
tail -f /quant_system/logs/bybit_alpha.log       # 实时日志
journalctl -u bybit-alpha.service -f             # systemd 日志
docker compose logs -f alpha-runner               # Docker 日志
```
