# Deployment Truth

> 更新时间: 2026-03-20
> 目标: 写清当前 host systemd、compose、CI/CD 之间的真实关系，而不是把它们包装成已经统一
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前部署真相

当前仓库同时存在两种部署面：

### 1.1 当前活跃 host deployment

| 服务 | 模板 | 实际入口 | 当前定位 |
|---|---|---|---|
| `bybit-alpha.service` | [`infra/systemd/bybit-alpha.service`](/quant_system/infra/systemd/bybit-alpha.service) | `python3 -m scripts.run_bybit_alpha --symbols ... --ws` | 当前活跃的方向性 alpha 交易服务 |
| `bybit-mm.service` | [`infra/systemd/bybit-mm.service`](/quant_system/infra/systemd/bybit-mm.service) | `python3 -m scripts.run_bybit_mm --symbol ETHUSDT ...` | 当前活跃的专用做市服务 |

### 1.2 仓库内的 compose / CI deployment

| Compose 服务 | 命令 | 当前定位 |
|---|---|---|
| `quant-paper` | `python3 -m scripts.run_bybit_alpha ... --dry-run` | containerized paper / smoke path |
| `quant-live` | `python3 -m scripts.run_bybit_alpha ...` | containerized live path |
| `quant-framework` | `python3 -m runner.live_runner --config /app/infra/config/examples/live.yaml` | framework candidate path |

结论：

- 当前实际在主机上跑交易的是 systemd 服务
- GitHub Actions deploy workflow 当前管理的是 compose 路径
- 这两者还没有统一成同一条发布链

---

## 2. 当前默认与非默认

| 类型 | 当前默认 | 非默认 / 候选 |
|---|---|---|
| Host 上的交易服务 | `bybit-alpha.service` / `bybit-mm.service` | `quant-runner.service` |
| 容器部署 | `quant-paper`（CI smoke / deploy 脚本默认） | `quant-live`、`quant-framework` |
| Framework service | `quant-framework` / `infra/systemd/quant-runner.service` | 仍属候选，不是主机上的默认交易入口 |

说明：

- `quant-paper` 是 deploy workflow 的默认 compose 服务，不等于“当前主机上的默认生产交易服务”
- `quant-runner.service` 是 framework systemd 模板，不等于当前活跃 directional alpha

---

## 3. 工件清单

### 3.1 当前活跃

- [`infra/systemd/bybit-alpha.service`](/quant_system/infra/systemd/bybit-alpha.service)
- [`infra/systemd/bybit-mm.service`](/quant_system/infra/systemd/bybit-mm.service)
- [`docker-compose.yml`](/quant_system/docker-compose.yml)
- [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)
- [`.github/workflows/ci.yml`](/quant_system/.github/workflows/ci.yml)
- [`.github/workflows/deploy.yml`](/quant_system/.github/workflows/deploy.yml)

### 3.2 候选 / 非默认

- [`infra/systemd/quant-runner.service`](/quant_system/infra/systemd/quant-runner.service)
- `quant-framework` compose profile
- [`deploy/README.md`](/quant_system/deploy/README.md) 中列出的 host tuning / candidate artifacts

---

## 4. 当前已确认的漂移

当前必须正视的事实：

1. GitHub Actions deploy workflow 当前只会执行 [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)，默认只重建 / 重启 `quant-paper`
2. 它不会同步或重启 `bybit-alpha.service`
3. 它也不会同步或重启 `bybit-mm.service`
4. `quant-framework` 也不在 deploy workflow 的默认目标里
5. 如果 diff 命中了 host-managed trading artifacts，workflow 现在会 fail fast，而不是继续做 compose deploy 并制造“已部署生产交易服务”的错觉

因此：

- “CI/CD 已覆盖当前 host trading services” 不是事实
- “compose 路径就是当前生产部署真相” 也不是事实

---

## 5. 同步规则

### 5.1 systemd

- `infra/systemd/*.service` 是仓库真相源
- 修改后必须显式同步到 `/etc/systemd/system/`

```bash
sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/
sudo cp infra/systemd/bybit-mm.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 5.2 compose

- `docker-compose.yml` 是容器路径真相源
- `quant-paper` / `quant-live` 运行 `scripts.run_bybit_alpha`
- `quant-framework` 运行 `runner.live_runner`

### 5.3 凭据

- compose 通过 `.env` 注入 `BYBIT_*`
- `infra/systemd/bybit-alpha.service` 与 `infra/systemd/bybit-mm.service` 现在都通过 `EnvironmentFile=/quant_system/.env` 注入 `BYBIT_*`
- host 上同步 systemd unit 后，方向性 alpha 与做市都默认从同一个 `.env` 读取 `BYBIT_API_KEY` / `BYBIT_API_SECRET` / `BYBIT_BASE_URL`
- [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh) 现在只接受 compose 服务名：`quant-paper` / `quant-live` / `quant-framework`

---

## 6. 验证命令

```bash
# systemd 服务
sudo systemctl status bybit-alpha.service --no-pager -l
sudo systemctl status bybit-mm.service --no-pager -l

# compose 配置
docker compose config >/dev/null

# deploy 脚本语法
bash -n scripts/deploy.sh

# workflow 目标核对
sed -n '1,220p' .github/workflows/deploy.yml
sed -n '1,220p' docker-compose.yml
```

---

## 7. 常用操作

### 7.1 重启当前活跃服务

```bash
sudo systemctl restart bybit-alpha.service
sudo systemctl restart bybit-mm.service
```

### 7.2 启动 compose 服务

```bash
docker compose up -d quant-paper
docker compose --profile live up -d quant-live
docker compose --profile framework up -d quant-framework
```

### 7.3 查看日志

```bash
tail -f /quant_system/logs/bybit_alpha.log
tail -f /quant_system/logs/bybit_mm.log
journalctl -u bybit-alpha.service -f
journalctl -u bybit-mm.service -f
docker compose logs -f quant-paper
```

---

## 8. 当前限制

- deploy workflow 与当前 host systemd services 仍然分叉
- `quant-paper` 代表的是 container deploy 默认目标，不代表 host 交易默认目标
- framework runtime、directional alpha、market maker 仍不是同一套部署制度
