# 构建与离线启动基线

本轮基于 master `e16f302f730273281255a8bcdd4c3af6c68b2860` 和上一轮未提交的 CI 安全补丁。
本报告记录本地验收阶段；当时未创建提交、分支或推送，未执行 Compose up、systemd 部署或真实账户操作。
仅启动了本地隔离的构建/验收容器，未修改策略数学、仓位规则和真实交易配置。

## 已修复

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 修正 Rust 源码输入；完整安装 native wheel；移除构建失败吞掉逻辑；固定基础镜像 digest、Rust 和 maturin；构建时执行真实离线启动检查 |
| `.dockerignore` | 保留必要入口、测试及工作流；排除外部脚本链接、本地扩展、订单数据库及运行状态 |
| `requirements/container.in`、`container.lock` | 为 Python 3.12 容器解析完整依赖版本和哈希；原宿主机 requirements.lock.txt 不作为运行锁使用 |
| `pyproject.toml` | 明确 Python 源码 wheel 文件选择，避免默认包发现失败及夹带 Rust/运行状态资源；固定 Hatchling |
| `scripts/run_bybit_alpha.py` | 将失效绝对链接替换为转发到 runner.alpha_main 的普通兼容入口 |
| `runner/alpha_cli.py`、`alpha_main.py`、`startup_check.py` | 增加 --check-startup，使用真实原生扩展和合成行情验证状态推进；缺失在线输入时在账户连接前失败 |
| `.github/workflows/ci.yml` | 加入镜像构建和禁网验收；原单测 job 复用运行依赖锁；移除复制 .so 的容错；检查真实 native 导出 |
| `tests/unit/runner/test_startup_check.py` | 9 项启动、网络隔离、入口兼容与失败路径测试 |
| `tests/offline/test_package_layout.py` | 4 项打包检查，包含实际构建 wheel 后验证内容 |

Docker 中 native wheel 用 Cargo 元数据构建为 quant_hotpath，Python 源码 wheel 用
Hatch 构建为 quant-system，两者可同时安装。Docker 运行镜像使用源码加 native wheel。
未修改其它历史脚本链接、候选 framework 入口或主机 systemd 文件。

## 修复前的实际失败

- 原 Docker paper 构建退出 1：COPY 的 `/regime/rust`、`/_quant_hotpath` 不存在。
- `python -m scripts.run_bybit_alpha --help` 退出 1：旧入口链接目标不在仓库内。
- 临时复制原 pyproject 和运行包后，`uv build --wheel` 退出 2：Hatch 默认无法确定分发文件。
- 首次新启动检查发现原生状态 close 是定点原始整数；检查改用现有 close_f 访问器，未改变计算逻辑。

## 本地验收结果

验收平台：Linux x86_64、Python 3.12、Rust 1.93.0。

| 验证 | 结果 |
| --- | --- |
| native release wheel 构建，Cargo --locked | 通过，实际编译，无本地扩展回退 |
| Docker ci 构建，含 paper 阶段启动检查 | 退出 0 |
| Python 源码 wheel 构建与内容检查 | 通过，包含 startup_check，不含状态、脚本链接或 Rust 文件 |
| 禁网容器启动检查 | 退出 0，event_index=1、close=101.0 |
| 启动 pytest | 9 项通过 |
| scripts/check_local_safety.py | 66 项通过，包含实际 wheel 构建测试 |
| runner.offline demo | 退出 0，合成数据，无外部模型、账户或网络 |
| pip check | 退出 0，无依赖冲突 |
| 源码 wheel 离线安装后从 /tmp 启动 | 通过，runner 从 site-packages 导入 |
| 屏蔽 native 模块 | 按预期退出 1，不报告成功 |
| 普通在线入口缺少 data.oi_cache | 按预期退出 1，账户连接前明确阻断 |
| 临时删除构建上下文 alpha/rust | 按预期构建退出 1 |
| 临时向上下文放入 .env、运行 JSON、DB/WAL/SHM | 实际 Docker 上下文导出均未包含它们，入口和工作流仍在 |
| 修改文件 Ruff E/W/F、工作流 YAML/run 语法、git diff --check | 通过 |

全部容器运行验收使用 `--network none`，不挂载账户环境或外部模型。
现有 offline 测试通过 unittest discover 运行；直接混入 pytest importlib 收集会因
原有跨文件导入失败，CI 保留各自正确入口，不把它算作全量 pytest 通过。

环境使用 VFS Docker 驱动，曾遇到构建层和并行容器副本占满磁盘。
已合并大构建层、清理本轮生成的缓存及确认无元数据引用的失败容器副本，并改为串行验收。
这些环境故障不计为通过；表内结果来自随后成功执行的验证。

## 可复用命令

```bash
docker build --target ci -t quant-ci:local .
docker run --rm --network none quant-ci:local python -m runner.alpha_main --check-startup
docker run --rm --network none quant-ci:local python -m pytest tests/unit/runner/test_startup_check.py -q
docker run --rm --network none quant-ci:local python scripts/check_local_safety.py
docker run --rm --network none quant-ci:local python -m runner.offline demo --out /tmp/offline-demo
uv build --wheel
```

代理使用私有 CA 时，构建可额外传 `--secret id=proxy_ca,src=<可信 CA bundle>`；
证书仅在安装/下载步骤临时挂载，不写入镜像，也不关闭 TLS 验证。
CI 新 job 只构建和本地载入镜像，push=false，不触发部署工作流。

## 明确未通过或未覆盖

**在线 Alpha 启动仍未通过。** data.oi_cache 实现不在仓库及已知历史内，
本轮没有编造 OI/LS 数据或用空实现替代；模型加载、实时数据完整性、真实执行与业务健康
均不属于离线检查覆盖范围。--dry-run 仍不应当作完全离线模式。

未执行远端 GitHub Actions、完整仓库 pytest、完整 framework 路径、跨架构构建或真实部署。
Docker ci 默认的完整 pytest 命令保留；新增 CI 验收明确选择启动和离线安全测试。
基础镜像与 Python/Rust 依赖已约束，Debian apt 仓库尚未快照化，不宣称逐字节可复现。
源码 wheel 不携带模型和配置资源，完整部署需另行定义这些外部资产契约。
Makefile 的历史 AVX512 构建选项及其它运行路径仍待单独处理。

详细命令输出保存在本工作区 `/workspace/scratch/build-baseline-evidence/`。
上述 SHA 是基线；后续发布的修复提交以 Git 历史为准。
