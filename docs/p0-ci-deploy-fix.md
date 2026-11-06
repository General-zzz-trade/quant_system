# CI 安全检查与部署入口修复验证

基线：`e16f302f730273281255a8bcdd4c3af6c68b2860`，分支 `master`。
本报告记录本地验证阶段；当时没有提交、推送、触发 GitHub Actions、运行真实部署或访问交易账户。
未修改交易策略、仓位管理、Dockerfile、Compose 或真实交易配置。

## 修改前复现

在临时目录创建仅用于测试的 BYBIT_API_KEY 和 BYBIT_API_SECRET 非空字符串赋值，
用 `bash -eo pipefail` 执行旧 security-scan 两条命令：退出码 **0**，说明扫描不阻断。

部署路径检查：

```text
scripts/deploy.sh exists=False symlink=False
scripts/check_deploy_scope.py exists=False symlink=True
python -B -m scripts.check_deploy_scope --guard-changed-files-stdin
exit=1: No module named scripts.check_deploy_scope
```

旧部署工作流仅在 ext/、Dockerfile、Makefile、pyproject.toml 变化时重建，
未涵盖实际 Python/Rust 源码；git diff 和回滚重建命令会吞掉失败。

## 文件与原因

| 文件 | 修改 |
| --- | --- |
| `.github/workflows/ci.yml` | 用直接传播退出码的 Python 检查替换 grep；为新增检查器接入 lint |
| `scripts/check_ci_security.py` | 扫描 Git 跟踪的 Python 凭据字面量赋值及私钥扩展名；仅输出位置；扫描/枚举错误也失败 |
| `scripts/check_deploy_scope.py` | 用普通跟踪文件替换失效绝对链接；阻止 Compose 工作流处理主机服务变更 |
| `scripts/deploy.sh` | 仅支持既有 quant-paper；先校验入口与 Compose，再执行不构建镜像的 recreate，并等待健康结果 |
| `.github/workflows/deploy.yml` | 部署前预检；每次只重建 quant-paper；diff 失败即停止；只使用本次备份的镜像回滚；回滚失败传播 |
| `tests/offline/test_ci_deploy_safety.py` | 12 项标准库回归测试，由已有 local-safety job 自动发现 |

成功部署 SHA 改为写入 runner 用户的
`$HOME/.local/state/quant-system/quant-paper-last-good-deploy`，使用临时文件和 rename，
避免 checkout 清理工作区时删除记录。该记录仅属于当前 self-hosted runner，不是集中发布账本。

## 验证结果

```text
python -B scripts/check_local_safety.py
Ran 62 tests
OK

ruff check --no-cache --select E,W,F \
  scripts/check_ci_security.py scripts/check_deploy_scope.py \
  tests/offline/test_ci_deploy_safety.py
All checks passed!

bash -n scripts/deploy.sh
exit=0

git diff --check
exit=0
```

测试全部在本地执行。部署测试通过临时 PATH 使用假 Docker 程序，
没有调用 Docker daemon；网络连接由已有离线测试入口阻断。
新增测试执行实际工作流中的安全扫描、revision 检查、scope 和回滚 run 块。

| 场景 | 结果 |
| --- | --- |
| 环境读取、空凭据赋值 | 扫描退出 0 |
| 假 key/secret/passphrase，包括带 example/os.environ/.env 注释的赋值 | 扫描退出 1，不输出值 |
| 跟踪的假 .pem 文件 | 扫描退出 1 |
| Python 源码损坏、Git 元数据不可用 | 检查退出 1 |
| 主机 systemd 变更、非法相对路径 | scope 退出 1 |
| 旧 revision 无效、git diff 失败 | 对应工作流 run 块退出 17（假 git 注入） |
| 有效 paper 预检/重建 | 假 Docker 路径退出 0；预检没有 up 调用 |
| 缺失入口、外部符号链接入口 | 非零退出，Docker 调用次数 0 |
| quant-live、额外服务、缺省服务参数 | 退出 2，Docker 调用次数 0 |
| Compose config 失败 | 退出 17，不执行 up |
| up/健康等待失败 | 退出 23 |
| 回滚成功 | 退出 0 |
| 回滚 up 失败、tag 失败、缺少本次备份 | 分别退出 23、19、1 |

在临时副本中故意引入回归，未修改交付补丁：

```text
security failure swallowed: regression detected=True failures=5 errors=0
rollback failure swallowed: regression detected=True failures=1 errors=0
```

两份工作流通过 YAML 解析；CI 的 10 个、Deploy 的 9 个 run 块分别通过
Bash 语法检查或 Python 编译检查。Compose 的 quant-paper 与 paper build target
引用已核实。保留手动部署确认、master 限制、同 SHA 成功 push CI 检查。
没有使用 actionlint；没有运行远端 Actions，不能将本地语法验证称为远端 CI 成功。

## 未通过的验证和剩余风险

**完整部署链未通过。** 对真实仓库运行只读预检得到：

```text
bash scripts/deploy.sh --check quant-paper
Deployment blocked: required runtime file unavailable: scripts/run_bybit_alpha.py
exit=1
```

该入口仍是上一轮确认的失效链接；Dockerfile 缺失输入、.dockerignore 排除入口、
导入式健康检查和运行依赖缺失均不在本次修复范围。没有构建或启动镜像，
也没有验证真实 Docker Compose 版本及业务健康。

安全检查是源码赋值检查，不能代替完整秘密检测：不检查历史、编码或动态拼接秘密、
所有配置格式；跳过 Python 符号链接目标，不读取本地 .env。后续可独立引入专用扫描器。

回滚仍基于本机可变标签，只回滚 paper 镜像，不处理数据库、订单状态或配置回退。
首次没有部署记录时，scope 比较基线退回 HEAD~1；多 runner 或首次引入时不能据此
证明历史变更全部已部署。部署状态记录写失败会使 job 失败，但不自动回滚已健康启动的服务。
这些限制需要后续单独任务处理，不能据本次测试宣布生产可部署。

上述 SHA 是审计与修改基线；后续发布的修复提交以 Git 历史为准。
