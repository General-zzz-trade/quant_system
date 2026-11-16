# ADR-0017：配置在启动时读一次，主机安装可重复

- 状态：**已批准**（2026-11-15，技术负责人"现在全部批准"）。原为提议稿；下面的内容是批准时的版本，实施中如有变动在本文更正并记入 `execution-log.md`。
- 日期：2026-11-15

## 背景

【代码】

- 配置读取函数（`exchange_settings`、`risk_limits`、`testnet_from_env`、`execution_scope`、`notifier_from_env`、`binance_config_from_env` 等）在启动阶段和命令行工具里共被调用 18 次【实际运行结果，按调用处统计：入口 9、手工工具 3、状态目录 3、组装 2、连接 1】；运行循环里没有（审计 L5）。
  - 启动时同一项配置会被读好几次，每次读到的都是当时的环境，没有一个统一的值。
  - 没有任何地方记录"这次运行实际用了什么配置"。
- `target.md` 2.1 原定 `settings.py` 只读一次，生成不可变对象（"环境变量 → 带校验的不可变配置对象"），实施时没有做到。
- 主机安装没有脚本。检出目录写死为 `/quant_system`（`CLAUDE.md` Gotchas），服务单元的用户写死为 `ubuntu`（`infra/systemd/*.service`）。

## 决定

1. **`RuntimeConfig`**（`trading/infrastructure/config.py`）：
   - 一个冻结的数据类，启动时由 `load_config(env)` 一次读出并校验；
   - 包含账户身份、网络、风险限额、交易所杠杆与保证金模式、止损距离、告警与心跳配置、状态目录；
   - 密钥和心跳地址只以"已设置/未设置"表示，值放在另一个只给适配器构造用的对象里。
   - 账户身份（ADR-0016 第 5 条）在这里派生一次。
2. **记录**：启动时把 `RuntimeConfig` 写进事件日志和进程日志，不含密钥。状态文件里放它的摘要哈希。
3. **之后只传对象**：组装根、手工工具、状态目录命令都从 `RuntimeConfig` 取值，不再读环境变量。
4. **主机安装**：`infra/install_host.sh`，幂等，不读、不写密钥。步骤：
   - 检查 Python 版本；
   - 按 `requirements/container.lock` 带哈希安装依赖；
   - 建日志目录和锁目录（ADR-0016 第 5 条）；
   - 运行 `infra/sync_systemd.sh` 和日志轮转部署；
   - 运行 `scripts/check_local_safety.py`。
   - 另外，**重写**已经存在但过时的 `.env.example`：它还列着 Bybit、OKX、Hyperliquid，缺 `RISK_SCOPE`、`BINANCE_TESTNET`（审计 L5）。新版本列出全部变量名和默认值，不含任何值。
   - 安装脚本从不启动服务。

## 状态未知与失败方向

- 配置缺失或非法：拒绝启动，并说明是哪一项。
- 安装脚本任何一步失败即退出，不留下半装好的服务。

## 备选方案

- **维持现状，只加一条"启动时记录配置"**：记录的是第一次读到的值，与后续实际使用的值可能不同。
- **用配置文件替代环境变量**：改动面更大，而且凭据仍然要单独存放。

## 后果

- 所有读配置的地方改为接收参数，测试的设置方式随之改变（只改设置、不改断言，逐个记录）。
- 风险参数的读取路径改变，所以按资金路径完整流程实施。

## 验收

- 架构测试：`trading/` 和 `runner/` 里只有 `trading/infrastructure/config.py` 读环境变量（加强现有的"只有组装层读环境变量"规则）。
- 新测试：
  - 配置日志不含任何密钥值，也不含心跳地址；
  - 同一组环境变量得到同一个摘要哈希；
  - 启动之后修改环境变量不影响运行。
- 安装脚本在一个干净容器里运行一次，并记录结果（不进 CI 部署流程）。

## 实施更正（2026-11-15，S4）

- **`trading/infrastructure/config.py`**：
  - 原来分散在 `settings.py` 里的环境变量解析函数原样搬到这里；`settings.py` 只做转发，原有导入照常可用。
  - 新增：
    - `RuntimeConfig`：冻结，含网络、执行与否、风险范围、止损距离、风险限额、交易所杠杆和保证金模式、告警方式、账户身份；
    - `Secrets`：Binance 凭据和 Telegram 令牌，打印时只显示"已隐藏"；
    - `load_config`、`with_credentials`、`account_identity`、`make_notifier`。
- **凭据在所有本地检查之后才读**（`with_credentials`），顺序和以前一样：本地检查失败时报的仍是本地问题，不会先报"没有凭据"。
  - 账户身份 = `sha256(网络:API Key)`，在读凭据时派生。
- **模拟盘不读任何账户配置**：没有风险限额，也没有杠杆和保证金设置，与 S10 的决定一致。
- **记录**：
  - 启动时写一条 `config` 事件（新的事件种类）和一行进程日志，不含任何密钥；
  - 状态文件里有 `config_digest`。
  - 心跳地址留到 S7 再加入配置。
- **之后只传对象**：
  - 入口、启动预检、组装、账户检查（`refuse_foreign_holdings` 收到杠杆和保证金模式）、适配器构造（`build_adapters` 收到凭据），都从配置对象取值。
  - 手工工具和状态目录命令不再自己读 `os.environ`，改为调用 `config.py` 里的函数。
- **"入口变薄"推迟到 S10 第 3 部分**：那一步要大改入口里模拟盘的组装，现在搬移入口会和副线冲突。
- **安装脚本和重写 `.env.example`** 按设计第 10 节放在 S8。

## 实施更正（2026-11-16，S10 第 3b 部分）

- **入口变薄已完成**：启动检查（`preflight`）、两种模式的组装（`TargetRuntime`）和主循环（`main`）搬到 `trading/infrastructure/cli/target.py`。
- `runner/binance_target.py` 只留 `trading` 按导入契约不能导入的两样：签名的规格加载器（`infra.model_signing`）和交易成本日志（`monitoring.tca`），并把它们交给 `cli/target.py`；原有的名字（`TargetRuntime`、`load_spec`、`preflight`、`main` 等）照旧可以导入，systemd 单元不用改。
- 模拟盘只用 `load_config(execution=False)` 的结果，不读任何账户配置；规格加载器读 `BINANCE_TESTNET` 只会让检查更严（S10 决定 3）。
