# 公共行情影子运行

`runner.shadow_main` 使用公共行情和本地模型记录信号，不构造账户客户端，不查询钱包或仓位，不提交订单、不生成成交，也不发送通知。它复用 Alpha 的 builder、特征和策略计算，不修改策略数学、仓位规则或 live 配置。

运行清单只有 [`config/shadow.json`](../config/shadow.json)。当前启用 `BTCUSDT`、`ETHUSDT`、`BTCUSDT_4h`、`ETHUSDT_4h`，显式禁用 `BTCUSDT_15m`、`ETHUSDT_15m`。启用和禁用列表必须完整覆盖策略配置中的 runner；未知、重复、遗漏和重叠均拒绝。CLI 不接受交易所、账户、执行开关或 runner 列表覆盖。

清单声明的 `10000 USDT` 是合成起始余额，不是账户资金。1h 发出的订单意图记录为 `not_submitted`，随后通过已有终态回调清理 pending 状态，不产生成交。4h 保留原有 signal-only 路径及其虚拟状态行为。因此两条路径的记录不能作为统一成交模拟、钱包余额或收益验收。

## 输入与启动条件

公共客户端只允许固定的 Bybit `api.bybit.com` K 线、ticker 路径，以及 Binance `fapi.binance.com` 的 OI/LS 历史路径；仅发出 GET，拒绝重定向，没有认证、账户或下单方法。它保留系统代理和 CA 配置。启动不需要交易所凭证或 `.env`。

从项目根目录运行，使用固定的 `models_v8/`、`data_files/` 和 `data/runtime/`。模型必须具有完整配置、声明的模型文件及可加载的可信产物；Python 依赖和 Rust 扩展也必须已安装。每个启用币种至少需要这些本地 CSV：

| 文件 | 必要列 |
| --- | --- |
| `BTCUSDT_1h.csv` / `ETHUSDT_1h.csv` | `open_time,open,high,low,close,volume` |
| `*_open_interest.csv` | `timestamp,sum_open_interest` |
| `*_ls_ratio.csv` | `timestamp,long_short_ratio` |

文件结构检查不能代替内容、时间覆盖和模型加载验证。1h 仍使用原有批量预测路径；缺失或不满足因果时间要求的 CSV 会阻断预测。影子入口不会修复或更新这些 CSV，需要事先准备完整输入并由独立的数据流程维护。公共行情预热不足、时间单位错误、冲突重复或缺口也会拒绝。

以下是准备完输入后手动运行的命令，不代表本轮已执行公共网络验证。仅对新的专用影子状态目录执行一次 `init`；已有目录使用 `check`，不要复用真实账户状态：

```bash
python3 -m scripts.check_runtime_state init \
  --state-dir "$PWD/data/runtime" --scope shadow-local
python3 -m scripts.check_runtime_state check \
  --state-dir "$PWD/data/runtime" --scope shadow-local
RISK_SCOPE=shadow-local RUNTIME_STATE_DIR="$PWD/data/runtime" \
  python3 -m runner.shadow_main \
  --manifest config/shadow.json --dry-run --max-cycles 1
```

`--dry-run` 必须显式提供。`--max-cycles 1` 在完成启动和一个轮询周期后停止；省略该参数才持续轮询。启动预热包含最新已闭合 bar，之后等待下一根新 bar 才记录信号，因此一次轮询可能没有新信号。进程成功退出只说明受限运行结束，不代表数据 ready 或线上验收通过。缺失输入、非法清单和启动错误会失败；运行中的输入阻断会记录在观察日志和健康状态中。待处理 bar 过期时进程失败，需要重新预热，不跳过历史缺口继续计算。

`data/runtime/shadow-observations.jsonl` 保存清单摘要、合成余额来源、输入阻断、信号和未提交意图。`runtime-status.json` 使用 `mode=shadow`，绑定实际加载的清单摘要，始终没有执行资格。退出后状态为 stopped；持续运行时可另行执行：

```bash
RISK_SCOPE=shadow-local RUNTIME_STATE_DIR="$PWD/data/runtime" \
  python3 -m scripts.check_runtime_health --require ready
```

## Compose 与发布路径

`quant-paper` 与 Docker `paper` 镜像的默认命令一致：

```text
python3 -m runner.shadow_main --manifest config/shadow.json --dry-run
```

此服务没有 `env_file` 或交易所凭证环境变量；默认 profile 可在没有 `.env` 的情况下渲染。`quant-live` 和 `quant-framework` 保持原配置及 profile 行为。

Compose 将 checkout 之外的持久目录挂载到容器 `/app/data/runtime`。必须先为专用影子 scope 初始化该目录，再设置 `QUANT_PAPER_STATE_DIR` 和 `QUANT_PAPER_RISK_SCOPE`。模型和数据保持只读挂载。`bash scripts/deploy.sh --check quant-paper` 只检查配置、入口和持久状态，不启动服务，也不证明模型加载或公共数据可用。

实际发布仍通过现有手动 paper 流程：精确 master SHA 的 CI、镜像、完整输入哈希、持久状态和发布后的真实健康证据必须一致。验收绑定清单文件、有效 runner 配置摘要及运行时加载摘要；影子验收始终不能授予实盘资格。本轮未调用部署操作。

## 已验证范围与限制

离线回归使用合成输入和可控公共传输，覆盖清单分区、凭证与参数注入拒绝、公共接口限制、时间和行情异常、发布输入替换、健康身份与过期数据。Compose 已渲染检查，live/framework 配置保持不变。测试数量和最终 CI 结论以本次提交对应的 CI 记录为准。

截至 2026-11-12，本地 checkout 缺少完整模型产物及 `data_files/`，真实模型、公共网络可用性、持续行情完整性和 7 天运行均未验收。当前也没有完整成交模拟或实盘批准。

健康检查保留已有新鲜度标准：bar 开放时间至今不得超过该周期加 5 分钟，接收时间不得超过 120 秒。仅轮询闭合 bar 时，两个 bar 之间仍可能被判为 unready；本轮未放宽这个标准。一次离线通过、一次轮询或一个成功退出码都不能替代持续运行和发布验收。
