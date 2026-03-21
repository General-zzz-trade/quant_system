# Config Examples Truth

这些示例 YAML 主要服务于 `runner.live_runner` / framework path。

## 当前定位

| 文件 | 当前定位 |
|---|---|
| [`live.yaml`](/quant_system/infra/config/examples/live.yaml) | `LiveRunner` 的 nested config 示例；偏 Binance / framework |
| [`paper_trading.yaml`](/quant_system/infra/config/examples/paper_trading.yaml) | framework paper 示例 |
| [`testnet_*.yaml`](/quant_system/infra/config/examples) | framework / validation / testnet 示例 |
| [`backtest.yaml`](/quant_system/infra/config/examples/backtest.yaml) | backtest 示例 |
| [`training.yaml`](/quant_system/infra/config/examples/training.yaml) | 训练配置示例 |

## 重要说明

- 这些示例文件不是 `bybit-alpha.service` 当前使用的部署配置
- `bybit-alpha.service` 当前直接执行 `scripts.run_bybit_alpha`，不消费这些 YAML
- `LiveRunner.from_config()` 支持 flat 和 nested 两种 schema；这些示例大多是 nested 形式

## 使用建议

- 如果你在调 framework path，用这些示例作为起点
- 如果你在排查当前 host 上的 directional alpha 或 market maker，优先看 systemd 模板和实际启动命令，而不是这里的 YAML
