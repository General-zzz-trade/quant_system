# Scripts Module

`scripts/` 目录承担三类职责：

1. 当前活跃交易入口
2. 运维 / 数据 / 研究脚本
3. 兼容包装层（转发到 `scripts/ops/data/research/`）

## 1. 核心入口

| 任务 | 入口 | 状态 |
|---|---|---|
| 方向性 alpha | `run_bybit_alpha.py` → `scripts/ops/run_bybit_alpha.py` | BTC+ETH only |
| 做市 | `run_bybit_mm.py` | inactive (spread < fee) |
| WF 验证 | `walkforward_validate.py` → `scripts/walkforward/` | 核心验证工具 |
| 自动重训练 | `auto_retrain.py` → `scripts/ops/auto_retrain.py` | 每3天一次 |
| 数据刷新 | `data_refresh.py` → `scripts/data/refresh_data.py` | 每6小时 |

## 2. 目录结构

```
scripts/
├── ops/           活跃 alpha runner、运维工具、健康检查
├── data/          数据下载与刷新 (klines, funding, OI, options)
├── training/      训练与重训练
├── walkforward/   walk-forward 验证
├── research/      研究脚本 (gate impact, bear model, pair trading, 4h ensemble)
├── shared/        共享 helper、catalog、signal_postprocess
└── *.py           兼容包装层 (36 files, down from 118)
```

## 3. Catalog Truth

[`scripts/shared/catalog.py`](/quant_system/scripts/shared/catalog.py) 是脚本分类真相源。

Catalog status 含义：

- `supported`: 当前维护、应优先参考的脚本入口
- `experimental`: 研究路径
- `legacy`: 兼容 / 历史对照路径
- `archive_candidate`: 待归档候选

## 4. Primary Entrypoints

```bash
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m --ws
python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20 --dry-run
python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo
python3 -m scripts.auto_retrain --dry-run
python3 -m scripts.data_refresh --symbols BTCUSDT ETHUSDT
python3 -m scripts.testnet_smoke --public-only
```

## 5. Supported Entrypoints

- `train_v7_alpha.py`
- `train_multi_horizon.py`
- `backtest_engine.py`
- `backtest_alpha_v8.py`
- `walkforward_validate.py`
- `auto_retrain.py`
- `run_bybit_alpha.py`
- `run_bybit_mm.py`
- `run_hft_signal.py`
- `run_binary_signal.py`
- `run_polymarket_dryrun.py`
- `run_liquidation_sniper.py`
- `download_binance_klines.py`
- `data_refresh.py`
- `run_paper_trading.py`
- `testnet_smoke.py`
- `monitor_paper_trading.py`
- `signal_postprocess.py`
- `check_deploy_scope.py`

## 6. Cleanup History

- 2026-03-21: Removed 82 dead/unreferenced scripts (118 → 36)
- All removed scripts preserved in git history
