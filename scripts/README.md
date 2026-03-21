# Scripts Module

当前 `scripts/` 目录同时承担三类职责：

1. 当前活跃交易入口的兼容包装层
2. 运维 / 数据 / 研究脚本层
3. 旧平铺脚本名到新子目录结构的兼容层

## 1. 当前最重要的入口

| 任务 | 入口 | 当前定位 |
|---|---|---|
| 方向性 alpha | [`scripts/run_bybit_alpha.py`](/quant_system/scripts/run_bybit_alpha.py) → [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) | 当前活跃 directional alpha host service 入口 |
| 高频做市 | [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) | 当前活跃 market-maker host service 入口 |
| framework CLI | [`scripts/cli.py`](/quant_system/scripts/cli.py) → [`scripts/shared/cli.py`](/quant_system/scripts/shared/cli.py) | model / ops 外部工具入口 |
| catalog | [`scripts/catalog.py`](/quant_system/scripts/catalog.py) → [`scripts/shared/catalog.py`](/quant_system/scripts/shared/catalog.py) | 维护中的脚本分类真相源 |

最重要的边界：

- `scripts/` 不再只是“工具层”
- 当前活跃的 directional alpha 和 market maker 都在 `scripts/` 里有正式入口
- `runner/live_runner.py` 仍然是 framework runtime，不等于 `scripts/` 当前所有交易入口都已经迁入 framework

## 2. 目录结构

```
scripts/
├── ops/           当前活跃 alpha、ops 工具、Bybit/Binance 运维脚本
├── data/          数据下载与刷新
├── training/      训练与重训练
├── backtesting/   回测入口
├── walkforward/   walk-forward 验证
├── research/      研究辅助脚本
├── shared/        共享 helper、catalog、CLI
└── *.py           向后兼容包装层
```

## 3. Catalog Truth

[`scripts/shared/catalog.py`](/quant_system/scripts/shared/catalog.py) 当前维护的是“精选主入口分类”，不是仓库里所有兼容包装文件的完整清单。

因此：

- `python3 -m scripts.catalog` / `quant catalog --scripts` 是当前维护入口的真相源
- 它不等于 `find scripts -name '*.py'` 的全量数量

Catalog status 含义：

- `supported`: 当前维护、应优先参考的脚本入口
- `experimental`: 研究或候选路径，不应被误写成默认生产 contract
- `legacy`: 兼容 / 历史对照路径
- `archive_candidate`: 待归档候选，不应再作为新工作默认入口

## 4. Primary Entrypoints

- `python3 -m scripts.run_bybit_alpha --symbols ... --ws`
- `python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20`
- `python3 -m scripts.walkforward_validate --symbol ETHUSDT --no-hpo`
- `python3 -m scripts.backtest_engine --config ...`
- `python3 -m scripts.auto_retrain --dry-run`
- `python3 -m scripts.testnet_smoke --public-only`
- `python3 -m scripts.cli model-inspect --model ...`

## 5. Supported Entrypoints Snapshot

以下脚本名需要与 [`scripts/shared/catalog.py`](/quant_system/scripts/shared/catalog.py) 的 `supported` 清单保持一致：

- `train_v11.py`
- `backtest_engine.py`
- `walkforward_validate.py`
- `run_paper_trading.py`
- `testnet_smoke.py`
- `refresh_data.py`
- `train_multi_horizon.py`
- `train_btc_production.py`
- `train_eth_production.py`
- `train_short_production.py`
- `train_4h_production.py`
- `backtest_portfolio.py`
- `backtest_multi_tf.py`
- `walkforward_short.py`
- `train_15m.py`
- `train_30m_production.py`
- `auto_retrain.py`
- `retrain_daemon.py`
- `train_unified.py`
- `train_v8_production.py`
- `train_bear_production.py`
- `train_sol_production.py`
- `backtest_alpha_v8.py`
- `backtest_honest.py`
- `backtest_kernel.py`
- `walkforward_portfolio.py`
- `alpha_rebuild.py`
- `run_alpha_research.py`
- `feature_ic_screen.py`
- `analyze_explore_log.py`
- `ensemble_eval.py`
- `oos_eval.py`
- `parity_test_binary.py`
- `parity_test_historical.py`
- `shadow_compare.py`
- `data_refresh.py`
- `download_binance_klines.py`
- `download_spot_klines.py`
- `download_funding_rates.py`
- `download_open_interest.py`
- `download_ls_ratio.py`
- `download_taker_ratio.py`
- `download_fear_greed.py`
- `download_macro.py`
- `download_deribit_iv.py`
- `download_eth_15m.py`
- `download_liquidations.py`
- `download_mempool.py`
- `download_onchain_metrics.py`
- `binance_um_klines_sync.py`
- `record_depth_data.py`
- `monitor_paper_trading.py`
- `latency_bench.py`
- `burnin_report.py`
- `rotate_api_keys.py`
- `export_model_to_json.py`
- `warm_bridge.py`
- `cli.py`
- `signal_postprocess.py`

## 6. Shared Helpers

[`scripts/shared/signal_postprocess.py`](/quant_system/scripts/shared/signal_postprocess.py) 是当前研究 / 脚本层共享后处理真相源，覆盖：

- `monthly_gate`
- `trend_hold`
- `vol_target`
- `min_hold`
- `rolling_zscore`
- `should_exit_position`
- `bear_mask`

## 7. 当前限制

- 当前 `run_bybit_alpha` 仍未切到 `LiveRunner`
- 当前 catalog 不试图枚举所有兼容包装脚本
- `scripts/` 目录仍然承载大量历史兼容层，不能把所有文件都当成同等维护级别入口
