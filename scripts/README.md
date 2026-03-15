# Scripts Module

## Quick Reference

| Task | Script | Status |
|------|--------|--------|
| Train production model | `train_v11.py` | supported |
| Auto-retrain (cron) | `auto_retrain.py` | supported |
| Retrain daemon | `retrain_daemon.py` | supported |
| Walk-forward validation | `walkforward_validate.py` | supported |
| Main backtest | `backtest_engine.py` | supported |
| Backtest alpha V8 | `backtest_alpha_v8.py` | supported |
| Shadow compare models | `shadow_compare.py` | supported |
| Download klines | `download_binance_klines.py` | supported |
| Refresh all data | `refresh_data.py` | supported |
| Unified data refresh | `data_refresh.py` | supported |
| Export model to JSON | `export_model_to_json.py` | supported |
| Paper trading | `run_paper_trading.py` | supported |
| Testnet smoke test | `testnet_smoke.py` | supported |
| Latency benchmark | `latency_bench.py` | supported |
| Monitor paper trading | `monitor_paper_trading.py` | supported |
| Train unified | `train_unified.py` | supported |
| Train V8 production | `train_v8_production.py` | supported |
| Train bear production | `train_bear_production.py` | supported |
| Train SOL production | `train_sol_production.py` | supported |
| Backtest honest | `backtest_honest.py` | supported |
| Backtest kernel | `backtest_kernel.py` | supported |
| Walk-forward portfolio | `walkforward_portfolio.py` | supported |
| Alpha rebuild | `alpha_rebuild.py` | supported |
| Alpha research CLI | `run_alpha_research.py` | supported |
| Feature IC screen | `feature_ic_screen.py` | supported |
| Analyze explore log | `analyze_explore_log.py` | supported |
| Ensemble evaluation | `ensemble_eval.py` | supported |
| OOS evaluation | `oos_eval.py` | supported |
| Parity test binary | `parity_test_binary.py` | supported |
| Parity test historical | `parity_test_historical.py` | supported |
| Download spot klines | `download_spot_klines.py` | supported |
| Download funding rates | `download_funding_rates.py` | supported |
| Download open interest | `download_open_interest.py` | supported |
| Download LS ratio | `download_ls_ratio.py` | supported |
| Download taker ratio | `download_taker_ratio.py` | supported |
| Download fear/greed | `download_fear_greed.py` | supported |
| Download macro data | `download_macro.py` | supported |
| Download Deribit IV | `download_deribit_iv.py` | supported |
| Download liquidations | `download_liquidations.py` | supported |
| Download mempool | `download_mempool.py` | supported |
| Download on-chain | `download_onchain_metrics.py` | supported |
| Binance klines sync | `binance_um_klines_sync.py` | supported |
| Record depth data | `record_depth_data.py` | supported |
| Burn-in report | `burnin_report.py` | supported |
| Grafana import | `grafana_import.py` | supported |
| Rotate API keys | `rotate_api_keys.py` | supported |
| Warm bridge checkpoint | `warm_bridge.py` | supported |
| CLI entry point | `cli.py` | supported |
| Signal postprocess | `signal_postprocess.py` | supported |
| Full catalog | `python3 -m scripts.catalog` | CLI |

> Source of truth: [`scripts/catalog.py`](catalog.py). Run `quant catalog --scripts` for the full list.

Status:

- This is the current maintained guide to the flat `scripts/` workspace.
- Script status and classification are defined in [`scripts/catalog.py`](/quant_system/scripts/catalog.py).
- Current runtime truth still lives outside `scripts/`, in [`runner/live_runner.py`](/quant_system/runner/live_runner.py) and [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md).
- Status meanings are:
  - `supported`: current maintained entrypoint
  - `experimental`: active but non-default path
  - `legacy`: historical comparison/reference path
  - `archive_candidate`: appears unused or orphaned

`scripts/` is the operator-facing tools layer around the core trading runtime. It is not the default production runtime entrypoint; that remains [`runner/live_runner.py`](/quant_system/runner/live_runner.py).

## Layout

Files are organized into 7 subdirectories. Symlinks at the top level maintain backward compatibility (`from scripts.xxx import yyy` still works).

```
scripts/
├── training/      (20) — train_v11, train_unified, train_*_production, ...
├── backtesting/   (17) — backtest_engine, backtest_alpha_v8, backtest_honest, ...
├── walkforward/    (7) — walkforward_validate, wf_regime_sweep, ...
├── data/          (17) — download_binance_klines, data_refresh, ...
├── research/      (19) — alpha_rebuild, ic_analysis_*, feature_ic_screen, ...
├── ops/           (13) — run_bybit_alpha, testnet_smoke, auto_retrain, ...
├── shared/         (8) — signal_postprocess, catalog, cli, parity_test_*, ...
└── *.py → symlinks (101) — backward compatibility
```

The maintained group definitions live in [`scripts/catalog.py`](/quant_system/scripts/catalog.py).

## Primary Entrypoints

- `supported / official`: [`scripts/train_v11.py`](/quant_system/scripts/train_v11.py) for production-oriented multi-horizon training
- `supported / official`: [`scripts/backtest_engine.py`](/quant_system/scripts/backtest_engine.py) for config-driven backtests
- `supported / official`: [`scripts/walkforward_validate.py`](/quant_system/scripts/walkforward_validate.py) for the main 1h walk-forward path
- `supported / recommended`: [`scripts/run_paper_trading.py`](/quant_system/scripts/run_paper_trading.py) for paper runtime checks
- `supported / recommended`: [`scripts/testnet_smoke.py`](/quant_system/scripts/testnet_smoke.py) for testnet connectivity and order-path smoke checks
- `supported / recommended`: [`scripts/refresh_data.py`](/quant_system/scripts/refresh_data.py) for local dataset refresh
- `supported / specialized`: [`scripts/train_multi_horizon.py`](/quant_system/scripts/train_multi_horizon.py) for richer multi-horizon gate-model research
- `supported / specialized`: [`scripts/train_btc_production.py`](/quant_system/scripts/train_btc_production.py) and [`scripts/train_eth_production.py`](/quant_system/scripts/train_eth_production.py) for symbol-specific production training
- `supported / specialized`: [`scripts/train_short_production.py`](/quant_system/scripts/train_short_production.py) for short-side production training
- `supported / specialized`: [`scripts/train_4h_production.py`](/quant_system/scripts/train_4h_production.py) for 4h deployment research
- `supported / specialized`: [`scripts/backtest_portfolio.py`](/quant_system/scripts/backtest_portfolio.py) for portfolio-level allocation validation
- `supported / specialized`: [`scripts/backtest_multi_tf.py`](/quant_system/scripts/backtest_multi_tf.py) for cross-timeframe backtests
- `supported / specialized`: [`scripts/walkforward_short.py`](/quant_system/scripts/walkforward_short.py) for short-side walk-forward validation
- `experimental / specialized`: [`scripts/sol_alpha_research.py`](/quant_system/scripts/sol_alpha_research.py) keeps SOL-specific research semantics
- `experimental / specialized`: [`scripts/backtest_hybrid_15m.py`](/quant_system/scripts/backtest_hybrid_15m.py) is an experimental hybrid-timeframe validation path
- `experimental / specialized`: [`scripts/backtest_adaptive.py`](/quant_system/scripts/backtest_adaptive.py) is an adaptive-parameter validation path rather than the default backtest route
- `supported / specialized`: [`scripts/train_15m.py`](/quant_system/scripts/train_15m.py) is the current dedicated 15m training path
- `supported / specialized`: [`scripts/train_30m_production.py`](/quant_system/scripts/train_30m_production.py) is the current 30m training path
- `experimental / specialized`: [`scripts/research_15m_alpha.py`](/quant_system/scripts/research_15m_alpha.py) is an experimental 15m alpha transfer research path
- `legacy / legacy-reference`: [`scripts/walk_forward.py`](/quant_system/scripts/walk_forward.py) is retained for comparison with the newer walk-forward stack
- `legacy / legacy-reference`: [`scripts/train_btc_v9.py`](/quant_system/scripts/train_btc_v9.py) and [`scripts/train_eth_v9.py`](/quant_system/scripts/train_eth_v9.py) are older generation training paths kept for regression comparison

## Shared Helpers

[`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py) is the shared post-processing truth source for script-level:

- `monthly_gate`
- `trend_hold`
- `vol_target`
- `min_hold`
- `rolling_zscore`
- `should_exit_position`
- `bear_mask`

## Auto-Retrain

- `supported / official`: [`scripts/auto_retrain.py`](/quant_system/scripts/auto_retrain.py) for automated walk-forward retraining with validation gates
  - Cron: `0 2 * * 0` (every Sunday 2am UTC)
  - Gates: IC > 0.02, Sharpe > 1.0, comparison > 70%, bootstrap p5 > 0
  - Auto-restores `ic_weighted` ensemble post-train
  - Logs to `logs/retrain_history.jsonl`

## Walk-Forward & Adaptive Research

- `experimental / specialized`: [`scripts/walk_forward.py`](/quant_system/scripts/walk_forward.py) for rolling N-fold walk-forward validation with per-fold training and config sweep
- `experimental / specialized`: [`scripts/backtest_adaptive.py`](/quant_system/scripts/backtest_adaptive.py) for adaptive parameter selection validation (Fixed vs Adaptive vs Robust vs Oracle)
- `experimental / specialized`: [`scripts/research_15m_alpha.py`](/quant_system/scripts/research_15m_alpha.py) for 15m alpha transfer research
- `experimental / specialized`: [`scripts/backtest_hybrid_15m.py`](/quant_system/scripts/backtest_hybrid_15m.py) for hybrid 1h signal + 15m execution comparison
- `supported / specialized`: [`scripts/train_15m.py`](/quant_system/scripts/train_15m.py) for dedicated 15m model training
- `supported / specialized`: [`scripts/download_eth_15m.py`](/quant_system/scripts/download_eth_15m.py) for ETH 15m data download from Binance API

## Catalog Summary

All 99 scripts in `scripts/` are classified in `catalog.py`. Counts by status:

| Status             | Count |
|--------------------|-------|
| `supported`        |    60 |
| `experimental`     |    32 |
| `legacy`           |     6 |
| `archive_candidate`|     1 |
| **Total**          |  **99** |

## Archive

Historical superseded training scripts live under [`scripts/archive`](/quant_system/scripts/archive). They remain in-repo for reference and regression comparison, but they are not the recommended current entrypoints.
