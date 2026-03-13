# Scripts Module

Status:

- This is the current maintained guide to the flat `scripts/` workspace.
- Script status and classification are defined in [`scripts/catalog.py`](/quant_system/scripts/catalog.py).
- Current runtime truth still lives outside `scripts/`, in [`runner/live_runner.py`](/quant_system/runner/live_runner.py) and [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md).
- Status meanings are:
  - `supported`: current maintained entrypoint
  - `experimental`: active but non-default path
  - `legacy`: historical comparison/reference path

`scripts/` is the operator-facing tools layer around the core trading runtime. It is not the default production runtime entrypoint; that remains [`runner/live_runner.py`](/quant_system/runner/live_runner.py).

## Layout

The directory is still intentionally flat for import compatibility, but it is now maintained as six logical groups:

- `train`: training, retraining, export
- `validate`: backtests, walk-forward validation, parity checks
- `research`: alpha studies, IC analysis, diagnostics
- `data`: dataset download and refresh
- `ops`: paper/testnet helpers, smoke tests, monitoring utilities
- `shared`: reusable helpers imported by multiple scripts

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

## Archive

Historical superseded training scripts live under [`scripts/archive`](/quant_system/scripts/archive). They remain in-repo for reference and regression comparison, but they are not the recommended current entrypoints.
