# Scripts Module

Status:

- This is the current maintained guide to the flat `scripts/` workspace.
- Script status and classification are defined in [`scripts/catalog.py`](/quant_system/scripts/catalog.py).
- Current runtime truth still lives outside `scripts/`, in [`runner/live_runner.py`](/quant_system/runner/live_runner.py) and [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md).

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

## Current Primary Entrypoints

- `official`: [`scripts/train_v11.py`](/quant_system/scripts/train_v11.py) for production-oriented multi-horizon training
- `official`: [`scripts/backtest_engine.py`](/quant_system/scripts/backtest_engine.py) for config-driven backtests
- `official`: [`scripts/walkforward_validate.py`](/quant_system/scripts/walkforward_validate.py) for the main 1h walk-forward path
- `recommended`: [`scripts/run_paper_trading.py`](/quant_system/scripts/run_paper_trading.py) for paper runtime checks
- `recommended`: [`scripts/testnet_smoke.py`](/quant_system/scripts/testnet_smoke.py) for testnet connectivity and order-path smoke checks
- `recommended`: [`scripts/refresh_data.py`](/quant_system/scripts/refresh_data.py) for local dataset refresh
- `specialized`: [`scripts/train_multi_horizon.py`](/quant_system/scripts/train_multi_horizon.py) for richer multi-horizon gate-model research
- `specialized`: [`scripts/train_btc_production.py`](/quant_system/scripts/train_btc_production.py) and [`scripts/train_eth_production.py`](/quant_system/scripts/train_eth_production.py) for symbol-specific production training
- `specialized`: [`scripts/train_short_production.py`](/quant_system/scripts/train_short_production.py) for short-side production training
- `specialized`: [`scripts/train_4h_production.py`](/quant_system/scripts/train_4h_production.py) for 4h deployment research
- `specialized`: [`scripts/backtest_portfolio.py`](/quant_system/scripts/backtest_portfolio.py) for portfolio-level allocation validation
- `specialized`: [`scripts/backtest_multi_tf.py`](/quant_system/scripts/backtest_multi_tf.py) for cross-timeframe backtests
- `specialized`: [`scripts/walkforward_short.py`](/quant_system/scripts/walkforward_short.py) for short-side walk-forward validation
- `specialized`: [`scripts/sol_alpha_research.py`](/quant_system/scripts/sol_alpha_research.py) keeps SOL-specific research semantics
- `specialized`: [`scripts/backtest_hybrid_15m.py`](/quant_system/scripts/backtest_hybrid_15m.py) is an experimental hybrid-timeframe validation path
- `specialized`: [`scripts/train_15m.py`](/quant_system/scripts/train_15m.py) is the current dedicated 15m training path
- `specialized`: [`scripts/train_30m_production.py`](/quant_system/scripts/train_30m_production.py) is the current 30m training path
- `specialized`: [`scripts/research_15m_alpha.py`](/quant_system/scripts/research_15m_alpha.py) is an experimental 15m alpha transfer research path
- `legacy-reference`: [`scripts/walk_forward.py`](/quant_system/scripts/walk_forward.py) is retained for comparison with the newer walk-forward stack
- `legacy-reference`: [`scripts/train_btc_v9.py`](/quant_system/scripts/train_btc_v9.py) and [`scripts/train_eth_v9.py`](/quant_system/scripts/train_eth_v9.py) are older generation training paths kept for regression comparison

## Shared Helpers

[`scripts/signal_postprocess.py`](/quant_system/scripts/signal_postprocess.py) is the shared post-processing truth source for script-level:

- `monthly_gate`
- `trend_hold`
- `vol_target`
- `min_hold`
- `rolling_zscore`
- `should_exit_position`
- `bear_mask`

## Archive

Historical superseded training scripts live under [`scripts/archive`](/quant_system/scripts/archive). They remain in-repo for reference and regression comparison, but they are not the recommended current entrypoints.
