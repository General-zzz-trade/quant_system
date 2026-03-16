"""Maintained catalog for the flat `scripts/` workspace.

This keeps a non-breaking classification of the scripts layer so the repo can
stay import-compatible while becoming easier to navigate.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED = "supported"
EXPERIMENTAL = "experimental"
LEGACY = "legacy"
ARCHIVE_CANDIDATE = "archive_candidate"


@dataclass(frozen=True)
class ScriptGroup:
    name: str
    purpose: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class ScriptEntrypoint:
    name: str
    status: str
    recommendation: str
    description: str


SCRIPT_GROUPS: tuple[ScriptGroup, ...] = (
    ScriptGroup(
        name="train",
        purpose="Train, retrain, and export production model artifacts.",
        examples=(
            "train_v11.py",
            "train_unified.py",
            "train_multi_horizon.py",
            "auto_retrain.py",
        ),
    ),
    ScriptGroup(
        name="validate",
        purpose="Run backtests, walk-forward validation, and parity checks.",
        examples=(
            "backtest_alpha_v8.py",
            "backtest_engine.py",
            "walkforward_validate.py",
            "walkforward_short.py",
        ),
    ),
    ScriptGroup(
        name="research",
        purpose="Explore factors, IC behavior, regime failures, and diagnostics.",
        examples=(
            "run_alpha_research.py",
            "feature_ic_screen.py",
            "sol_alpha_research.py",
            "ic_analysis_short_features.py",
        ),
    ),
    ScriptGroup(
        name="data",
        purpose="Download, sync, and refresh market, macro, and alt data inputs.",
        examples=(
            "download_binance_klines.py",
            "download_funding_rates.py",
            "download_open_interest.py",
            "refresh_data.py",
        ),
    ),
    ScriptGroup(
        name="ops",
        purpose="Paper/testnet helpers, smoke tests, monitoring, and operational tooling.",
        examples=(
            "run_paper_trading.py",
            "testnet_smoke.py",
            "monitor_paper_trading.py",
            "latency_bench.py",
        ),
    ),
    ScriptGroup(
        name="shared",
        purpose="Shared helpers used by multiple scripts; not primary CLI entrypoints.",
        examples=("signal_postprocess.py",),
    ),
)


PRIMARY_ENTRYPOINTS: tuple[ScriptEntrypoint, ...] = (
    ScriptEntrypoint(
        name="train_v11.py",
        status=SUPPORTED,
        recommendation="official",
        description="Current production-oriented multi-horizon training entrypoint.",
    ),
    ScriptEntrypoint(
        name="backtest_engine.py",
        status=SUPPORTED,
        recommendation="official",
        description="Backtest runner aligned to model config-driven constraints.",
    ),
    ScriptEntrypoint(
        name="walkforward_validate.py",
        status=SUPPORTED,
        recommendation="official",
        description="Main walk-forward validator for the 1h alpha stack.",
    ),
    ScriptEntrypoint(
        name="run_paper_trading.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Paper trading helper around the live runtime.",
    ),
    ScriptEntrypoint(
        name="testnet_smoke.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Testnet connectivity and order-path smoke checks.",
    ),
    ScriptEntrypoint(
        name="refresh_data.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Refresh local market and factor datasets.",
    ),
    ScriptEntrypoint(
        name="train_multi_horizon.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Feature-rich multi-horizon research training path used to iterate on gate-style models.",
    ),
    ScriptEntrypoint(
        name="train_btc_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Current BTC-specific production training path retained alongside the generic v11 trainer.",
    ),
    ScriptEntrypoint(
        name="train_eth_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Current ETH-specific production training path retained alongside the generic v11 trainer.",
    ),
    ScriptEntrypoint(
        name="train_short_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Current short/bear production training path with intentionally wrapped bear-mask semantics.",
    ),
    ScriptEntrypoint(
        name="train_4h_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Current 4h production training path for slower-horizon deployment and comparison.",
    ),
    ScriptEntrypoint(
        name="backtest_portfolio.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Portfolio-level backtest path for multi-asset allocation validation.",
    ),
    ScriptEntrypoint(
        name="backtest_multi_tf.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Multi-timeframe backtest path used when validating cross-timeframe signal composition.",
    ),
    ScriptEntrypoint(
        name="walkforward_short.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Short-side walk-forward validator for bear/short production research.",
    ),
    ScriptEntrypoint(
        name="sol_alpha_research.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Deep-dive SOL-specific research path with intentionally custom regime semantics.",
    ),
    ScriptEntrypoint(
        name="backtest_hybrid_15m.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Hybrid timeframe backtest experiment rather than the default validation path.",
    ),
    ScriptEntrypoint(
        name="train_15m.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Dedicated 15m model training path for shorter-horizon research and deployment.",
    ),
    ScriptEntrypoint(
        name="walk_forward.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="Older walk-forward framework retained for comparison against the current validator path.",
    ),
    ScriptEntrypoint(
        name="backtest_adaptive.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Adaptive-parameter backtest used for regime adaptation research rather than the default backtest path.",  # noqa: E501
    ),
    ScriptEntrypoint(
        name="train_btc_v9.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="Superseded BTC training path kept for historical comparison against newer production pipelines.",
    ),
    ScriptEntrypoint(
        name="train_eth_v9.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="Superseded ETH training path kept for historical comparison against newer production pipelines.",
    ),
    ScriptEntrypoint(
        name="train_30m_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Current 30m production training path for longer-hold intraday experiments.",
    ),
    ScriptEntrypoint(
        name="research_15m_alpha.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="15m alpha transfer research path used to evaluate shorter-horizon model viability.",
    ),
    # ── Auto-retrain & retraining ──────────────────────────────────────
    ScriptEntrypoint(
        name="auto_retrain.py",
        status=SUPPORTED,
        recommendation="official",
        description="Automated walk-forward retraining with IC/Sharpe validation gates and model backup.",
    ),
    ScriptEntrypoint(
        name="retrain_daemon.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Long-running retrain daemon with drift detection and model registry promotion.",
    ),
    # ── Training scripts ───────────────────────────────────────────────
    ScriptEntrypoint(
        name="train_unified.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Parameterized training pipeline covering target, horizon, feature set, and regime splits.",
    ),
    ScriptEntrypoint(
        name="train_v8_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="V8 ensemble LGBM+XGB production trainer with validated walk-forward config.",
    ),
    ScriptEntrypoint(
        name="train_btc_v9b.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="Conservative V9b BTC upgrade path; superseded by V11 training pipeline.",
    ),
    ScriptEntrypoint(
        name="train_bear_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Train full-dataset bear-C binary classifier for production short-side deployment.",
    ),
    ScriptEntrypoint(
        name="train_sol_production.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="SOL-specific production trainer with BTC-lead features and gate_v3 config.",
    ),
    ScriptEntrypoint(
        name="train_v7_alpha.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="V7 multi-timeframe training pipeline; superseded by V8/V11 trainers.",
    ),
    ScriptEntrypoint(
        name="train_1m_alpha.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="1-minute alpha trainer with multi-resolution features and IC gate.",
    ),
    ScriptEntrypoint(
        name="train_1m_v2.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Improved 1-minute alpha V2 with walk-forward OOS evaluation and high deadzone.",
    ),
    ScriptEntrypoint(
        name="train_1m_v3.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="1-minute alpha V3 exploring mean reversion, vol filter, classification, and hybrid approaches.",
    ),
    ScriptEntrypoint(
        name="train_30m_test.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Test 30m/15m/5m resampled models to measure alpha decay across timeframes.",
    ),
    # ── Backtest scripts ───────────────────────────────────────────────
    ScriptEntrypoint(
        name="backtest_alpha_v8.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Realistic OOS trade simulation for alpha_rebuild V8 models with cost modeling.",
    ),
    ScriptEntrypoint(
        name="backtest_honest.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Bias-corrected backtest fixing lookahead, costs, funding, and embargo issues.",
    ),
    ScriptEntrypoint(
        name="backtest_kernel.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Kernel-aware backtester replaying bars through the production Rust pipeline.",
    ),
    ScriptEntrypoint(
        name="backtest_30m_full.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Full-history 30m backtest with rolling walk-forward and OOS validation.",
    ),
    ScriptEntrypoint(
        name="backtest_4h_full.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Full-history 4h backtest with rolling walk-forward and production holdout.",
    ),
    ScriptEntrypoint(
        name="backtest_multi_coin.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Multi-asset portfolio backtest (BTC+ETH) with adaptive deadzone and shorts.",
    ),
    ScriptEntrypoint(
        name="backtest_multi_tf_full.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Full-history multi-timeframe backtest with 1h+4h blend and dynamic leverage.",
    ),
    ScriptEntrypoint(
        name="backtest_p0_improvements.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="P0 improvements backtest comparing adaptive deadzone and short signal variants.",
    ),
    ScriptEntrypoint(
        name="backtest_small_cap.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Small capital growth backtest with correct Binance minimum notional constraints.",
    ),
    ScriptEntrypoint(
        name="backtest_staged_risk.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Staged risk manager backtest comparing fixed vs adaptive risk for small capital.",
    ),
    ScriptEntrypoint(
        name="backtest_tick_v2.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Tick-level V2 strategy backtest simulating aggTrade signals from 1m klines.",
    ),
    ScriptEntrypoint(
        name="backtest_v11_verify.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="V11 architecture verification backtest comparing V10 baseline against V11 variants.",
    ),
    # ── Walk-forward validators ────────────────────────────────────────
    ScriptEntrypoint(
        name="walkforward_portfolio.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Portfolio-level expanding-window walk-forward across BTC+ETH+SOL.",
    ),
    ScriptEntrypoint(
        name="walkforward_validate_1m.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Walk-forward validation for 1-minute alpha models with cost-adjusted metrics.",
    ),
    ScriptEntrypoint(
        name="wf_regime_sweep.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Sweep regime-gating configurations over walk-forward folds for comparison.",
    ),
    ScriptEntrypoint(
        name="wf_signal_sweep.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="3-stage signal parameter sweep (deadzone, min-hold, sizing) over walk-forward folds.",
    ),
    # ── Research & IC analysis ─────────────────────────────────────────
    ScriptEntrypoint(
        name="alpha_rebuild.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Systematic 6-step experiment pipeline to find OOS-robust signals with anti-overfit controls.",
    ),
    ScriptEntrypoint(
        name="run_alpha_research.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="End-to-end alpha research CLI for factor evaluation and comparison.",
    ),
    ScriptEntrypoint(
        name="bear_alpha_research.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Bear market BTC alpha research with 7 strategy variants and comparison report.",
    ),
    ScriptEntrypoint(
        name="research_funding_alpha.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Research funding rate reversal alpha with IC, win rate, and conditional Sharpe.",
    ),
    ScriptEntrypoint(
        name="research_ic_decay.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC decay analysis across horizons to identify fast-decay vs slow-decay features.",
    ),
    ScriptEntrypoint(
        name="research_microstructure_alpha.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis for kline microstructure features at multiple horizons.",
    ),
    ScriptEntrypoint(
        name="feature_ic_screen.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Screen all features by IC and IC_IR to find unused high-value features.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_btc_lead.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis of BTC features as leading indicators for SOL forward returns.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_liquidation.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis for liquidation proxy features derived from OI+volume data.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_macro.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis for macro features (DXY/SPX/VIX) against forward returns.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_mempool.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis for mempool fee features against forward returns.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_onchain.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="IC analysis for on-chain metrics (exchange flows, active addresses, hash rate).",
    ),
    ScriptEntrypoint(
        name="ic_analysis_short_features.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Bear IC analysis ranking features by negative-return-bar predictive power.",
    ),
    ScriptEntrypoint(
        name="ic_analysis_v9.py",
        status=LEGACY,
        recommendation="legacy-reference",
        description="V9 feature IC analysis; superseded by enriched_computer-based IC screens.",
    ),
    ScriptEntrypoint(
        name="analyze_eth_losses.py",
        status=ARCHIVE_CANDIDATE,
        recommendation="legacy-reference",
        description="One-off ETH V10 loss analysis for diagnosing regime/timing/z-score failure patterns.",
    ),
    ScriptEntrypoint(
        name="analyze_explore_log.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Analyze quant_trader dry-run logs to evaluate signal quality and threshold sweeps.",
    ),
    ScriptEntrypoint(
        name="growth_simulation.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Capital growth simulation testing compounding scenarios from small starting equity.",
    ),
    ScriptEntrypoint(
        name="ensemble_eval.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Evaluate LightGBM vs XGBoost ensemble members per symbol for model comparison.",
    ),
    ScriptEntrypoint(
        name="oos_eval.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Shared OOS evaluation utilities for IC, direction accuracy, and cost-adjusted Sharpe.",
    ),
    ScriptEntrypoint(
        name="parity_ablation.py",
        status=EXPERIMENTAL,
        recommendation="specialized",
        description="Parity gap ablation isolating factors behind batch vs kernel backtest divergence.",
    ),
    ScriptEntrypoint(
        name="parity_test_binary.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Parity test verifying Python RustTickProcessor determinism against binary code path.",
    ),
    ScriptEntrypoint(
        name="parity_test_historical.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Historical parity test verifying RustTickProcessor consistency across runs with real data.",
    ),
    ScriptEntrypoint(
        name="shadow_compare.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Shadow comparison of candidate vs production model with auto-promote gating.",
    ),
    # ── Data download & sync ───────────────────────────────────────────
    ScriptEntrypoint(
        name="data_refresh.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Unified daily incremental download of klines, funding, and external data sources.",
    ),
    ScriptEntrypoint(
        name="download_binance_klines.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Download Binance Futures perpetual historical klines (batch or single symbol).",
    ),
    ScriptEntrypoint(
        name="download_spot_klines.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download Binance Spot historical klines using the spot API endpoint.",
    ),
    ScriptEntrypoint(
        name="download_funding_rates.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download historical funding rates from Binance Futures API.",
    ),
    ScriptEntrypoint(
        name="download_open_interest.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download historical open interest data from Binance Futures API.",
    ),
    ScriptEntrypoint(
        name="download_ls_ratio.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download historical long/short ratio from Binance Futures API.",
    ),
    ScriptEntrypoint(
        name="download_taker_ratio.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download taker buy/sell ratio and top trader position ratio from Binance.",
    ),
    ScriptEntrypoint(
        name="download_fear_greed.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download Bitcoin Fear and Greed Index full history from alternative.me API.",
    ),
    ScriptEntrypoint(
        name="download_macro.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download macro data (DXY, SPX, VIX) from Yahoo Finance.",
    ),
    ScriptEntrypoint(
        name="download_deribit_iv.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download historical implied volatility from Deribit public API.",
    ),
    ScriptEntrypoint(
        name="download_eth_15m.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download ETHUSDT 15m kline data from Binance Futures API.",
    ),
    ScriptEntrypoint(
        name="download_liquidations.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Generate liquidation proxy data from OI+volume (Binance deprecated allForceOrders).",
    ),
    ScriptEntrypoint(
        name="download_mempool.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download historical mempool/fee data from mempool.space API.",
    ),
    ScriptEntrypoint(
        name="download_onchain_metrics.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Download on-chain metrics from Coin Metrics Community API.",
    ),
    ScriptEntrypoint(
        name="binance_um_klines_sync.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Sync Binance UM klines from data.binance.vision daily ZIP archives.",
    ),
    ScriptEntrypoint(
        name="record_depth_data.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Record live orderbook depth snapshots aggregated into bar-level CSV for backtesting.",
    ),
    # ── Ops & monitoring ───────────────────────────────────────────────
    ScriptEntrypoint(
        name="monitor_paper_trading.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Monitor paper trading logs showing signal/fill stats and anomaly detection.",
    ),
    ScriptEntrypoint(
        name="latency_bench.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Tick-to-trade latency benchmark measuring full production pipeline timing.",
    ),
    ScriptEntrypoint(
        name="burnin_report.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Burn-in report generator validating paper/shadow/testnet runs against exit criteria.",
    ),
ScriptEntrypoint(
        name="rotate_api_keys.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="API key rotation helper that validates new Binance keys before manual swap.",
    ),
    ScriptEntrypoint(
        name="export_model_to_json.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Export LightGBM/XGBoost .pkl models to .json for Rust native inference.",
    ),
    ScriptEntrypoint(
        name="warm_bridge.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="Generate bridge checkpoint from historical data for z-score warmup on startup.",
    ),
    ScriptEntrypoint(
        name="cli.py",
        status=SUPPORTED,
        recommendation="specialized",
        description="CLI entry point for quant system tools (backtest, sync, catalog subcommands).",
    ),
    # ── Shared helpers ─────────────────────────────────────────────────
    ScriptEntrypoint(
        name="signal_postprocess.py",
        status=SUPPORTED,
        recommendation="shared",
        description="Shared signal post-processing helpers (z-score, monthly gate, bear mask, min hold).",
    ),
)

SUPPORTED_ENTRYPOINTS: tuple[ScriptEntrypoint, ...] = tuple(
    entry for entry in PRIMARY_ENTRYPOINTS if entry.status == SUPPORTED
)


ARCHIVE_NOTE = (
    "Historical superseded training scripts (v2-v6) have been removed. "
    "Git history retains them for reference if needed."
)


def render_catalog() -> str:
    """Render a compact human-readable catalog for CLI/help output."""
    lines = [
        "Scripts Catalog",
        "===============",
        "",
        "Primary entrypoints:",
    ]
    for entry in PRIMARY_ENTRYPOINTS:
        lines.append(
            f"- {entry.name} [{entry.status}/{entry.recommendation}]: {entry.description}"
        )

    lines.extend(["", "Groups:"])
    for group in SCRIPT_GROUPS:
        lines.append(f"- {group.name}: {group.purpose}")
        lines.append(f"  examples: {', '.join(group.examples)}")

    lines.extend(["", f"Archive: {ARCHIVE_NOTE}"])
    return "\n".join(lines)
