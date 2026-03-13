"""Maintained catalog for the flat `scripts/` workspace.

This keeps a non-breaking classification of the scripts layer so the repo can
stay import-compatible while becoming easier to navigate.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED = "supported"
EXPERIMENTAL = "experimental"
LEGACY = "legacy"


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
        description="Adaptive-parameter backtest used for regime adaptation research rather than the default backtest path.",
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
