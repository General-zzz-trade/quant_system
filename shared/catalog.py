"""Maintained catalog for the flat `scripts/` workspace.

This keeps a non-breaking classification of the scripts layer so the repo can
stay import-compatible while becoming easier to navigate.

Updated: 2026-03-21 — cleaned to match files on disk after dead script removal.
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
            "train_v7_alpha.py",
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
        ),
    ),
    ScriptGroup(
        name="data",
        purpose="Download, sync, and refresh market, macro, and alt data inputs.",
        examples=(
            "download_binance_klines.py",
            "data_refresh.py",
        ),
    ),
    ScriptGroup(
        name="ops",
        purpose="Paper/testnet helpers, smoke tests, monitoring, and operational tooling.",
        examples=(
            "run_paper_trading.py",
            "testnet_smoke.py",
            "monitor_paper_trading.py",
        ),
    ),
    ScriptGroup(
        name="shared",
        purpose="Shared helpers used by multiple scripts; not primary CLI entrypoints.",
        examples=("signal_postprocess.py",),
    ),
)


PRIMARY_ENTRYPOINTS: tuple[ScriptEntrypoint, ...] = (
    # ── Core production ───────────────────────────────────────────────
    ScriptEntrypoint(
        name="train_v7_alpha.py",
        status=SUPPORTED,
        recommendation="official",
        description="Core training pipeline (imported by WF and auto_retrain).",
    ),
    ScriptEntrypoint(
        name="train_multi_horizon.py",
        status=SUPPORTED,
        recommendation="official",
        description="Multi-horizon research training path for gate-style models.",
    ),
    ScriptEntrypoint(
        name="backtest_engine.py",
        status=SUPPORTED,
        recommendation="official",
        description="Backtest runner aligned to model config-driven constraints.",
    ),
    ScriptEntrypoint(
        name="backtest_alpha_v8.py",
        status=SUPPORTED,
        recommendation="official",
        description="Realistic OOS trade simulation for alpha V8 models with cost modeling.",
    ),
    ScriptEntrypoint(
        name="walkforward_validate.py",
        status=SUPPORTED,
        recommendation="official",
        description="Main walk-forward validator for the 1h alpha stack.",
    ),
    ScriptEntrypoint(
        name="auto_retrain.py",
        status=SUPPORTED,
        recommendation="official",
        description="Automated walk-forward retraining with IC/Sharpe gates (every 3 days).",
    ),
    # ── Trading entry points ──────────────────────────────────────────
    ScriptEntrypoint(
        name="run_bybit_alpha.py",
        status=SUPPORTED,
        recommendation="official",
        description="Directional alpha entry (forwards to scripts.ops.run_bybit_alpha).",
    ),
    ScriptEntrypoint(
        name="run_bybit_mm.py",
        status=SUPPORTED,
        recommendation="official",
        description="Market maker entry (inactive, spread < fee).",
    ),
    ScriptEntrypoint(
        name="run_hft_signal.py",
        status=SUPPORTED,
        recommendation="legacy-reference",
        description="HFT 8-layer signal (inactive, Sharpe -5).",
    ),
    ScriptEntrypoint(
        name="run_binary_signal.py",
        status=SUPPORTED,
        recommendation="legacy-reference",
        description="Binary 5m signal (inactive, wrong payoff).",
    ),
    ScriptEntrypoint(
        name="run_polymarket_dryrun.py",
        status=SUPPORTED,
        recommendation="official",
        description="Polymarket RSI(30/70) taker dry-run validation.",
    ),
    # ── Data ──────────────────────────────────────────────────────────
    ScriptEntrypoint(
        name="download_binance_klines.py",
        status=SUPPORTED,
        recommendation="official",
        description="Download Binance Futures klines (batch or single symbol).",
    ),
    ScriptEntrypoint(
        name="data_refresh.py",
        status=SUPPORTED,
        recommendation="official",
        description="Unified incremental download of all data sources.",
    ),
    # ── Ops ───────────────────────────────────────────────────────────
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
        name="monitor_paper_trading.py",
        status=SUPPORTED,
        recommendation="recommended",
        description="Monitor paper trading logs with anomaly detection.",
    ),
    ScriptEntrypoint(
        name="signal_postprocess.py",
        status=SUPPORTED,
        recommendation="shared",
        description="Shared signal post-processing helpers (z-score, monthly gate, min hold).",
    ),
    ScriptEntrypoint(
        name="check_deploy_scope.py",
        status=SUPPORTED,
        recommendation="shared",
        description="CI deploy scope checker.",
    ),
)

SUPPORTED_ENTRYPOINTS: tuple[ScriptEntrypoint, ...] = tuple(
    entry for entry in PRIMARY_ENTRYPOINTS if entry.status == SUPPORTED
)


ARCHIVE_NOTE = (
    "82 dead/unreferenced scripts removed 2026-03-21. "
    "Git history retains them. See commit log for details."
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
