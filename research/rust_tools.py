"""Rust-accelerated research tools -- re-exports from _quant_hotpath.

Provides fast Monte Carlo simulation, feature selection (IC-based),
cross-sectional analytics, and portfolio covariance estimation for
research scripts.
"""
from __future__ import annotations

from _quant_hotpath import (  # type: ignore[import-untyped]
    BootstrapResult,
    MCResult,
    cpp_simulate_paths,
    cpp_greedy_ic_select,
    cpp_correlation_select,
    cpp_mutual_info_select,
    cpp_momentum_rank,
    cpp_rolling_beta,
    cpp_relative_strength,
    cpp_vol_normalized_target,
    cpp_rolling_correlation,
    cpp_sample_covariance,
    cpp_ewma_covariance,
    cpp_portfolio_variance,
)

__all__ = [
    "BootstrapResult",
    "MCResult",
    "cpp_simulate_paths",
    "cpp_greedy_ic_select",
    "cpp_correlation_select",
    "cpp_mutual_info_select",
    "cpp_momentum_rank",
    "cpp_rolling_beta",
    "cpp_relative_strength",
    "cpp_vol_normalized_target",
    "cpp_rolling_correlation",
    "cpp_sample_covariance",
    "cpp_ewma_covariance",
    "cpp_portfolio_variance",
]
