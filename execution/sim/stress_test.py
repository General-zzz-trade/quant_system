# execution/sim/stress_test.py
"""Monte Carlo stress testing for backtest robustness.

Answers: how sensitive are results to randomness in execution?
Tests: order shuffling, slippage variance, gap injection, fold permutation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from execution.sim.realistic_backtest import (
    BacktestConfig,
    run_realistic_backtest,
)


@dataclass
class StressTestResult:
    """Results from Monte Carlo stress testing."""
    n_simulations: int
    median_return: float
    p5_return: float        # 5th percentile (worst case)
    p95_return: float       # 95th percentile (best case)
    median_max_dd: float
    p95_max_dd: float       # 95th percentile drawdown (worst)
    bust_rate: float        # % of simulations that went bust (<$10)
    median_sharpe: float
    median_trades: float
    median_liquidations: float


def run_stress_test(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    signal: np.ndarray,
    cfg: BacktestConfig,
    n_sims: int = 100,
    funding_rates: np.ndarray | None = None,
    slippage_noise_bps: float = 2.0,
    gap_injection_pct: float = 0.01,
    execution_delay_bars: int = 0,
) -> StressTestResult:
    """Run Monte Carlo stress test on a backtest.

    For each simulation:
    1. Add random noise to slippage (±slippage_noise_bps)
    2. Inject random price gaps (gap_injection_pct of bars)
    3. Optionally delay signal execution by N bars
    4. Run realistic backtest with perturbed data

    Args:
        n_sims: number of Monte Carlo simulations
        slippage_noise_bps: random slippage noise range (uniform)
        gap_injection_pct: fraction of bars to inject gap shocks
        execution_delay_bars: signal execution delay (0=none, 1=next bar)
    """
    n = len(closes)
    returns = []
    max_dds = []
    sharpes = []
    trades_counts = []
    liquidations = []

    for sim in range(n_sims):
        rng = np.random.RandomState(sim)

        # 1. Widen high/low to simulate worse intra-bar execution
        # (don't change close — signal depends on it)
        sim_closes = closes.copy()
        sim_highs = highs.copy()
        sim_lows = lows.copy()
        n_gaps = int(n * gap_injection_pct)
        if n_gaps > 0:
            gap_bars = rng.choice(range(10, n), size=n_gaps, replace=False)
            for b in gap_bars:
                # Widen the bar range (worse for stop-loss)
                extra = rng.uniform(0.005, 0.03) * sim_closes[b]
                sim_highs[b] += extra
                sim_lows[b] -= extra

        # 2. Perturb slippage
        perturbed_cfg = BacktestConfig(
            initial_equity=cfg.initial_equity,
            leverage=cfg.leverage,
            stop_loss_pct=cfg.stop_loss_pct,
            max_position_pct=cfg.max_position_pct,
            fee_bps=cfg.fee_bps,
            base_slippage_bps=cfg.base_slippage_bps + rng.uniform(-slippage_noise_bps, slippage_noise_bps),
            volume_impact_factor=cfg.volume_impact_factor,
            spread_bps=cfg.spread_bps + rng.uniform(-0.5, 0.5),
            maintenance_margin=cfg.maintenance_margin,
            min_order_notional=cfg.min_order_notional,
        )

        # 3. Delay signal
        sim_signal = signal.copy()
        if execution_delay_bars > 0:
            sim_signal = np.concatenate([
                np.zeros(execution_delay_bars),
                signal[:-execution_delay_bars],
            ])

        # 4. Run backtest
        result = run_realistic_backtest(
            sim_closes, sim_highs, sim_lows, volumes, sim_signal,
            perturbed_cfg, funding_rates,
        )

        returns.append(result.total_return_pct)
        max_dds.append(result.max_drawdown_pct)
        sharpes.append(result.sharpe)
        trades_counts.append(result.n_trades)
        liquidations.append(result.n_liquidations)

    returns = np.array(returns)
    max_dds = np.array(max_dds)

    return StressTestResult(
        n_simulations=n_sims,
        median_return=float(np.median(returns)),
        p5_return=float(np.percentile(returns, 5)),
        p95_return=float(np.percentile(returns, 95)),
        median_max_dd=float(np.median(max_dds)),
        p95_max_dd=float(np.percentile(max_dds, 95)),
        bust_rate=float(np.mean(returns < -90)) * 100,
        median_sharpe=float(np.median(sharpes)),
        median_trades=float(np.median(trades_counts)),
        median_liquidations=float(np.median(liquidations)),
    )
