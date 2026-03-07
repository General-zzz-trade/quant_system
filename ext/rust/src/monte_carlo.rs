use pyo3::prelude::*;

use crate::fast_rng::FastRNG;

#[pyclass]
#[derive(Clone)]
pub struct MCResult {
    #[pyo3(get)]
    pub paths: i32,
    #[pyo3(get)]
    pub mean_final: f64,
    #[pyo3(get)]
    pub median_final: f64,
    #[pyo3(get)]
    pub percentile_5: f64,
    #[pyo3(get)]
    pub percentile_95: f64,
    #[pyo3(get)]
    pub prob_loss: f64,
    #[pyo3(get)]
    pub prob_target: f64,
    #[pyo3(get)]
    pub max_drawdown_mean: f64,
    #[pyo3(get)]
    pub max_drawdown_95: f64,
}

/// Max drawdown of an equity curve
fn mc_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;
    for i in 1..equity.len() {
        if equity[i] > peak {
            peak = equity[i];
        }
        let dd = (peak - equity[i]) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Linear interpolation percentile on sorted values
fn mc_percentile(sorted_vals: &[f64], p: f64) -> f64 {
    let n = sorted_vals.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_vals[0];
    }
    let k = (n as f64 - 1.0) * p / 100.0;
    let lo = k as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = k - lo as f64;
    sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
}

/// Monte Carlo path simulation.
/// Parametric mode: Box-Muller Gaussian using FastRNG.
/// Bootstrap mode: block bootstrap from returns.
#[pyfunction]
#[pyo3(signature = (returns, n_paths = 1000, horizon = 252, parametric = false, target_return = 0.0, block_size = 5, seed = 42))]
pub fn cpp_simulate_paths(
    returns: Vec<f64>,
    n_paths: i32,
    horizon: i32,
    parametric: bool,
    target_return: f64,
    block_size: i32,
    seed: u64,
) -> MCResult {
    let n = returns.len();
    let np = n_paths as usize;
    let hz = horizon as usize;
    let bs = block_size as usize;

    if n == 0 || np < 1 {
        return MCResult {
            paths: 0,
            mean_final: 1.0,
            median_final: 1.0,
            percentile_5: 1.0,
            percentile_95: 1.0,
            prob_loss: 0.0,
            prob_target: 0.0,
            max_drawdown_mean: 0.0,
            max_drawdown_95: 0.0,
        };
    }

    let mut rng = FastRNG::new(seed);
    let target_wealth = 1.0 + target_return;

    // Pre-compute parametric params
    let mut mu = 0.0f64;
    let mut sigma = 0.0f64;
    if parametric {
        for i in 0..n {
            mu += returns[i];
        }
        mu /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            let d = returns[i] - mu;
            var += d * d;
        }
        sigma = (var / (n.max(2) - 1) as f64).sqrt();
    }

    let mut finals = vec![0.0f64; np];
    let mut drawdowns = vec![0.0f64; np];
    let mut equity = vec![0.0f64; hz + 1];

    for p in 0..np {
        equity[0] = 1.0;

        if parametric {
            for t in 0..hz {
                let r = rng.gauss(mu, sigma);
                equity[t + 1] = equity[t] * (1.0 + r);
            }
        } else {
            // Block bootstrap
            let mut pos = 0usize;
            while pos < hz {
                let start = rng.randint(n);
                for j in 0..bs {
                    if pos >= hz {
                        break;
                    }
                    let r = returns[(start + j) % n];
                    equity[pos + 1] = equity[pos] * (1.0 + r);
                    pos += 1;
                }
            }
        }

        finals[p] = equity[hz];
        drawdowns[p] = mc_max_drawdown(&equity[..hz + 1]);
    }

    let mut finals_sorted = finals.clone();
    let mut dd_sorted = drawdowns.clone();
    finals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dd_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum_f = 0.0f64;
    let mut sum_dd = 0.0f64;
    let mut n_loss = 0i32;
    let mut n_target = 0i32;
    for i in 0..np {
        sum_f += finals[i];
        sum_dd += drawdowns[i];
        if finals[i] < 1.0 {
            n_loss += 1;
        }
        if finals[i] >= target_wealth {
            n_target += 1;
        }
    }

    MCResult {
        paths: n_paths,
        mean_final: sum_f / np as f64,
        median_final: mc_percentile(&finals_sorted, 50.0),
        percentile_5: mc_percentile(&finals_sorted, 5.0),
        percentile_95: mc_percentile(&finals_sorted, 95.0),
        prob_loss: n_loss as f64 / np as f64,
        prob_target: n_target as f64 / np as f64,
        max_drawdown_mean: sum_dd / np as f64,
        max_drawdown_95: mc_percentile(&dd_sorted, 95.0),
    }
}
