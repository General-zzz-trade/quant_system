use pyo3::prelude::*;

use crate::fast_rng::FastRNG;

#[pyclass]
#[derive(Clone)]
pub struct BootstrapResult {
    #[pyo3(get)]
    pub sharpe_mean: f64,
    #[pyo3(get)]
    pub sharpe_95ci_lo: f64,
    #[pyo3(get)]
    pub sharpe_95ci_hi: f64,
    #[pyo3(get)]
    pub p_sharpe_gt_0: f64,
    #[pyo3(get)]
    pub p_sharpe_gt_05: f64,
}

/// Block bootstrap Sharpe ratio confidence interval.
/// Annualization: sqrt(8760) (hourly -> annual).
#[pyfunction]
#[pyo3(signature = (returns, n_bootstrap = 10000, block_size = 5, seed = 42))]
pub fn cpp_bootstrap_sharpe_ci(
    returns: Vec<f64>,
    n_bootstrap: i32,
    block_size: i32,
    seed: u64,
) -> BootstrapResult {
    let n = returns.len();
    if n < 10 {
        return BootstrapResult {
            sharpe_mean: 0.0,
            sharpe_95ci_lo: 0.0,
            sharpe_95ci_hi: 0.0,
            p_sharpe_gt_0: 0.0,
            p_sharpe_gt_05: 0.0,
        };
    }

    let annualize = (8760.0f64).sqrt();
    let mut rng = FastRNG::new(seed);
    let nb = n_bootstrap as usize;
    let bs = block_size as usize;

    let mut sharpes = vec![0.0f64; nb];
    let mut sample = vec![0.0f64; n];

    for b in 0..nb {
        let mut pos = 0usize;
        while pos < n {
            let start = rng.randint(n);
            for j in 0..bs {
                if pos >= n {
                    break;
                }
                sample[pos] = returns[(start + j) % n];
                pos += 1;
            }
        }

        let mut sum = 0.0f64;
        for i in 0..n {
            sum += sample[i];
        }
        let mu = sum / n as f64;

        let mut sumsq = 0.0f64;
        for i in 0..n {
            let d = sample[i] - mu;
            sumsq += d * d;
        }
        let std_dev = (sumsq / (n as f64 - 1.0)).sqrt();
        sharpes[b] = if std_dev > 1e-15 {
            mu / std_dev * annualize
        } else {
            0.0
        };
    }

    let mut sorted_sharpes = sharpes.clone();
    sorted_sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let s_sum: f64 = sharpes.iter().sum();
    let s_mean = s_sum / nb as f64;

    let percentile = |p: f64| -> f64 {
        let k = (nb as f64 - 1.0) * p / 100.0;
        let lo = k as usize;
        let hi = (lo + 1).min(nb - 1);
        let frac = k - lo as f64;
        sorted_sharpes[lo] + frac * (sorted_sharpes[hi] - sorted_sharpes[lo])
    };

    let mut gt0 = 0i32;
    let mut gt05 = 0i32;
    for &s in &sharpes {
        if s > 0.0 {
            gt0 += 1;
        }
        if s > 0.5 {
            gt05 += 1;
        }
    }

    BootstrapResult {
        sharpe_mean: s_mean,
        sharpe_95ci_lo: percentile(2.5),
        sharpe_95ci_hi: percentile(97.5),
        p_sharpe_gt_0: gt0 as f64 / nb as f64,
        p_sharpe_gt_05: gt05 as f64 / nb as f64,
    }
}
