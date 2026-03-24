use pyo3::prelude::*;

/// Cross-sectional momentum rank (NaN-sentinel fast path).
#[pyfunction]
#[pyo3(signature = (returns_matrix, lookback))]
pub fn cpp_momentum_rank(
    returns_matrix: Vec<Vec<f64>>,
    lookback: i32,
) -> PyResult<Vec<Vec<f64>>> {
    if lookback <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lookback must be positive",
        ));
    }
    let lookback = lookback as usize;
    let m = returns_matrix.len();
    if m == 0 {
        return Ok(vec![]);
    }

    let t = returns_matrix.iter().map(|r| r.len()).min().unwrap_or(0);
    let half_lookback = lookback / 2;

    let mut result = vec![vec![f64::NAN; t]; m];
    let mut cum_rets = vec![0.0; m];
    let mut indices = vec![0usize; m];

    for ti in 0..t {
        if ti < lookback {
            continue;
        }

        let mut valid_count = 0usize;
        let start = ti - lookback + 1;

        for mi in 0..m {
            let mut cum = 1.0;
            let mut n_valid = 0;
            for j in start..=ti {
                let r = returns_matrix[mi][j];
                if !r.is_nan() {
                    cum *= 1.0 + r;
                    n_valid += 1;
                }
            }

            if n_valid >= half_lookback {
                cum_rets[mi] = cum - 1.0;
                indices[valid_count] = mi;
                valid_count += 1;
            }
        }

        if valid_count < 2 {
            continue;
        }

        indices[..valid_count].sort_by(|&a, &b| {
            cum_rets[a].partial_cmp(&cum_rets[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let denom = (valid_count - 1).max(1) as f64;
        for rank in 0..valid_count {
            result[indices[rank]][ti] = rank as f64 / denom;
        }
    }

    Ok(result)
}

/// Rolling beta: cov(asset, market) / var(market).
#[pyfunction]
#[pyo3(signature = (asset_returns, market_returns, window))]
pub fn cpp_rolling_beta(
    asset_returns: Vec<f64>,
    market_returns: Vec<f64>,
    window: i32,
) -> PyResult<Vec<f64>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    let window = window as usize;
    let n = asset_returns.len().min(market_returns.len());
    let half_window = window / 2;

    let mut valid = vec![false; n];
    let mut a_vals = vec![0.0; n];
    let mut m_vals = vec![0.0; n];

    for i in 0..n {
        if !asset_returns[i].is_nan() && !market_returns[i].is_nan() {
            valid[i] = true;
            a_vals[i] = asset_returns[i];
            m_vals[i] = market_returns[i];
        }
    }

    let mut out = vec![f64::NAN; n];
    let mut sum_a = 0.0;
    let mut sum_m = 0.0;
    let mut sum_am = 0.0;
    let mut sum_mm = 0.0;
    let mut count = 0i32;

    for i in 0..n {
        if valid[i] {
            sum_a += a_vals[i];
            sum_m += m_vals[i];
            sum_am += a_vals[i] * m_vals[i];
            sum_mm += m_vals[i] * m_vals[i];
            count += 1;
        }

        if i >= window {
            let drop = i - window;
            if valid[drop] {
                sum_a -= a_vals[drop];
                sum_m -= m_vals[drop];
                sum_am -= a_vals[drop] * m_vals[drop];
                sum_mm -= m_vals[drop] * m_vals[drop];
                count -= 1;
            }
        }

        if i < window - 1 {
            continue;
        }
        if count < half_window as i32 {
            continue;
        }

        let cf = count as f64;
        let mean_a = sum_a / cf;
        let mean_m = sum_m / cf;
        let cov = sum_am / cf - mean_a * mean_m;
        let var_m = sum_mm / cf - mean_m * mean_m;

        if var_m > 0.0 {
            out[i] = cov / var_m;
        }
    }

    Ok(out)
}

/// Relative strength vs benchmark over a rolling window.
#[pyfunction]
#[pyo3(signature = (target_returns, benchmark_returns, window))]
pub fn cpp_relative_strength(
    target_returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
    window: i32,
) -> PyResult<Vec<f64>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    let window = window as usize;
    let n = target_returns.len().min(benchmark_returns.len());
    let mut out = vec![f64::NAN; n];

    for i in 0..n {
        if i < window - 1 {
            continue;
        }

        let start = i - window + 1;
        let mut t_cum = 1.0;
        let mut b_cum = 1.0;
        let mut all_valid = true;

        for j in start..=i {
            if target_returns[j].is_nan() || benchmark_returns[j].is_nan() {
                all_valid = false;
                break;
            }
            t_cum *= 1.0 + target_returns[j];
            b_cum *= 1.0 + benchmark_returns[j];
        }

        if all_valid && b_cum != 0.0 {
            out[i] = t_cum / b_cum;
        }
    }

    Ok(out)
}
