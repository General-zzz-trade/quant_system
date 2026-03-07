use pyo3::prelude::*;

/// Sample covariance matrix. Exploits symmetry.
#[pyfunction]
pub fn cpp_sample_covariance(
    returns_matrix: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let m = returns_matrix.len();
    if m == 0 {
        return vec![];
    }

    let n_obs = returns_matrix.iter().map(|r| r.len()).min().unwrap_or(0);
    let mut result = vec![vec![0.0; m]; m];
    if n_obs < 2 {
        return result;
    }

    let mut means = vec![0.0; m];
    for mi in 0..m {
        let s: f64 = returns_matrix[mi][..n_obs].iter().sum();
        means[mi] = s / n_obs as f64;
    }

    let mut dm = vec![vec![0.0; n_obs]; m];
    for mi in 0..m {
        for t in 0..n_obs {
            dm[mi][t] = returns_matrix[mi][t] - means[mi];
        }
    }

    let inv_n1 = 1.0 / (n_obs - 1) as f64;
    for i in 0..m {
        for j in i..m {
            let mut cov = 0.0;
            for t in 0..n_obs {
                cov += dm[i][t] * dm[j][t];
            }
            cov *= inv_n1;
            result[i][j] = cov;
            result[j][i] = cov;
        }
    }

    result
}

/// EWMA covariance matrix.
#[pyfunction]
pub fn cpp_ewma_covariance(
    returns_matrix: Vec<Vec<f64>>,
    alpha: f64,
) -> Vec<Vec<f64>> {
    let m = returns_matrix.len();
    if m == 0 {
        return vec![];
    }

    let n_obs = returns_matrix.iter().map(|r| r.len()).min().unwrap_or(0);
    let mut result = vec![vec![0.0; m]; m];
    if n_obs < 2 {
        return result;
    }

    let mut cov = vec![0.0; m * m];

    // Initialize with first observation outer product (symmetric)
    for i in 0..m {
        let ri = returns_matrix[i][0];
        for j in i..m {
            let val = ri * returns_matrix[j][0];
            cov[i * m + j] = val;
            cov[j * m + i] = val;
        }
    }

    let one_minus_alpha = 1.0 - alpha;
    for t in 1..n_obs {
        for i in 0..m {
            let ari = alpha * returns_matrix[i][t];
            for j in i..m {
                let rj = returns_matrix[j][t];
                let val = ari * rj + one_minus_alpha * cov[i * m + j];
                cov[i * m + j] = val;
                cov[j * m + i] = val;
            }
        }
    }

    for i in 0..m {
        for j in 0..m {
            result[i][j] = cov[i * m + j];
        }
    }

    result
}

/// Rolling Pearson correlation matrix using last `window` observations.
#[pyfunction]
#[pyo3(signature = (returns_matrix, window))]
pub fn cpp_rolling_correlation(
    returns_matrix: Vec<Vec<f64>>,
    window: i32,
) -> PyResult<Vec<Vec<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    let w = window as usize;
    let m = returns_matrix.len();
    if m == 0 {
        return Ok(vec![]);
    }

    // Extract last `window` observations
    let data: Vec<Vec<f64>> = returns_matrix
        .iter()
        .map(|r| {
            let actual_w = r.len().min(w);
            r[r.len() - actual_w..].to_vec()
        })
        .collect();

    let n = data.iter().map(|d| d.len()).min().unwrap_or(0);
    let mut result = vec![vec![0.0; m]; m];

    if n < 2 {
        for i in 0..m {
            result[i][i] = 1.0;
        }
        return Ok(result);
    }

    let inv_n1 = 1.0 / (n - 1) as f64;

    let mut means = vec![0.0; m];
    let mut stds = vec![0.0; m];
    let mut dm = vec![vec![0.0; n]; m];

    for mi in 0..m {
        let s: f64 = data[mi][..n].iter().sum();
        means[mi] = s / n as f64;

        let mut var_sum = 0.0;
        for t in 0..n {
            let d = data[mi][t] - means[mi];
            dm[mi][t] = d;
            var_sum += d * d;
        }
        stds[mi] = (var_sum * inv_n1).sqrt();
    }

    for i in 0..m {
        result[i][i] = 1.0;
        for j in (i + 1)..m {
            if stds[i] < 1e-12 || stds[j] < 1e-12 {
                continue;
            }
            let mut cov = 0.0;
            for t in 0..n {
                cov += dm[i][t] * dm[j][t];
            }
            cov *= inv_n1;
            let corr = (cov / (stds[i] * stds[j])).clamp(-1.0, 1.0);
            result[i][j] = corr;
            result[j][i] = corr;
        }
    }

    Ok(result)
}

/// Portfolio variance: w' * Cov * w.
#[pyfunction]
pub fn cpp_portfolio_variance(
    weights: Vec<f64>,
    cov: Vec<Vec<f64>>,
) -> f64 {
    let n = weights.len();
    if n == 0 || cov.len() < n {
        return 0.0;
    }

    let mut variance = 0.0;
    for i in 0..n {
        variance += weights[i] * weights[i] * cov[i][i];
        for j in (i + 1)..n {
            variance += 2.0 * weights[i] * weights[j] * cov[i][j];
        }
    }
    variance
}
