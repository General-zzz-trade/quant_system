use pyo3::prelude::*;

/// Batch factor exposure (beta) computation.
#[pyfunction]
pub fn cpp_compute_exposures(
    asset_returns: Vec<Vec<f64>>,
    factor_returns: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let n = asset_returns.len();
    let f_count = factor_returns.len();
    if n == 0 || f_count == 0 {
        return vec![vec![0.0; f_count]; n];
    }

    let mut t = asset_returns[0].len();
    for i in 1..n {
        t = t.min(asset_returns[i].len());
    }
    for f in 0..f_count {
        t = t.min(factor_returns[f].len());
    }

    let mut result = vec![vec![0.0; f_count]; n];
    if t < 2 {
        return result;
    }

    let inv_t = 1.0 / t as f64;
    let inv_t1 = 1.0 / (t - 1) as f64;

    // Precompute factor means, demeaned data, and variances
    let mut f_means = vec![0.0; f_count];
    let mut f_dm = vec![vec![0.0; t]; f_count];
    let mut f_var = vec![0.0; f_count];

    for f in 0..f_count {
        let s: f64 = factor_returns[f][..t].iter().sum();
        f_means[f] = s * inv_t;

        let mut var_sum = 0.0;
        for ti in 0..t {
            let d = factor_returns[f][ti] - f_means[f];
            f_dm[f][ti] = d;
            var_sum += d * d;
        }
        f_var[f] = var_sum * inv_t1;
    }

    for i in 0..n {
        let s: f64 = asset_returns[i][..t].iter().sum();
        let a_mean = s * inv_t;

        for f in 0..f_count {
            if f_var[f] < 1e-12 {
                continue;
            }

            let mut cov = 0.0;
            for ti in 0..t {
                cov += (asset_returns[i][ti] - a_mean) * f_dm[f][ti];
            }
            cov *= inv_t1;

            result[i][f] = cov / f_var[f];
        }
    }

    result
}

/// Factor model covariance: Sigma = B * F * B' + D.
#[pyfunction]
pub fn cpp_factor_model_covariance(
    exposures: Vec<Vec<f64>>,
    factor_cov: Vec<Vec<f64>>,
    specific_risk: Vec<f64>,
) -> Vec<Vec<f64>> {
    let n = exposures.len();
    let f_count = factor_cov.len();

    let mut result = vec![vec![0.0; n]; n];

    if n == 0 || f_count == 0 {
        for i in 0..n {
            if i < specific_risk.len() {
                result[i][i] = specific_risk[i];
            }
        }
        return result;
    }

    // BF = B * F  (N x F)
    let mut bf = vec![vec![0.0; f_count]; n];
    for i in 0..n {
        for f2 in 0..f_count {
            let mut s = 0.0;
            for f1 in 0..f_count {
                let b = if f1 < exposures[i].len() { exposures[i][f1] } else { 0.0 };
                let fc = if f2 < factor_cov[f1].len() { factor_cov[f1][f2] } else { 0.0 };
                s += b * fc;
            }
            bf[i][f2] = s;
        }
    }

    // Sigma = BF * B' + D
    for i in 0..n {
        for j in 0..n {
            let mut cov = 0.0;
            for f in 0..f_count {
                let bj = if f < exposures[j].len() { exposures[j][f] } else { 0.0 };
                cov += bf[i][f] * bj;
            }
            if i == j && i < specific_risk.len() {
                cov += specific_risk[i];
            }
            result[i][j] = cov;
        }
    }

    result
}

/// Estimate specific (idiosyncratic) risk for each asset.
#[pyfunction]
pub fn cpp_estimate_specific_risk(
    asset_returns: Vec<Vec<f64>>,
    factor_returns: Vec<Vec<f64>>,
    exposures: Vec<Vec<f64>>,
) -> Vec<f64> {
    let n = asset_returns.len();
    let f_count = factor_returns.len();
    let mut result = vec![0.0; n];
    if n == 0 {
        return result;
    }

    let mut t = asset_returns[0].len();
    for i in 1..n {
        t = t.min(asset_returns[i].len());
    }
    for f in 0..f_count {
        t = t.min(factor_returns[f].len());
    }

    if t < 2 {
        return result;
    }

    let inv_t1 = 1.0 / (t - 1) as f64;

    for i in 0..n {
        let mut sum_r = 0.0;
        let mut residuals = vec![0.0; t];

        for ti in 0..t {
            let mut predicted = 0.0;
            for f in 0..f_count {
                let beta = if f < exposures[i].len() { exposures[i][f] } else { 0.0 };
                predicted += beta * factor_returns[f][ti];
            }
            residuals[ti] = asset_returns[i][ti] - predicted;
            sum_r += residuals[ti];
        }

        let mean_r = sum_r / t as f64;
        let mut var_sum = 0.0;
        for ti in 0..t {
            let d = residuals[ti] - mean_r;
            var_sum += d * d;
        }
        result[i] = var_sum * inv_t1;
    }

    result
}
