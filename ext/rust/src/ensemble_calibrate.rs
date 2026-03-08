//! Adaptive ensemble calibration — IC-weighted, inverse-vol, ridge regression.
//!
//! Migrates computation from `decision/signals/adaptive_ensemble.py`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// ── Pearson correlation ─────────────────────────────────────────────────────

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return 0.0;
    }
    let mx: f64 = x.iter().sum::<f64>() / n as f64;
    let my: f64 = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

// ── Gaussian elimination with partial pivoting ──────────────────────────────

fn solve_linear(a: &[Vec<f64>], b: &[f64], k: usize) -> Vec<f64> {
    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination
    for col in 0..k {
        // Partial pivoting
        let mut max_row = col;
        for row in (col + 1)..k {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }

        for row in (col + 1)..k {
            let factor = aug[row][col] / pivot;
            for j in col..=k {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        if aug[i][i].abs() < 1e-12 {
            continue;
        }
        x[i] = aug[i][k];
        for j in (i + 1)..k {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

// ── Calibration methods ─────────────────────────────────────────────────────

fn ic_weights(
    score_history: &HashMap<String, Vec<f64>>,
    return_history: &[f64],
    names: &[String],
) -> HashMap<String, f64> {
    let n = return_history.len();
    let mut ics: HashMap<String, f64> = HashMap::new();

    for name in names {
        let scores = match score_history.get(name) {
            Some(s) => s,
            None => {
                ics.insert(name.clone(), 0.0);
                continue;
            }
        };
        let m = scores.len().min(n);
        if m < 20 {
            ics.insert(name.clone(), 0.0);
            continue;
        }
        let xs = &scores[scores.len() - m..];
        let ys = &return_history[n - m..];
        // Clip negative IC to 0
        ics.insert(name.clone(), pearson(xs, ys).max(0.0));
    }

    let total: f64 = ics.values().sum();
    if total < 1e-12 {
        let k = ics.len().max(1) as f64;
        for v in ics.values_mut() {
            *v = 1.0 / k;
        }
    } else {
        for v in ics.values_mut() {
            *v /= total;
        }
    }
    ics
}

fn inverse_vol_weights(
    score_history: &HashMap<String, Vec<f64>>,
    names: &[String],
    lookback: usize,
) -> HashMap<String, f64> {
    let mut inv_vols: HashMap<String, f64> = HashMap::new();

    for name in names {
        let scores = match score_history.get(name) {
            Some(s) => s,
            None => {
                inv_vols.insert(name.clone(), 1.0);
                continue;
            }
        };
        if scores.len() < 20 {
            inv_vols.insert(name.clone(), 1.0);
            continue;
        }
        let start = if scores.len() > lookback {
            scores.len() - lookback
        } else {
            0
        };
        let recent = &scores[start..];
        let n = recent.len() as f64;
        let mean = recent.iter().sum::<f64>() / n;
        let var = recent.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let iv = if var > 1e-12 { 1.0 / var.sqrt() } else { 1.0 };
        inv_vols.insert(name.clone(), iv);
    }

    let total: f64 = inv_vols.values().sum();
    if total < 1e-12 {
        let k = inv_vols.len().max(1) as f64;
        for v in inv_vols.values_mut() {
            *v = 1.0 / k;
        }
    } else {
        for v in inv_vols.values_mut() {
            *v /= total;
        }
    }
    inv_vols
}

fn ridge_weights(
    score_history: &HashMap<String, Vec<f64>>,
    return_history: &[f64],
    names: &[String],
) -> HashMap<String, f64> {
    let n = return_history.len();
    let k = names.len();

    if k == 0 {
        return HashMap::new();
    }

    // Get signal data
    let mut signals_data: Vec<Vec<f64>> = Vec::with_capacity(k);
    for name in names {
        let scores = score_history.get(name).cloned().unwrap_or_default();
        let m = scores.len().min(n);
        if m > 0 {
            signals_data.push(scores[scores.len() - m..].to_vec());
        } else {
            signals_data.push(Vec::new());
        }
    }

    // Minimum overlapping length
    let mut min_len = signals_data.iter().map(|s| s.len()).min().unwrap_or(0);
    min_len = min_len.min(n);

    if min_len < 20 {
        let eq = 1.0 / k.max(1) as f64;
        return names.iter().map(|n| (n.clone(), eq)).collect();
    }

    let target = &return_history[n - min_len..];

    // Truncate signals to min_len
    let x: Vec<&[f64]> = signals_data
        .iter()
        .map(|s| &s[s.len() - min_len..])
        .collect();

    // Compute means
    let y_mean = target.iter().sum::<f64>() / min_len as f64;
    let x_means: Vec<f64> = x.iter().map(|s| s.iter().sum::<f64>() / min_len as f64).collect();

    // Build X'X + lambda*I and X'y
    let reg = 0.01;
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..min_len {
        for j in 0..k {
            let xj = x[j][i] - x_means[j];
            xty[j] += xj * (target[i] - y_mean);
            for m in 0..k {
                xtx[j][m] += xj * (x[m][i] - x_means[m]);
            }
        }
    }

    // Regularization
    for j in 0..k {
        xtx[j][j] += reg * min_len as f64;
    }

    let beta = solve_linear(&xtx, &xty, k);

    // Normalize positive weights
    let mut raw: HashMap<String, f64> = HashMap::new();
    for (j, name) in names.iter().enumerate() {
        raw.insert(name.clone(), beta[j].max(0.0));
    }
    let total: f64 = raw.values().sum();
    if total < 1e-12 {
        let eq = 1.0 / k.max(1) as f64;
        return names.iter().map(|n| (n.clone(), eq)).collect();
    }
    for v in raw.values_mut() {
        *v /= total;
    }
    raw
}

// ── PyO3 export ─────────────────────────────────────────────────────────────

/// Calibrate adaptive ensemble weights.
///
/// Parameters
/// ----------
/// method : "ic_weighted" | "inverse_vol" | "ridge"
/// score_history : dict[str, list[float]] — per-signal score history
/// return_history : list[float] — realized return history
/// shrinkage : float — blend toward equal weight (0 = pure calibrated, 1 = equal)
/// lookback : int — max history length for inverse_vol
///
/// Returns
/// -------
/// dict[str, float] — calibrated weights summing to ~1.0
#[pyfunction]
#[pyo3(signature = (method, score_history, return_history, shrinkage, lookback=200))]
pub fn rust_adaptive_ensemble_calibrate<'py>(
    py: Python<'py>,
    method: &str,
    score_history: &Bound<'py, PyDict>,
    return_history: Vec<f64>,
    shrinkage: f64,
    lookback: usize,
) -> PyResult<Bound<'py, PyDict>> {
    // Parse score_history
    let mut sh: HashMap<String, Vec<f64>> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    for (k, v) in score_history.iter() {
        let name: String = k.extract()?;
        let scores: Vec<f64> = v.extract()?;
        names.push(name.clone());
        sh.insert(name, scores);
    }

    if return_history.len() < 20 || names.is_empty() {
        let result = PyDict::new(py);
        let eq = 1.0 / names.len().max(1) as f64;
        for name in &names {
            result.set_item(name, eq)?;
        }
        return Ok(result);
    }

    let raw_weights = match method {
        "ic_weighted" => ic_weights(&sh, &return_history, &names),
        "inverse_vol" => inverse_vol_weights(&sh, &names, lookback),
        "ridge" => ridge_weights(&sh, &return_history, &names),
        _ => {
            let eq = 1.0 / names.len().max(1) as f64;
            names.iter().map(|n| (n.clone(), eq)).collect()
        }
    };

    // Apply shrinkage toward equal weight
    let k = names.len().max(1) as f64;
    let equal_w = 1.0 / k;
    let result = PyDict::new(py);
    for name in &names {
        let raw = raw_weights.get(name).copied().unwrap_or(equal_w);
        let blended = shrinkage * equal_w + (1.0 - shrinkage) * raw;
        result.set_item(name, blended)?;
    }

    Ok(result)
}
