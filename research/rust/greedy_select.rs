use pyo3::prelude::*;

const EPS: f64 = 1e-12;

/// Greedy forward IC selection with OLS residualization.
/// Uses precomputed Gram matrix to avoid O(n) work per candidate per step.
/// X_flat: row-major flattened (n_samples x n_features), y: (n_samples,)
/// Returns indices of selected features (0-based).
#[pyfunction]
#[pyo3(signature = (x_flat, y, n_samples, n_features, top_k = 20))]
pub fn cpp_greedy_ic_select(
    x_flat: Vec<f64>,
    y: Vec<f64>,
    n_samples: i32,
    n_features: i32,
    top_k: i32,
) -> Vec<i32> {
    let n = n_samples as usize;
    let p = n_features as usize;

    if n < 50 || p == 0 {
        return vec![];
    }

    let k_max = (top_k as usize).min(p);

    // 1. Precompute column sums, sum-of-squares, and X'y
    let mut col_sum = vec![0.0f64; p];
    let mut col_sumsq = vec![0.0f64; p];
    let mut xty = vec![0.0f64; p];
    let mut y_sum = 0.0f64;
    let mut y_sumsq = 0.0f64;

    for i in 0..n {
        let row_off = i * p;
        let yi = y[i];
        y_sum += yi;
        y_sumsq += yi * yi;
        for j in 0..p {
            let xv = x_flat[row_off + j];
            col_sum[j] += xv;
            col_sumsq[j] += xv * xv;
            xty[j] += xv * yi;
        }
    }

    let y_mean = y_sum / n as f64;
    let y_var = y_sumsq / n as f64 - y_mean * y_mean;
    if y_var < EPS {
        return vec![];
    }

    // Column means and variances (population)
    let mut col_mean = vec![0.0f64; p];
    let mut col_var = vec![0.0f64; p];
    for j in 0..p {
        col_mean[j] = col_sum[j] / n as f64;
        col_var[j] = col_sumsq[j] / n as f64 - col_mean[j] * col_mean[j];
    }

    // 2. Precompute Gram matrix G = X'X (upper triangle + mirror)
    let mut g = vec![0.0f64; p * p];
    for i in 0..n {
        let row_off = i * p;
        for a in 0..p {
            let ra = x_flat[row_off + a];
            for b in a..p {
                g[a * p + b] += ra * x_flat[row_off + b];
            }
        }
    }
    // Mirror
    for a in 0..p {
        for b in (a + 1)..p {
            g[b * p + a] = g[a * p + b];
        }
    }

    // Centered X'y
    let mut c_xty = vec![0.0f64; p];
    for j in 0..p {
        c_xty[j] = xty[j] - n as f64 * col_mean[j] * y_mean;
    }

    // Centered Gram: cG[a][b] = G[a][b] - n * mean_a * mean_b
    let mut cg = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            cg[a * p + b] = g[a * p + b] - n as f64 * col_mean[a] * col_mean[b];
        }
    }

    // Correlation helper
    let corr_col_y = |j: usize| -> f64 {
        let cov = xty[j] / n as f64 - col_mean[j] * y_mean;
        let den = (col_var[j] * y_var).sqrt();
        if den < 1e-15 { 0.0 } else { cov / den }
    };

    // 3. Greedy selection
    let mut selected: Vec<usize> = Vec::with_capacity(k_max);
    let mut used = vec![false; p];

    for step in 0..k_max {
        let mut best_ic = -1.0f64;
        let mut best_idx: Option<usize> = None;

        if step == 0 {
            // First step: pick highest |corr(col_j, y)|
            for j in 0..p {
                if col_var[j] < EPS {
                    continue;
                }
                let ic = corr_col_y(j).abs();
                if ic > best_ic {
                    best_ic = ic;
                    best_idx = Some(j);
                }
            }
        } else {
            let k = selected.len();

            // Build centered Gram sub-matrix for selected features
            let mut cg_sel = vec![0.0f64; k * k];
            for a in 0..k {
                for b in 0..k {
                    cg_sel[a * k + b] = cg[selected[a] * p + selected[b]];
                }
            }

            // LU factorization with partial pivoting
            let mut lu = cg_sel.clone();
            let mut piv: Vec<usize> = (0..k).collect();

            for col in 0..k {
                // Partial pivot
                let mut max_row = col;
                let mut max_val = lu[col * k + col].abs();
                for r in (col + 1)..k {
                    let v = lu[r * k + col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = r;
                    }
                }
                if max_val < 1e-14 {
                    break;
                }
                if max_row != col {
                    piv.swap(col, max_row);
                    for c in 0..k {
                        let idx1 = col * k + c;
                        let idx2 = max_row * k + c;
                        lu.swap(idx1, idx2);
                    }
                }
                for r in (col + 1)..k {
                    let factor = lu[r * k + col] / lu[col * k + col];
                    lu[r * k + col] = factor; // store L factor
                    for c in (col + 1)..k {
                        lu[r * k + c] -= factor * lu[col * k + c];
                    }
                }
            }

            // LU solve closure
            let lu_solve = |rhs: &[f64], out: &mut [f64]| {
                let mut b = vec![0.0f64; k];
                for i in 0..k {
                    b[i] = rhs[piv[i]];
                }
                // Forward sub (L)
                for i in 1..k {
                    for j in 0..i {
                        b[i] -= lu[i * k + j] * b[j];
                    }
                }
                // Back sub (U)
                for i in (0..k).rev() {
                    for j in (i + 1)..k {
                        b[i] -= lu[i * k + j] * b[j];
                    }
                    b[i] /= lu[i * k + i];
                }
                out[..k].copy_from_slice(&b);
            };

            let mut v = vec![0.0f64; k];
            let mut coef = vec![0.0f64; k];

            for j in 0..p {
                if used[j] {
                    continue;
                }

                // v = cG[selected, j]
                for a in 0..k {
                    v[a] = cg[selected[a] * p + j];
                }

                // coef = cG_sel^{-1} @ v
                lu_solve(&v, &mut coef);

                // var(residual) * n = cG[j,j] - v' @ coef
                let mut res_var_n = cg[j * p + j];
                for a in 0..k {
                    res_var_n -= v[a] * coef[a];
                }

                if res_var_n / n as f64 <= EPS {
                    continue;
                }

                // cov(residual, y) * n = cXty[j] - sum coef[a] * cXty[selected[a]]
                let mut res_cov_y_n = c_xty[j];
                for a in 0..k {
                    res_cov_y_n -= coef[a] * c_xty[selected[a]];
                }

                let ic = res_cov_y_n.abs() / (res_var_n * y_var * n as f64).sqrt();
                if ic > best_ic {
                    best_ic = ic;
                    best_idx = Some(j);
                }
            }
        }

        match best_idx {
            Some(idx) => {
                selected.push(idx);
                used[idx] = true;
            }
            None => break,
        }
    }

    selected.iter().map(|&i| i as i32).collect()
}

/// Numpy-accepting version: x is n_samples x n_features (list of rows).
/// Flattens to row-major and calls the flat version.
#[pyfunction]
#[pyo3(signature = (x, y, top_k = 20))]
pub fn cpp_greedy_ic_select_np(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    top_k: i32,
) -> Vec<i32> {
    if x.is_empty() {
        return vec![];
    }
    let n_samples = x.len() as i32;
    let n_features = x[0].len() as i32;

    // Flatten row-major
    let mut x_flat = Vec::with_capacity(x.len() * x[0].len());
    for row in &x {
        x_flat.extend_from_slice(row);
    }

    cpp_greedy_ic_select(x_flat, y, n_samples, n_features, top_k)
}
