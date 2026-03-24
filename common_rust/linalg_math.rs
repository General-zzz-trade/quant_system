use pyo3::prelude::*;

type Vec1 = Vec<f64>;
type Mat = Vec<Vec<f64>>;

fn mat_zeros(n: usize, m: usize) -> Mat {
    vec![vec![0.0; m]; n]
}

fn mat_transpose(a: &Mat) -> Mat {
    if a.is_empty() {
        return vec![];
    }
    let r = a.len();
    let c = a[0].len();
    let mut result = mat_zeros(c, r);
    for i in 0..r {
        for j in 0..c {
            result[j][i] = a[i][j];
        }
    }
    result
}

fn mat_multiply(a: &Mat, b: &Mat) -> Mat {
    let ra = a.len();
    let ca = a[0].len();
    let cb = b[0].len();
    let mut result = mat_zeros(ra, cb);
    for i in 0..ra {
        for k in 0..ca {
            let aik = a[i][k];
            for j in 0..cb {
                result[i][j] += aik * b[k][j];
            }
        }
    }
    result
}

fn mat_scale(a: &Mat, s: f64) -> Mat {
    a.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

fn mat_add(a: &Mat, b: &Mat) -> Mat {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&va, &vb)| va + vb).collect())
        .collect()
}

fn mat_vec_multiply(m: &Mat, v: &Vec1) -> Vec1 {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

fn mat_inverse(mat: &Mat) -> Result<Mat, String> {
    let n = mat.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err("Singular matrix".to_string());
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * n) {
            aug[col][j] *= inv_pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut inv = mat_zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Black-Litterman posterior computation.
/// Returns (posterior_returns, posterior_covariance, equilibrium_returns).
#[pyfunction]
#[pyo3(signature = (sigma, market_weights, p, q, confidences, tau, risk_aversion))]
pub fn cpp_black_litterman_posterior(
    sigma: Vec<Vec<f64>>,
    market_weights: Vec<f64>,
    p: Vec<Vec<f64>>,
    q: Vec<f64>,
    confidences: Vec<f64>,
    tau: f64,
    risk_aversion: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
    let n = sigma.len();
    let k = p.len();

    // pi = delta * Sigma @ w
    let sigma_w = mat_vec_multiply(&sigma, &market_weights);
    let pi: Vec<f64> = sigma_w.iter().map(|&v| risk_aversion * v).collect();

    if k == 0 {
        return Ok((pi.clone(), sigma, pi));
    }

    let tau_sigma = mat_scale(&sigma, tau);

    // Omega (KxK diagonal)
    let mut omega = mat_zeros(k, k);
    for v in 0..k {
        let tau_sigma_p = mat_vec_multiply(&tau_sigma, &p[v]);
        let mut view_var = 0.0;
        for j in 0..n {
            view_var += p[v][j] * tau_sigma_p[j];
        }
        let conf = if confidences[v] <= 0.0 { 1e-6 } else { confidences[v] };
        omega[v][v] = view_var / conf;
    }

    let tau_sigma_inv = mat_inverse(&tau_sigma)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let omega_inv = mat_inverse(&omega)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let pt = mat_transpose(&p);
    let pt_omega_inv = mat_multiply(&pt, &omega_inv);
    let pt_omega_inv_p = mat_multiply(&pt_omega_inv, &p);

    let m_inv = mat_add(&tau_sigma_inv, &pt_omega_inv_p);
    let m_mat = mat_inverse(&m_inv)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let tsi_pi = mat_vec_multiply(&tau_sigma_inv, &pi);
    let poi_q = mat_vec_multiply(&pt_omega_inv, &q);

    let combined: Vec<f64> = tsi_pi.iter().zip(poi_q.iter()).map(|(&a, &b)| a + b).collect();
    let mu_post = mat_vec_multiply(&m_mat, &combined);
    let post_cov = mat_add(&sigma, &m_mat);

    Ok((mu_post, post_cov, pi))
}
