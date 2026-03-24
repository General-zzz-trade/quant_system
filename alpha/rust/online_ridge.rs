// alpha/rust/online_ridge.rs — Online Ridge regression via Recursive Least Squares.
//
// Replaces Python numpy matrix operations with pure Rust.
// Matches the interface of alpha/online_ridge.py::OnlineRidge.

use pyo3::prelude::*;

/// Online Ridge regression with RLS (Recursive Least Squares) updates.
///
/// Starts from zero (or loaded) weights and incrementally updates them
/// as new (features, return) pairs arrive.  Forgetting factor lambda
/// controls how quickly old observations are down-weighted.
#[pyclass]
pub struct RustOnlineRidge {
    weights: Vec<f64>,
    p_matrix: Vec<f64>,        // Flattened n x n covariance matrix (row-major)
    intercept: f64,
    forgetting_factor: f64,
    n_features: usize,
    n_updates: u64,
    regularization: f64,
    max_update_magnitude: f64,
    min_samples: u64,
    // Static weights for reset_to_static
    static_weights: Option<Vec<f64>>,
    static_intercept: f64,
}

impl RustOnlineRidge {
    #[inline]
    fn p(&self, i: usize, j: usize) -> f64 {
        self.p_matrix[i * self.n_features + j]
    }

    #[inline]
    fn p_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        let n = self.n_features;
        &mut self.p_matrix[i * n + j]
    }

    fn reset_p(&mut self) {
        let n = self.n_features;
        let p_init = 1.0 / self.regularization;
        self.p_matrix.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            *self.p_mut(i, i) = p_init;
        }
    }
}

#[pymethods]
impl RustOnlineRidge {
    #[new]
    #[pyo3(signature = (n_features, forgetting_factor=0.997, regularization=1.0, max_update_magnitude=0.1, min_samples=50))]
    fn new(
        n_features: usize,
        forgetting_factor: f64,
        regularization: f64,
        max_update_magnitude: f64,
        min_samples: u64,
    ) -> Self {
        let p_init = 1.0 / regularization;
        let mut p_matrix = vec![0.0; n_features * n_features];
        for i in 0..n_features {
            p_matrix[i * n_features + i] = p_init;
        }
        Self {
            weights: vec![0.0; n_features],
            p_matrix,
            intercept: 0.0,
            forgetting_factor,
            n_features,
            n_updates: 0,
            regularization,
            max_update_magnitude,
            min_samples,
            static_weights: None,
            static_intercept: 0.0,
        }
    }

    /// Load weights from explicit vectors (matches load_from_weights).
    #[pyo3(signature = (weights, intercept=0.0))]
    fn load_from_weights(&mut self, weights: Vec<f64>, intercept: f64) {
        if weights.len() != self.n_features {
            return;
        }
        self.weights = weights.clone();
        self.intercept = intercept;
        self.static_weights = Some(weights);
        self.static_intercept = intercept;
        self.reset_p();
        self.n_updates = 0;
    }

    /// RLS update: P = (1/lam)(P - P*x*x'*P / (lam + x'*P*x))
    ///              w = w + K*(y - x'*w)  with magnitude clamping.
    ///
    /// Returns prediction error before update.
    fn update(&mut self, x: Vec<f64>, y: f64) -> f64 {
        let n = self.n_features;
        if x.len() != n {
            return 0.0;
        }

        // Skip NaN/Inf
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return 0.0;
        }

        self.n_updates += 1;

        // Prediction before update
        let pred: f64 = x.iter().zip(self.weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + self.intercept;
        let error = y - pred;

        // Don't update weights until we have enough samples
        if self.n_updates < self.min_samples {
            return error;
        }

        let lam = self.forgetting_factor;

        // Compute P*x
        let mut px = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p(i, j) * x[j];
            }
            px[i] = s;
        }

        // Compute x'*P*x (scalar)
        let xpx: f64 = x.iter().zip(px.iter()).map(|(xi, pxi)| xi * pxi).sum();

        // Gain: K = P*x / (lam + x'*P*x)
        let denom = lam + xpx;
        if denom.abs() < 1e-12 {
            return error;
        }

        let mut k = vec![0.0; n];
        for i in 0..n {
            k[i] = px[i] / denom;
        }

        // Compute delta_w = K * error, then clamp magnitude
        let mut delta_w = vec![0.0; n];
        let mut delta_norm_sq = 0.0;
        for i in 0..n {
            delta_w[i] = k[i] * error;
            delta_norm_sq += delta_w[i] * delta_w[i];
        }
        let delta_norm = delta_norm_sq.sqrt();
        if delta_norm > self.max_update_magnitude {
            let scale = self.max_update_magnitude / delta_norm;
            for i in 0..n {
                delta_w[i] *= scale;
            }
        }

        // Update weights: w = w + delta_w
        for i in 0..n {
            self.weights[i] += delta_w[i];
        }

        // Update P: P = (1/lam)(P - K * (P*x)')
        // Since px = P*x, this is P = (1/lam)(P - K * px')
        let inv_lam = 1.0 / lam;
        for i in 0..n {
            for j in 0..n {
                *self.p_mut(i, j) = inv_lam * (self.p(i, j) - k[i] * px[j]);
            }
        }

        // Enforce symmetry
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.p(i, j) + self.p(j, i));
                *self.p_mut(i, j) = avg;
                *self.p_mut(j, i) = avg;
            }
        }

        // Intercept update (simple EMA)
        self.intercept += 0.001 * error;

        error
    }

    /// Predict using current weights.
    fn predict(&self, x: Vec<f64>) -> f64 {
        if x.len() != self.n_features {
            return 0.0;
        }
        x.iter().zip(self.weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + self.intercept
    }

    /// Predict for a batch of samples (list of lists).
    fn predict_batch(&self, rows: Vec<Vec<f64>>) -> Vec<f64> {
        rows.iter()
            .map(|x| {
                if x.len() != self.n_features {
                    0.0
                } else {
                    x.iter().zip(self.weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + self.intercept
                }
            })
            .collect()
    }

    /// Reset to the original static weights.
    fn reset_to_static(&mut self) {
        if let Some(ref sw) = self.static_weights {
            self.weights = sw.clone();
            self.intercept = self.static_intercept;
        }
        self.reset_p();
        self.n_updates = 0;
    }

    /// L2 distance between current and static weights.
    #[getter]
    fn weight_drift(&self) -> f64 {
        match &self.static_weights {
            Some(sw) => {
                let sum_sq: f64 = self.weights.iter().zip(sw.iter())
                    .map(|(w, s)| (w - s) * (w - s))
                    .sum();
                sum_sq.sqrt()
            }
            None => 0.0,
        }
    }

    #[getter]
    fn n_updates(&self) -> u64 {
        self.n_updates
    }

    #[getter]
    fn weights_list(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[getter]
    fn intercept(&self) -> f64 {
        self.intercept
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.n_features
    }

    /// Stats dict matching Python OnlineRidge.stats.
    fn stats(&self) -> std::collections::HashMap<String, f64> {
        let mut m = std::collections::HashMap::new();
        m.insert("n_updates".into(), self.n_updates as f64);
        m.insert("weight_drift".into(), self.weight_drift());
        m.insert("forgetting_factor".into(), self.forgetting_factor);
        let w_norm: f64 = self.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        m.insert("w_norm".into(), w_norm);
        m.insert("intercept".into(), self.intercept);
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "RustOnlineRidge(n_features={}, n_updates={}, drift={:.6})",
            self.n_features, self.n_updates, self.weight_drift()
        )
    }
}
