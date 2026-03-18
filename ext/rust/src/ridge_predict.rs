// ridge_predict.rs — Pure Rust Ridge (linear) model prediction.
//
// Loads coefficients + intercept from JSON, predicts via dot product.
// Used by the standalone binary trader for Ridge 60% + LightGBM 40% ensemble.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::Deserialize;

/// JSON schema for exported Ridge model
#[derive(Deserialize, Clone, Debug)]
pub struct RidgeModelJson {
    pub format: String,           // "ridge"
    pub features: Vec<String>,
    pub num_features: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

/// Compiled Ridge model (ready for prediction)
#[derive(Clone, Debug)]
pub struct RidgeModel {
    pub features: Vec<String>,
    pub coefs: Vec<f64>,
    pub intercept: f64,
}

impl RidgeModel {
    /// Load from JSON string
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let schema: RidgeModelJson = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse Ridge JSON: {}", e))?;

        if schema.coefficients.len() != schema.num_features {
            return Err(format!(
                "Coefficient count {} != num_features {}",
                schema.coefficients.len(),
                schema.num_features
            ));
        }

        Ok(Self {
            features: schema.features,
            coefs: schema.coefficients,
            intercept: schema.intercept,
        })
    }

    /// Predict: intercept + sum(coef_i * x_i)
    /// NaN features are treated as 0.0 (neutral)
    pub fn predict(&self, feature_values: &[f64]) -> f64 {
        let mut sum = self.intercept;
        for (i, &coef) in self.coefs.iter().enumerate() {
            let val = if i < feature_values.len() {
                let v = feature_values[i];
                if v.is_finite() { v } else { 0.0 }
            } else {
                0.0
            };
            sum += coef * val;
        }
        sum
    }

    /// Predict from a feature name→value map
    pub fn predict_map(&self, features: &std::collections::HashMap<String, f64>) -> f64 {
        let mut sum = self.intercept;
        for (i, name) in self.features.iter().enumerate() {
            let val = features.get(name).copied().unwrap_or(0.0);
            let val = if val.is_finite() { val } else { 0.0 };
            sum += self.coefs[i] * val;
        }
        sum
    }
}

/// PyO3 wrapper for Ridge model
#[cfg(feature = "python")]
#[pyclass(name = "RustRidgePredictor")]
pub struct RustRidgePredictor {
    model: RidgeModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl RustRidgePredictor {
    /// Load from a JSON file path
    #[new]
    fn new(json_path: &str) -> PyResult<Self> {
        let json_str = std::fs::read_to_string(json_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        let model = RidgeModel::from_json(&json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { model })
    }

    /// Predict from a list of feature values (ordered same as features)
    fn predict(&self, values: Vec<f64>) -> f64 {
        self.model.predict(&values)
    }

    /// Predict from a dict of feature_name → value
    fn predict_dict(&self, features: std::collections::HashMap<String, f64>) -> f64 {
        self.model.predict_map(&features)
    }

    /// Get feature names
    fn feature_names(&self) -> Vec<String> {
        self.model.features.clone()
    }

    fn num_features(&self) -> usize {
        self.model.coefs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_predict() {
        let model = RidgeModel {
            features: vec!["a".into(), "b".into()],
            coefs: vec![0.5, -0.3],
            intercept: 1.0,
        };
        // 1.0 + 0.5*2.0 + (-0.3)*3.0 = 1.0 + 1.0 - 0.9 = 1.1
        let pred = model.predict(&[2.0, 3.0]);
        assert!((pred - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_ridge_nan_handling() {
        let model = RidgeModel {
            features: vec!["a".into(), "b".into()],
            coefs: vec![0.5, -0.3],
            intercept: 1.0,
        };
        // NaN treated as 0: 1.0 + 0.5*0.0 + (-0.3)*3.0 = 0.1
        let pred = model.predict(&[f64::NAN, 3.0]);
        assert!((pred - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_ridge_from_json() {
        let json = r#"{
            "format": "ridge",
            "features": ["x1", "x2"],
            "num_features": 2,
            "coefficients": [1.0, 2.0],
            "intercept": 0.5
        }"#;
        let model = RidgeModel::from_json(json).unwrap();
        assert_eq!(model.features.len(), 2);
        assert!((model.predict(&[1.0, 1.0]) - 3.5).abs() < 1e-10);
    }
}
