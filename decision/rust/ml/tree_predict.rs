// tree_predict.rs — Pure Rust decision tree traversal for LightGBM/XGBoost models.
//
// Loads model JSON (exported by scripts/export_model_to_json.py) and predicts
// without Python ML libraries. Supports NaN handling (default_left/right).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;

// ── JSON schema ──

#[derive(Deserialize, Clone)]
pub(crate) struct ModelJson {
    pub(crate) format: String,
    pub(crate) features: Vec<String>,
    pub(crate) num_features: usize,
    pub(crate) num_trees: usize,
    #[serde(default)]
    pub(crate) is_classifier: bool,
    #[serde(default)]
    pub(crate) base_score: f64,
    pub(crate) trees: Vec<TreeJson>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct TreeJson {
    #[serde(default = "default_shrinkage")]
    pub(crate) shrinkage: f64,
    pub(crate) nodes: Vec<NodeJson>,
}

fn default_shrinkage() -> f64 { 1.0 }

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub(crate) enum NodeJson {
    #[serde(rename = "split")]
    Split {
        feature: usize,
        threshold: f64,
        default_left: bool,
        #[serde(default)]
        nan_as_zero: bool,
        left: usize,
        right: usize,
    },
    #[serde(rename = "leaf")]
    Leaf { value: f64 },
}

// ── Compiled model (optimized for prediction) ──

#[derive(Clone)]
pub(crate) enum Node {
    Split {
        feature: u16,
        threshold: f64,
        default_left: bool,
        nan_as_zero: bool,
        left: u32,
        right: u32,
    },
    Leaf(f64),
}

#[derive(Clone)]
pub(crate) struct Tree {
    pub(crate) shrinkage: f64,
    pub(crate) nodes: Vec<Node>,
}

impl Tree {
    pub(crate) fn predict(&self, features: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            match &self.nodes[idx] {
                Node::Leaf(v) => return v * self.shrinkage,
                Node::Split { feature, threshold, default_left, nan_as_zero, left, right } => {
                    let mut val = features[*feature as usize];
                    if val.is_nan() {
                        if *nan_as_zero {
                            // Feature had no NaN in training: LightGBM replaces NaN with 0.0
                            val = 0.0;
                        } else {
                            // Feature had NaN in training: use learned default direction
                            idx = if *default_left { *left as usize } else { *right as usize };
                            continue;
                        }
                    }
                    if val <= *threshold {
                        idx = *left as usize;
                    } else {
                        idx = *right as usize;
                    }
                }
            }
        }
    }
}

// ── PyO3 wrapper ──

#[pyclass]
pub struct RustTreePredictor {
    pub(crate) trees: Vec<Tree>,
    pub(crate) features: Vec<String>,
    pub(crate) num_features: usize,
    pub(crate) is_classifier: bool,
    pub(crate) base_score: f64,
    pub(crate) format: String,
}

#[pymethods]
impl RustTreePredictor {
    /// Load model from JSON file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot read {}: {}", path, e)))?;
        let model_json: ModelJson = serde_json::from_str(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let trees: Vec<Tree> = model_json.trees.iter().map(|tj| {
            let nodes: Vec<Node> = tj.nodes.iter().map(|nj| match nj {
                NodeJson::Split { feature, threshold, default_left, nan_as_zero, left, right } => {
                    Node::Split {
                        feature: *feature as u16,
                        threshold: *threshold,
                        default_left: *default_left,
                        nan_as_zero: *nan_as_zero,
                        left: *left as u32,
                        right: *right as u32,
                    }
                }
                NodeJson::Leaf { value } => Node::Leaf(*value),
            }).collect();
            Tree { shrinkage: tj.shrinkage, nodes }
        }).collect();

        Ok(Self {
            format: model_json.format,
            features: model_json.features,
            num_features: model_json.num_features,
            is_classifier: model_json.is_classifier,
            base_score: model_json.base_score,
            trees,
        })
    }

    /// Load model from JSON string.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let model_json: ModelJson = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let trees: Vec<Tree> = model_json.trees.iter().map(|tj| {
            let nodes: Vec<Node> = tj.nodes.iter().map(|nj| match nj {
                NodeJson::Split { feature, threshold, default_left, nan_as_zero, left, right } => {
                    Node::Split {
                        feature: *feature as u16,
                        threshold: *threshold,
                        default_left: *default_left,
                        nan_as_zero: *nan_as_zero,
                        left: *left as u32,
                        right: *right as u32,
                    }
                }
                NodeJson::Leaf { value } => Node::Leaf(*value),
            }).collect();
            Tree { shrinkage: tj.shrinkage, nodes }
        }).collect();

        Ok(Self {
            format: model_json.format,
            features: model_json.features,
            num_features: model_json.num_features,
            is_classifier: model_json.is_classifier,
            base_score: model_json.base_score,
            trees,
        })
    }

    /// Predict from a feature dict {name: value}.
    /// Returns raw prediction (regression) or probability-centered (classifier).
    fn predict_dict(&self, features: &Bound<'_, PyDict>) -> PyResult<f64> {
        let mut x = vec![f64::NAN; self.num_features];
        for (i, name) in self.features.iter().enumerate() {
            if let Some(val) = features.get_item(name)? {
                if let Ok(v) = val.extract::<f64>() {
                    x[i] = v;
                }
            }
        }
        Ok(self.predict_raw(&x))
    }

    /// Predict from a flat feature array (must match feature order).
    fn predict_array(&self, features: Vec<f64>) -> PyResult<f64> {
        if features.len() != self.num_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} features, got {}", self.num_features, features.len())
            ));
        }
        Ok(self.predict_raw(&features))
    }

    /// Get feature names.
    fn feature_names(&self) -> Vec<String> {
        self.features.clone()
    }

    /// Get model info.
    fn info(&self) -> String {
        format!(
            "{}:{} trees={} features={} classifier={}",
            self.format,
            if self.format == "xgb" { format!(" base_score={:.6}", self.base_score) } else { String::new() },
            self.trees.len(),
            self.num_features,
            self.is_classifier,
        )
    }

    /// Number of trees.
    fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Is classifier model.
    fn is_classifier(&self) -> bool {
        self.is_classifier
    }
}

impl RustTreePredictor {
    pub(crate) fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut sum = if self.format == "xgb" { self.base_score } else { 0.0 };
        for tree in &self.trees {
            sum += tree.predict(features);
        }
        if self.is_classifier {
            // Sigmoid for classifier, then center at 0
            let prob = 1.0 / (1.0 + (-sum).exp());
            prob - 0.5
        } else {
            sum
        }
    }
}
