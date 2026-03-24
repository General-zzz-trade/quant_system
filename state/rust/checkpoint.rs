use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "RustCheckpointStore")]
pub struct RustCheckpointStore {
    items: HashMap<String, String>,
}

#[pymethods]
impl RustCheckpointStore {
    #[new]
    fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    fn save_latest(&mut self, run_id: &str, name: &str, payload_json: &str) {
        self.items
            .insert(make_key(run_id, name), payload_json.to_owned());
    }

    fn load_latest(&self, run_id: &str, name: &str) -> Option<String> {
        self.items.get(&make_key(run_id, name)).cloned()
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

fn make_key(run_id: &str, name: &str) -> String {
    format!("{}\x1f{}", run_id, name)
}
