use pyo3::prelude::*;

/// O(1) rolling window with running sum and sum-of-squares.
#[pyclass]
pub struct RollingWindow {
    size: usize,
    head: usize,
    count: usize,
    sum: f64,
    sumsq: f64,
    buf: Vec<f64>,
}

#[pymethods]
impl RollingWindow {
    #[new]
    fn new(size: i32) -> PyResult<Self> {
        if size <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "size must be positive",
            ));
        }
        let sz = size as usize;
        Ok(Self {
            size: sz,
            head: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
            buf: vec![0.0; sz],
        })
    }

    fn push(&mut self, x: f64) {
        if self.count < self.size {
            self.buf[self.count] = x;
            self.count += 1;
        } else {
            let old = self.buf[self.head];
            self.sum -= old;
            self.sumsq -= old * old;
            self.buf[self.head] = x;
            self.head = (self.head + 1) % self.size;
        }
        self.sum += x;
        self.sumsq += x * x;
    }

    #[getter]
    fn full(&self) -> bool {
        self.count == self.size
    }

    #[getter]
    fn n(&self) -> i32 {
        self.count as i32
    }

    #[getter]
    fn size(&self) -> i32 {
        self.size as i32
    }

    #[getter]
    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(self.sum / self.count as f64)
    }

    #[getter]
    fn variance(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mu = self.sum / self.count as f64;
        let v = self.sumsq / self.count as f64 - mu * mu;
        Some(v.max(0.0))
    }

    #[getter]
    fn std(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }
}

/// O(1) rolling VWAP window.
#[pyclass]
pub struct VWAPWindow {
    size: usize,
    head: usize,
    count: usize,
    sum_pv_val: f64,
    sum_v_val: f64,
    prices: Vec<f64>,
    volumes: Vec<f64>,
}

#[pymethods]
impl VWAPWindow {
    #[new]
    fn new(size: i32) -> PyResult<Self> {
        if size <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "size must be positive",
            ));
        }
        let sz = size as usize;
        Ok(Self {
            size: sz,
            head: 0,
            count: 0,
            sum_pv_val: 0.0,
            sum_v_val: 0.0,
            prices: vec![0.0; sz],
            volumes: vec![0.0; sz],
        })
    }

    fn push(&mut self, price: f64, volume: f64) {
        if self.count < self.size {
            self.prices[self.count] = price;
            self.volumes[self.count] = volume;
            self.count += 1;
        } else {
            let old_p = self.prices[self.head];
            let old_v = self.volumes[self.head];
            self.sum_pv_val -= old_p * old_v;
            self.sum_v_val -= old_v;
            self.prices[self.head] = price;
            self.volumes[self.head] = volume;
            self.head = (self.head + 1) % self.size;
        }
        self.sum_pv_val += price * volume;
        self.sum_v_val += volume;
    }

    #[getter]
    fn full(&self) -> bool {
        self.count == self.size
    }

    #[getter]
    fn n(&self) -> i32 {
        self.count as i32
    }

    #[getter]
    fn size(&self) -> i32 {
        self.size as i32
    }

    #[getter]
    fn vwap(&self) -> Option<f64> {
        if self.count == 0 || self.sum_v_val <= 0.0 {
            return None;
        }
        Some(self.sum_pv_val / self.sum_v_val)
    }

    #[getter]
    fn sum_pv(&self) -> f64 {
        self.sum_pv_val
    }

    #[getter]
    fn sum_v(&self) -> f64 {
        self.sum_v_val
    }
}
