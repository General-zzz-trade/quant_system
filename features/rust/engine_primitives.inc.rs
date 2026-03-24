// primitives.inc.rs — Included by engine.rs via include!() macro.
// ============================================================
// CircBuf — runtime-sized circular buffer
// ============================================================

pub(crate) struct CircBuf {
    pub(crate) buf: Vec<f64>,
    pub(crate) max_size: usize,
    pub(crate) count: usize,
    pub(crate) head: usize,
}

impl CircBuf {
    fn new(max_size: usize) -> Self {
        CircBuf {
            buf: vec![0.0; max_size],
            max_size,
            count: 0,
            head: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.count < self.max_size {
            self.buf[self.count] = x;
            self.count += 1;
        } else {
            self.buf[self.head] = x;
            self.head = (self.head + 1) % self.max_size;
        }
    }

    fn size(&self) -> usize {
        self.count
    }

    fn full(&self) -> bool {
        self.count == self.max_size
    }

    fn get(&self, i: usize) -> f64 {
        self.buf[(self.head + i) % self.max_size]
    }

    fn back(&self) -> f64 {
        if self.count < self.max_size {
            self.buf[self.count - 1]
        } else {
            self.buf[(self.head + self.max_size - 1) % self.max_size]
        }
    }

    fn back_n(&self, n: usize) -> f64 {
        if self.count < self.max_size {
            self.buf[self.count - 1 - n]
        } else {
            self.buf[(self.head + self.max_size - 1 - n) % self.max_size]
        }
    }

    fn sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.count {
            s += self.buf[(self.head + i) % self.max_size];
        }
        s
    }
}

// ============================================================
// RollingWindow — online mean/std via sum/sumsq
// ============================================================

pub(crate) struct RollingWindow {
    pub(crate) buf: Vec<f64>,
    pub(crate) size: usize,
    pub(crate) head: usize,
    pub(crate) count: usize,
    pub(crate) sum: f64,
    pub(crate) sumsq: f64,
}

impl RollingWindow {
    fn new(size: usize) -> Self {
        RollingWindow {
            buf: vec![0.0; size],
            size,
            head: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
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

    fn full(&self) -> bool {
        self.count == self.size
    }

    fn n(&self) -> usize {
        self.count
    }

    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }

    fn std_dev(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mu = self.sum / self.count as f64;
        let v = self.sumsq / self.count as f64 - mu * mu;
        Some(v.max(0.0).sqrt())
    }
}

// ============================================================
// EMAState
// ============================================================

pub(crate) struct EMAState {
    pub(crate) alpha: f64,
    pub(crate) value: f64,
    pub(crate) n: i32,
}

impl EMAState {
    fn new(span: i32) -> Self {
        EMAState {
            alpha: 2.0 / (span as f64 + 1.0),
            value: 0.0,
            n: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.n == 0 {
            self.value = x;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.n += 1;
    }

    fn ready(&self, span: i32) -> bool {
        self.n >= span
    }

    fn get_value(&self) -> f64 {
        if self.n > 0 {
            self.value
        } else {
            f64::NAN
        }
    }
}

// ============================================================
// RSIState
// ============================================================

pub(crate) struct RSIState {
    pub(crate) period: i32,
    pub(crate) avg_gain: f64,
    pub(crate) avg_loss: f64,
    pub(crate) n: i32,
    pub(crate) prev_close: f64,
    pub(crate) init_gains: f64,
    pub(crate) init_losses: f64,
}

impl RSIState {
    fn new(period: i32) -> Self {
        RSIState {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            n: 0,
            prev_close: f64::NAN,
            init_gains: 0.0,
            init_losses: 0.0,
        }
    }

    fn push(&mut self, close: f64) {
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return;
        }
        let change = close - self.prev_close;
        self.prev_close = close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.n += 1;

        if self.n <= self.period {
            self.init_gains += gain;
            self.init_losses += loss;
            if self.n == self.period {
                self.avg_gain = self.init_gains / self.period as f64;
                self.avg_loss = self.init_losses / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.avg_gain = (self.avg_gain * (p - 1.0) + gain) / p;
            self.avg_loss = (self.avg_loss * (p - 1.0) + loss) / p;
        }
    }

    fn get_value(&self) -> f64 {
        if self.n < self.period {
            return f64::NAN;
        }
        if self.avg_loss == 0.0 {
            return 100.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

// ============================================================
// ATRState
// ============================================================

pub(crate) struct ATRState {
    pub(crate) period: i32,
    pub(crate) atr: f64,
    pub(crate) n: i32,
    pub(crate) prev_close: f64,
    pub(crate) init_sum: f64,
}

impl ATRState {
    fn new(period: i32) -> Self {
        ATRState {
            period,
            atr: 0.0,
            n: 0,
            prev_close: f64::NAN,
            init_sum: 0.0,
        }
    }

    fn push(&mut self, high: f64, low: f64, close: f64) {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let a = high - low;
            let b = (high - self.prev_close).abs();
            let c = (low - self.prev_close).abs();
            a.max(b).max(c)
        };
        self.prev_close = close;
        self.n += 1;

        if self.n <= self.period {
            self.init_sum += tr;
            if self.n == self.period {
                self.atr = self.init_sum / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.atr = (self.atr * (p - 1.0) + tr) / p;
        }
    }

    fn get_value(&self) -> f64 {
        if self.n >= self.period {
            self.atr
        } else {
            f64::NAN
        }
    }
}

// ============================================================
// zscore_buf helper
// ============================================================

fn zscore_buf(data: &[f64], min_std: f64) -> f64 {
    let count = data.len();
    if count == 0 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &v in data {
        sum += v;
    }
    let mean = sum / count as f64;
    let mut var = 0.0;
    for &v in data {
        let d = v - mean;
        var += d * d;
    }
    var /= count as f64;
    let std = var.sqrt();
    if std <= min_std {
        return 0.0;
    }
    (data[count - 1] - mean) / std
}

fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
