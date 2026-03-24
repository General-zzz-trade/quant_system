// cursors.inc.rs — Included by engine.rs via include!() macro.
// ============================================================
// Schedule cursors
// ============================================================

struct StridedCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    current: f64,
}

impl StridedCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 2);
        for row in sched {
            if row.len() >= 2 {
                data.push(row[0]);
                data.push(row[1]);
            }
        }
        let actual_rows = data.len() / 2;
        StridedCursor {
            data,
            rows: actual_rows,
            idx: 0,
            current: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) -> f64 {
        while self.idx < self.rows && self.data[self.idx * 2] <= bar_ts {
            self.current = self.data[self.idx * 2 + 1];
            self.idx += 1;
        }
        self.current
    }
}

struct OnchainCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    flow_in: f64,
    flow_out: f64,
    supply: f64,
    addr: f64,
    tx: f64,
    hashrate: f64,
}

impl OnchainCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 7);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 7 {
                for j in 0..7 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        OnchainCursor {
            data,
            rows: actual_rows,
            idx: 0,
            flow_in: f64::NAN,
            flow_out: f64::NAN,
            supply: f64::NAN,
            addr: f64::NAN,
            tx: f64::NAN,
            hashrate: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 7] <= bar_ts {
            let base = self.idx * 7;
            self.flow_in = self.data[base + 1];
            self.flow_out = self.data[base + 2];
            self.supply = self.data[base + 3];
            self.addr = self.data[base + 4];
            self.tx = self.data[base + 5];
            self.hashrate = self.data[base + 6];
            self.idx += 1;
        }
    }
}

struct LiqCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    total_vol: f64,
    buy_vol: f64,
    sell_vol: f64,
}

impl LiqCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        LiqCursor {
            data,
            rows: actual_rows,
            idx: 0,
            total_vol: f64::NAN,
            buy_vol: f64::NAN,
            sell_vol: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            self.total_vol = self.data[base + 1];
            self.buy_vol = self.data[base + 2];
            self.sell_vol = self.data[base + 3];
            self.idx += 1;
        }
    }
}

struct MempoolCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    fastest_fee: f64,
    economy_fee: f64,
    mempool_size: f64,
}

impl MempoolCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        MempoolCursor {
            data,
            rows: actual_rows,
            idx: 0,
            fastest_fee: f64::NAN,
            economy_fee: f64::NAN,
            mempool_size: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            self.fastest_fee = self.data[base + 1];
            self.economy_fee = self.data[base + 2];
            self.mempool_size = self.data[base + 3];
            self.idx += 1;
        }
    }
}

struct MacroCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    dxy: f64,
    spx: f64,
    vix: f64,
    day: i64,
}

impl MacroCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        MacroCursor {
            data,
            rows: actual_rows,
            idx: 0,
            dxy: f64::NAN,
            spx: f64::NAN,
            vix: f64::NAN,
            day: -1,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            let ts_sec = (self.data[base] / 1000.0) as i64;
            self.day = ts_sec / 86400;
            if !self.data[base + 1].is_nan() {
                self.dxy = self.data[base + 1];
            }
            if !self.data[base + 2].is_nan() {
                self.spx = self.data[base + 2];
            }
            if !self.data[base + 3].is_nan() {
                self.vix = self.data[base + 3];
            }
            self.idx += 1;
        }
    }
}
