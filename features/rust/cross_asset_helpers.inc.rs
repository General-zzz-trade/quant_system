// cross_asset_helpers.inc.rs — Helper types for cross-asset computation.
// AssetState, EMA, RollingStats, PairState, utility fns.
// Included by cross_asset.rs via include!().

// ── Per-asset state ─────────────────────────────────────────

struct AssetState {
    close_history: VecDeque<f64>, // maxlen 70
    last_funding_rate: Option<f64>,
    bar_count: usize,
    high_history: VecDeque<f64>, // maxlen 25
    low_history: VecDeque<f64>,  // maxlen 25
    rsi_gain_ema: Option<f64>,
    rsi_loss_ema: Option<f64>,
    ema_12: Option<f64>,
    ema_26: Option<f64>,
}

impl AssetState {
    fn new() -> Self {
        Self {
            close_history: VecDeque::with_capacity(71),
            last_funding_rate: None,
            bar_count: 0,
            high_history: VecDeque::with_capacity(26),
            low_history: VecDeque::with_capacity(26),
            rsi_gain_ema: None,
            rsi_loss_ema: None,
            ema_12: None,
            ema_26: None,
        }
    }

    fn push(&mut self, close: f64, funding_rate: Option<f64>, high: Option<f64>, low: Option<f64>) {
        let prev_close = self.close_history.back().copied();
        if self.close_history.len() >= 70 {
            self.close_history.pop_front();
        }
        self.close_history.push_back(close);
        self.bar_count += 1;

        if let Some(fr) = funding_rate {
            self.last_funding_rate = Some(fr);
        }
        if let Some(h) = high {
            if self.high_history.len() >= 25 {
                self.high_history.pop_front();
            }
            self.high_history.push_back(h);
        }
        if let Some(l) = low {
            if self.low_history.len() >= 25 {
                self.low_history.pop_front();
            }
            self.low_history.push_back(l);
        }

        // RSI EMAs
        if let Some(prev) = prev_close {
            let delta = close - prev;
            let gain = delta.max(0.0);
            let loss = (-delta).max(0.0);
            match self.rsi_gain_ema {
                None => {
                    self.rsi_gain_ema = Some(gain);
                    self.rsi_loss_ema = Some(loss);
                }
                Some(ge) => {
                    let alpha = 1.0 / 14.0;
                    self.rsi_gain_ema = Some(alpha * gain + (1.0 - alpha) * ge);
                    self.rsi_loss_ema = Some(alpha * loss + (1.0 - alpha) * self.rsi_loss_ema.unwrap());
                }
            }
        }

        // MACD EMAs
        match self.ema_12 {
            None => {
                self.ema_12 = Some(close);
                self.ema_26 = Some(close);
            }
            Some(e12) => {
                self.ema_12 = Some((2.0 / 13.0) * close + (11.0 / 13.0) * e12);
                self.ema_26 = Some((2.0 / 27.0) * close + (25.0 / 27.0) * self.ema_26.unwrap());
            }
        }
    }

    fn ret(&self, lag: usize) -> Option<f64> {
        let n = self.close_history.len();
        if n <= lag {
            return None;
        }
        let base = self.close_history[n - 1 - lag];
        if base == 0.0 {
            return None;
        }
        Some((self.close_history[n - 1] - base) / base)
    }

    fn rsi_14(&self) -> Option<f64> {
        if self.bar_count < 15 {
            return None;
        }
        let ge = self.rsi_gain_ema?;
        let le = self.rsi_loss_ema?;
        if le < 1e-12 {
            return Some(100.0);
        }
        let rs = ge / le;
        Some(100.0 - 100.0 / (1.0 + rs))
    }

    fn macd_line(&self) -> Option<f64> {
        if self.bar_count < 27 {
            return None;
        }
        let e12 = self.ema_12?;
        let e26 = self.ema_26?;
        Some(e12 - e26)
    }

    fn mean_reversion_20(&self) -> Option<f64> {
        let n = self.close_history.len();
        if n < 20 {
            return None;
        }
        let start = n - 20;
        let sum: f64 = self.close_history.iter().skip(start).sum();
        let sma = sum / 20.0;
        if sma == 0.0 {
            return None;
        }
        Some((self.close_history[n - 1] - sma) / sma)
    }

    fn atr_norm_14(&self) -> Option<f64> {
        let nh = self.high_history.len();
        let nl = self.low_history.len();
        let nc = self.close_history.len();
        if nh < 14 || nl < 14 || nc < 15 {
            return None;
        }
        let mut atr_sum = 0.0;
        for i in 0..14 {
            let hi = self.high_history[nh - 14 + i];
            let lo = self.low_history[nl - 14 + i];
            let prev_c = self.close_history[nc - 15 + i];
            let tr = (hi - lo).max((hi - prev_c).abs()).max((lo - prev_c).abs());
            atr_sum += tr;
        }
        let atr = atr_sum / 14.0;
        let cur = *self.close_history.back()?;
        if cur == 0.0 {
            return None;
        }
        Some(atr / cur)
    }

    fn bb_width_20(&self) -> Option<f64> {
        let n = self.close_history.len();
        if n < 20 {
            return None;
        }
        let start = n - 20;
        let sum: f64 = self.close_history.iter().skip(start).sum();
        let sma = sum / 20.0;
        if sma == 0.0 {
            return None;
        }
        let var: f64 = self.close_history.iter().skip(start)
            .map(|x| (x - sma) * (x - sma))
            .sum::<f64>() / 20.0;
        let std = var.sqrt();
        Some((4.0 * std) / sma)
    }
}

// ── Per-pair state ──────────────────────────────────────────

struct EMA {
    alpha: f64,
    value: f64,
    n: usize,
}

impl EMA {
    fn new(span: usize) -> Self {
        Self {
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

    fn ready(&self) -> bool {
        self.n > 0
    }
}

struct RollingStats {
    window: VecDeque<f64>,
    maxlen: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    fn new(maxlen: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(maxlen + 1),
            maxlen,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, x: f64) {
        self.sum += x;
        self.sum_sq += x * x;
        self.window.push_back(x);
        if self.window.len() > self.maxlen {
            let old = self.window.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }
    }

    fn full(&self) -> bool {
        self.window.len() >= self.maxlen
    }

    fn mean(&self) -> f64 {
        if self.window.is_empty() { 0.0 } else { self.sum / self.window.len() as f64 }
    }

    fn std(&self) -> f64 {
        let n = self.window.len() as f64;
        if n < 2.0 { return 0.0; }
        let var = (self.sum_sq / n) - (self.sum / n).powi(2);
        var.max(0.0).sqrt()
    }
}

struct PairState {
    sym_rets_30: VecDeque<f64>,
    bench_rets_30: VecDeque<f64>,
    sym_rets_60: VecDeque<f64>,
    bench_rets_60: VecDeque<f64>,
    sym_cum_20: VecDeque<f64>,
    bench_cum_20: VecDeque<f64>,
    spread_window_20: RollingStats,
    funding_diff_ema: EMA,
    last_funding_diff: Option<f64>,
}

impl PairState {
    fn new() -> Self {
        Self {
            sym_rets_30: VecDeque::with_capacity(31),
            bench_rets_30: VecDeque::with_capacity(31),
            sym_rets_60: VecDeque::with_capacity(61),
            bench_rets_60: VecDeque::with_capacity(61),
            sym_cum_20: VecDeque::with_capacity(21),
            bench_cum_20: VecDeque::with_capacity(21),
            spread_window_20: RollingStats::new(20),
            funding_diff_ema: EMA::new(8),
            last_funding_diff: None,
        }
    }

    fn push(&mut self, sym_ret: f64, bench_ret: f64, funding_diff: Option<f64>) {
        let beta = beta_from_deques(&self.sym_rets_30, &self.bench_rets_30, 30);

        push_capped(&mut self.sym_rets_30, sym_ret, 30);
        push_capped(&mut self.bench_rets_30, bench_ret, 30);
        push_capped(&mut self.sym_rets_60, sym_ret, 60);
        push_capped(&mut self.bench_rets_60, bench_ret, 60);
        push_capped(&mut self.sym_cum_20, sym_ret, 20);
        push_capped(&mut self.bench_cum_20, bench_ret, 20);

        if let Some(b) = beta {
            let spread = sym_ret - b * bench_ret;
            self.spread_window_20.push(spread);
        }

        if let Some(fd) = funding_diff {
            self.last_funding_diff = Some(fd);
            self.funding_diff_ema.push(fd);
        }
    }
}

fn push_capped(deque: &mut VecDeque<f64>, val: f64, maxlen: usize) {
    if deque.len() >= maxlen {
        deque.pop_front();
    }
    deque.push_back(val);
}

fn beta_from_deques(sym: &VecDeque<f64>, bench: &VecDeque<f64>, maxlen: usize) -> Option<f64> {
    let n = sym.len();
    if n < maxlen || n != bench.len() {
        return None;
    }
    let s_mean: f64 = sym.iter().sum::<f64>() / n as f64;
    let b_mean: f64 = bench.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_b = 0.0;
    for (s, b) in sym.iter().zip(bench.iter()) {
        cov += (s - s_mean) * (b - b_mean);
        var_b += (b - b_mean) * (b - b_mean);
    }
    cov /= n as f64;
    var_b /= n as f64;
    if var_b < 1e-20 {
        return None;
    }
    Some(cov / var_b)
}

fn pearson(x: &VecDeque<f64>, y: &VecDeque<f64>) -> Option<f64> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return None;
    }
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let mut vx = 0.0;
    let mut vy = 0.0;
    let mut cov = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        vx += dx * dx;
        vy += dy * dy;
        cov += dx * dy;
    }
    if vx < 1e-12 || vy < 1e-12 {
        return None;
    }
    Some(cov / (vx * vy).sqrt())
}

// ── Feature names ───────────────────────────────────────────

const FEATURE_NAMES: &[&str] = &[
    "btc_ret_1", "btc_ret_3", "btc_ret_6",
    "btc_ret_12", "btc_ret_24",
    "btc_rsi_14",
    "btc_macd_line",
    "btc_mean_reversion_20",
    "btc_atr_norm_14",
    "btc_bb_width_20",
    "rolling_beta_30", "rolling_beta_60",
    "relative_strength_20",
    "rolling_corr_30",
    "funding_diff", "funding_diff_ma8",
    "spread_zscore_20",
];
