// types.inc.rs — Types, state structs, and order logic for quant_trader binary.
// ── Prometheus metrics (shared with HTTP server) ──

struct MetricsSnapshot {
    equity: f64,
    peak_equity: f64,
    drawdown_pct: f64,
    balance: f64,
    tick_count: u64,
    micro_orders: u64,
    uptime_secs: u64,
    positions: HashMap<String, f64>,    // symbol → qty
    ml_scores: HashMap<String, f64>,    // symbol → last ml_score
    raw_scores: HashMap<String, f64>,   // symbol → last raw_score
}

impl MetricsSnapshot {
    fn new() -> Self {
        Self {
            equity: 0.0,
            peak_equity: 0.0,
            drawdown_pct: 0.0,
            balance: 0.0,
            tick_count: 0,
            micro_orders: 0,
            uptime_secs: 0,
            positions: HashMap::new(),
            ml_scores: HashMap::new(),
            raw_scores: HashMap::new(),
        }
    }

    fn to_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("# HELP quant_equity Current equity in USDT\n");
        out.push_str("# TYPE quant_equity gauge\n");
        out.push_str(&format!("quant_equity {:.2}\n", self.equity));
        out.push_str("# HELP quant_peak_equity High water mark equity\n");
        out.push_str("# TYPE quant_peak_equity gauge\n");
        out.push_str(&format!("quant_peak_equity {:.2}\n", self.peak_equity));
        out.push_str("# HELP quant_drawdown_pct Current drawdown percentage\n");
        out.push_str("# TYPE quant_drawdown_pct gauge\n");
        out.push_str(&format!("quant_drawdown_pct {:.4}\n", self.drawdown_pct));
        out.push_str("# HELP quant_balance Wallet balance in USDT\n");
        out.push_str("# TYPE quant_balance gauge\n");
        out.push_str(&format!("quant_balance {:.2}\n", self.balance));
        out.push_str("# HELP quant_ticks_total Total bars processed\n");
        out.push_str("# TYPE quant_ticks_total counter\n");
        out.push_str(&format!("quant_ticks_total {}\n", self.tick_count));
        out.push_str("# HELP quant_micro_orders_total Micro-alpha orders submitted\n");
        out.push_str("# TYPE quant_micro_orders_total counter\n");
        out.push_str(&format!("quant_micro_orders_total {}\n", self.micro_orders));
        out.push_str("# HELP quant_uptime_seconds Process uptime\n");
        out.push_str("# TYPE quant_uptime_seconds counter\n");
        out.push_str(&format!("quant_uptime_seconds {}\n", self.uptime_secs));
        out.push_str("# HELP quant_position Current position quantity\n");
        out.push_str("# TYPE quant_position gauge\n");
        for (sym, qty) in &self.positions {
            out.push_str(&format!("quant_position{{symbol=\"{}\"}} {:.6}\n", sym, qty));
        }
        out.push_str("# HELP quant_ml_score Last ML score\n");
        out.push_str("# TYPE quant_ml_score gauge\n");
        for (sym, score) in &self.ml_scores {
            out.push_str(&format!("quant_ml_score{{symbol=\"{}\"}} {:.6}\n", sym, score));
        }
        out.push_str("# HELP quant_raw_score Last raw prediction score\n");
        out.push_str("# TYPE quant_raw_score gauge\n");
        for (sym, score) in &self.raw_scores {
            out.push_str(&format!("quant_raw_score{{symbol=\"{}\"}} {:.6}\n", sym, score));
        }
        out
    }
}

// ── Exchange state (synced from user data stream) ──

struct ExchangePosition {
    qty: f64,      // positive = long, negative = short
    notional: f64, // abs(qty * entry_price)
}

struct ExchangeState {
    /// USDT balance (wallet balance)
    balance: f64,
    /// Per-symbol positions from exchange
    positions: HashMap<String, ExchangePosition>,
    /// Whether we've received at least one update
    synced: bool,
    /// Last update timestamp
    last_update: std::time::Instant,
}

impl ExchangeState {
    fn new(starting_balance: f64) -> Self {
        Self {
            balance: starting_balance,
            positions: HashMap::new(),
            synced: false,
            last_update: std::time::Instant::now(),
        }
    }

    fn gross_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.qty.abs()).sum::<f64>()
    }

    fn position_qty(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }

    fn position_notional(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.notional).unwrap_or(0.0)
    }
}

// ── Pending order tracking ──

struct PendingOrder {
    symbol: String,
    side: String,
    qty: f64,
    #[allow(dead_code)]
    sent_at: std::time::Instant,
}

// ── Per-symbol state for tick-by-tick decisions ──

struct PerSymbolState {
    /// Last ML score from kline bar close
    last_ml_score: f64,
    /// Last price seen (from aggTrade or kline)
    last_price: f64,
    /// Timestamp of last order submission (for throttling)
    last_order_time: std::time::Instant,
    /// Total aggTrade count (for logging)
    trade_count: u64,
}

// ── Trade aggregator: builds bars from aggTrade stream ──

struct TradeAggregator {
    /// Current bar bucket (floor to interval_ms)
    bucket_ts: i64,
    interval_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    trade_count: u32,
}

impl TradeAggregator {
    fn new(interval_ms: i64) -> Self {
        Self {
            bucket_ts: 0,
            interval_ms,
            open: 0.0,
            high: f64::MIN,
            low: f64::MAX,
            close: 0.0,
            volume: 0.0,
            trade_count: 0,
        }
    }

    /// Push a trade. Returns Some(completed bar) if a bar just closed.
    fn push(&mut self, ts_ms: i64, price: f64, qty: f64) -> Option<AggBar> {
        let bucket = (ts_ms / self.interval_ms) * self.interval_ms;

        if self.bucket_ts == 0 {
            // First trade
            self.start_new_bar(bucket, price);
            self.volume += qty;
            self.trade_count += 1;
            return None;
        }

        if bucket > self.bucket_ts {
            // New bucket → close previous bar
            let bar = AggBar {
                ts_ms: self.bucket_ts,
                open: self.open,
                high: self.high,
                low: self.low,
                close: self.close,
                volume: self.volume,
            };
            self.start_new_bar(bucket, price);
            self.volume += qty;
            self.trade_count += 1;
            return Some(bar);
        }

        // Same bucket
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
        self.volume += qty;
        self.trade_count += 1;
        None
    }

    fn start_new_bar(&mut self, bucket_ts: i64, price: f64) {
        self.bucket_ts = bucket_ts;
        self.open = price;
        self.high = price;
        self.low = price;
        self.close = price;
        self.volume = 0.0;
        self.trade_count = 0;
    }
}

struct AggBar {
    ts_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

// ── Risk params (extracted from config for helper function) ──

struct RiskParams {
    fraction: f64,
    max_leverage: f64,
    max_drawdown_pct: f64,
    max_position_notional: Option<f64>,
    circuit_breaker_threshold: u32,
    min_order_interval: std::time::Duration,
}

/// Check risk gates and submit order if allowed.
/// Returns true if order was submitted.
/// Uses ExchangeState (from user data stream) for position/balance checks when synced,
/// falls back to tick_processor internal state otherwise.
#[allow(clippy::too_many_arguments)]
fn try_submit_order(
    symbol: &str,
    score: f64,
    price: f64,
    source: &str,
    processor: &RustTickProcessor,
    sym_state: &mut PerSymbolState,
    exchange_state: &ExchangeState,
    risk: &RiskParams,
    lot_size: f64,
    peak_equity: f64,
    order_gateway: &RustWsOrderGateway,
    order_tx: &mpsc::Sender<String>,
    pending_orders: &Arc<tokio::sync::Mutex<HashMap<String, PendingOrder>>>,
    order_failure_count: &Arc<AtomicU32>,
    dry_run: bool,
    micro_sig: Option<&MicroAlphaSignal>,
) -> bool {
    if score.abs() < 1e-9 {
        return false;
    }

    let desired_side = if score > 0.0 { "BUY" } else { "SELL" };

    // Check position direction — prefer exchange state when synced
    let current_qty_f64 = if exchange_state.synced {
        exchange_state.position_qty(symbol)
    } else {
        processor
            .get_position_native(symbol)
            .map(|p| p.qty as f64 / SCALE)
            .unwrap_or(0.0)
    };

    let need_order = (desired_side == "BUY" && current_qty_f64 <= 0.0)
        || (desired_side == "SELL" && current_qty_f64 >= 0.0);
    if !need_order {
        return false;
    }

    // Throttle check
    if sym_state.last_order_time.elapsed() < risk.min_order_interval {
        return false;
    }

    // Circuit breaker
    if order_failure_count.load(Ordering::Relaxed) >= risk.circuit_breaker_threshold {
        warn!(
            symbol = symbol,
            failures = order_failure_count.load(Ordering::Relaxed),
            "Circuit breaker OPEN — skipping order"
        );
        return false;
    }

    // Get equity — prefer exchange state
    let equity = if exchange_state.synced {
        exchange_state.balance
    } else {
        processor.account_native().balance as f64 / SCALE
    };

    // Drawdown check
    let drawdown_pct = if peak_equity > 0.0 {
        (peak_equity - equity) / peak_equity
    } else {
        0.0
    };
    if drawdown_pct > risk.max_drawdown_pct {
        warn!(
            symbol = symbol,
            drawdown = format!("{:.2}%", drawdown_pct * 100.0),
            limit = format!("{:.2}%", risk.max_drawdown_pct * 100.0),
            "RISK: drawdown exceeded"
        );
        return false;
    }

    // Position sizing
    let qty = rust_fixed_fraction_qty(equity, price, risk.fraction, score.abs(), lot_size);
    if qty < lot_size {
        return false;
    }

    let order_notional = qty * price;

    // Leverage check — prefer exchange state
    let gross_exposure = if exchange_state.synced {
        exchange_state.gross_exposure() * price // approximate: sum of abs qty * current price
    } else {
        let port = processor.portfolio_native();
        port.gross_exposure.parse::<f64>().unwrap_or(0.0)
    };
    let projected_leverage = if equity > 0.0 {
        (gross_exposure + order_notional) / equity
    } else {
        f64::MAX
    };
    if projected_leverage > risk.max_leverage {
        warn!(
            symbol = symbol,
            leverage = format!("{:.2}x", projected_leverage),
            limit = format!("{:.2}x", risk.max_leverage),
            "RISK: leverage exceeded"
        );
        return false;
    }

    // Max position notional
    if let Some(max_notional) = risk.max_position_notional {
        let current_pos_notional = if exchange_state.synced {
            exchange_state.position_notional(symbol)
        } else {
            processor
                .get_position_native(symbol)
                .map(|p| (p.qty as f64 / SCALE) * price)
                .unwrap_or(0.0)
                .abs()
        };
        if current_pos_notional + order_notional > max_notional {
            warn!(
                symbol = symbol,
                projected = format!("{:.2}", current_pos_notional + order_notional),
                limit = format!("{:.2}", max_notional),
                "RISK: position notional exceeded"
            );
            return false;
        }
    }

    // Build and send order
    let qty_str = format_qty(qty, lot_size);
    let (msg, req_id) = order_gateway.build_order_message_native(
        symbol,
        desired_side,
        "MARKET",
        Some(&qty_str),
        None,
        None,
        None,
        None,
    );

    let micro_score_str = micro_sig
        .map(|s| format!("{:.4}", s.micro_score))
        .unwrap_or_else(|| "n/a".to_string());

    if dry_run {
        info!(
            symbol = symbol,
            side = desired_side,
            qty = %qty_str,
            notional = format!("{:.2}", order_notional),
            leverage = format!("{:.2}x", projected_leverage),
            equity = format!("{:.2}", equity),
            score = format!("{:.4}", score),
            micro = micro_score_str,
            source = source,
            "DRY-RUN order (simulated)"
        );
        sym_state.last_order_time = std::time::Instant::now();
        return true;
    }

    info!(
        symbol = symbol,
        side = desired_side,
        qty = %qty_str,
        notional = format!("{:.2}", order_notional),
        leverage = format!("{:.2}x", projected_leverage),
        equity = format!("{:.2}", equity),
        exchange_synced = exchange_state.synced,
        req_id = %req_id,
        score = format!("{:.4}", score),
        micro = micro_score_str,
        source = source,
        "Submitting order"
    );

    // Track pending
    if let Ok(mut pending) = pending_orders.try_lock() {
        pending.insert(
            req_id,
            PendingOrder {
                symbol: symbol.to_string(),
                side: desired_side.to_string(),
                qty,
                sent_at: std::time::Instant::now(),
            },
        );
    }

    if let Err(e) = order_tx.try_send(msg) {
        error!(error = %e, "Failed to queue order");
        return false;
    }

    sym_state.last_order_time = std::time::Instant::now();
    true
}

/// Compute combined ML + micro-alpha score for bar-close decisions.
fn combined_bar_score(ml_score: f64, micro_sig: Option<&MicroAlphaSignal>, tick_by_tick: bool) -> f64 {
    let micro_score = micro_sig.map(|s| s.micro_score).unwrap_or(0.0);
    let micro_valid = micro_sig.map(|s| s.valid).unwrap_or(false);

    if ml_score.abs() > 0.0 && micro_valid {
        let agreement = ml_score.signum() == micro_score.signum() || micro_score.abs() < 0.1;
        if agreement {
            ml_score * (1.0 + micro_score.abs() * 0.3)
        } else {
            ml_score * (1.0 - micro_score.abs() * 0.5).max(0.1)
        }
    } else if ml_score.abs() > 0.0 {
        ml_score
    } else if !tick_by_tick && micro_valid && micro_score.abs() > 0.5 {
        micro_score * 0.3
    } else {
        0.0
    }
}
