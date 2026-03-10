// quant_trader — Standalone Rust trading binary
//
// Runs the full trading loop without Python:
//   Market data (WS) → RustTickProcessor (features+predict+state) → Decision → Order (WS)
//
// Modes:
//   Bar-based (default): decisions on kline bar close, micro-alpha confirms/dampens ML
//   Tick-by-tick (micro_alpha.tick_by_tick=true): decisions on every aggTrade, ~50-100ms
//
// Production features:
//   - Position sizing via fixed_fraction_qty (Phase 1)
//   - Risk gating: leverage/drawdown/notional checks (Phase 2)
//   - Order confirmation: JSON-RPC response parsing + circuit breaker (Phase 3)
//   - Reconnection with exponential backoff (Phase 4)
//   - Health monitoring (Phase 4)
//   - Checkpoint persistence: SIGINT saves state, restart replays bars (Phase 5)
//   - Micro-alpha: aggTrade-driven trade flow / volume / large trade signals (Phase 6)
//   - Tick-by-tick: sub-second decision on every aggTrade (Phase 6)
//
// Usage:
//   quant_trader --config config.yaml [--dry-run]

mod config;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use _quant_hotpath::decision_math::rust_fixed_fraction_qty;
use _quant_hotpath::json_parse::{parse_agg_trade_native, parse_kline_native};
use _quant_hotpath::micro_alpha::{MicroAlpha, MicroAlphaConfig, MicroAlphaSignal};
use _quant_hotpath::order_submit::RustWsOrderGateway;
use _quant_hotpath::tick_processor::RustTickProcessor;

use config::Config;

const SCALE: f64 = 100_000_000.0;

#[derive(Parser)]
#[command(name = "quant_trader", about = "Standalone Rust trading binary")]
struct Cli {
    /// Path to YAML config file
    #[arg(short, long)]
    config: PathBuf,

    /// Dry-run mode: process ticks but don't submit orders
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

const WS_FAPI_MAINNET: &str = "wss://fstream.binance.com/stream";
const WS_FAPI_TESTNET: &str = "wss://fstream.binancefuture.com/stream";
const WS_ORDER_MAINNET: &str = "wss://ws-fapi.binance.com/ws-fapi/v1";
const WS_ORDER_TESTNET: &str = "wss://testnet.binancefuture.com/ws-fapi/v1";
const WS_USERDATA_MAINNET: &str = "wss://fstream.binance.com/ws/";
const WS_USERDATA_TESTNET: &str = "wss://fstream.binancefuture.com/ws/";

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

// ── Historical kline backfill for instant warmup ──

const FAPI_REST_MAINNET: &str = "https://fapi.binance.com";
const FAPI_REST_TESTNET: &str = "https://testnet.binancefuture.com";

async fn backfill_historical_klines(
    symbol: &str,
    interval: &str,
    testnet: bool,
    processor: &mut RustTickProcessor,
    limit: usize,
) -> usize {
    let base = if testnet { FAPI_REST_TESTNET } else { FAPI_REST_MAINNET };
    let url = format!(
        "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
        base, symbol, interval, limit
    );

    info!(symbol = %symbol, limit = limit, "Fetching historical klines for warmup");

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "Failed to build HTTP client, skipping backfill");
            return 0;
        }
    };

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Failed to fetch historical klines, skipping backfill");
            return 0;
        }
    };

    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, "Failed to read kline response body");
            return 0;
        }
    };

    // Parse kline array: each element is [open_time, open, high, low, close, volume, ...]
    let klines: Vec<serde_json::Value> = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "Failed to parse kline JSON");
            return 0;
        }
    };

    let mut count = 0usize;
    // Skip last bar (it's still open / current)
    let n = if klines.len() > 1 { klines.len() - 1 } else { 0 };

    for kline in &klines[..n] {
        let arr = match kline.as_array() {
            Some(a) if a.len() >= 6 => a,
            _ => continue,
        };

        let ts_ms = arr[0].as_i64().unwrap_or(0);
        let open: f64 = arr[1].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let high: f64 = arr[2].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let low: f64 = arr[3].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let close: f64 = arr[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let volume: f64 = arr[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);

        if close == 0.0 {
            continue;
        }

        let hour_key = ts_ms / 3_600_000;
        let ts_str = chrono_ts(ts_ms);

        processor.process_tick_native(
            symbol, close, volume, high, low, open, hour_key, Some(ts_str),
        );
        count += 1;
    }

    if count > 0 {
        info!(symbol = %symbol, bars = count, "Historical backfill complete — warmup done");
    }
    count
}

// ── Sync exchange state at startup via REST API ──

fn hmac_sha256_hex(secret: &str, payload: &str) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC key");
    mac.update(payload.as_bytes());
    let result = mac.finalize();
    let bytes = result.into_bytes();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

async fn sync_exchange_state(
    testnet: bool,
    api_key: &str,
    api_secret: &str,
    exchange_state: &std::sync::Mutex<ExchangeState>,
) {
    let base = if testnet { FAPI_REST_TESTNET } else { FAPI_REST_MAINNET };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let query = format!("timestamp={}", ts);
    let signature = hmac_sha256_hex(api_secret, &query);
    let url = format!("{}/fapi/v2/account?{}&signature={}", base, query, signature);

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "Failed to build HTTP client for sync");
            return;
        }
    };

    let resp = match client
        .get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Failed to query account, skipping sync");
            return;
        }
    };

    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, "Failed to read account response");
            return;
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "Failed to parse account JSON");
            return;
        }
    };

    let mut state = exchange_state.lock().unwrap();

    // Sync balance
    if let Some(bal) = v.get("totalWalletBalance").and_then(|b| b.as_str()).and_then(|s| s.parse::<f64>().ok()) {
        let old = state.balance;
        state.balance = bal;
        info!(old = format!("{:.2}", old), new = format!("{:.2}", bal), "Exchange balance synced");
    }

    // Sync positions
    if let Some(positions) = v.get("positions").and_then(|p| p.as_array()) {
        for pos in positions {
            let qty: f64 = pos.get("positionAmt")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            if qty.abs() < 1e-12 {
                continue;
            }
            let symbol = match pos.get("symbol").and_then(|s| s.as_str()) {
                Some(s) => s,
                None => continue,
            };
            let entry: f64 = pos.get("entryPrice")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let notional = (qty * entry).abs();

            state.positions.insert(symbol.to_string(), ExchangePosition { qty, notional });
            info!(
                symbol = %symbol,
                qty = format!("{:.6}", qty),
                entry = format!("{:.2}", entry),
                notional = format!("{:.2}", notional),
                "Exchange position synced"
            );
        }
    }

    state.synced = true;
    state.last_update = std::time::Instant::now();

    let pos_count = state.positions.len();
    info!(positions = pos_count, "Exchange state sync complete");
}

#[tokio::main]
async fn main() {
    load_dotenv();

    let cli = Cli::parse();

    let config = match Config::load(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    };

    init_logging(&config.logging);

    let tick_by_tick = config.micro_alpha.tick_by_tick;

    info!(
        symbols = ?config.trading.symbols,
        mode = %config.trading.mode,
        testnet = config.trading.testnet,
        dry_run = cli.dry_run,
        interval = %config.trading.interval,
        tick_by_tick = tick_by_tick,
        fraction = config.position_sizing.fraction,
        max_leverage = config.risk.max_leverage,
        max_drawdown = format!("{:.0}%", config.risk.max_drawdown_pct * 100.0),
        min_order_interval_ms = config.micro_alpha.min_order_interval_ms,
        micro_threshold = config.micro_alpha.micro_threshold,
        "quant_trader starting"
    );

    // Resolve credentials
    let api_key = match config.resolve_api_key() {
        Ok(k) => k,
        Err(e) => {
            error!(error = %e, "Failed to resolve API key");
            std::process::exit(1);
        }
    };
    let api_secret = match config.resolve_api_secret() {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "Failed to resolve API secret");
            std::process::exit(1);
        }
    };

    // Build tick processors (one per symbol)
    let mut processors = Vec::new();
    for symbol in &config.trading.symbols {
        let resolved = match config.resolve_models_for(symbol) {
            Ok(r) => r,
            Err(e) => {
                error!(symbol = %symbol, error = %e, "Failed to resolve models, skipping");
                continue;
            }
        };

        let mut tp = match RustTickProcessor::create_native(
            vec![symbol.clone()],
            config.trading.currency.clone(),
            config.trading.starting_balance,
            resolved.json_paths.clone(),
            resolved.ensemble_weights.clone(),
            resolved.bear_model_path.as_deref(),
            resolved.short_model_path.as_deref(),
            720,
            168,
        ) {
            Ok(tp) => tp,
            Err(e) => {
                error!(symbol = %symbol, error = %e, "Failed to create tick processor");
                continue;
            }
        };

        // Configure symbol strategy
        let strat = &config.strategy;
        let yaml_ovr = strat.per_symbol.get(symbol.as_str());
        let model_ovr = &resolved.strategy_override;

        tp.configure_symbol_native(
            symbol,
            model_ovr
                .min_hold
                .or_else(|| yaml_ovr.and_then(|o| o.min_hold))
                .unwrap_or(strat.min_hold),
            model_ovr
                .deadzone
                .or_else(|| yaml_ovr.and_then(|o| o.deadzone))
                .unwrap_or(strat.deadzone),
            model_ovr
                .long_only
                .or_else(|| yaml_ovr.and_then(|o| o.long_only))
                .unwrap_or(strat.long_only),
            yaml_ovr
                .and_then(|o| o.trend_follow)
                .unwrap_or(strat.trend_follow),
            strat.trend_threshold,
            &strat.trend_indicator,
            strat.max_hold,
            model_ovr
                .monthly_gate
                .or_else(|| yaml_ovr.and_then(|o| o.monthly_gate))
                .unwrap_or(strat.monthly_gate),
            model_ovr
                .monthly_gate_window
                .unwrap_or(strat.monthly_gate_window),
            model_ovr
                .vol_target
                .or_else(|| yaml_ovr.and_then(|o| o.vol_target))
                .or(strat.vol_target),
            model_ovr
                .vol_feature
                .as_deref()
                .unwrap_or(&strat.vol_feature),
            model_ovr
                .bear_thresholds
                .clone()
                .or_else(|| yaml_ovr.and_then(|o| o.bear_thresholds.clone()))
                .unwrap_or_else(|| strat.bear_thresholds.clone()),
        );

        info!(
            symbol = %symbol,
            models = resolved.json_paths.len(),
            bear = resolved.bear_model_path.is_some(),
            short = resolved.short_model_path.is_some(),
            deadzone = model_ovr.deadzone.unwrap_or(strat.deadzone),
            min_hold = model_ovr.min_hold.unwrap_or(strat.min_hold),
            "Tick processor ready"
        );

        processors.push((symbol.clone(), tp));
    }

    if processors.is_empty() {
        error!("No tick processors created, exiting");
        std::process::exit(1);
    }

    // Phase 5: Restore checkpoint
    let checkpoint_path = cli
        .config
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("checkpoint.json");
    if checkpoint_path.exists() {
        match std::fs::read_to_string(&checkpoint_path) {
            Ok(json) => {
                let mut restored_count = 0usize;
                for (sym, tp) in &mut processors {
                    match tp.restore_checkpoint_native(&json) {
                        Ok(n) => {
                            restored_count += n;
                            info!(symbol = %sym, bars = n, "Checkpoint restored");
                        }
                        Err(e) => warn!(symbol = %sym, error = %e, "Failed to restore checkpoint"),
                    }
                }
                if restored_count > 0 {
                    info!(total_bars = restored_count, "Checkpoint replay complete");
                }
            }
            Err(e) => warn!(error = %e, "Failed to read checkpoint file"),
        }
    }

    // Historical kline backfill for instant warmup (skip 8-hour wait)
    // Fetch 200 closed bars from REST API and replay through processor
    let backfill_limit = 200usize; // > 168 warmup threshold
    for (symbol, processor) in &mut processors {
        backfill_historical_klines(
            symbol,
            &config.trading.interval,
            config.trading.testnet,
            processor,
            backfill_limit,
        )
        .await;
    }

    // Build order gateway
    let order_gateway = RustWsOrderGateway::new_native(&api_key, &api_secret, 5000);

    // Exchange state (synced from user data stream)
    let exchange_state = Arc::new(std::sync::Mutex::new(ExchangeState::new(
        config.trading.starting_balance,
    )));

    // Sync exchange state from REST API at startup (picks up existing positions)
    sync_exchange_state(
        config.trading.testnet,
        &api_key,
        &api_secret,
        &exchange_state,
    )
    .await;

    // Shared metrics for Prometheus endpoint
    let metrics = Arc::new(std::sync::Mutex::new(MetricsSnapshot::new()));
    {
        let mut m = metrics.lock().unwrap();
        m.equity = config.trading.starting_balance;
        m.peak_equity = config.trading.starting_balance;
        m.balance = config.trading.starting_balance;
    }

    // Spawn Prometheus metrics HTTP server on port 9090
    // GET /metrics → Prometheus text format
    // POST /kill → create KILL_SWITCH file, triggers emergency close-all
    // DELETE /kill → remove KILL_SWITCH file, resume trading
    let metrics_clone = metrics.clone();
    let kill_path_clone = cli
        .config
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("KILL_SWITCH");
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = match tokio::net::TcpListener::bind("0.0.0.0:9090").await {
            Ok(l) => {
                info!(port = 9090, "Metrics server listening");
                l
            }
            Err(e) => {
                warn!(error = %e, "Failed to bind metrics server");
                return;
            }
        };
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = [0u8; 1024];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);

            let resp = if req.starts_with("POST /kill") {
                let _ = std::fs::write(&kill_path_clone, "kill");
                let body = "KILL SWITCH ACTIVATED\n";
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(), body
                )
            } else if req.starts_with("DELETE /kill") {
                let _ = std::fs::remove_file(&kill_path_clone);
                let body = "KILL SWITCH DEACTIVATED\n";
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(), body
                )
            } else {
                let body = {
                    let m = metrics_clone.lock().unwrap();
                    m.to_prometheus()
                };
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(), body
                )
            };
            let _ = socket.write_all(resp.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });

    // Listen key channel (order WS → main thread → user data WS)
    let (listen_key_tx, mut listen_key_rx) = mpsc::channel::<String>(4);

    // Shutdown signal
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("SIGINT received, shutting down...");
        r.store(false, Ordering::SeqCst);
    });

    // Build micro-alpha engines
    let mut micro_alphas: HashMap<String, MicroAlpha> = HashMap::new();
    for symbol in &config.trading.symbols {
        micro_alphas.insert(
            symbol.clone(),
            MicroAlpha::new(MicroAlphaConfig {
                window_ms: config.micro_alpha.window_ms,
                large_trade_mult: config.micro_alpha.large_trade_mult,
                ..Default::default()
            }),
        );
    }

    // Per-symbol state
    let mut sym_states: HashMap<String, PerSymbolState> = HashMap::new();
    for symbol in &config.trading.symbols {
        sym_states.insert(
            symbol.clone(),
            PerSymbolState {
                last_ml_score: 0.0,
                last_price: 0.0,
                last_order_time: std::time::Instant::now()
                    - std::time::Duration::from_secs(3600), // allow immediate first order
                trade_count: 0,
            },
        );
    }

    // Parse interval to milliseconds for trade aggregator
    let interval = &config.trading.interval;
    let interval_ms: i64 = parse_interval_ms(interval);

    // Build trade aggregators (one per symbol) for building bars from aggTrade
    let mut aggregators: HashMap<String, TradeAggregator> = HashMap::new();
    for symbol in &config.trading.symbols {
        aggregators.insert(symbol.clone(), TradeAggregator::new(interval_ms));
    }

    // Build market data WS URL
    // For intervals >= 1m: subscribe to kline + aggTrade
    // For sub-minute (1s): aggTrade only (futures don't support kline_1s)
    let mut streams: Vec<String> = Vec::new();
    let use_kline_stream = interval_ms >= 60_000;
    if use_kline_stream {
        for s in &config.trading.symbols {
            streams.push(format!("{}@kline_{}", s.to_lowercase(), interval));
        }
    }
    for s in &config.trading.symbols {
        streams.push(format!("{}@aggTrade", s.to_lowercase()));
    }
    let base_url = if config.trading.testnet {
        WS_FAPI_TESTNET
    } else {
        WS_FAPI_MAINNET
    };
    let ws_url = format!("{}?streams={}", base_url, streams.join("/"));

    let order_ws_url = if config.trading.testnet {
        WS_ORDER_TESTNET
    } else {
        WS_ORDER_MAINNET
    };

    info!(url = %ws_url, "Connecting to market data WebSocket");

    // Order channel + response channel
    let (order_tx, mut order_rx) = mpsc::channel::<String>(64);
    let (resp_tx, mut resp_rx) = mpsc::channel::<String>(64);

    // Pending orders tracking
    let pending_orders: Arc<tokio::sync::Mutex<HashMap<String, PendingOrder>>> =
        Arc::new(tokio::sync::Mutex::new(HashMap::new()));

    // Circuit breaker
    let order_failure_count = Arc::new(AtomicU32::new(0));

    // Risk params
    let risk = RiskParams {
        fraction: config.position_sizing.fraction,
        max_leverage: config.risk.max_leverage,
        max_drawdown_pct: config.risk.max_drawdown_pct,
        max_position_notional: config.risk.max_position_notional,
        circuit_breaker_threshold: 5,
        min_order_interval: std::time::Duration::from_millis(
            config.micro_alpha.min_order_interval_ms,
        ),
    };

    let micro_threshold = config.micro_alpha.micro_threshold;

    // Lot sizes lookup
    let lot_sizes = config.position_sizing.lot_sizes.clone();

    // Spawn order WS (if not dry-run)
    // Also handles: userDataStream.start on connect, keepalive every 30 min
    if !cli.dry_run {
        let order_url = order_ws_url.to_string();
        let running_clone = running.clone();
        let resp_tx_clone = resp_tx.clone();
        let listen_key_tx_clone = listen_key_tx.clone();
        let order_gw_for_ws = RustWsOrderGateway::new_native(&api_key, &api_secret, 5000);
        tokio::spawn(async move {
            let mut backoff_secs = 1u64;
            let mut current_listen_key: Option<String> = None;
            loop {
                if !running_clone.load(Ordering::Relaxed) {
                    break;
                }
                match connect_async(&order_url).await {
                    Ok((ws, _)) => {
                        info!("Order WebSocket connected");
                        backoff_secs = 1;
                        let (mut write, mut read) = ws.split();

                        // Request listenKey for user data stream
                        let (lk_msg, lk_req_id) = order_gw_for_ws.build_listen_key_start_native();
                        info!(req_id = %lk_req_id, "Requesting listenKey");
                        if let Err(e) = write.send(Message::Text(lk_msg.into())).await {
                            error!(error = %e, "Failed to send listenKey request");
                        }

                        let running_inner = running_clone.clone();
                        let resp_tx_inner = resp_tx_clone.clone();
                        let lk_tx_inner = listen_key_tx_clone.clone();
                        let read_handle = tokio::spawn(async move {
                            while running_inner.load(Ordering::Relaxed) {
                                match read.next().await {
                                    Some(Ok(Message::Text(text))) => {
                                        let text_str = text.to_string();
                                        // Check if this is a listenKey response
                                        if text_str.contains("listenKey") {
                                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text_str) {
                                                if let Some(lk) = v.get("result")
                                                    .and_then(|r| r.get("listenKey"))
                                                    .and_then(|v| v.as_str())
                                                {
                                                    info!(listen_key = &lk[..8], "listenKey received");
                                                    let _ = lk_tx_inner.try_send(lk.to_string());
                                                }
                                            }
                                        } else {
                                            let _ = resp_tx_inner.try_send(text_str);
                                        }
                                    }
                                    Some(Ok(Message::Ping(_))) => continue,
                                    Some(Ok(_)) => continue,
                                    Some(Err(e)) => {
                                        warn!(error = %e, "Order WS read error");
                                        break;
                                    }
                                    None => {
                                        warn!("Order WS stream ended");
                                        break;
                                    }
                                }
                            }
                        });

                        let mut keepalive_interval = tokio::time::interval(
                            std::time::Duration::from_secs(30 * 60), // 30 minutes
                        );
                        keepalive_interval.tick().await; // consume first immediate tick

                        loop {
                            if !running_clone.load(Ordering::Relaxed) {
                                break;
                            }
                            tokio::select! {
                                msg = order_rx.recv() => {
                                    match msg {
                                        Some(m) => {
                                            if let Err(e) = write.send(Message::Text(m.into())).await {
                                                error!(error = %e, "Failed to send order");
                                                break;
                                            }
                                        }
                                        None => break,
                                    }
                                }
                                _ = keepalive_interval.tick() => {
                                    // Send listenKey keepalive
                                    if let Some(ref lk) = current_listen_key {
                                        let (ka_msg, _) = order_gw_for_ws.build_listen_key_ping_native(lk);
                                        if let Err(e) = write.send(Message::Text(ka_msg.into())).await {
                                            warn!(error = %e, "Failed to send listenKey keepalive");
                                            break;
                                        }
                                        info!("listenKey keepalive sent");
                                    }
                                    // Also send WS ping
                                    if write.send(Message::Ping(vec![].into())).await.is_err() {
                                        break;
                                    }
                                }
                                _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {
                                    if write.send(Message::Ping(vec![].into())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }

                        read_handle.abort();
                        current_listen_key = None;
                    }
                    Err(e) => {
                        error!(error = %e, backoff_s = backoff_secs, "Failed to connect order WS");
                    }
                }

                if !running_clone.load(Ordering::Relaxed) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(60);
            }
        });

        // Spawn user data stream WS
        let userdata_base = if config.trading.testnet {
            WS_USERDATA_TESTNET
        } else {
            WS_USERDATA_MAINNET
        };
        let exchange_state_clone = exchange_state.clone();
        let running_clone2 = running.clone();
        tokio::spawn(async move {
            loop {
                if !running_clone2.load(Ordering::Relaxed) {
                    break;
                }

                // Wait for listenKey from order WS
                let listen_key = match listen_key_rx.recv().await {
                    Some(lk) => lk,
                    None => break,
                };

                let url = format!("{}{}", userdata_base, listen_key);
                info!(url = &url[..url.len().min(60)], "Connecting user data stream");

                match connect_async(&url).await {
                    Ok((ws, _)) => {
                        info!("User data stream connected");
                        let (_, mut read) = ws.split();

                        loop {
                            if !running_clone2.load(Ordering::Relaxed) {
                                break;
                            }

                            let msg = tokio::select! {
                                msg = read.next() => msg,
                                _ = tokio::time::sleep(std::time::Duration::from_secs(300)) => {
                                    warn!("User data stream: no data for 5 min");
                                    break;
                                }
                            };

                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    let mut state = exchange_state_clone.lock().unwrap();
                                    process_user_data_event(&text, &mut state);
                                }
                                Some(Ok(Message::Ping(_))) => continue,
                                Some(Ok(_)) => continue,
                                Some(Err(e)) => {
                                    warn!(error = %e, "User data stream error");
                                    break;
                                }
                                None => {
                                    warn!("User data stream ended");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to connect user data stream");
                    }
                }

                // Will reconnect when a new listenKey arrives
            }
        });
    }

    // Tracking
    let mut peak_equity = config.trading.starting_balance;
    let mut tick_count: u64 = 0;
    let mut micro_order_count: u64 = 0;
    let start_time = std::time::Instant::now();
    let mut last_data_time = std::time::Instant::now();
    let mut md_backoff_secs = 1u64;
    let mut kill_switch_active = false;
    let kill_switch_path = cli
        .config
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("KILL_SWITCH");

    // ── Main loop ──

    'outer: loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        let result = connect_async(&ws_url).await;
        let (_, mut read) = match result {
            Ok((ws, _)) => {
                info!("Market data WebSocket connected");
                md_backoff_secs = 1;
                ws.split()
            }
            Err(e) => {
                error!(error = %e, backoff_s = md_backoff_secs, "Market data WS connect failed");
                tokio::time::sleep(std::time::Duration::from_secs(md_backoff_secs)).await;
                md_backoff_secs = (md_backoff_secs * 2).min(60);
                continue;
            }
        };

        loop {
            if !running.load(Ordering::Relaxed) {
                break 'outer;
            }

            // Kill switch: check file every iteration, close all positions once
            if !kill_switch_active && kill_switch_path.exists() {
                kill_switch_active = true;
                error!("KILL SWITCH activated — closing all positions");

                let ex = exchange_state.lock().unwrap();
                let open_positions: Vec<(String, f64)> = ex
                    .positions
                    .iter()
                    .filter(|(_, p)| p.qty.abs() > 1e-12)
                    .map(|(s, p)| (s.clone(), p.qty))
                    .collect();
                drop(ex);

                for (sym, qty) in &open_positions {
                    let close_side = if *qty > 0.0 { "SELL" } else { "BUY" };
                    let close_qty = format!("{:.6}", qty.abs());
                    let lot = lot_sizes.get(sym.as_str()).copied().unwrap_or(0.001);
                    let close_qty_str = format_qty(qty.abs(), lot);
                    let (msg, req_id) = order_gateway.build_order_message_native(
                        sym,
                        close_side,
                        "MARKET",
                        Some(&close_qty_str),
                        None,
                        None,
                        Some(true), // reduce_only
                        None,
                    );
                    error!(
                        symbol = %sym,
                        side = close_side,
                        qty = close_qty,
                        req_id = %req_id,
                        "KILL: closing position"
                    );
                    let _ = order_tx.try_send(msg);
                }

                if open_positions.is_empty() {
                    error!("KILL: no open positions to close");
                }
            }

            // Process order responses (non-blocking)
            while let Ok(resp_text) = resp_rx.try_recv() {
                process_order_response(&resp_text, &pending_orders, &order_failure_count).await;
            }

            let msg = tokio::select! {
                msg = read.next() => msg,
                _ = tokio::time::sleep(std::time::Duration::from_secs(120)) => {
                    warn!("No market data for 120s, reconnecting");
                    break;
                }
            };

            let msg = match msg {
                Some(Ok(Message::Text(text))) => text,
                Some(Ok(Message::Ping(_))) => {
                    last_data_time = std::time::Instant::now();
                    continue;
                }
                Some(Ok(_)) => continue,
                Some(Err(e)) => {
                    error!(error = %e, "Market data WS error, reconnecting");
                    break;
                }
                None => {
                    warn!("Market data WS stream ended, reconnecting");
                    break;
                }
            };

            last_data_time = std::time::Instant::now();

            // ── aggTrade path ──
            if let Some(trade) = parse_agg_trade_native(&msg) {
                // Push to micro-alpha engine
                if let Some(ma) = micro_alphas.get_mut(&trade.symbol) {
                    ma.push_trade(trade.ts_ms, trade.price, trade.qty, trade.is_buy);
                }

                // Update per-symbol state
                if let Some(ss) = sym_states.get_mut(&trade.symbol) {
                    ss.last_price = trade.price;
                    ss.trade_count += 1;
                }

                // Aggregate trades into bars (only when exchange doesn't provide kline stream)
                let agg_bar = if !use_kline_stream {
                    aggregators
                        .get_mut(&trade.symbol)
                        .and_then(|agg| agg.push(trade.ts_ms, trade.price, trade.qty))
                } else {
                    None
                };

                // If a bar closed, feed it to the ML pipeline
                if let Some(bar) = agg_bar {
                    if let Some((_, processor)) =
                        processors.iter_mut().find(|(s, _)| *s == trade.symbol)
                    {
                        let hour_key = bar.ts_ms / 3_600_000;
                        let ts_str = chrono_ts(bar.ts_ms);

                        let result = processor.process_tick_native(
                            &trade.symbol,
                            bar.close,
                            bar.volume,
                            bar.high,
                            bar.low,
                            bar.open,
                            hour_key,
                            Some(ts_str),
                        );

                        tick_count += 1;

                        // Update ML score
                        if let Some(ss) = sym_states.get_mut(&trade.symbol) {
                            ss.last_ml_score = result.ml_score;
                        }

                        // Compute micro signal at bar close
                        let micro_sig = micro_alphas
                            .get(&trade.symbol)
                            .map(|ma| ma.compute(bar.ts_ms));

                        // Update equity tracking
                        let equity = processor.account_native().balance as f64 / SCALE;
                        if equity > peak_equity {
                            peak_equity = equity;
                        }
                        let drawdown_pct = if peak_equity > 0.0 {
                            (peak_equity - equity) / peak_equity
                        } else {
                            0.0
                        };

                        // Health log every 60 bars
                        if tick_count % 60 == 0 {
                            let current_qty = processor
                                .get_position_native(&trade.symbol)
                                .map(|p| p.qty as f64 / SCALE)
                                .unwrap_or(0.0);
                            let micro_str = micro_sig
                                .as_ref()
                                .map(|s| format!("{:.4}", s.micro_score))
                                .unwrap_or_else(|| "n/a".to_string());
                            info!(
                                ticks = tick_count,
                                equity = format!("{:.2}", equity),
                                drawdown = format!("{:.2}%", drawdown_pct * 100.0),
                                position = format!("{:.6}", current_qty),
                                ml_score = format!("{:.4}", result.ml_score),
                                micro = micro_str,
                                micro_orders = micro_order_count,
                                "health"
                            );
                        }

                        // Tick log
                        if tick_count <= 10 || tick_count % 10 == 0 || result.ml_score.abs() > 0.0 {
                            let (micro_str, flow_str, vol_str, large_str) = match &micro_sig {
                                Some(s) => (
                                    format!("{:.4}", s.micro_score),
                                    format!("{:.3}", s.trade_flow_imbalance),
                                    format!("{:.2}", s.volume_spike),
                                    format!("{:.3}", s.large_trade_signal),
                                ),
                                None => ("n/a".into(), "n/a".into(), "n/a".into(), "n/a".into()),
                            };
                            info!(
                                symbol = %trade.symbol,
                                close = format!("{:.2}", bar.close),
                                ml = format!("{:.4}", result.ml_score),
                                raw = format!("{:.6}", result.raw_score),
                                micro = micro_str,
                                flow = flow_str,
                                vol_spike = vol_str,
                                large = large_str,
                                idx = result.event_index,
                                ticks = tick_count,
                                "bar"
                            );
                        }

                        // Bar-close combined decision
                        if !cli.dry_run && !kill_switch_active {
                            let micro_score =
                                micro_sig.as_ref().map(|s| s.micro_score).unwrap_or(0.0);
                            let micro_valid =
                                micro_sig.as_ref().map(|s| s.valid).unwrap_or(false);

                            let combined_score = if result.ml_score.abs() > 0.0 && micro_valid {
                                let agreement = result.ml_score.signum()
                                    == micro_score.signum()
                                    || micro_score.abs() < 0.1;
                                if agreement {
                                    result.ml_score * (1.0 + micro_score.abs() * 0.3)
                                } else {
                                    result.ml_score * (1.0 - micro_score.abs() * 0.5).max(0.1)
                                }
                            } else if result.ml_score.abs() > 0.0 {
                                result.ml_score
                            } else if !tick_by_tick && micro_valid && micro_score.abs() > 0.5 {
                                micro_score * 0.3
                            } else {
                                0.0
                            };

                            if combined_score.abs() > 0.0 {
                                let lot_size = lot_sizes
                                    .get(trade.symbol.as_str())
                                    .copied()
                                    .unwrap_or(0.001);
                                if let Some(ss) = sym_states.get_mut(&trade.symbol) {
                                    let ex_state = exchange_state.lock().unwrap();
                                    try_submit_order(
                                        &trade.symbol,
                                        combined_score,
                                        bar.close,
                                        "bar_agg",
                                        processor,
                                        ss,
                                        &ex_state,
                                        &risk,
                                        lot_size,
                                        peak_equity,
                                        &order_gateway,
                                        &order_tx,
                                        &pending_orders,
                                        &order_failure_count,
                                        micro_sig.as_ref(),
                                    );
                                }
                            }
                        }
                    }
                }

                // Tick-by-tick decision on every aggTrade
                if tick_by_tick && !cli.dry_run && !kill_switch_active {
                    let sym = &trade.symbol;
                    let micro_sig = micro_alphas.get(sym).map(|ma| ma.compute(trade.ts_ms));

                    if let Some(ref sig) = micro_sig {
                        if sig.valid && sig.micro_score.abs() >= micro_threshold {
                            let ml = sym_states.get(sym).map(|s| s.last_ml_score).unwrap_or(0.0);

                            let score = if ml.abs() > 0.0 {
                                if ml.signum() == sig.micro_score.signum() {
                                    sig.micro_score * (1.0 + ml.abs() * 0.5)
                                } else {
                                    0.0
                                }
                            } else {
                                sig.micro_score * 0.5
                            };

                            if score.abs() > 0.0 {
                                let processor =
                                    match processors.iter().find(|(s, _)| s == sym) {
                                        Some((_, tp)) => tp,
                                        None => continue,
                                    };
                                let lot_size =
                                    lot_sizes.get(sym.as_str()).copied().unwrap_or(0.001);

                                if let Some(ss) = sym_states.get_mut(sym) {
                                    let ex_state = exchange_state.lock().unwrap();
                                    let submitted = try_submit_order(
                                        sym,
                                        score,
                                        trade.price,
                                        "tick",
                                        processor,
                                        ss,
                                        &ex_state,
                                        &risk,
                                        lot_size,
                                        peak_equity,
                                        &order_gateway,
                                        &order_tx,
                                        &pending_orders,
                                        &order_failure_count,
                                        micro_sig.as_ref(),
                                    );
                                    if submitted {
                                        micro_order_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                // Periodic aggTrade stats (every 1000 trades)
                if let Some(ss) = sym_states.get(&trade.symbol) {
                    if ss.trade_count % 1000 == 0 && ss.trade_count > 0 {
                        let sig = micro_alphas
                            .get(&trade.symbol)
                            .map(|ma| ma.compute(trade.ts_ms));
                        if let Some(sig) = sig {
                            info!(
                                symbol = %trade.symbol,
                                trades = ss.trade_count,
                                flow = format!("{:.3}", sig.trade_flow_imbalance),
                                vol_spike = format!("{:.2}", sig.volume_spike),
                                large = format!("{:.3}", sig.large_trade_signal),
                                micro = format!("{:.4}", sig.micro_score),
                                micro_orders = micro_order_count,
                                valid = sig.valid,
                                "aggTrade stats"
                            );
                        }
                    }
                }

                continue;
            }

            // ── kline path (for intervals >= 1m where exchange provides kline stream) ──
            if !use_kline_stream {
                continue; // sub-minute: bars built from aggTrade above
            }
            let bar = match parse_kline_native(&msg) {
                Some(bar) => bar,
                _ => continue,
            };

            if !bar.closed {
                continue;
            }

            // Find processor
            let processor = match processors.iter_mut().find(|(s, _)| *s == bar.symbol) {
                Some((_, tp)) => tp,
                None => continue,
            };

            // Process tick through ML pipeline
            let hour_key = bar.ts_ms / 3_600_000;
            let ts_str = chrono_ts(bar.ts_ms);

            let result = processor.process_tick_native(
                &bar.symbol,
                bar.close,
                bar.volume,
                bar.high,
                bar.low,
                bar.open,
                hour_key,
                Some(ts_str),
            );

            tick_count += 1;

            // Update per-symbol state with ML score
            if let Some(ss) = sym_states.get_mut(&bar.symbol) {
                ss.last_ml_score = result.ml_score;
                ss.last_price = bar.close;
            }

            // Compute micro signal at bar close
            let micro_sig = micro_alphas
                .get(&bar.symbol)
                .map(|ma| ma.compute(bar.ts_ms));

            // Update peak equity / drawdown
            let equity = processor.account_native().balance as f64 / SCALE;
            if equity > peak_equity {
                peak_equity = equity;
            }
            let drawdown_pct = if peak_equity > 0.0 {
                (peak_equity - equity) / peak_equity
            } else {
                0.0
            };

            // Update Prometheus metrics
            if let Ok(mut m) = metrics.try_lock() {
                m.equity = equity;
                m.peak_equity = peak_equity;
                m.drawdown_pct = drawdown_pct;
                m.tick_count = tick_count;
                m.micro_orders = micro_order_count;
                m.uptime_secs = start_time.elapsed().as_secs();
                m.ml_scores.insert(bar.symbol.clone(), result.ml_score);
                m.raw_scores.insert(bar.symbol.clone(), result.raw_score);
                if let Ok(ex) = exchange_state.try_lock() {
                    m.balance = ex.balance;
                    for (sym, pos) in &ex.positions {
                        m.positions.insert(sym.clone(), pos.qty);
                    }
                }
            }

            // Health log every 60 bars
            if tick_count % 60 == 0 {
                let current_qty = processor
                    .get_position_native(&bar.symbol)
                    .map(|p| p.qty as f64 / SCALE)
                    .unwrap_or(0.0);
                let micro_str = micro_sig
                    .as_ref()
                    .map(|s| format!("{:.4}", s.micro_score))
                    .unwrap_or_else(|| "n/a".to_string());
                let trade_cnt = micro_sig.as_ref().map(|s| s.trade_count).unwrap_or(0);
                info!(
                    ticks = tick_count,
                    ticks_per_min = format!(
                        "{:.1}",
                        tick_count as f64 / start_time.elapsed().as_secs_f64() * 60.0
                    ),
                    equity = format!("{:.2}", equity),
                    drawdown = format!("{:.2}%", drawdown_pct * 100.0),
                    position = format!("{:.6}", current_qty),
                    ml_score = format!("{:.4}", result.ml_score),
                    micro = micro_str,
                    micro_trades = trade_cnt,
                    micro_orders = micro_order_count,
                    last_data_age_s = last_data_time.elapsed().as_secs(),
                    "health"
                );
            }

            // Tick log
            if tick_count <= 5 || tick_count % 10 == 0 || result.ml_score.abs() > 0.0 {
                let (micro_str, flow_str, vol_str, large_str) = match &micro_sig {
                    Some(s) => (
                        format!("{:.4}", s.micro_score),
                        format!("{:.3}", s.trade_flow_imbalance),
                        format!("{:.2}", s.volume_spike),
                        format!("{:.3}", s.large_trade_signal),
                    ),
                    None => (
                        "n/a".to_string(),
                        "n/a".to_string(),
                        "n/a".to_string(),
                        "n/a".to_string(),
                    ),
                };
                info!(
                    symbol = %bar.symbol,
                    close = format!("{:.2}", bar.close),
                    ml_score = format!("{:.4}", result.ml_score),
                    raw = format!("{:.6}", result.raw_score),
                    micro = micro_str,
                    flow = flow_str,
                    vol_spike = vol_str,
                    large = large_str,
                    idx = result.event_index,
                    ticks = tick_count,
                    "bar"
                );
            }

            // Bar-close combined decision
            if !cli.dry_run && !kill_switch_active {
                let ms = &micro_sig;
                let micro_score = ms.as_ref().map(|s| s.micro_score).unwrap_or(0.0);
                let micro_valid = ms.as_ref().map(|s| s.valid).unwrap_or(false);

                let combined_score = if result.ml_score.abs() > 0.0 && micro_valid {
                    let agreement =
                        result.ml_score.signum() == micro_score.signum() || micro_score.abs() < 0.1;
                    if agreement {
                        result.ml_score * (1.0 + micro_score.abs() * 0.3)
                    } else {
                        result.ml_score * (1.0 - micro_score.abs() * 0.5).max(0.1)
                    }
                } else if result.ml_score.abs() > 0.0 {
                    result.ml_score
                } else if !tick_by_tick && micro_valid && micro_score.abs() > 0.5 {
                    // Only use pure micro on bar close if tick-by-tick is disabled
                    micro_score * 0.3
                } else {
                    0.0
                };

                if combined_score.abs() > 0.0 {
                    let lot_size = lot_sizes
                        .get(bar.symbol.as_str())
                        .copied()
                        .unwrap_or(0.001);

                    if let Some(ss) = sym_states.get_mut(&bar.symbol) {
                        let ex_state = exchange_state.lock().unwrap();
                        try_submit_order(
                            &bar.symbol,
                            combined_score,
                            bar.close,
                            "bar",
                            processor,
                            ss,
                            &ex_state,
                            &risk,
                            lot_size,
                            peak_equity,
                            &order_gateway,
                            &order_tx,
                            &pending_orders,
                            &order_failure_count,
                            ms.as_ref(),
                        );
                    }
                }
            }
        }

        // Backoff before reconnect
        if running.load(Ordering::Relaxed) {
            warn!(backoff_s = md_backoff_secs, "Reconnecting market data WS");
            tokio::time::sleep(std::time::Duration::from_secs(md_backoff_secs)).await;
            md_backoff_secs = (md_backoff_secs * 2).min(60);
        }
    }

    // Save checkpoint on shutdown — merge all processors into one JSON
    info!("Saving checkpoint...");
    {
        let mut merged_symbols = serde_json::Map::new();
        let mut max_event_index: u64 = 0;
        for (sym, tp) in &processors {
            match tp.checkpoint_native() {
                Ok(json) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                        if let Some(idx) = v.get("event_index").and_then(|v| v.as_u64()) {
                            max_event_index = max_event_index.max(idx);
                        }
                        if let Some(syms) = v.get("symbols").and_then(|v| v.as_object()) {
                            for (s, data) in syms {
                                merged_symbols.insert(s.clone(), data.clone());
                            }
                        }
                    }
                    info!(symbol = %sym, "Checkpoint data collected");
                }
                Err(e) => error!(symbol = %sym, error = %e, "Failed to serialize checkpoint"),
            }
        }
        let sym_count = merged_symbols.len();
        let merged = serde_json::json!({
            "version": 1,
            "symbols": merged_symbols,
            "event_index": max_event_index,
        });
        match serde_json::to_string(&merged) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&checkpoint_path, &json) {
                    error!(error = %e, "Failed to write checkpoint");
                } else {
                    info!(
                        symbols = sym_count,
                        path = %checkpoint_path.display(),
                        "Checkpoint saved"
                    );
                }
            }
            Err(e) => error!(error = %e, "Failed to serialize merged checkpoint"),
        }
    }

    let elapsed = start_time.elapsed();
    info!(
        ticks = tick_count,
        micro_orders = micro_order_count,
        elapsed_secs = elapsed.as_secs(),
        "quant_trader shutdown complete"
    );
}

// ── User data stream event processing ──

fn process_user_data_event(text: &str, exchange_state: &mut ExchangeState) {
    let v: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return,
    };

    let event_type = v.get("e").and_then(|v| v.as_str()).unwrap_or("");

    match event_type {
        "ACCOUNT_UPDATE" => {
            // Parse balance updates
            if let Some(data) = v.get("a") {
                // Balances
                if let Some(balances) = data.get("B").and_then(|b| b.as_array()) {
                    for b in balances {
                        let asset = b.get("a").and_then(|v| v.as_str()).unwrap_or("");
                        if asset == "USDT" {
                            if let Some(wb) = b.get("wb").and_then(|v| v.as_str()) {
                                if let Ok(bal) = wb.parse::<f64>() {
                                    let old = exchange_state.balance;
                                    exchange_state.balance = bal;
                                    exchange_state.synced = true;
                                    exchange_state.last_update = std::time::Instant::now();
                                    if (old - bal).abs() > 0.01 {
                                        info!(
                                            old = format!("{:.2}", old),
                                            new = format!("{:.2}", bal),
                                            "Exchange balance updated"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Positions
                if let Some(positions) = data.get("P").and_then(|p| p.as_array()) {
                    for p in positions {
                        let symbol = p.get("s").and_then(|v| v.as_str()).unwrap_or("");
                        let qty: f64 = p
                            .get("pa")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0);
                        let entry_price: f64 = p
                            .get("ep")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0);

                        if !symbol.is_empty() {
                            let notional = qty.abs() * entry_price;
                            info!(
                                symbol = symbol,
                                qty = format!("{:.6}", qty),
                                entry_price = format!("{:.2}", entry_price),
                                notional = format!("{:.2}", notional),
                                "Exchange position updated"
                            );
                            if qty.abs() < 1e-12 {
                                exchange_state.positions.remove(symbol);
                            } else {
                                exchange_state.positions.insert(
                                    symbol.to_string(),
                                    ExchangePosition { qty, notional },
                                );
                            }
                        }
                    }
                }
            }
        }
        "ORDER_TRADE_UPDATE" => {
            if let Some(o) = v.get("o") {
                let symbol = o.get("s").and_then(|v| v.as_str()).unwrap_or("");
                let side = o.get("S").and_then(|v| v.as_str()).unwrap_or("");
                let status = o.get("X").and_then(|v| v.as_str()).unwrap_or("");
                let filled_qty: f64 = o
                    .get("z")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let avg_price: f64 = o
                    .get("ap")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let order_id = o.get("i").and_then(|v| v.as_i64()).unwrap_or(0);
                let realized_pnl: f64 = o
                    .get("rp")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);

                info!(
                    symbol = symbol,
                    side = side,
                    status = status,
                    filled_qty = format!("{:.6}", filled_qty),
                    avg_price = format!("{:.2}", avg_price),
                    realized_pnl = format!("{:.4}", realized_pnl),
                    order_id = order_id,
                    "ORDER_TRADE_UPDATE"
                );
            }
        }
        "MARGIN_CALL" => {
            warn!(event = %text, "MARGIN CALL received");
        }
        _ => {}
    }
}

// ── Order response processing ──

async fn process_order_response(
    text: &str,
    pending: &Arc<tokio::sync::Mutex<HashMap<String, PendingOrder>>>,
    failure_count: &Arc<AtomicU32>,
) {
    let v: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return,
    };

    let req_id = v.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let status = v.get("status").and_then(|v| v.as_i64()).unwrap_or(0);

    let pending_info = {
        let mut map = pending.lock().await;
        map.remove(req_id)
    };

    if status == 200 {
        let order_id = v
            .get("result")
            .and_then(|r| r.get("orderId"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if let Some(p) = &pending_info {
            info!(
                req_id = req_id,
                order_id = order_id,
                symbol = %p.symbol,
                side = %p.side,
                qty = format!("{:.6}", p.qty),
                "Order ACCEPTED"
            );
        } else {
            info!(req_id = req_id, order_id = order_id, "Order ACCEPTED");
        }
        failure_count.store(0, Ordering::Relaxed);
    } else {
        let error_code = v
            .get("error")
            .and_then(|e| e.get("code"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let error_msg = v
            .get("error")
            .and_then(|e| e.get("msg"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        if let Some(p) = &pending_info {
            error!(
                req_id = req_id,
                status = status,
                error_code = error_code,
                error_msg = error_msg,
                symbol = %p.symbol,
                side = %p.side,
                "Order REJECTED"
            );
        } else {
            error!(
                req_id = req_id,
                status = status,
                error_code = error_code,
                error_msg = error_msg,
                "Order REJECTED"
            );
        }
        failure_count.fetch_add(1, Ordering::Relaxed);
    }
}

// ── Qty formatting ──

fn format_qty(qty: f64, lot_size: f64) -> String {
    if lot_size >= 1.0 {
        format!("{:.0}", qty)
    } else {
        let decimals = (-lot_size.log10()).ceil() as usize;
        format!("{:.prec$}", qty, prec = decimals)
    }
}

/// Load .env file from current directory or parent directories.
fn load_dotenv() {
    for dir in &[".", ".."] {
        let path = std::path::Path::new(dir).join(".env");
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((key, value)) = line.split_once('=') {
                        let key = key.trim();
                        let value = value.trim();
                        if !key.is_empty() && std::env::var(key).is_err() {
                            std::env::set_var(key, value);
                        }
                    }
                }
            }
            break;
        }
    }
}

fn init_logging(config: &config::LoggingConfig) {
    use tracing_subscriber::EnvFilter;

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    if config.json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
    }
}

fn chrono_ts(ts_ms: i64) -> String {
    let secs = ts_ms / 1000;
    let nanos = ((ts_ms % 1000) * 1_000_000) as u32;
    let dt = std::time::UNIX_EPOCH + std::time::Duration::new(secs as u64, nanos);
    let elapsed = dt.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}000", elapsed.as_millis())
}

/// Parse interval string to milliseconds: "1s" → 1000, "1m" → 60000, etc.
fn parse_interval_ms(interval: &str) -> i64 {
    let s = interval.trim();
    if s.ends_with('s') {
        s[..s.len() - 1].parse::<i64>().unwrap_or(1) * 1_000
    } else if s.ends_with('m') {
        s[..s.len() - 1].parse::<i64>().unwrap_or(1) * 60_000
    } else if s.ends_with('h') {
        s[..s.len() - 1].parse::<i64>().unwrap_or(1) * 3_600_000
    } else {
        60_000 // default 1m
    }
}
