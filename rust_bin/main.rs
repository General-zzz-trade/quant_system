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

use _quant_hotpath::decision::math::rust_fixed_fraction_qty;
use _quant_hotpath::common::json_parse::{parse_agg_trade_native, parse_kline_native};
use _quant_hotpath::decision::micro_alpha::{MicroAlpha, MicroAlphaConfig, MicroAlphaSignal};
use _quant_hotpath::execution::order_submit::{hmac_sha256_hex, RustWsOrderGateway};
use _quant_hotpath::engine::tick_processor::RustTickProcessor;

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

// ── Types, state structs, and order/signal logic ──
include!("types.inc.rs");

// ── Historical kline backfill and exchange state sync ──
include!("startup.inc.rs");

// ── Main async entry point and event loop ──
include!("runner.inc.rs");

// ── Utility functions: user data, order response, parsing, logging ──
include!("utils.inc.rs");
