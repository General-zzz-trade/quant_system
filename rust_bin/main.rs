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
//   quant_trader --config config.yaml              # dry run (default)
//   quant_trader --config config.yaml --live       # orders; also needs trading.mode: live

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

    /// Send real orders. Without this flag nothing is ever submitted.
    /// Live trading additionally requires `trading.mode: live` in the config.
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Force dry run (already the default; kept for existing command lines).
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

/// Hard ceiling on total notional / equity, the same 3x cap as
/// `shared/exposure_gate.py`. Configuration may lower it, never raise it.
const MAX_TOTAL_LEVERAGE: f64 = 3.0;

/// Orders are sent only with `--live`, without `--dry-run`, and with
/// `trading.mode: live`. Every other combination is a dry run.
fn orders_enabled(live_flag: bool, dry_run_flag: bool, mode: &str) -> bool {
    live_flag && !dry_run_flag && mode == "live"
}

/// The configured leverage limit, capped at `MAX_TOTAL_LEVERAGE`.
fn effective_max_leverage(configured: f64) -> Result<f64, String> {
    if !configured.is_finite() || configured <= 0.0 {
        return Err(format!("risk.max_leverage must be a positive number, got {}", configured));
    }
    Ok(configured.min(MAX_TOTAL_LEVERAGE))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_need_live_flag_and_live_mode() {
        assert!(!orders_enabled(false, false, "live"));
        assert!(!orders_enabled(true, false, "paper"));
        assert!(!orders_enabled(true, true, "live"));
        assert!(orders_enabled(true, false, "live"));
    }

    #[test]
    fn leverage_is_capped_and_validated() {
        assert_eq!(effective_max_leverage(5.0), Ok(3.0));
        assert_eq!(effective_max_leverage(1.5), Ok(1.5));
        assert!(effective_max_leverage(0.0).is_err());
        assert!(effective_max_leverage(f64::NAN).is_err());
    }

    #[test]
    fn cli_defaults_to_dry_run() {
        let cli = Cli::try_parse_from(["quant_trader", "--config", "c.yaml"]).unwrap();
        assert!(!cli.live);
        assert!(!orders_enabled(cli.live, cli.dry_run, "live"));
    }
}

