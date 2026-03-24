// utils.inc.rs — Utility functions: user data, order response, parsing, logging.
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

/// Parse depth5 message: returns (symbol, best_bid, best_ask, bid_qty_sum, ask_qty_sum)
fn parse_depth5(msg: &str) -> Option<(String, f64, f64, f64, f64)> {
    let v: serde_json::Value = serde_json::from_str(msg).ok()?;
    // Combined stream format: {"stream":"btcusdt@depth5@100ms","data":{...}}
    let data = v.get("data").unwrap_or(&v);
    let stream = v.get("stream").and_then(|s| s.as_str()).unwrap_or("");
    let symbol = stream.split('@').next().unwrap_or("").to_uppercase();
    if symbol.is_empty() { return None; }

    let bids = data.get("b")?.as_array()?;
    let asks = data.get("a")?.as_array()?;
    if bids.is_empty() || asks.is_empty() { return None; }

    let best_bid: f64 = bids[0].get(0)?.as_str()?.parse().ok()?;
    let best_ask: f64 = asks[0].get(0)?.as_str()?.parse().ok()?;

    let bid_qty: f64 = bids.iter()
        .filter_map(|b| b.get(1)?.as_str()?.parse::<f64>().ok())
        .sum();
    let ask_qty: f64 = asks.iter()
        .filter_map(|a| a.get(1)?.as_str()?.parse::<f64>().ok())
        .sum();

    Some((symbol, best_bid, best_ask, bid_qty, ask_qty))
}

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
