// startup.inc.rs — Historical kline backfill and exchange state sync.
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
