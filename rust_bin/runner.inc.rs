// runner.inc.rs — Main async entry point and event loop.
// Single async fn main() — 1178 lines. Cannot split via include!() because Rust's
// include!() macro expands to a single syntactic element, not multiple statements.
// The function shares ~30 local variables across init/setup/loop/shutdown phases.
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
    // V2: subscribe to depth5@100ms for tick-level depth imbalance signals
    if tick_by_tick {
        for s in &config.trading.symbols {
            streams.push(format!("{}@depth5@100ms", s.to_lowercase()));
        }
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
    let mut last_kill_check = std::time::Instant::now() - std::time::Duration::from_secs(2);
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

            // Kill switch: check file at most once per second (avoid stat() on every WS message)
            if !kill_switch_active && last_kill_check.elapsed() >= std::time::Duration::from_secs(1) {
                last_kill_check = std::time::Instant::now();
                if kill_switch_path.exists() {
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
                        qty = %close_qty_str,
                        req_id = %req_id,
                        "KILL: closing position"
                    );
                    let _ = order_tx.try_send(msg);
                }

                if open_positions.is_empty() {
                    error!("KILL: no open positions to close");
                }
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
                        if !kill_switch_active {
                            let combined_score = combined_bar_score(result.ml_score, micro_sig.as_ref(), tick_by_tick);

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
                                        cli.dry_run,
                                        micro_sig.as_ref(),
                                    );
                                }
                            }
                        }
                    }
                }

                // Tick-by-tick decision on every aggTrade (V2: pure tick signals, no ML dependency)
                if tick_by_tick && !kill_switch_active {
                    let sym = &trade.symbol;
                    // V2: Use enhanced multi-timeframe tick signal
                    let tick_sig = micro_alphas.get(sym).map(|ma| ma.compute_tick_signal(trade.ts_ms));
                    // Legacy signal for logging/compatibility
                    let micro_sig = micro_alphas.get(sym).map(|ma| ma.compute(trade.ts_ms));

                    if let Some(ref tsig) = tick_sig {
                        if tsig.valid && tsig.tick_score.abs() >= micro_threshold
                            && tsig.confidence >= 0.5
                        {
                            // Pure tick score — no ML gating
                            let score = tsig.tick_score;

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
                                    "tick_v2",
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
                                    cli.dry_run,
                                    micro_sig.as_ref(),
                                );
                                if submitted {
                                    micro_order_count += 1;
                                }
                            }
                        }
                    }
                }

                // Periodic aggTrade stats (every 1000 trades)
                if let Some(ss) = sym_states.get(&trade.symbol) {
                    if ss.trade_count % 1000 == 0 && ss.trade_count > 0 {
                        let tsig = micro_alphas
                            .get(&trade.symbol)
                            .map(|ma| ma.compute_tick_signal(trade.ts_ms));
                        if let Some(tsig) = tsig {
                            info!(
                                symbol = %trade.symbol,
                                trades = ss.trade_count,
                                f3s = format!("{:.3}", tsig.flow_3s),
                                f10s = format!("{:.3}", tsig.flow_10s),
                                f30s = format!("{:.3}", tsig.flow_30s),
                                vol_spike = format!("{:.2}", tsig.volume_spike),
                                cluster = format!("{:.3}", tsig.large_cluster),
                                momentum = format!("{:.3}", tsig.price_momentum),
                                depth = format!("{:.3}", tsig.depth_imbalance),
                                tick = format!("{:.4}", tsig.tick_score),
                                conf = format!("{:.2}", tsig.confidence),
                                micro_orders = micro_order_count,
                                "tick_v2 stats"
                            );
                        }
                    }
                }

                continue;
            }

            // ── depth5 path (V2: orderbook depth for tick-level signals) ──
            if msg.contains("\"depthUpdate\"") || msg.contains("\"lastUpdateId\"") {
                if let Some(depth) = parse_depth5(&msg) {
                    if let Some(ma) = micro_alphas.get_mut(&depth.0) {
                        ma.update_depth(_quant_hotpath::decision::micro_alpha::DepthSnapshot {
                            ts_ms: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as i64,
                            best_bid: depth.1,
                            best_ask: depth.2,
                            bid_qty: depth.3,
                            ask_qty: depth.4,
                        });
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
            if !kill_switch_active {
                let combined_score = combined_bar_score(result.ml_score, micro_sig.as_ref(), tick_by_tick);

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
                            cli.dry_run,
                            micro_sig.as_ref(),
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
