// micro_alpha_tests.inc.rs — Unit tests for MicroAlpha.
// Included by micro_alpha.rs via include!() macro.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_signals() {
        let ma = MicroAlpha::new(MicroAlphaConfig::default());
        let sig = ma.compute(1000);
        assert!(!sig.valid);
        assert_eq!(sig.micro_score, 0.0);
    }

    #[test]
    fn test_buy_flow_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, true);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.trade_flow_imbalance > 0.9, "Expected strong buy imbalance, got {}", sig.trade_flow_imbalance);
        assert!(sig.micro_score > 0.0);
    }

    #[test]
    fn test_sell_flow_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, false);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.trade_flow_imbalance < -0.9);
        assert!(sig.micro_score < 0.0);
    }

    #[test]
    fn test_large_trade_detection() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            large_trade_mult: 3.0,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..30 {
            ma.push_trade(now - 15000 + i * 500, 50000.0, 0.01, true);
        }
        ma.push_trade(now - 100, 50000.0, 1.0, true);

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.large_trade_signal > 0.0, "Expected positive large trade signal");
    }

    #[test]
    fn test_volume_spike() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 5_000,
            min_trades: 3,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 9000 + i * 500, 50000.0, 0.001, true);
        }
        for i in 0..10 {
            ma.push_trade(now - 4000 + i * 400, 50000.0, 0.1, true);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.volume_spike > 1.5, "Expected volume spike > 1.5, got {}", sig.volume_spike);
    }

    #[test]
    fn test_multi_timeframe_agreement() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Sustained buy pressure across 30s
        for i in 0..60 {
            ma.push_trade(now - 30000 + i * 500, 50000.0, 0.1, true);
        }

        let sig = ma.compute_tick_signal(now);
        assert!(sig.valid);
        assert!(sig.flow_3s > 0.5);
        assert!(sig.flow_10s > 0.5);
        assert!(sig.flow_30s > 0.5);
        assert!(sig.tick_score > 0.0, "Expected positive tick score for all-buy flow");
    }

    #[test]
    fn test_conflicting_timeframes() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Sell pressure in the past (30s)
        for i in 0..30 {
            ma.push_trade(now - 25000 + i * 500, 50000.0, 0.1, false);
        }
        // Buy pressure recently (3s)
        for i in 0..20 {
            ma.push_trade(now - 2500 + i * 125, 50000.0, 0.1, true);
        }

        let sig = ma.compute_tick_signal(now);
        assert!(sig.valid);
        // 3s should be buy, 30s should be sell → conflicting
        assert!(sig.flow_3s > 0.0);
        assert!(sig.flow_30s < 0.0);
        // Score should be dampened due to disagreement
        assert!(sig.tick_score.abs() < 0.5, "Expected dampened score, got {}", sig.tick_score);
    }

    #[test]
    fn test_tick_position_roundtrip() {
        let mut pos = TickPosition::new();

        // Open long
        let cost = pos.enter(1000, 50000.0, 0.1, 2e-4);
        assert!(cost < 0.0); // paid cost
        assert_eq!(pos.qty, 0.1);

        // Close long at profit
        let pnl = pos.enter(2000, 50100.0, -0.1, 2e-4);
        assert!(pnl > 0.0); // 50100-50000 = 100 * 0.1 = 10 - costs
        assert_eq!(pos.qty, 0.0);
        assert_eq!(pos.trade_count, 1);
        assert_eq!(pos.win_count, 1);
    }

    #[test]
    fn test_depth_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, true);
        }

        // Push depth with bid-heavy imbalance
        ma.update_depth(DepthSnapshot {
            ts_ms: now - 100,
            best_bid: 49999.0,
            best_ask: 50001.0,
            bid_qty: 10.0,
            ask_qty: 2.0,
        });

        let sig = ma.compute_tick_signal(now);
        assert!(sig.depth_imbalance > 0.0, "Expected positive depth imbalance");
    }

    #[test]
    fn test_large_trade_cluster() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            large_trade_mult: 3.0,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Build baseline
        for i in 0..30 {
            ma.push_trade(now - 20000 + i * 500, 50000.0, 0.01, true);
        }
        // 3 consecutive large buy trades in 1s
        ma.push_trade(now - 800, 50000.0, 1.0, true);
        ma.push_trade(now - 400, 50000.0, 1.5, true);
        ma.push_trade(now - 100, 50000.0, 2.0, true);

        let sig = ma.compute_tick_signal(now);
        assert!(sig.large_cluster > 0.0, "Expected positive cluster signal, got {}", sig.large_cluster);
    }
}
