// gate_chain_tests.inc.rs — Unit tests for RustGateChain.
#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> GateContext {
        GateContext {
            symbol: "ETHUSDT".to_string(),
            side: "buy".to_string(),
            signal: 1,
            qty: 10.0,
            price: 2000.0,
            equity: 1000.0,
            peak_equity: 1200.0,
            drawdown_pct: 0.0,
            z_score: 1.5,
            avg_correlation: 0.3,
            alpha_health_scale: 1.0,
            staged_risk_scale: 1.0,
            regime_scale: 1.0,
            consensus: HashMap::new(),
        }
    }

    #[test]
    fn test_bracket_leverage_default() {
        let brackets = EquityLeverageGate::default_brackets();
        assert_eq!(bracket_leverage(500.0, &brackets), 1.5);
        assert_eq!(bracket_leverage(10_000.0, &brackets), 1.5);
        assert_eq!(bracket_leverage(30_000.0, &brackets), 1.0);
        assert_eq!(bracket_leverage(100_000.0, &brackets), 1.0);
    }

    #[test]
    fn test_z_scale_factor() {
        assert_eq!(z_scale_factor(2.5), 1.5);
        assert_eq!(z_scale_factor(-2.5), 1.5);
        assert_eq!(z_scale_factor(1.5), 1.0);
        assert_eq!(z_scale_factor(0.7), 0.7);
        assert_eq!(z_scale_factor(0.3), 0.5);
        assert_eq!(z_scale_factor(0.0), 0.5);
    }

    #[test]
    fn test_equity_leverage_gate() {
        let gate = EquityLeverageGate::new(EquityLeverageGate::default_brackets());
        let mut ctx = make_ctx();
        ctx.equity = 1000.0;
        ctx.z_score = 1.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        // equity=1000 -> bracket 1.5, z=1.5 -> z_scale=1.0, total=1.5
        assert!((r.scale - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_contrarian_boost() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.consensus.insert("BTCUSDT".into(), -1);
        ctx.consensus.insert("SOLUSDT".into(), -1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.3).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_strong_agreement() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.consensus.insert("BTCUSDT".into(), 1);
        ctx.consensus.insert("SOLUSDT".into(), 1);
        ctx.consensus.insert("SUIUSDT".into(), 1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_flat_signal() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_drawdown_gate_allows() {
        let gate = DrawdownGate {
            max_drawdown_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.10;
        let r = gate.check(&ctx);
        assert!(r.allowed);
    }

    #[test]
    fn test_drawdown_gate_rejects() {
        let gate = DrawdownGate {
            max_drawdown_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.25;
        let r = gate.check(&ctx);
        assert!(!r.allowed);
        assert!(r.reason.contains("drawdown"));
    }

    #[test]
    fn test_correlation_gate() {
        let gate = CorrelationGate {
            max_avg_correlation: 0.70,
        };
        let mut ctx = make_ctx();

        ctx.avg_correlation = 0.5;
        assert!(gate.check(&ctx).allowed);

        ctx.avg_correlation = 0.8;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_alpha_health_gate() {
        let gate = AlphaHealthGate;
        let mut ctx = make_ctx();

        ctx.alpha_health_scale = 0.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.5).abs() < 1e-6);

        ctx.alpha_health_scale = 0.0;
        assert!(!gate.check(&ctx).allowed);

        ctx.alpha_health_scale = -0.1;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_staged_risk_gate() {
        let gate = StagedRiskGate;
        let mut ctx = make_ctx();

        ctx.staged_risk_scale = 0.8;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.8).abs() < 1e-6);

        ctx.staged_risk_scale = 0.0;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_regime_sizer_gate() {
        let gate = RegimeSizerGate;
        let mut ctx = make_ctx();

        ctx.regime_scale = 0.6;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.6).abs() < 1e-6);

        // Negative clamped to 0
        ctx.regime_scale = -0.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_notional_limit_gate() {
        let gate = NotionalLimitGate {
            max_notional: 5_000.0,
        };
        let mut ctx = make_ctx();

        // Within limit: qty=1.0, price=2000 -> notional=2000
        ctx.qty = 1.0;
        ctx.price = 2000.0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);

        // Over limit: qty=5.0, price=2000 -> notional=10000
        ctx.qty = 5.0;
        ctx.price = 2000.0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        // Should clamp: 5000/2000 = 2.5, scale = 2.5/5.0 = 0.5
        assert!((r.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_min_qty_gate() {
        let gate = MinQtyGate { min_qty: 0.01 };
        let mut ctx = make_ctx();

        ctx.qty = 0.1;
        assert!(gate.check(&ctx).allowed);

        ctx.qty = 0.005;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_chain_short_circuits_on_rejection() {
        // Build a chain: drawdown (will reject) -> alpha_health (should not run)
        let gates: Vec<Box<dyn Gate>> = vec![
            Box::new(DrawdownGate {
                max_drawdown_pct: 0.10,
            }),
            Box::new(AlphaHealthGate),
        ];

        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.15; // above 10% limit

        // Simulate chain logic
        let mut rejected = false;
        let mut reject_gate = String::new();
        for gate in &gates {
            let r = gate.check(&ctx);
            if !r.allowed {
                rejected = true;
                reject_gate = gate.name().to_string();
                break;
            }
        }
        assert!(rejected);
        assert_eq!(reject_gate, "drawdown");
    }

    #[test]
    fn test_cumulative_scaling() {
        let gates: Vec<Box<dyn Gate>> = vec![
            Box::new(AlphaHealthGate),
            Box::new(RegimeSizerGate),
            Box::new(StagedRiskGate),
        ];

        let mut ctx = make_ctx();
        ctx.alpha_health_scale = 0.5;
        ctx.regime_scale = 0.8;
        ctx.staged_risk_scale = 0.9;

        let mut cumulative = 1.0_f64;
        for gate in &gates {
            let r = gate.check(&ctx);
            assert!(r.allowed);
            cumulative *= r.scale;
        }
        // 0.5 * 0.8 * 0.9 = 0.36
        assert!((cumulative - 0.36).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_mixed_signals() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        // 2 agree, 2 disagree -> 50% agree -> mixed -> 0.7
        ctx.consensus.insert("BTCUSDT".into(), 1);
        ctx.consensus.insert("SOLUSDT".into(), 1);
        ctx.consensus.insert("SUIUSDT".into(), -1);
        ctx.consensus.insert("AXSUSDT".into(), -1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_self_excluded() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.symbol = "ETHUSDT".to_string();
        // Only self in consensus -> no other signals -> 1.0
        ctx.consensus.insert("ETHUSDT".into(), 1);
        let r = gate.check(&ctx);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_ignores_flat_others() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        // Others are flat (signal=0) -> no active others -> 1.0
        ctx.consensus.insert("BTCUSDT".into(), 0);
        ctx.consensus.insert("SOLUSDT".into(), 0);
        let r = gate.check(&ctx);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }
}
