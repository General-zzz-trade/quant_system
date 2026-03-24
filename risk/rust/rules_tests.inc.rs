// rules_tests.inc.rs — Unit tests for risk rules.
// Included by rules.rs via include!() macro.

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> EvalContext {
        EvalContext {
            symbol: "ETHUSDT".to_string(),
            side: "buy".to_string(),
            qty: 1.0,
            price: 3000.0,
            notional: 3000.0,
            current_position_qty: 0.0,
            current_position_notional: 0.0,
            account_equity: 10000.0,
            gross_exposure: 5000.0,
            net_exposure: 2000.0,
            max_symbol_concentration: 0.3,
            drawdown_pct: 0.05,
            recent_order_count: 0,
            recent_order_window_secs: 0.0,
            avg_correlation: 0.0,
            portfolio_var_pct: 0.0,
            is_reduce_only: false,
        }
    }

    #[test]
    fn test_max_position_allow() {
        let rule = MaxPositionRule {
            name: "max_pos".to_string(),
            max_qty: 10.0,
            max_notional: 100000.0,
        };
        let ctx = make_ctx();
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Allow));
    }

    #[test]
    fn test_max_position_reject() {
        let rule = MaxPositionRule {
            name: "max_pos".to_string(),
            max_qty: 0.5,
            max_notional: 0.0,
        };
        let ctx = make_ctx();
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reduce { .. }));
    }

    #[test]
    fn test_max_position_reduce_only_allows() {
        let rule = MaxPositionRule {
            name: "max_pos".to_string(),
            max_qty: 0.1,
            max_notional: 0.0,
        };
        let mut ctx = make_ctx();
        ctx.current_position_qty = 2.0;
        ctx.side = "sell".to_string(); // reducing long
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Allow));
    }

    #[test]
    fn test_leverage_cap_reject() {
        let rule = LeverageCapRule {
            name: "lev".to_string(),
            max_gross_leverage: 0.1,
            max_net_leverage: 0.0,
        };
        let ctx = make_ctx();
        // gross_exposure=5000, equity=10000, cap=0.1 → max_gross=1000 < 5000 → no headroom → REJECT
        assert!(matches!(
            rule.evaluate(&ctx),
            RiskVerdict::Reject { .. }
        ));
    }

    #[test]
    fn test_drawdown_kill() {
        let rule = MaxDrawdownRule {
            name: "dd".to_string(),
            warning_pct: 0.10,
            kill_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.25;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }

    #[test]
    fn test_drawdown_warning() {
        let rule = MaxDrawdownRule {
            name: "dd".to_string(),
            warning_pct: 0.10,
            kill_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.15;
        match rule.evaluate(&ctx) {
            RiskVerdict::Reduce { factor, .. } => {
                assert!((factor - 0.5).abs() < 1e-9);
            }
            _ => panic!("Expected Reduce"),
        }
    }

    #[test]
    fn test_drawdown_reduce_only_past_kill() {
        let rule = MaxDrawdownRule {
            name: "dd".to_string(),
            warning_pct: 0.10,
            kill_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.25;
        ctx.current_position_qty = 5.0;
        ctx.side = "sell".to_string();
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Allow));
    }

    #[test]
    fn test_nan_price_rejects() {
        let rule = MaxPositionRule {
            name: "max_pos".to_string(),
            max_qty: 10.0,
            max_notional: 0.0,
        };
        let mut ctx = make_ctx();
        ctx.price = f64::NAN;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }

    #[test]
    fn test_nan_price_reduce_only_allows() {
        let rule = MaxPositionRule {
            name: "max_pos".to_string(),
            max_qty: 10.0,
            max_notional: 0.0,
        };
        let mut ctx = make_ctx();
        ctx.price = f64::NAN;
        ctx.is_reduce_only = true;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Allow));
    }

    #[test]
    fn test_order_frequency_reject() {
        let rule = OrderFrequencyRule::new("freq".to_string(), 5, 60.0);
        let mut ctx = make_ctx();
        ctx.recent_order_count = 100;
        ctx.recent_order_window_secs = 60.0;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }

    #[test]
    fn test_correlation_reject() {
        let rule = CorrelationLimitRule {
            name: "corr".to_string(),
            max_avg_correlation: 0.7,
        };
        let mut ctx = make_ctx();
        ctx.avg_correlation = 0.85;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }

    #[test]
    fn test_var_reject() {
        let rule = VaRLimitRule {
            name: "var".to_string(),
            max_var_pct: 5.0,
        };
        let mut ctx = make_ctx();
        ctx.portfolio_var_pct = 8.0;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }

    #[test]
    fn test_portfolio_limits_concentration() {
        let rule = PortfolioLimitsRule {
            name: "port".to_string(),
            max_concentration: 0.2,
            max_gross_exposure: 10.0,
        };
        let mut ctx = make_ctx();
        ctx.max_symbol_concentration = 0.5;
        assert!(matches!(rule.evaluate(&ctx), RiskVerdict::Reject { .. }));
    }
}
