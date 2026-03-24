// portfolio_allocator_tests.inc.rs — Unit tests for portfolio allocator.
// Included by portfolio_allocator.rs via include!().

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signum() {
        assert_eq!(1.0_f64.signum(), 1.0);
        assert_eq!((-1.0_f64).signum(), -1.0);
        // Rust: 0.0_f64.signum() == 1.0 (positive zero)
        assert_eq!(0.0_f64.signum(), 1.0);
    }

    #[test]
    fn test_round_to_lot() {
        // Positive qty, step=0.1 -> floor
        assert!((round_to_lot(1.57, 0.1) - 1.5).abs() < 1e-12);
        // Negative qty -> floor toward zero
        assert!((round_to_lot(-1.57, 0.1) - (-1.5)).abs() < 1e-12);
        // Exact multiple
        assert!((round_to_lot(2.0, 0.5) - 2.0).abs() < 1e-12);
        // Step=0 -> return qty unchanged
        assert!((round_to_lot(1.57, 0.0) - 1.57).abs() < 1e-12);
        // Large step
        assert!((round_to_lot(7.0, 10.0) - 0.0).abs() < 1e-12);
        assert!((round_to_lot(15.0, 10.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_scale_order_notional_cap() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 5000.0, 0.4, 5.0, 1.0);
        // qty=10, price=1000 -> notional=10000 > 5000 cap
        let scale = alloc.scale_order("ETH", 10.0, 100000.0, 1000.0);
        assert!((scale - 0.5).abs() < 1e-9); // 5000/10000
    }

    #[test]
    fn test_scale_order_leverage_cap() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 50000.0, 0.4, 5.0, 1.0);
        // qty=10, price=1000 -> notional=10000, equity=1000 -> leverage=10 > 3
        let scale = alloc.scale_order("ETH", 10.0, 1000.0, 1000.0);
        assert!((scale - 0.3).abs() < 1e-9); // 3*1000/10000
    }

    #[test]
    fn test_scale_order_invalid_inputs() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 5000.0, 0.4, 5.0, 1.0);
        assert_eq!(alloc.scale_order("X", f64::NAN, 1000.0, 100.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, 0.0, 100.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, 1000.0, 0.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, -1.0, 100.0), 0.0);
    }
}
