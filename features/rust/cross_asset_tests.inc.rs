// cross_asset_tests.inc.rs — Unit tests for cross-asset computer.
// Included by cross_asset.rs via include!().

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_state_ret() {
        let mut a = AssetState::new();
        a.push(100.0, None, None, None);
        a.push(110.0, None, None, None);
        assert!((a.ret(1).unwrap() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_beta_positive() {
        let mut sym = VecDeque::new();
        let mut bench = VecDeque::new();
        for i in 0..30 {
            sym.push_back(0.01 * (i as f64 + 1.0));
            bench.push_back(0.005 * (i as f64 + 1.0));
        }
        let b = beta_from_deques(&sym, &bench, 30).unwrap();
        assert!(b > 0.0);
    }

    #[test]
    fn test_pearson_corr() {
        let x: VecDeque<f64> = (0..10).map(|i| i as f64).collect();
        let y: VecDeque<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let r = pearson(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_range() {
        let mut a = AssetState::new();
        for i in 0..20 {
            a.push(100.0 + i as f64, None, None, None);
        }
        let rsi = a.rsi_14().unwrap();
        assert!(rsi > 0.0 && rsi <= 100.0);
    }
}
