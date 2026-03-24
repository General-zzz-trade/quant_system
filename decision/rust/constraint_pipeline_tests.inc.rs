// constraint_pipeline_tests.inc.rs — Unit tests for constraint pipeline.
// Included by constraint_pipeline.rs via include!().

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretize() {
        assert_eq!(discretize(0.6, 0.5), 1.0);
        assert_eq!(discretize(-0.6, 0.5), -1.0);
        assert_eq!(discretize(0.3, 0.5), 0.0);
        assert_eq!(discretize(0.5, 0.5), 0.0); // boundary: not >
        assert_eq!(discretize(-0.5, 0.5), 0.0); // boundary: not <
    }

    #[test]
    fn test_long_only_clip() {
        assert_eq!(long_only_clip(-1.0, true), 0.0);
        assert_eq!(long_only_clip(1.0, true), 1.0);
        assert_eq!(long_only_clip(-1.0, false), -1.0);
    }

    #[test]
    fn test_enforce_hold_step_lockout() {
        // During min-hold lockout, previous signal is maintained
        let (sig, hold) = enforce_hold_step(1.0, -1.0, 3, 10, false, f64::NAN, 0.0, 120);
        assert_eq!(sig, -1.0);
        assert_eq!(hold, 4);
    }

    #[test]
    fn test_enforce_hold_step_change() {
        // After min-hold, signal can change
        let (sig, hold) = enforce_hold_step(1.0, -1.0, 10, 10, false, f64::NAN, 0.0, 120);
        assert_eq!(sig, 1.0);
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_enforce_hold_step_trend_extend() {
        // Trend hold: extend long when trend is up
        let (sig, hold) = enforce_hold_step(0.0, 1.0, 30, 10, true, 0.5, 0.0, 120);
        assert_eq!(sig, 1.0); // held
        assert_eq!(hold, 31);
    }

    #[test]
    fn test_enforce_hold_step_trend_no_extend_at_max() {
        // Don't extend past max_hold
        let (sig, hold) = enforce_hold_step(0.0, 1.0, 120, 10, true, 0.5, 0.0, 120);
        assert_eq!(sig, 0.0); // released
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_enforce_hold_step_trend_extend_short() {
        // Trend hold: extend short when trend is down
        let (sig, hold) = enforce_hold_step(0.0, -1.0, 30, 10, true, -0.5, 0.0, 120);
        assert_eq!(sig, -1.0); // held
        assert_eq!(hold, 31);
    }

    #[test]
    fn test_enforce_hold_step_short_no_extend_wrong_trend() {
        // Don't extend short when trend is up
        let (sig, hold) = enforce_hold_step(0.0, -1.0, 30, 10, true, 0.5, 0.0, 120);
        assert_eq!(sig, 0.0); // released
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_enforce_hold_step_short_no_extend_at_max() {
        // Don't extend short past max_hold
        let (sig, hold) = enforce_hold_step(0.0, -1.0, 120, 10, true, -0.5, 0.0, 120);
        assert_eq!(sig, 0.0); // released — max_hold reached
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_vol_scale() {
        assert!((vol_scale(1.0, 0.02, 0.01) - 0.5).abs() < 1e-10);
        assert!((vol_scale(1.0, 0.005, 0.01) - 1.0).abs() < 1e-10); // capped at 1.0
        assert_eq!(vol_scale(0.0, 0.02, 0.01), 0.0); // zero signal unchanged
        assert_eq!(vol_scale(1.0, f64::NAN, 0.01), 1.0); // NaN vol unchanged
    }

    #[test]
    fn test_discretize_short() {
        assert_eq!(discretize_short(-0.6, 0.5), -1.0);
        assert_eq!(discretize_short(0.6, 0.5), 0.0);
        assert_eq!(discretize_short(0.0, 0.5), 0.0);
    }

    #[test]
    fn test_enforce_hold_array_basic() {
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0];
        let signal = enforce_hold_array(&raw, 3, false, None, 0.0, 120);
        // hold_count starts at 1 for raw[0]=0
        // i=1: hold=1<3 → keep 0, hold=2
        // i=2: hold=2<3 → keep 0, hold=3
        // i=3: hold=3>=3, desired=1.0 != prev=0.0 → 1.0, hold=1
        // i=4: hold=1<3 → keep 1.0, hold=2
        // i=5: hold=2<3 → keep 1.0, hold=3
        // i=6: hold=3>=3, desired=-1.0 != prev=1.0 → -1.0, hold=1
        assert_eq!(signal, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_compute_bear_mask_warmup() {
        let closes = vec![100.0, 101.0, 99.0];
        let mask = compute_bear_mask(&closes, 5);
        assert_eq!(mask, vec![false, false, false]); // all warmup
    }

    #[test]
    fn test_enforce_hold_with_gate_no_gate() {
        // Without gate, should behave identically to enforce_hold_array
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0];
        let signal_no_gate = enforce_hold_with_gate_array(&raw, 3, false, None, 0.0, 120, None, None);
        let signal_plain = enforce_hold_array(&raw, 3, false, None, 0.0, 120);
        assert_eq!(signal_no_gate, signal_plain);
    }

    #[test]
    fn test_enforce_hold_with_gate_overrides_mid_hold() {
        // Gate overrides signal even during min-hold lockout
        // raw:       [0, 0, 1, 1, 0]
        // min_hold=3
        // Without gate: [0, 0, 0, 1, 1]  (hold locks bar 2)
        // Gate at bar 3: [0, 0, 0, 0, 0]  (gate overrides bar 3 to 0)
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        let gate_mask = vec![false, false, false, true, false];
        let signal = enforce_hold_with_gate_array(
            &raw, 3, false, None, 0.0, 120,
            Some(&gate_mask), None,
        );
        // bar 0: raw=0, signal=0, hold=1
        // bar 1: hold=1<3, keep 0, hold=2
        // bar 2: hold=2<3, keep 0, hold=3
        // bar 3: hold=3>=3, desired=1!=0 → 1, hold=1; then gate → 0, hold=1
        // bar 4: hold=1<3, keep 0, hold=2
        assert_eq!(signal, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_enforce_hold_with_gate_uses_gate_scores() {
        // Gate replaces signal with bear model score
        let raw = vec![1.0, 1.0, 1.0, 1.0];
        let gate_mask = vec![false, false, true, true];
        let gate_scores = vec![0.0, 0.0, -1.0, -1.0];
        let signal = enforce_hold_with_gate_array(
            &raw, 1, false, None, 0.0, 120,
            Some(&gate_mask), Some(&gate_scores),
        );
        // bar 0: raw=1, no gate → 1, hold=1
        // bar 1: hold>=1, desired=1, same → 1, hold=2
        // bar 2: hold>=1, desired=1, same → 1, hold=3; gate → -1, hold=1
        // bar 3: hold=1>=1, desired=1, !=prev(-1) → 1, hold=1; gate → -1, hold=1
        assert_eq!(signal, vec![1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_enforce_hold_with_gate_matches_live_semantics() {
        // Scenario: holding long for 5 bars, gate triggers at bar 5
        // Live behavior: gate overrides immediately (bypasses min-hold)
        // Old backtest: gate zeros signal, then re-min-hold would lock it
        let raw = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let gate_mask = vec![false, false, false, false, false, true, true, true];
        let signal = enforce_hold_with_gate_array(
            &raw, 3, false, None, 0.0, 120,
            Some(&gate_mask), None,
        );
        // bar 5: min-hold ok (hold>=3), desired=0!=1 → 0, hold=1; gate → 0 (same), hold=1
        // bar 6: hold=1<3, keep 0; gate → 0 (same)
        // bar 7: hold=2<3, keep 0; gate → 0 (same)
        assert_eq!(signal[5], 0.0);
        assert_eq!(signal[6], 0.0);
        assert_eq!(signal[7], 0.0);
    }
}
