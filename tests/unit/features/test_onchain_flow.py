from features.onchain_flow import OnchainFlowComputer, ONCHAIN_FEATURES


class TestOnchainFlowComputer:
    def test_insufficient_history_returns_none(self):
        comp = OnchainFlowComputer()
        feats = comp.update(100.0)
        assert feats["exchange_inflow_zscore"] is None

    def test_ma_ratio_after_24_bars(self):
        comp = OnchainFlowComputer()
        for _ in range(24):
            feats = comp.update(100.0)
        assert feats["exchange_inflow_ma_ratio"] is not None
        assert abs(feats["exchange_inflow_ma_ratio"] - 1.0) < 1e-10

    def test_zscore_after_warmup(self):
        comp = OnchainFlowComputer()
        for i in range(200):
            feats = comp.update(100.0 + i * 0.1)
        assert feats["exchange_inflow_zscore"] is not None

    def test_none_input(self):
        comp = OnchainFlowComputer()
        feats = comp.update(None)
        assert all(v is None for v in feats.values())

    def test_spike_gives_high_zscore(self):
        comp = OnchainFlowComputer()
        for _ in range(200):
            comp.update(100.0)
        feats = comp.update(500.0)  # 5x spike
        assert feats["exchange_inflow_zscore"] is not None
        assert feats["exchange_inflow_zscore"] > 2.0

    def test_feature_names(self):
        assert len(ONCHAIN_FEATURES) == 2
