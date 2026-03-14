"""Tests for RegimeGate checkpoint/restore persistence."""
from __future__ import annotations


from decision.regime_gate import RegimeGate


def _make_regime_gate(enabled=True) -> RegimeGate:
    from alpha.v11_config import RegimeGateConfig
    return RegimeGate(config=RegimeGateConfig(enabled=enabled))


class TestRegimeGatePersistence:
    def test_checkpoint_restore_roundtrip(self):
        rg = _make_regime_gate()

        # Populate buffers
        for i in range(200):
            rg.evaluate({"bb_width_20": 0.01 * i, "vol_of_vol": 0.005 * i})

        assert len(rg._bb_width_buf) == 200
        assert len(rg._vol_of_vol_buf) == 200

        data = rg.checkpoint()
        assert len(data["bb_width_buf"]) == 200
        assert len(data["vol_of_vol_buf"]) == 200

        # Restore into fresh gate
        rg2 = _make_regime_gate()
        assert len(rg2._bb_width_buf) == 0
        rg2.restore(data)

        assert len(rg2._bb_width_buf) == 200
        assert len(rg2._vol_of_vol_buf) == 200
        assert list(rg2._bb_width_buf) == data["bb_width_buf"]
        assert list(rg2._vol_of_vol_buf) == data["vol_of_vol_buf"]

    def test_regime_detection_same_before_after_restore(self):
        rg = _make_regime_gate()

        # Build up enough data for percentile computation
        for i in range(150):
            rg.evaluate({"bb_width_20": 0.01 + 0.001 * i, "vol_of_vol": 0.005 + 0.0005 * i})

        # Get result before checkpoint
        label1, scale1 = rg.evaluate({"bb_width_20": 0.5, "vol_of_vol": 0.3, "adx_14": 30.0})

        data = rg.checkpoint()
        rg2 = _make_regime_gate()
        rg2.restore(data)

        label2, scale2 = rg2.evaluate({"bb_width_20": 0.5, "vol_of_vol": 0.3, "adx_14": 30.0})
        assert label1 == label2
        assert scale1 == scale2

    def test_restore_empty(self):
        rg = _make_regime_gate()
        rg.restore({})
        assert len(rg._bb_width_buf) == 0
        assert len(rg._vol_of_vol_buf) == 0

    def test_restore_preserves_maxlen(self):
        rg = _make_regime_gate()
        # Restore with data that exceeds maxlen — should be truncated
        data = {"bb_width_buf": list(range(1000)), "vol_of_vol_buf": list(range(1000))}
        rg.restore(data)
        assert len(rg._bb_width_buf) == 720  # maxlen
        assert len(rg._vol_of_vol_buf) == 720
