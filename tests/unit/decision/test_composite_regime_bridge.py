# tests/unit/decision/test_composite_regime_bridge.py
"""Test per-symbol CompositeRegime enable in RegimeAwareDecisionModule."""
from unittest.mock import MagicMock


class TestPerSymbolCompositeRegime:
    def test_btc_uses_composite_regime(self):
        """BTC in composite_regime_symbols should use ParamRouter."""
        from decision.regime_bridge import RegimeAwareDecisionModule
        inner = MagicMock()
        inner.decide.return_value = []
        module = RegimeAwareDecisionModule(
            inner=inner,
            composite_regime_symbols=("BTCUSDT",),
        )
        # BTC should have composite detector
        assert module._is_composite_symbol("BTCUSDT") is True
        assert module._is_composite_symbol("ETHUSDT") is False

    def test_non_composite_symbol_uses_fixed_params(self):
        """ETH not in composite_regime_symbols should not route params."""
        from decision.regime_bridge import RegimeAwareDecisionModule
        inner = MagicMock()
        inner.decide.return_value = []
        module = RegimeAwareDecisionModule(
            inner=inner,
            composite_regime_symbols=("BTCUSDT",),
        )
        # ETH should not get param routing
        assert module._should_route_params("ETHUSDT") is False
        assert module._should_route_params("BTCUSDT") is True

    def test_empty_composite_symbols_all_fixed(self):
        """With empty composite_regime_symbols, no symbol uses param routing."""
        from decision.regime_bridge import RegimeAwareDecisionModule
        inner = MagicMock()
        inner.decide.return_value = []
        module = RegimeAwareDecisionModule(
            inner=inner,
            composite_regime_symbols=(),
        )
        assert module._is_composite_symbol("BTCUSDT") is False
        assert module._should_route_params("BTCUSDT") is False
        assert module._should_route_params("ETHUSDT") is False

    def test_multiple_composite_symbols(self):
        """Multiple symbols can be designated as composite."""
        from decision.regime_bridge import RegimeAwareDecisionModule
        inner = MagicMock()
        inner.decide.return_value = []
        module = RegimeAwareDecisionModule(
            inner=inner,
            composite_regime_symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert module._is_composite_symbol("BTCUSDT") is True
        assert module._is_composite_symbol("ETHUSDT") is True
        assert module._should_route_params("BTCUSDT") is True
        assert module._should_route_params("ETHUSDT") is True
        assert module._is_composite_symbol("SUIUSDT") is False
        assert module._should_route_params("SUIUSDT") is False
