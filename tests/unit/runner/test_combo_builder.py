"""Test combo builder and signal combination logic."""
from runner.builders.combo_builder import (
    ComboConfig,
    CombinedSignal,
    combine_signals,
    build_combo,
)

_AGREE_CFG = ComboConfig(
    mode="agree",
    conviction_both=1.0,
    conviction_single=0.5,
    per_symbol_cap=0.3,
)


class TestCombineSignals:
    def test_both_agree_long(self):
        result = combine_signals(1, 1, _AGREE_CFG)
        assert result.direction == 1
        assert result.conviction == 1.0
        assert result.source == "both_agree"

    def test_both_agree_short(self):
        result = combine_signals(-1, -1, _AGREE_CFG)
        assert result.direction == -1
        assert result.conviction == 1.0

    def test_disagree_returns_flat(self):
        result = combine_signals(1, -1, _AGREE_CFG)
        assert result.direction == 0
        assert result.conviction == 0.0
        assert result.source == "disagree"

    def test_one_long_one_flat(self):
        result = combine_signals(1, 0, _AGREE_CFG)
        assert result.direction == 1
        assert result.conviction == 0.5
        assert result.source == "single_1h"

    def test_flat_and_short(self):
        result = combine_signals(0, -1, _AGREE_CFG)
        assert result.direction == -1
        assert result.conviction == 0.5
        assert result.source == "single_15m"

    def test_both_flat(self):
        result = combine_signals(0, 0, _AGREE_CFG)
        assert result.direction == 0
        assert result.conviction == 0.0

    def test_per_symbol_cap_stored(self):
        assert _AGREE_CFG.per_symbol_cap == 0.3

    def test_returns_named_tuple(self):
        result = combine_signals(1, 1, _AGREE_CFG)
        assert isinstance(result, CombinedSignal)

    def test_disagree_reverse(self):
        result = combine_signals(-1, 1, _AGREE_CFG)
        assert result.direction == 0
        assert result.source == "disagree"


class TestCombineSignalsAnyMode:
    _ANY_CFG = ComboConfig(
        mode="any",
        conviction_both=1.0,
        conviction_single=0.5,
        per_symbol_cap=0.3,
    )

    def test_1h_triggers(self):
        result = combine_signals(1, 0, self._ANY_CFG)
        assert result.direction == 1
        assert result.source == "signal_1h"

    def test_15m_triggers(self):
        result = combine_signals(0, -1, self._ANY_CFG)
        assert result.direction == -1
        assert result.source == "signal_15m"

    def test_both_flat_returns_flat(self):
        result = combine_signals(0, 0, self._ANY_CFG)
        assert result.direction == 0
        assert result.source == "flat"

    def test_1h_takes_priority_over_15m(self):
        # When both are non-zero in "any" mode, 1h takes priority
        result = combine_signals(1, -1, self._ANY_CFG)
        assert result.direction == 1
        assert result.source == "signal_1h"


class TestBuildCombo:
    def test_disabled_returns_none(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(enable_combo=False)
        assert build_combo(cfg) is None

    def test_enabled_returns_config(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enable_combo=True,
            combo_mode="agree",
            combo_conviction_both=1.0,
            combo_conviction_single=0.5,
            combo_per_symbol_cap=0.3,
        )
        result = build_combo(cfg)
        assert result is not None
        assert result.mode == "agree"

    def test_enabled_fields_match(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enable_combo=True,
            combo_mode="any",
            combo_conviction_both=0.9,
            combo_conviction_single=0.4,
            combo_per_symbol_cap=0.25,
        )
        result = build_combo(cfg)
        assert result.mode == "any"
        assert result.conviction_both == 0.9
        assert result.conviction_single == 0.4
        assert result.per_symbol_cap == 0.25

    def test_returns_combo_config_type(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enable_combo=True,
            combo_mode="agree",
            combo_conviction_both=1.0,
            combo_conviction_single=0.5,
            combo_per_symbol_cap=0.3,
        )
        result = build_combo(cfg)
        assert isinstance(result, ComboConfig)


class TestLiveRunnerConfigComboFields:
    """Verify config fields were added to LiveRunnerConfig."""

    def test_default_combo_disabled(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig()
        assert cfg.enable_combo is False

    def test_default_combo_mode(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig()
        assert cfg.combo_mode == "agree"

    def test_default_conviction_both(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig()
        assert cfg.combo_conviction_both == 1.0

    def test_default_conviction_single(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig()
        assert cfg.combo_conviction_single == 0.5

    def test_default_per_symbol_cap(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig()
        assert cfg.combo_per_symbol_cap == 0.3

    def test_can_enable_via_override(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig(enable_combo=True, combo_mode="any")
        assert cfg.enable_combo is True
        assert cfg.combo_mode == "any"

    def test_build_combo_from_live_runner_config(self):
        from runner.config import LiveRunnerConfig
        cfg = LiveRunnerConfig(enable_combo=True)
        result = build_combo(cfg)
        assert result is not None
        assert result.mode == "agree"
        assert result.conviction_both == 1.0
        assert result.per_symbol_cap == 0.3
