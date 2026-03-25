"""Parity tests: Rust rust_adaptive_target_qty vs Python AdaptivePositionSizer."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# ── Import Rust function (skip if not built) ──────────────────────
try:
    from _quant_hotpath import rust_adaptive_target_qty
except ImportError:
    pytest.skip("Rust extension not available", allow_module_level=True)

from decision.sizing.adaptive import AdaptivePositionSizer


# ── Helpers ───────────────────────────────────────────────────────

def _snap(equity: float, price: float, symbol: str = "BTCUSDT") -> MagicMock:
    """Build a minimal MagicMock StateSnapshot."""
    snap = MagicMock()
    snap.account.balance = equity
    market = MagicMock()
    market.close = price
    snap.markets = {symbol: market}
    return snap


def _python_target_qty(
    runner_key: str,
    equity: float,
    price: float,
    step_size: float = 0.001,
    min_size: float = 0.001,
    max_qty: float = 0.0,
    weight: float = 1.0,
    leverage: float = 10.0,
    ic_scale: float = 1.0,
    regime_active: bool = True,
    z_scale: float = 1.0,
    symbol: str = "BTCUSDT",
) -> Decimal:
    """Call AdaptivePositionSizer with Rust disabled to get pure-Python result."""
    import decision.sizing.adaptive as mod
    # Temporarily disable Rust path
    orig = mod._RUST_SIZER
    mod._RUST_SIZER = False
    try:
        sizer = AdaptivePositionSizer(
            runner_key=runner_key,
            step_size=step_size,
            min_size=min_size,
            max_qty=max_qty,
        )
        snap = _snap(equity, price, symbol)
        return sizer.target_qty(
            snap, symbol,
            weight=Decimal(str(weight)),
            leverage=leverage,
            ic_scale=ic_scale,
            regime_active=regime_active,
            z_scale=z_scale,
        )
    finally:
        mod._RUST_SIZER = orig


def _rust_target_qty(
    runner_key: str,
    equity: float,
    price: float,
    step_size: float = 0.001,
    min_size: float = 0.001,
    max_qty: float = 0.0,
    weight: float = 1.0,
    leverage: float = 10.0,
    ic_scale: float = 1.0,
    regime_active: bool = True,
    z_scale: float = 1.0,
) -> Decimal:
    """Call Rust function directly, return as Decimal for comparison."""
    result = rust_adaptive_target_qty(
        runner_key, equity, price,
        step_size, min_size, max_qty,
        weight, leverage, ic_scale,
        regime_active, z_scale,
    )
    return Decimal(str(result))


# ── Parity tests ─────────────────────────────────────────────────

CASES = [
    # (runner_key, equity, price, symbol, kwargs)
    ("BTCUSDT_4h", 400.0, 60000.0, "BTCUSDT", {}),
    ("BTCUSDT_4h", 5000.0, 60000.0, "BTCUSDT", {}),
    ("BTCUSDT_4h", 50000.0, 60000.0, "BTCUSDT", {}),
    ("ETHUSDT_4h", 1000.0, 3000.0, "ETHUSDT", {"ic_scale": 1.2}),
    ("ETHUSDT_4h", 1000.0, 3000.0, "ETHUSDT", {"ic_scale": 0.4}),
    ("BTCUSDT", 2000.0, 60000.0, "BTCUSDT", {"regime_active": True}),
    ("BTCUSDT", 2000.0, 60000.0, "BTCUSDT", {"regime_active": False}),
    ("BTCUSDT", 2000.0, 60000.0, "BTCUSDT", {"z_scale": 1.5}),
    ("ETHUSDT", 800.0, 3500.0, "ETHUSDT", {"leverage": 5.0}),
    ("BTCUSDT", 15000.0, 70000.0, "BTCUSDT", {"weight": 0.5}),
]


class TestAdaptiveSizerRustParity:
    @pytest.mark.parametrize("runner_key,equity,price,symbol,kwargs", CASES)
    def test_parity(self, runner_key, equity, price, symbol, kwargs):
        py_result = _python_target_qty(runner_key, equity, price, symbol=symbol, **kwargs)
        rs_result = _rust_target_qty(runner_key, equity, price, **kwargs)
        assert py_result == rs_result, (
            f"Parity mismatch for {runner_key} eq={equity} px={price} {kwargs}: "
            f"python={py_result} rust={rs_result}"
        )

    def test_zero_equity(self):
        py = _python_target_qty("BTCUSDT", 0.0, 60000.0)
        rs = _rust_target_qty("BTCUSDT", 0.0, 60000.0)
        assert py == rs == Decimal("0.001")

    def test_zero_price(self):
        py = _python_target_qty("BTCUSDT", 5000.0, 0.0)
        rs = _rust_target_qty("BTCUSDT", 5000.0, 0.0)
        assert py == rs == Decimal("0.001")

    def test_negative_equity(self):
        py = _python_target_qty("BTCUSDT", -100.0, 60000.0)
        rs = _rust_target_qty("BTCUSDT", -100.0, 60000.0)
        assert py == rs == Decimal("0.001")

    def test_max_qty_clamp(self):
        py = _python_target_qty("BTCUSDT_4h", 50000.0, 60000.0, max_qty=0.01)
        rs = _rust_target_qty("BTCUSDT_4h", 50000.0, 60000.0, max_qty=0.01)
        assert py == rs
        assert py <= Decimal("0.01")

    def test_unknown_runner_key_uses_default_cap(self):
        py = _python_target_qty("SOLUSDT", 2000.0, 100.0, symbol="SOLUSDT")
        rs = _rust_target_qty("SOLUSDT", 2000.0, 100.0)
        # Both should use _DEFAULT_CAP = 0.15
        assert py == rs

    def test_step_size_zero(self):
        """step_size=0 means no rounding."""
        py = _python_target_qty("BTCUSDT", 5000.0, 60000.0, step_size=0.0)
        rs = _rust_target_qty("BTCUSDT", 5000.0, 60000.0, step_size=0.0)
        assert py == rs


class TestAdaptiveSizerRustNaN:
    """NaN guard tests (Rust-only, Python won't see NaN via snapshot)."""

    def test_nan_equity_returns_min(self):
        result = rust_adaptive_target_qty(
            "BTCUSDT", float("nan"), 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        assert result == 0.001

    def test_nan_price_returns_min(self):
        result = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, float("nan"), 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        assert result == 0.001


class TestAdaptiveSizerRustDirect:
    """Direct Rust function tests (not parity, just correctness)."""

    def test_basic_computation(self):
        # small tier, BTCUSDT_4h cap=0.35, lev=10 → notional=400*0.35*10=1400
        # size = 1400/60000 ≈ 0.02333 → 0.023
        qty = rust_adaptive_target_qty(
            "BTCUSDT_4h", 400.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        assert qty == 0.023

    def test_regime_inactive_60pct(self):
        active = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        inactive = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, False, 1.0,
        )
        # inactive should be exactly 60% of active (before rounding)
        assert inactive < active
        # Ratio check with tolerance for rounding
        ratio = inactive / active
        assert 0.55 <= ratio <= 0.65

    def test_large_tier_lower_cap(self):
        # large tier BTCUSDT cap=0.12, medium=0.18 → large should be smaller
        large = rust_adaptive_target_qty(
            "BTCUSDT", 50000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        medium = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, True, 1.0,
        )
        # Per-dollar allocation: large should be lower
        per_dollar_large = large * 60000.0 / 50000.0
        per_dollar_medium = medium * 60000.0 / 5000.0
        assert per_dollar_large < per_dollar_medium
