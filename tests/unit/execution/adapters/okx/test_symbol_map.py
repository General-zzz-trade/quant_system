"""Unit tests for OKX symbol + quantity conversion.

These tests are **critical safety gates**: a wrong conversion could place
an order 100× larger (or smaller) than intended.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.adapters.okx.symbol_map import (
    InstrumentMeta,
    coin_to_contracts,
    contracts_to_coin,
    from_okx_symbol,
    round_price_to_tick,
    to_okx_symbol,
)


BTC_META = InstrumentMeta(
    inst_id="BTC-USDT-SWAP",
    ct_val=Decimal("0.01"),   # 1 contract = 0.01 BTC
    ct_val_ccy="BTC",
    lot_sz=Decimal("0.01"),
    min_sz=Decimal("0.01"),
    tick_sz=Decimal("0.1"),
    max_lever=100,
)

ETH_META = InstrumentMeta(
    inst_id="ETH-USDT-SWAP",
    ct_val=Decimal("0.1"),    # 1 contract = 0.1 ETH
    ct_val_ccy="ETH",
    lot_sz=Decimal("0.01"),
    min_sz=Decimal("0.01"),
    tick_sz=Decimal("0.01"),
    max_lever=100,
)

SOL_META = InstrumentMeta(
    inst_id="SOL-USDT-SWAP",
    ct_val=Decimal("1"),      # 1 contract = 1 SOL (different from BTC/ETH!)
    ct_val_ccy="SOL",
    lot_sz=Decimal("0.01"),
    min_sz=Decimal("0.01"),
    tick_sz=Decimal("0.01"),
    max_lever=50,
)


# ── Symbol mapping ─────────────────────────────────────────────────
class TestSymbolMap:
    def test_btc_usdt_to_okx(self):
        assert to_okx_symbol("BTCUSDT") == "BTC-USDT-SWAP"

    def test_eth_usdt_to_okx(self):
        assert to_okx_symbol("ETHUSDT") == "ETH-USDT-SWAP"

    def test_strips_4h_suffix(self):
        assert to_okx_symbol("BTCUSDT_4h") == "BTC-USDT-SWAP"
        assert to_okx_symbol("ETHUSDT_4h") == "ETH-USDT-SWAP"

    def test_sol_mapping(self):
        assert to_okx_symbol("SOLUSDT") == "SOL-USDT-SWAP"
        assert to_okx_symbol("SOLUSDT_4h") == "SOL-USDT-SWAP"
        assert from_okx_symbol("SOL-USDT-SWAP") == "SOLUSDT"

    def test_strips_1h_suffix(self):
        assert to_okx_symbol("BTCUSDT_1h") == "BTC-USDT-SWAP"

    def test_lowercase_input(self):
        assert to_okx_symbol("btcusdt") == "BTC-USDT-SWAP"

    def test_unmapped_raises(self):
        # SOL is now mapped (2026-04-11) — use a truly unsupported symbol
        with pytest.raises(KeyError):
            to_okx_symbol("DOGEUSDT")

    def test_from_okx_roundtrip(self):
        assert from_okx_symbol("BTC-USDT-SWAP") == "BTCUSDT"
        assert from_okx_symbol("ETH-USDT-SWAP") == "ETHUSDT"


# ── BTC quantity conversion ────────────────────────────────────────
class TestBtcQuantity:
    """BTC: ctVal=0.01 so 1 BTC = 100 contracts, 0.01 BTC = 1 contract."""

    def test_one_btc_is_100_contracts(self):
        assert coin_to_contracts(1.0, BTC_META) == Decimal("100")

    def test_half_btc_is_50_contracts(self):
        assert coin_to_contracts(0.5, BTC_META) == Decimal("50")

    def test_exact_lot_size(self):
        # 0.01 BTC = exactly 1 contract
        assert coin_to_contracts(0.01, BTC_META) == Decimal("1")

    def test_below_min_returns_zero(self):
        # min_sz=0.01 contracts, so min coin = 0.01 * 0.01 = 0.0001 BTC
        assert coin_to_contracts(0.00005, BTC_META) == Decimal("0")

    def test_at_min_size(self):
        # 0.0001 BTC / 0.01 = 0.01 contracts = exactly min_sz
        result = coin_to_contracts(0.0001, BTC_META)
        assert result == Decimal("0.01")

    def test_rounds_down_never_over(self):
        # 0.999 BTC / 0.01 = 99.9 → rounds down to 99.9 (lotSz=0.01 allows .9)
        # But 0.9991 BTC / 0.01 = 99.91 → round down to 99.91
        assert coin_to_contracts(0.9991, BTC_META) == Decimal("99.91")
        # Never rounds UP
        assert coin_to_contracts(0.9999, BTC_META) == Decimal("99.99")

    def test_never_exceeds_requested(self):
        """Guarantee: round-trip coin_qty never increases after conversion."""
        for coin in [0.001, 0.01, 0.123, 0.9, 1.5, 10.0]:
            contracts = coin_to_contracts(coin, BTC_META)
            back = contracts_to_coin(contracts, BTC_META)
            assert back <= Decimal(str(coin)), f"round-trip increased: {coin} → {back}"


# ── ETH quantity conversion ────────────────────────────────────────
class TestEthQuantity:
    """ETH: ctVal=0.1 so 1 ETH = 10 contracts, 0.1 ETH = 1 contract."""

    def test_one_eth_is_10_contracts(self):
        assert coin_to_contracts(1.0, ETH_META) == Decimal("10")

    def test_point_one_eth_is_one_contract(self):
        assert coin_to_contracts(0.1, ETH_META) == Decimal("1")

    def test_below_min_returns_zero(self):
        # min_sz=0.01 contracts, so min coin = 0.01 * 0.1 = 0.001 ETH
        assert coin_to_contracts(0.0005, ETH_META) == Decimal("0")

    def test_at_min_size(self):
        assert coin_to_contracts(0.001, ETH_META) == Decimal("0.01")

    def test_ten_eth_is_100_contracts(self):
        assert coin_to_contracts(10.0, ETH_META) == Decimal("100")


# ── SOL quantity conversion ────────────────────────────────────────
class TestSolQuantity:
    """SOL: ctVal=1 (unlike BTC/ETH) so coin qty ≡ contract count 1:1."""

    def test_one_sol_is_one_contract(self):
        assert coin_to_contracts(1.0, SOL_META) == Decimal("1")

    def test_ten_sol_is_ten_contracts(self):
        assert coin_to_contracts(10.0, SOL_META) == Decimal("10")

    def test_fractional_sol_rounds_down(self):
        # 0.567 SOL → 0.56 contracts (lot_sz=0.01)
        assert coin_to_contracts(0.567, SOL_META) == Decimal("0.56")

    def test_at_min_size(self):
        # min_sz=0.01 → 0.01 SOL
        assert coin_to_contracts(0.01, SOL_META) == Decimal("0.01")

    def test_below_min_returns_zero(self):
        assert coin_to_contracts(0.005, SOL_META) == Decimal("0")

    def test_never_confuses_with_btc_conversion(self):
        """Guardrail: SOL (ctVal=1) must NOT be treated as 0.01 ctVal."""
        # If SOL were mistakenly using BTC's ctVal (0.01), 0.5 SOL → 50 contracts.
        # Correct SOL: 0.5 / 1.0 = 0.5 contracts.
        assert coin_to_contracts(0.5, SOL_META) == Decimal("0.5")
        assert coin_to_contracts(0.5, SOL_META) != Decimal("50")


# ── Round-trip safety ──────────────────────────────────────────────
class TestRoundTrip:
    """The adversarial tests: ensure no confusion between contracts and coins."""

    @pytest.mark.parametrize(
        "coin_qty, expected_contracts",
        [
            (0.01, Decimal("1")),    # 0.01 BTC = 1 contract (NOT 0.01!)
            (0.1,  Decimal("10")),   # 0.1  BTC = 10 contracts
            (1.0,  Decimal("100")),  # 1.0  BTC = 100 contracts
        ],
    )
    def test_btc_exact(self, coin_qty, expected_contracts):
        assert coin_to_contracts(coin_qty, BTC_META) == expected_contracts

    @pytest.mark.parametrize(
        "coin_qty, expected_contracts",
        [
            (0.1,  Decimal("1")),    # 0.1 ETH = 1 contract
            (1.0,  Decimal("10")),
            (10.0, Decimal("100")),
        ],
    )
    def test_eth_exact(self, coin_qty, expected_contracts):
        assert coin_to_contracts(coin_qty, ETH_META) == expected_contracts

    def test_never_100x_mistake(self):
        """Regression: catch the worst-case bug where conversion is missed."""
        # If someone forgets /ctVal, 0.5 BTC would become 0.5 contracts
        # instead of 50 contracts.  This test asserts we never return 0.5.
        assert coin_to_contracts(0.5, BTC_META) != Decimal("0.5")
        assert coin_to_contracts(0.5, BTC_META) == Decimal("50")

    def test_zero_and_negative(self):
        assert coin_to_contracts(0, BTC_META) == Decimal("0")
        assert coin_to_contracts(-1.0, BTC_META) == Decimal("0")


# ── Price rounding ─────────────────────────────────────────────────
class TestPriceRounding:
    def test_btc_price_rounds_to_0_1(self):
        assert round_price_to_tick(72690.57, BTC_META) == Decimal("72690.5")
        assert round_price_to_tick(72690.00, BTC_META) == Decimal("72690.0")

    def test_eth_price_rounds_to_0_01(self):
        assert round_price_to_tick(2232.865, ETH_META) == Decimal("2232.86")
        assert round_price_to_tick(2232.8, ETH_META) == Decimal("2232.8")

    def test_zero_price_returns_zero(self):
        assert round_price_to_tick(0, BTC_META) == Decimal("0")


# ── InstrumentMeta.from_api ────────────────────────────────────────
class TestInstrumentMetaFromApi:
    def test_parses_real_okx_response(self):
        # Actual shape from /api/v5/public/instruments?instId=BTC-USDT-SWAP
        raw = {
            "instId": "BTC-USDT-SWAP",
            "ctVal": "0.01",
            "ctValCcy": "BTC",
            "lotSz": "0.01",
            "minSz": "0.01",
            "tickSz": "0.1",
            "lever": "100",
        }
        meta = InstrumentMeta.from_api(raw)
        assert meta.inst_id == "BTC-USDT-SWAP"
        assert meta.ct_val == Decimal("0.01")
        assert meta.ct_val_ccy == "BTC"
        assert meta.lot_sz == Decimal("0.01")
        assert meta.min_sz == Decimal("0.01")
        assert meta.tick_sz == Decimal("0.1")
        assert meta.max_lever == 100
