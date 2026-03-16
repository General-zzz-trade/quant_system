"""Extended tests for Bybit mapper — instruments, positions, orders, fills, balances."""
from __future__ import annotations

from decimal import Decimal

from execution.adapters.bybit.mapper import (
    map_balance,
    map_fill,
    map_instrument,
    map_order,
    map_position,
)


class TestMapInstrumentExtended:
    def test_all_fields_populated(self):
        raw = {
            "symbol": "ETHUSDT",
            "baseCoin": "ETH",
            "quoteCoin": "USDT",
            "status": "Trading",
            "contractType": "LinearPerpetual",
            "settleCoin": "USDT",
            "lotSizeFilter": {
                "qtyStep": "0.01",
                "minOrderQty": "0.01",
                "maxOrderQty": "1000",
                "minNotionalValue": "5",
            },
            "priceFilter": {"tickSize": "0.01"},
        }
        inst = map_instrument(raw)
        assert inst.symbol == "ETHUSDT"
        assert inst.base_asset == "ETH"
        assert inst.quote_asset == "USDT"
        assert inst.tick_size == Decimal("0.01")
        assert inst.lot_size == Decimal("0.01")
        assert inst.min_qty == Decimal("0.01")
        assert inst.max_qty == Decimal("1000")
        assert inst.min_notional == Decimal("5")
        assert inst.contract_type == "perpetual"
        assert inst.margin_asset == "USDT"
        assert inst.trading_enabled is True

    def test_not_trading_status(self):
        raw = {
            "symbol": "XYZUSDT",
            "status": "PreLaunch",
            "lotSizeFilter": {},
            "priceFilter": {},
        }
        inst = map_instrument(raw)
        assert inst.trading_enabled is False

    def test_missing_max_qty(self):
        raw = {
            "symbol": "BTCUSDT",
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"},
            "priceFilter": {"tickSize": "0.10"},
        }
        inst = map_instrument(raw)
        assert inst.max_qty is None

    def test_missing_contract_type(self):
        raw = {
            "symbol": "BTCUSDT",
            "lotSizeFilter": {},
            "priceFilter": {},
        }
        inst = map_instrument(raw)
        assert inst.contract_type is None

    def test_precision_calculation(self):
        raw = {
            "symbol": "SUIUSDT",
            "lotSizeFilter": {"qtyStep": "0.1"},
            "priceFilter": {"tickSize": "0.0001"},
        }
        inst = map_instrument(raw)
        assert inst.qty_precision == 1
        assert inst.price_precision == 4


class TestMapPositionExtended:
    def test_sell_side_negative_qty(self):
        raw = {
            "symbol": "BTCUSDT", "side": "Sell", "size": "0.5",
            "avgPrice": "70000", "markPrice": "69500",
            "unrealisedPnl": "250", "leverage": "20",
            "tradeMode": "0", "updatedTime": "1700000000000",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("-0.5")
        assert pos.is_short
        assert pos.leverage == 20
        assert pos.margin_type == "cross"

    def test_zero_size_is_flat(self):
        raw = {
            "symbol": "BTCUSDT", "side": "None", "size": "0",
            "avgPrice": "0", "markPrice": "70000",
            "unrealisedPnl": "0",
            "tradeMode": "0", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.is_flat
        assert pos.abs_qty == Decimal("0")

    def test_isolated_margin(self):
        raw = {
            "symbol": "ETHUSDT", "side": "Buy", "size": "1",
            "avgPrice": "3500", "markPrice": "3600",
            "unrealisedPnl": "100", "leverage": "5",
            "tradeMode": "1", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.margin_type == "isolated"

    def test_missing_leverage(self):
        raw = {
            "symbol": "BTCUSDT", "side": "Buy", "size": "0.1",
            "avgPrice": "70000", "markPrice": "70000",
            "unrealisedPnl": "0",
            "tradeMode": "0", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.leverage is None

    def test_liq_price_present(self):
        raw = {
            "symbol": "BTCUSDT", "side": "Buy", "size": "0.1",
            "avgPrice": "70000", "markPrice": "70000",
            "liqPrice": "60000",
            "unrealisedPnl": "0", "leverage": "10",
            "tradeMode": "0", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.liquidation_price == Decimal("60000")

    def test_liq_price_absent(self):
        raw = {
            "symbol": "BTCUSDT", "side": "Buy", "size": "0.1",
            "avgPrice": "70000", "markPrice": "70000",
            "unrealisedPnl": "0",
            "tradeMode": "0", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.liquidation_price is None


class TestMapOrderExtended:
    def test_all_status_codes(self):
        statuses = {
            "New": "new",
            "PartiallyFilled": "partially_filled",
            "Filled": "filled",
            "Cancelled": "canceled",
            "PartiallyFilledCanceled": "canceled",
            "Rejected": "rejected",
            "Deactivated": "expired",
            "Untriggered": "new",
            "Triggered": "new",
        }
        for bybit_status, expected in statuses.items():
            raw = {
                "symbol": "BTCUSDT", "orderId": "o1",
                "orderStatus": bybit_status, "side": "Buy",
                "orderType": "Limit", "qty": "0.01",
                "timeInForce": "GTC", "createdTime": "0",
            }
            order = map_order(raw)
            assert order.status == expected, f"Failed for {bybit_status}"

    def test_market_order_type(self):
        raw = {
            "symbol": "ETHUSDT", "orderId": "o2",
            "orderStatus": "Filled", "side": "Sell",
            "orderType": "Market", "qty": "1.0",
            "cumExecQty": "1.0", "avgPrice": "3500",
            "timeInForce": "IOC", "createdTime": "0",
        }
        order = map_order(raw)
        assert order.order_type == "market"
        assert order.side == "sell"

    def test_missing_order_link_id(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "o3",
            "orderStatus": "New", "side": "Buy",
            "orderType": "Limit", "qty": "0.01",
            "timeInForce": "GTC", "createdTime": "0",
        }
        order = map_order(raw)
        assert order.client_order_id is None

    def test_zero_price_becomes_none(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "o4",
            "orderStatus": "New", "side": "Buy",
            "orderType": "Market", "qty": "0.01",
            "price": "0", "avgPrice": "0",
            "timeInForce": "GTC", "createdTime": "0",
        }
        order = map_order(raw)
        assert order.price is None
        assert order.avg_price is None


class TestMapFillExtended:
    def test_fill_with_fees(self):
        raw = {
            "symbol": "ETHUSDT", "orderId": "o1",
            "execId": "e100", "side": "Sell",
            "execQty": "2.5", "execPrice": "3500",
            "execFee": "-1.75", "feeCurrency": "USDT",
            "isMaker": "true", "execTime": "1700000000000",
        }
        fill = map_fill(raw)
        assert fill.side == "sell"
        assert fill.qty == Decimal("2.5")
        assert fill.price == Decimal("3500")
        assert fill.fee == Decimal("1.75")  # abs value
        assert fill.fee_asset == "USDT"
        assert fill.liquidity == "maker"
        assert fill.fill_id == "bybit:ETHUSDT:e100"

    def test_fill_taker(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "o2",
            "execId": "e200", "side": "Buy",
            "execQty": "0.01", "execPrice": "70000",
            "execFee": "0.42", "feeCurrency": "USDT",
            "isMaker": "false", "execTime": "1700000000000",
        }
        fill = map_fill(raw)
        assert fill.liquidity == "taker"

    def test_fill_zero_fee(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "o3",
            "execId": "e300", "side": "Buy",
            "execQty": "0.001", "execPrice": "70000",
            "execFee": "0", "feeCurrency": "USDT",
            "isMaker": "false", "execTime": "0",
        }
        fill = map_fill(raw)
        assert fill.fee == Decimal("0")


class TestMapBalanceExtended:
    def test_multi_asset_balance(self):
        raw = {
            "list": [{
                "coin": [
                    {"coin": "USDT", "walletBalance": "10000", "locked": "500"},
                    {"coin": "BTC", "walletBalance": "0.5", "locked": "0"},
                ]
            }]
        }
        snap = map_balance(raw)
        assert len(snap.balances) == 2
        usdt = snap.get("USDT")
        assert usdt is not None
        assert usdt.free == Decimal("9500")
        assert usdt.locked == Decimal("500")
        btc = snap.get("BTC")
        assert btc is not None
        assert btc.free == Decimal("0.5")

    def test_zero_balance_excluded(self):
        raw = {
            "list": [{
                "coin": [
                    {"coin": "USDT", "walletBalance": "0", "locked": "0"},
                ]
            }]
        }
        snap = map_balance(raw)
        assert len(snap.balances) == 0

    def test_empty_coin_list(self):
        raw = {"list": [{"coin": []}]}
        snap = map_balance(raw)
        assert len(snap.balances) == 0

    def test_empty_account_list(self):
        raw = {"list": []}
        snap = map_balance(raw)
        assert len(snap.balances) == 0
