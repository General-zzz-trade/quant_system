# tests/unit/ib/test_ib_adapter.py
"""Unit tests for IB adapter — uses mocks (no IB Gateway required)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from execution.adapters.ib.config import IBConfig
from execution.adapters.ib.mapper import (
    make_contract,
    map_account_values,
    map_contract_details,
    map_fill,
    map_position,
    map_trade,
)


# -- Mock IB data structures --


@dataclass
class MockContract:
    symbol: str = "AAPL"
    localSymbol: str = "AAPL"
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    conId: int = 265598


@dataclass
class MockContractDetails:
    contract: MockContract = None
    minTick: float = 0.01

    def __post_init__(self):
        if self.contract is None:
            self.contract = MockContract()


@dataclass
class MockPosition:
    account: str = "DU12345"
    contract: MockContract = None
    position: float = 100.0
    avgCost: float = 150.25

    def __post_init__(self):
        if self.contract is None:
            self.contract = MockContract()


@dataclass
class MockOrderStatus:
    status: str = "Submitted"
    filled: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class MockOrder:
    orderId: int = 1001
    permId: int = 2001
    action: str = "BUY"
    orderType: str = "LMT"
    totalQuantity: float = 100.0
    lmtPrice: float = 150.0
    tif: str = "GTC"


@dataclass
class MockTrade:
    contract: MockContract = None
    order: MockOrder = None
    orderStatus: MockOrderStatus = None

    def __post_init__(self):
        if self.contract is None:
            self.contract = MockContract()
        if self.order is None:
            self.order = MockOrder()
        if self.orderStatus is None:
            self.orderStatus = MockOrderStatus()


@dataclass
class MockExecution:
    execId: str = "0001"
    orderId: int = 1001
    acctNumber: str = "DU12345"
    side: str = "BOT"
    shares: float = 100.0
    price: float = 150.30
    exchange: str = "SMART"
    liquidation: int = 0


@dataclass
class MockCommissionReport:
    commission: float = 1.0
    currency: str = "USD"


@dataclass
class MockFill:
    contract: MockContract = None
    execution: MockExecution = None
    commissionReport: MockCommissionReport = None

    def __post_init__(self):
        if self.contract is None:
            self.contract = MockContract()
        if self.execution is None:
            self.execution = MockExecution()
        if self.commissionReport is None:
            self.commissionReport = MockCommissionReport()


@dataclass
class MockAccountValue:
    tag: str = ""
    value: str = "0"
    currency: str = "USD"
    account: str = "DU12345"


# -- Config tests --


class TestIBConfig:
    def test_defaults(self):
        cfg = IBConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 4002
        assert cfg.is_paper is True
        assert cfg.client_id == 1
        assert cfg.readonly is False

    def test_paper_factory(self):
        cfg = IBConfig.paper(client_id=5)
        assert cfg.port == 4002
        assert cfg.is_paper is True
        assert cfg.client_id == 5

    def test_live_factory(self):
        cfg = IBConfig.live(client_id=2)
        assert cfg.port == 4001
        assert cfg.is_paper is False
        assert cfg.readonly is False

    def test_paper_ports(self):
        assert IBConfig(port=4002).is_paper is True
        assert IBConfig(port=7497).is_paper is True
        assert IBConfig(port=4001).is_paper is False
        assert IBConfig(port=7496).is_paper is False


# -- Contract builder tests --


class TestMakeContract:
    def test_stock(self):
        from ib_insync import Stock
        c = make_contract("AAPL", "STK")
        assert isinstance(c, Stock)
        assert c.symbol == "AAPL"

    def test_forex(self):
        from ib_insync import Forex
        c = make_contract("EURUSD", "CASH")
        assert isinstance(c, Forex)

    def test_future(self):
        from ib_insync import Future
        c = make_contract("ES", "FUT", expiry="202412")
        assert isinstance(c, Future)
        assert c.lastTradeDateOrContractMonth == "202412"

    def test_option(self):
        from ib_insync import Option
        c = make_contract("AAPL", "OPT", expiry="20241220", strike=200, right="C")
        assert isinstance(c, Option)
        assert c.strike == 200

    def test_cfd(self):
        from ib_insync import CFD
        c = make_contract("XAUUSD", "CFD")
        assert isinstance(c, CFD)

    def test_crypto(self):
        from ib_insync import Crypto
        c = make_contract("BTC", "CRYPTO")
        assert isinstance(c, Crypto)


# -- Mapper tests --


class TestMapContractDetails:
    def test_stock(self):
        cd = MockContractDetails(contract=MockContract(symbol="AAPL", secType="STK"), minTick=0.01)
        inst = map_contract_details(cd)
        assert inst.venue == "ib"
        assert inst.symbol == "AAPL"
        assert inst.base_asset == "AAPL"
        assert inst.quote_asset == "USD"
        assert inst.tick_size == Decimal("0.01")
        assert inst.contract_type is None  # stock

    def test_forex(self):
        cd = MockContractDetails(
            contract=MockContract(symbol="EUR", localSymbol="EUR.USD", secType="CASH", currency="USD"),
            minTick=0.00005,
        )
        inst = map_contract_details(cd)
        assert inst.symbol == "EUR.USD"
        assert inst.tick_size == Decimal("0.00005")
        assert inst.contract_type == "forex"

    def test_futures(self):
        cd = MockContractDetails(
            contract=MockContract(symbol="ES", localSymbol="ESZ4", secType="FUT"),
            minTick=0.25,
        )
        inst = map_contract_details(cd)
        assert inst.symbol == "ESZ4"
        assert inst.contract_type == "futures"


class TestMapPosition:
    def test_long_stock(self):
        pos = MockPosition(position=100, avgCost=150.25)
        vp = map_position(pos)
        assert vp.venue == "ib"
        assert vp.symbol == "AAPL"
        assert vp.qty == Decimal("100")
        assert vp.is_long
        assert vp.entry_price == Decimal("150.25")

    def test_short_stock(self):
        pos = MockPosition(position=-50, avgCost=155.0)
        vp = map_position(pos)
        assert vp.qty == Decimal("-50")
        assert vp.is_short

    def test_forex_position(self):
        pos = MockPosition(
            contract=MockContract(symbol="EUR", localSymbol="EUR.USD", secType="CASH"),
            position=20000, avgCost=1.08500,
        )
        vp = map_position(pos)
        assert vp.symbol == "EUR.USD"
        assert vp.qty == Decimal("20000")


class TestMapTrade:
    def test_buy_limit(self):
        trade = MockTrade(
            order=MockOrder(action="BUY", orderType="LMT", totalQuantity=100, lmtPrice=150.0),
            orderStatus=MockOrderStatus(status="Submitted"),
        )
        co = map_trade(trade)
        assert co.venue == "ib"
        assert co.side == "buy"
        assert co.order_type == "limit"
        assert co.status == "new"
        assert co.qty == Decimal("100")
        assert co.price == Decimal("150.0")

    def test_sell_market_filled(self):
        trade = MockTrade(
            order=MockOrder(action="SELL", orderType="MKT", totalQuantity=50, lmtPrice=0),
            orderStatus=MockOrderStatus(status="Filled", filled=50, avgFillPrice=152.3),
        )
        co = map_trade(trade)
        assert co.side == "sell"
        assert co.order_type == "market"
        assert co.status == "filled"
        assert co.filled_qty == Decimal("50")
        assert co.avg_price == Decimal("152.3")

    def test_cancelled(self):
        trade = MockTrade(
            orderStatus=MockOrderStatus(status="Cancelled"),
        )
        co = map_trade(trade)
        assert co.status == "canceled"


class TestMapFill:
    def test_buy_fill(self):
        fill = MockFill(
            execution=MockExecution(side="BOT", shares=100, price=150.30),
            commissionReport=MockCommissionReport(commission=1.0),
        )
        cf = map_fill(fill)
        assert cf.venue == "ib"
        assert cf.side == "buy"
        assert cf.qty == Decimal("100")
        assert cf.price == Decimal("150.3")
        assert cf.fee == Decimal("1.0")
        assert cf.fill_id == "ib:0001"

    def test_sell_fill(self):
        fill = MockFill(
            execution=MockExecution(side="SLD", shares=50, price=155.0),
            commissionReport=MockCommissionReport(commission=0.5),
        )
        cf = map_fill(fill)
        assert cf.side == "sell"
        assert cf.qty == Decimal("50")
        assert cf.fee == Decimal("0.5")

    def test_fill_no_commission_yet(self):
        fill = MockFill(
            commissionReport=MockCommissionReport(commission=1e10),  # IB "not reported" value
        )
        cf = map_fill(fill)
        assert cf.fee == Decimal("0")


class TestMapAccountValues:
    def test_basic_balance(self):
        values = [
            MockAccountValue(tag="TotalCashBalance", value="50000", currency="USD"),
            MockAccountValue(tag="MaintMarginReq", value="5000", currency="USD"),
            MockAccountValue(tag="NetLiquidation", value="55000", currency="USD"),
        ]
        snap = map_account_values(values)
        assert snap.venue == "ib"
        assert len(snap.balances) >= 1
        usd = snap.get("USD")
        assert usd is not None
        assert usd.free == Decimal("45000")
        assert usd.locked == Decimal("5000")
        assert usd.total == Decimal("50000")

    def test_multi_currency(self):
        values = [
            MockAccountValue(tag="TotalCashBalance", value="50000", currency="USD"),
            MockAccountValue(tag="TotalCashBalance", value="10000", currency="EUR"),
            MockAccountValue(tag="MaintMarginReq", value="2000", currency="USD"),
            MockAccountValue(tag="MaintMarginReq", value="1000", currency="EUR"),
        ]
        snap = map_account_values(values)
        assert len(snap.balances) == 2
        usd = snap.get("USD")
        eur = snap.get("EUR")
        assert usd.free == Decimal("48000")
        assert eur.free == Decimal("9000")


# -- Adapter tests (mocked IB connection) --


class TestIBAdapter:
    def _make_adapter(self):
        from execution.adapters.ib.adapter import IBAdapter
        cfg = IBConfig.paper()
        adapter = IBAdapter(cfg)
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ["DU12345"]
        adapter._ib = mock_ib
        adapter._connected = True
        return adapter, mock_ib

    def test_get_positions(self):
        adapter, ib = self._make_adapter()
        ib.positions.return_value = [MockPosition()]
        positions = adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].is_long

    def test_get_positions_filters_flat(self):
        adapter, ib = self._make_adapter()
        p1 = MockPosition(position=100)
        p2 = MockPosition(position=0)  # flat, should be excluded
        ib.positions.return_value = [p1, p2]
        positions = adapter.get_positions()
        assert len(positions) == 1

    def test_get_open_orders(self):
        adapter, ib = self._make_adapter()
        ib.openTrades.return_value = [MockTrade()]
        orders = adapter.get_open_orders()
        assert len(orders) == 1
        assert orders[0].order_type == "limit"

    def test_get_balances(self):
        adapter, ib = self._make_adapter()
        ib.accountValues.return_value = [
            MockAccountValue(tag="TotalCashBalance", value="50000", currency="USD"),
            MockAccountValue(tag="MaintMarginReq", value="5000", currency="USD"),
        ]
        snap = adapter.get_balances()
        assert snap.balances[0].asset == "USD"
        assert snap.balances[0].free == Decimal("45000")

    def test_get_recent_fills(self):
        adapter, ib = self._make_adapter()
        ib.fills.return_value = [MockFill()]
        fills = adapter.get_recent_fills()
        assert len(fills) == 1
        assert fills[0].side == "buy"

    def test_not_connected_raises(self):
        from execution.adapters.ib.adapter import IBAdapter
        adapter = IBAdapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.list_instruments()

    def test_send_market_order(self):
        adapter, ib = self._make_adapter()

        mock_contract = MockContract()
        ib.qualifyContracts.return_value = [mock_contract]

        mock_trade = MockTrade()
        mock_trade.order.orderId = 5001
        mock_trade.orderStatus.status = "Submitted"
        ib.placeOrder.return_value = mock_trade

        result = adapter.send_market_order("AAPL", "buy", 100)
        assert result["orderId"] == 5001
        assert result["status"] == "Submitted"
        ib.placeOrder.assert_called_once()

    def test_send_limit_order(self):
        adapter, ib = self._make_adapter()

        mock_contract = MockContract()
        ib.qualifyContracts.return_value = [mock_contract]

        mock_trade = MockTrade()
        mock_trade.order.orderId = 5002
        ib.placeOrder.return_value = mock_trade

        result = adapter.send_limit_order("AAPL", "buy", 100, 150.0)
        assert result["orderId"] == 5002
        assert result["price"] == 150.0

    def test_cancel_order(self):
        adapter, ib = self._make_adapter()
        mock_trade = MockTrade()
        mock_trade.order.orderId = 5001
        ib.openTrades.return_value = [mock_trade]

        result = adapter.cancel_order(5001)
        assert result["status"] == "cancel_submitted"
        ib.cancelOrder.assert_called_once()

    def test_cancel_order_not_found(self):
        adapter, ib = self._make_adapter()
        ib.openTrades.return_value = []
        result = adapter.cancel_order(9999)
        assert result["status"] == "not_found"

    def test_cancel_all(self):
        adapter, ib = self._make_adapter()
        result = adapter.cancel_all()
        assert result["status"] == "global_cancel_submitted"
        ib.reqGlobalCancel.assert_called_once()

    def test_close_position(self):
        adapter, ib = self._make_adapter()
        ib.positions.return_value = [MockPosition(position=100)]

        mock_contract = MockContract()
        ib.qualifyContracts.return_value = [mock_contract]

        mock_trade = MockTrade()
        mock_trade.order.orderId = 6001
        ib.placeOrder.return_value = mock_trade

        result = adapter.close_position("AAPL")
        assert result["orderId"] == 6001
        assert result["side"] == "sell"  # closing long = sell

    def test_get_portfolio(self):
        adapter, ib = self._make_adapter()
        item = MagicMock()
        item.contract = MockContract()
        item.position = 100
        item.marketPrice = 155.0
        item.marketValue = 15500.0
        item.averageCost = 150.25
        item.unrealizedPNL = 475.0
        item.realizedPNL = 0.0
        ib.portfolio.return_value = [item]

        portfolio = adapter.get_portfolio()
        assert len(portfolio) == 1
        assert portfolio[0]["symbol"] == "AAPL"
        assert portfolio[0]["unrealizedPNL"] == 475.0
