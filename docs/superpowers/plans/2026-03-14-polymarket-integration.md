# Polymarket Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add crypto-related Polymarket prediction market trading as an independent vertical within quant_system.

**Architecture:** Lightweight independent module — new `polymarket/` top-level package + `execution/adapters/polymarket/` adapter. Reuses execution safety infra (state machine, dedup, kill switch) but has own runner, features, and decision logic. Does NOT share coordinator or constraint pipeline with Binance futures path.

**Tech Stack:** Python 3.12, stdlib urllib/json for REST, websocket-client for WS, existing Decimal/dataclass patterns.

**Spec:** `docs/superpowers/specs/2026-03-14-polymarket-integration-design.md`

---

## File Map

### New files to create:

```
execution/adapters/polymarket/
├── __init__.py                    # Package init
├── types.py                       # Market, Outcome, Token dataclasses
├── auth.py                        # API key + HMAC signing
├── client.py                      # REST client (CLOB API v2)
├── ws_client.py                   # WebSocket orderbook/trades
├── mapper.py                      # Polymarket → CanonicalOrder/CanonicalFill
└── market_discovery.py            # Crypto market filter

polymarket/
├── __init__.py                    # Package init
├── config.py                      # PolymarketConfig dataclass
├── data.py                        # MarketScanner + snapshot storage
├── features.py                    # ~30 probability market features
├── signals.py                     # Rule-based signal generation
├── sizing.py                      # Kelly criterion position sizing
├── decision.py                    # PolymarketDecisionModule
└── runner.py                      # Independent PolymarketRunner

config/polymarket.yaml             # Default config

tests/unit/polymarket/
├── __init__.py
├── test_types.py
├── test_auth.py
├── test_client.py
├── test_mapper.py
├── test_market_discovery.py
├── test_features.py
├── test_signals.py
├── test_sizing.py
└── test_decision.py

tests/integration/
└── test_polymarket_flow.py        # End-to-end with mocks
```

### Existing files NOT modified:
- `runner/live_runner.py` — no changes (independent runner)
- `engine/coordinator.py` — no changes
- `event/types.py` — reuse existing types, no changes

---

## Chunk 1: Foundation Layer (Types + Auth + Config)

### Task 1: Polymarket Types

**Files:**
- Create: `execution/adapters/polymarket/__init__.py`
- Create: `execution/adapters/polymarket/types.py`
- Test: `tests/unit/polymarket/__init__.py`
- Test: `tests/unit/polymarket/test_types.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p execution/adapters/polymarket
mkdir -p polymarket
mkdir -p tests/unit/polymarket
touch execution/adapters/polymarket/__init__.py
touch polymarket/__init__.py
touch tests/unit/polymarket/__init__.py
```

- [ ] **Step 2: Write failing test for types**

```python
# tests/unit/polymarket/test_types.py
from decimal import Decimal
from execution.adapters.polymarket.types import PolymarketMarket, PolymarketOutcome


def test_market_creation():
    market = PolymarketMarket(
        condition_id="0xabc123",
        slug="btc-above-100k-march",
        question="Will BTC be above $100K on March 31?",
        outcomes=("Yes", "No"),
        token_ids=("token_yes_123", "token_no_456"),
        end_date_iso="2026-03-31T00:00:00Z",
        active=True,
        volume_24h=Decimal("50000"),
    )
    assert market.slug == "btc-above-100k-march"
    assert len(market.token_ids) == 2


def test_market_symbol_format():
    market = PolymarketMarket(
        condition_id="0xabc",
        slug="btc-above-100k",
        question="BTC above 100K?",
        outcomes=("Yes", "No"),
        token_ids=("t1", "t2"),
        end_date_iso="2026-03-31T00:00:00Z",
        active=True,
        volume_24h=Decimal("10000"),
    )
    assert market.symbol("Yes") == "POLY:btc-above-100k:YES"
    assert market.symbol("No") == "POLY:btc-above-100k:NO"


def test_market_is_crypto():
    m1 = PolymarketMarket(
        condition_id="0x1", slug="btc-price", question="Will Bitcoin exceed 100K?",
        outcomes=("Yes", "No"), token_ids=("a", "b"),
        end_date_iso="2026-12-31T00:00:00Z", active=True, volume_24h=Decimal("5000"),
    )
    assert m1.is_crypto(("BTC", "Bitcoin", "ETH"))

    m2 = PolymarketMarket(
        condition_id="0x2", slug="election", question="Who wins the election?",
        outcomes=("A", "B"), token_ids=("c", "d"),
        end_date_iso="2026-12-31T00:00:00Z", active=True, volume_24h=Decimal("5000"),
    )
    assert not m2.is_crypto(("BTC", "Bitcoin", "ETH"))
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
pytest tests/unit/polymarket/test_types.py -v
```

- [ ] **Step 4: Implement types**

```python
# execution/adapters/polymarket/types.py
"""Polymarket domain types — markets, outcomes, tokens."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class PolymarketMarket:
    """A single prediction market on Polymarket."""
    condition_id: str
    slug: str
    question: str
    outcomes: Tuple[str, ...]
    token_ids: Tuple[str, ...]
    end_date_iso: str
    active: bool
    volume_24h: Decimal
    description: str = ""
    category: str = ""

    def symbol(self, outcome: str) -> str:
        """System symbol format: POLY:{slug}:{outcome_upper}."""
        return f"POLY:{self.slug}:{outcome.upper()}"

    def token_id_for(self, outcome: str) -> Optional[str]:
        """Get token_id for a given outcome name."""
        for i, o in enumerate(self.outcomes):
            if o.lower() == outcome.lower() and i < len(self.token_ids):
                return self.token_ids[i]
        return None

    def is_crypto(self, keywords: Sequence[str] = ()) -> bool:
        """Check if market is crypto-related by keyword match."""
        text = f"{self.question} {self.slug} {self.description}".lower()
        return any(kw.lower() in text for kw in keywords)

    def hours_to_expiry(self, now_iso: str) -> float:
        """Hours until market expiry."""
        from datetime import datetime, timezone
        end = datetime.fromisoformat(self.end_date_iso.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        return max(0.0, (end - now).total_seconds() / 3600)


@dataclass(frozen=True, slots=True)
class PolymarketOrderbook:
    """L2 orderbook snapshot."""
    token_id: str
    bids: Tuple[Tuple[Decimal, Decimal], ...]  # (price, size) sorted desc
    asks: Tuple[Tuple[Decimal, Decimal], ...]  # (price, size) sorted asc
    timestamp_ms: int = 0

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
```

- [ ] **Step 5: Run test — expect PASS**

```bash
pytest tests/unit/polymarket/test_types.py -v
```

- [ ] **Step 6: Commit**

```bash
git add execution/adapters/polymarket/ polymarket/ tests/unit/polymarket/
git commit -m "feat(polymarket): add types — Market, Orderbook dataclasses"
```

---

### Task 2: Auth + Config

**Files:**
- Create: `execution/adapters/polymarket/auth.py`
- Create: `polymarket/config.py`
- Create: `config/polymarket.yaml`
- Test: `tests/unit/polymarket/test_auth.py`

- [ ] **Step 1: Write failing test for auth**

```python
# tests/unit/polymarket/test_auth.py
from execution.adapters.polymarket.auth import PolymarketAuth


def test_sign_request():
    auth = PolymarketAuth(api_key="test-key", api_secret="test-secret")
    headers = auth.sign_request("GET", "/markets", "")
    assert "POLY_ADDRESS" in headers or "Authorization" in headers


def test_auth_headers_contain_timestamp():
    auth = PolymarketAuth(api_key="k", api_secret="s")
    headers = auth.sign_request("POST", "/order", '{"side":"BUY"}')
    assert "POLY_TIMESTAMP" in headers or "X-Timestamp" in headers
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/unit/polymarket/test_auth.py -v
```

- [ ] **Step 3: Implement auth**

```python
# execution/adapters/polymarket/auth.py
"""Polymarket CLOB API authentication — HMAC signing."""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PolymarketAuth:
    """Signs requests for Polymarket CLOB API."""
    api_key: str
    api_secret: str

    def sign_request(
        self, method: str, path: str, body: str = "",
    ) -> dict[str, str]:
        """Generate auth headers for a CLOB API request."""
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "POLY_ADDRESS": self.api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": timestamp,
        }
```

- [ ] **Step 4: Implement config**

```python
# polymarket/config.py
"""PolymarketConfig — configuration for prediction market trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class PolymarketConfig:
    """Configuration for Polymarket runner."""
    # --- Auth ---
    api_key: str = ""
    api_secret: str = ""

    # --- API endpoints ---
    base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # --- Market discovery ---
    scan_interval_sec: int = 3600
    min_liquidity_usd: float = 10_000
    min_hours_to_expiry: float = 24.0
    crypto_keywords: tuple[str, ...] = (
        "BTC", "Bitcoin", "ETH", "Ethereum", "crypto", "Solana", "SOL",
    )

    # --- Risk limits ---
    max_position_pct: float = 0.10
    max_total_pct: float = 0.30
    stop_loss_pct: float = 0.15

    # --- Signal ---
    signal_threshold: float = 0.15
    min_probability: float = 0.15
    max_probability: float = 0.85
    kelly_fraction: float = 0.5

    # --- Drawdown ---
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0

    # --- Persistence ---
    data_dir: str = "data/polymarket"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "PolymarketConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Flatten nested keys if needed
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}_{k2}"] = v2
            else:
                flat[k] = v
        # Convert crypto_keywords list to tuple
        if "crypto_keywords" in flat and isinstance(flat["crypto_keywords"], list):
            flat["crypto_keywords"] = tuple(flat["crypto_keywords"])
        return cls(**{k: v for k, v in flat.items() if hasattr(cls, k)})
```

- [ ] **Step 5: Create default config YAML**

```yaml
# config/polymarket.yaml
# Polymarket prediction market configuration
# API credentials should be set via environment variables

api_key: ${POLYMARKET_API_KEY}
api_secret: ${POLYMARKET_API_SECRET}

# Market discovery
scan_interval_sec: 3600
min_liquidity_usd: 10000
min_hours_to_expiry: 24

# Risk
max_position_pct: 0.10
max_total_pct: 0.30
stop_loss_pct: 0.15
kelly_fraction: 0.5

# Signal
signal_threshold: 0.15
min_probability: 0.15
max_probability: 0.85

# Data
data_dir: data/polymarket
log_level: INFO
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/unit/polymarket/ -v
```

- [ ] **Step 7: Commit**

```bash
git add execution/adapters/polymarket/auth.py polymarket/config.py config/polymarket.yaml tests/unit/polymarket/test_auth.py
git commit -m "feat(polymarket): add auth signing + config + default YAML"
```

---

## Chunk 2: API Client + Mapper

### Task 3: REST Client

**Files:**
- Create: `execution/adapters/polymarket/client.py`
- Test: `tests/unit/polymarket/test_client.py`

- [ ] **Step 1: Write failing test for client**

```python
# tests/unit/polymarket/test_client.py
import json
from unittest.mock import patch, MagicMock
from decimal import Decimal
from execution.adapters.polymarket.client import PolymarketClient
from execution.adapters.polymarket.auth import PolymarketAuth


def _make_client():
    auth = PolymarketAuth(api_key="test", api_secret="secret")
    return PolymarketClient(auth=auth, base_url="https://test.polymarket.com")


def test_get_markets_parses_response():
    client = _make_client()
    mock_response = json.dumps([{
        "condition_id": "0xabc",
        "market_slug": "btc-above-100k",
        "question": "Will BTC exceed 100K?",
        "outcomes": '["Yes","No"]',
        "tokens": [{"token_id": "t1", "outcome": "Yes"}, {"token_id": "t2", "outcome": "No"}],
        "end_date_iso": "2026-03-31T00:00:00Z",
        "active": True,
        "volume": "50000",
        "description": "Bitcoin prediction",
    }]).encode()

    with patch("execution.adapters.polymarket.client.urlopen") as mock_url:
        mock_url.return_value.__enter__ = lambda s: MagicMock(read=lambda: mock_response)
        mock_url.return_value.__exit__ = MagicMock(return_value=False)
        markets = client.get_markets()

    assert len(markets) == 1
    assert markets[0].slug == "btc-above-100k"
    assert markets[0].is_crypto(("BTC",))


def test_get_orderbook_parses_bids_asks():
    client = _make_client()
    mock_response = json.dumps({
        "bids": [{"price": "0.65", "size": "100"}, {"price": "0.60", "size": "200"}],
        "asks": [{"price": "0.70", "size": "150"}],
    }).encode()

    with patch("execution.adapters.polymarket.client.urlopen") as mock_url:
        mock_url.return_value.__enter__ = lambda s: MagicMock(read=lambda: mock_response)
        mock_url.return_value.__exit__ = MagicMock(return_value=False)
        ob = client.get_orderbook("token_123")

    assert ob.best_bid == Decimal("0.65")
    assert ob.best_ask == Decimal("0.70")
    assert ob.mid_price == Decimal("0.675")
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/unit/polymarket/test_client.py -v
```

- [ ] **Step 3: Implement client**

Implement `PolymarketClient` with methods:
- `get_markets() -> list[PolymarketMarket]`
- `get_orderbook(token_id) -> PolymarketOrderbook`
- `get_trades(token_id, limit) -> list[dict]`
- `create_order(token_id, side, size, price) -> dict`
- `cancel_order(order_id) -> bool`
- `get_positions() -> list[dict]`
- `get_balances() -> dict`

Use stdlib `urllib.request` (following existing Binance REST pattern). Parse JSON responses into domain types.

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/unit/polymarket/test_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add execution/adapters/polymarket/client.py tests/unit/polymarket/test_client.py
git commit -m "feat(polymarket): REST client for CLOB API v2"
```

---

### Task 4: Mapper (Polymarket JSON → Canonical Types)

**Files:**
- Create: `execution/adapters/polymarket/mapper.py`
- Test: `tests/unit/polymarket/test_mapper.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/polymarket/test_mapper.py
from decimal import Decimal
from execution.adapters.polymarket.mapper import fill_from_polymarket, order_from_polymarket


def test_fill_from_polymarket():
    raw = {
        "id": "fill-123",
        "market": "btc-above-100k",
        "asset_id": "token_yes",
        "side": "BUY",
        "size": "10",
        "price": "0.65",
        "fee_rate_bps": "0",
        "status": "MATCHED",
        "associate_trades": [{"id": "trade-1"}],
    }
    fill = fill_from_polymarket(raw, venue="polymarket")
    assert fill.venue == "polymarket"
    assert fill.symbol == "POLY:btc-above-100k:YES"
    assert fill.qty == Decimal("10")
    assert fill.price == Decimal("0.65")
    assert fill.side == "buy"


def test_order_from_polymarket():
    raw = {
        "id": "order-456",
        "market": "btc-above-100k",
        "asset_id": "token_yes",
        "side": "BUY",
        "original_size": "20",
        "price": "0.60",
        "size_matched": "5",
        "status": "LIVE",
        "outcome": "Yes",
    }
    order = order_from_polymarket(raw, venue="polymarket")
    assert order.symbol.startswith("POLY:")
    assert order.qty == Decimal("20")
    assert order.filled_qty == Decimal("5")
    assert order.side == "buy"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/unit/polymarket/test_mapper.py -v
```

- [ ] **Step 3: Implement mapper**

Map Polymarket CLOB API JSON to existing `CanonicalOrder` and `CanonicalFill` types. Key mappings:
- `asset_id` + `outcome` → `symbol` (format: `POLY:{slug}:{OUTCOME}`)
- `side` BUY/SELL → `side` buy/sell (lowercase)
- `size` → `qty` (Decimal)
- `price` → `price` (Decimal, 0-1 probability)

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/unit/polymarket/test_mapper.py -v
```

- [ ] **Step 5: Commit**

```bash
git add execution/adapters/polymarket/mapper.py tests/unit/polymarket/test_mapper.py
git commit -m "feat(polymarket): mapper — Polymarket JSON to CanonicalOrder/Fill"
```

---

### Task 5: Market Discovery

**Files:**
- Create: `execution/adapters/polymarket/market_discovery.py`
- Test: `tests/unit/polymarket/test_market_discovery.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/polymarket/test_market_discovery.py
from decimal import Decimal
from execution.adapters.polymarket.types import PolymarketMarket
from execution.adapters.polymarket.market_discovery import filter_crypto_markets

CRYPTO_KEYWORDS = ("BTC", "Bitcoin", "ETH", "Ethereum", "crypto")

def _make_market(slug, question, volume=50000, active=True, end="2026-12-31T00:00:00Z"):
    return PolymarketMarket(
        condition_id="0x1", slug=slug, question=question,
        outcomes=("Yes", "No"), token_ids=("a", "b"),
        end_date_iso=end, active=active, volume_24h=Decimal(str(volume)),
    )


def test_filters_crypto_markets():
    markets = [
        _make_market("btc-100k", "Will Bitcoin reach 100K?"),
        _make_market("election", "Who wins the election?"),
        _make_market("eth-merge", "Will Ethereum merge succeed?"),
    ]
    result = filter_crypto_markets(markets, CRYPTO_KEYWORDS, min_volume=10000)
    assert len(result) == 2
    assert result[0].slug == "btc-100k"
    assert result[1].slug == "eth-merge"


def test_filters_low_liquidity():
    markets = [_make_market("btc-low", "BTC prediction?", volume=500)]
    result = filter_crypto_markets(markets, CRYPTO_KEYWORDS, min_volume=10000)
    assert len(result) == 0


def test_filters_inactive():
    markets = [_make_market("btc-old", "BTC old?", active=False)]
    result = filter_crypto_markets(markets, CRYPTO_KEYWORDS, min_volume=1000)
    assert len(result) == 0
```

- [ ] **Step 2: Implement, test, commit**

```bash
pytest tests/unit/polymarket/test_market_discovery.py -v
git add execution/adapters/polymarket/market_discovery.py tests/unit/polymarket/test_market_discovery.py
git commit -m "feat(polymarket): market discovery — crypto keyword filter"
```

---

## Chunk 3: Features + Signals + Sizing

### Task 6: Feature Engine (~30 features)

**Files:**
- Create: `polymarket/features.py`
- Test: `tests/unit/polymarket/test_features.py`

- [ ] **Step 1: Write failing test**

Test that feature computation produces expected keys from probability history + orderbook data.

```python
# tests/unit/polymarket/test_features.py
import numpy as np
from polymarket.features import compute_features


def test_compute_features_returns_expected_keys():
    prob_history = np.array([0.5, 0.52, 0.55, 0.53, 0.58] * 25)  # 125 hourly points
    orderbook = {"best_bid": 0.57, "best_ask": 0.59, "bid_depth": 1000, "ask_depth": 800}
    trades = {"count_1h": 50, "buy_volume": 300, "sell_volume": 200}
    expiry_hours = 72.0
    btc_price = 99500.0
    btc_strike = 100000.0

    feats = compute_features(prob_history, orderbook, trades, expiry_hours, btc_price, btc_strike)

    assert "prob_ret_1h" in feats
    assert "prob_ret_24h" in feats
    assert "prob_zscore_24h" in feats
    assert "bid_ask_spread" in feats
    assert "depth_imbalance" in feats
    assert "hours_to_expiry" in feats
    assert "btc_price_vs_strike" in feats


def test_features_handles_short_history():
    prob_history = np.array([0.5, 0.52, 0.55])
    orderbook = {"best_bid": 0.5, "best_ask": 0.55, "bid_depth": 100, "ask_depth": 100}
    trades = {"count_1h": 5, "buy_volume": 50, "sell_volume": 50}

    feats = compute_features(prob_history, orderbook, trades, 100.0, 0.0, 0.0)
    # Should not crash, features with insufficient data should be NaN
    assert "prob_ret_1h" in feats
```

- [ ] **Step 2: Implement features.py**

Compute ~30 features from probability history, orderbook, trade flow, expiry, and cross-market BTC price. Return `dict[str, float]`.

- [ ] **Step 3: Run test, commit**

```bash
pytest tests/unit/polymarket/test_features.py -v
git commit -m "feat(polymarket): feature engine — 30 probability market features"
```

---

### Task 7: Signal Generation (V1 Rules)

**Files:**
- Create: `polymarket/signals.py`
- Test: `tests/unit/polymarket/test_signals.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/polymarket/test_signals.py
from polymarket.signals import generate_signal


def test_bullish_signal_when_btc_above_strike_and_prob_underpriced():
    features = {
        "btc_price_vs_strike": 0.05,  # BTC 5% above strike
        "prob_zscore_24h": -1.5,       # probability is below 24h mean
        "hours_to_expiry": 48.0,
        "mid_price": 0.60,
    }
    signal = generate_signal(features, threshold=0.15)
    assert signal > 0  # buy YES (underpriced)


def test_no_signal_when_conflicting():
    features = {
        "btc_price_vs_strike": 0.05,  # BTC above strike
        "prob_zscore_24h": 1.5,        # but probability already high
        "hours_to_expiry": 48.0,
        "mid_price": 0.70,
    }
    signal = generate_signal(features, threshold=0.15)
    assert abs(signal) < 0.15  # no signal (conflicting)


def test_no_signal_near_expiry():
    features = {
        "btc_price_vs_strike": 0.10,
        "prob_zscore_24h": -2.0,
        "hours_to_expiry": 3.0,  # too close to expiry
        "mid_price": 0.50,
    }
    signal = generate_signal(features, threshold=0.15, min_hours=6.0)
    assert signal == 0.0
```

- [ ] **Step 2: Implement, test, commit**

```bash
pytest tests/unit/polymarket/test_signals.py -v
git commit -m "feat(polymarket): V1 rule-based signal generation"
```

---

### Task 8: Kelly Sizing

**Files:**
- Create: `polymarket/sizing.py`
- Test: `tests/unit/polymarket/test_sizing.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/polymarket/test_sizing.py
from polymarket.sizing import kelly_size


def test_kelly_basic():
    # Estimated prob 0.7, market price 0.5 → positive edge
    size = kelly_size(
        estimated_prob=0.7, market_price=0.5,
        bankroll=1000, kelly_fraction=0.5,
        max_position_pct=0.10,
    )
    assert 0 < size <= 100  # max 10% of 1000


def test_kelly_no_edge():
    # Estimated prob equals market price → no edge → zero size
    size = kelly_size(estimated_prob=0.5, market_price=0.5,
                      bankroll=1000, kelly_fraction=0.5, max_position_pct=0.10)
    assert size == 0.0


def test_kelly_negative_edge():
    # Estimated prob BELOW market price → negative edge → zero size
    size = kelly_size(estimated_prob=0.4, market_price=0.6,
                      bankroll=1000, kelly_fraction=0.5, max_position_pct=0.10)
    assert size == 0.0


def test_kelly_respects_max_position():
    # Very high edge should still be capped
    size = kelly_size(estimated_prob=0.99, market_price=0.10,
                      bankroll=10000, kelly_fraction=1.0, max_position_pct=0.10)
    assert size <= 1000  # 10% cap
```

- [ ] **Step 2: Implement, test, commit**

```bash
pytest tests/unit/polymarket/test_sizing.py -v
git commit -m "feat(polymarket): Kelly criterion position sizing"
```

---

## Chunk 4: Decision Module + Runner

### Task 9: Decision Module

**Files:**
- Create: `polymarket/decision.py`
- Test: `tests/unit/polymarket/test_decision.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/polymarket/test_decision.py
from decimal import Decimal
from polymarket.decision import PolymarketDecisionModule
from polymarket.config import PolymarketConfig


def test_generates_intent_for_strong_signal():
    config = PolymarketConfig(signal_threshold=0.15, max_position_pct=0.10)
    module = PolymarketDecisionModule(config)

    market_data = {
        "symbol": "POLY:btc-above-100k:YES",
        "features": {
            "btc_price_vs_strike": 0.05,
            "prob_zscore_24h": -1.5,
            "hours_to_expiry": 48.0,
            "mid_price": 0.55,
        },
        "market_price": Decimal("0.55"),
        "bankroll": Decimal("1000"),
    }
    intents = module.evaluate(market_data)
    assert len(intents) > 0
    assert intents[0]["side"] == "buy"


def test_skips_extreme_probability():
    config = PolymarketConfig(min_probability=0.15, max_probability=0.85)
    module = PolymarketDecisionModule(config)

    market_data = {
        "symbol": "POLY:btc-sure-thing:YES",
        "features": {"btc_price_vs_strike": 0.10, "prob_zscore_24h": -2.0,
                      "hours_to_expiry": 48.0, "mid_price": 0.92},
        "market_price": Decimal("0.92"),
        "bankroll": Decimal("1000"),
    }
    intents = module.evaluate(market_data)
    assert len(intents) == 0  # probability too extreme


def test_generates_exit_on_stop_loss():
    config = PolymarketConfig(stop_loss_pct=0.15)
    module = PolymarketDecisionModule(config)

    position = {
        "symbol": "POLY:btc-above-100k:YES",
        "entry_price": Decimal("0.60"),
        "current_price": Decimal("0.40"),  # -33% → exceeds 15% stop
        "qty": Decimal("10"),
    }
    exits = module.check_exits([position])
    assert len(exits) == 1
    assert exits[0]["action"] == "close"
```

- [ ] **Step 2: Implement, test, commit**

```bash
pytest tests/unit/polymarket/test_decision.py -v
git commit -m "feat(polymarket): decision module — entry/exit/skip logic"
```

---

### Task 10: Runner

**Files:**
- Create: `polymarket/runner.py`
- Create: `polymarket/__main__.py`
- Test: `tests/integration/test_polymarket_flow.py`

- [ ] **Step 1: Write integration test with mocks**

```python
# tests/integration/test_polymarket_flow.py
"""End-to-end flow test: discovery → features → signal → sizing → order."""
from unittest.mock import MagicMock, patch
from decimal import Decimal
from polymarket.runner import PolymarketRunner
from polymarket.config import PolymarketConfig


def test_runner_main_loop_with_mock_client():
    config = PolymarketConfig(
        api_key="test", api_secret="secret",
        signal_threshold=0.10,
    )
    runner = PolymarketRunner(config)

    # Mock the API client
    runner._client = MagicMock()
    runner._client.get_markets.return_value = []  # no markets found

    # One iteration should not crash
    runner.run_once()

    runner._client.get_markets.assert_called_once()
```

- [ ] **Step 2: Implement runner**

`PolymarketRunner` with:
- `__init__(config)`: create client, decision module, kill switch
- `run_once()`: single cycle (scan → features → decide → execute)
- `start()`: main loop with interval sleep
- `stop()`: graceful shutdown
- Entry point: `polymarket/__main__.py`

- [ ] **Step 3: Run all tests**

```bash
pytest tests/unit/polymarket/ tests/integration/test_polymarket_flow.py -v
```

- [ ] **Step 4: Commit**

```bash
git add polymarket/ tests/
git commit -m "feat(polymarket): runner + main loop + integration test"
```

---

## Chunk 5: Final Verification

### Task 11: Full Test Suite + Documentation

- [ ] **Step 1: Run complete Polymarket test suite**

```bash
pytest tests/unit/polymarket/ tests/integration/test_polymarket_flow.py -v --tb=short
```

- [ ] **Step 2: Verify no regression on existing system**

```bash
pytest tests/unit/runner/ -x -q -k "not test_control_plane_flush"
pytest execution/tests/ -x -q
cd ext/rust && cargo test
```

- [ ] **Step 3: Update CLAUDE.md**

Add Polymarket entry to Key Files and Architecture sections.

- [ ] **Step 4: Update scripts/catalog.py**

Add `polymarket.runner` as a supported entry point.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(polymarket): complete V1 — types, client, features, signals, sizing, decision, runner"
```
