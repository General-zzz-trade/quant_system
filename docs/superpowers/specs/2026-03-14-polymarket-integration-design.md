# Polymarket Integration Design

> Date: 2026-03-14
> Status: Approved
> Scope: Crypto-related prediction markets, directional trading, pure market data alpha

---

## 1. Overview

Add Polymarket prediction market support as a new market vertical within quant_system. Trades crypto-related binary outcome markets (e.g., "BTC above $100K by March") using market microstructure signals (probability momentum, orderbook imbalance, cross-market Binance price).

**Architecture**: Lightweight independent module (approach 1) — reuses execution safety infrastructure (state machine, dedup, reconcile, risk) but does NOT share coordinator, feature engine, or constraint pipeline with the existing Binance perpetual futures path.

**Constraints**:
- Pure CLOB API (no direct chain interaction in V1)
- Pure market data signals (no NLP/news in V1)
- Crypto-related markets only (keyword-filtered)
- Independent runner (does not pollute live_runner.py)

---

## 2. File Structure

```
execution/adapters/polymarket/
├── __init__.py
├── client.py              # REST client (CLOB API v2)
├── ws_client.py           # WebSocket orderbook/trades stream
├── auth.py                # API key + HMAC/EIP-712 signing
├── mapper.py              # Polymarket JSON → CanonicalOrder/CanonicalFill
├── market_discovery.py    # Discover/filter crypto-related markets
└── types.py               # Market, Outcome, Token types

polymarket/
├── __init__.py
├── data.py                # MarketScanner + historical data collection
├── features.py            # ~30 probability market features
├── signals.py             # Signal generation (rules V1, ML V2)
├── decision.py            # Binary outcome decision module
├── sizing.py              # Kelly criterion position sizing
├── runner.py              # Independent PolymarketRunner
└── config.py              # PolymarketConfig dataclass

config/polymarket.yaml     # Production config
```

---

## 3. Polymarket API Adapter

### 3.1 client.py
REST client wrapping Polymarket CLOB API v2:
- `get_markets()` — list active markets with metadata
- `get_orderbook(token_id)` — L2 orderbook snapshot
- `get_trades(token_id)` — recent trade history
- `create_order(token_id, side, size, price)` — limit order on CLOB
- `cancel_order(order_id)` — cancel active order
- `get_positions()` — current share holdings
- `get_balances()` — USDC balance on Polygon

### 3.2 ws_client.py
WebSocket client for real-time data:
- Subscribe to orderbook updates per market
- Subscribe to trade stream
- Push events into the polymarket event pipeline

### 3.3 auth.py
Polymarket uses API key + HMAC signing for CLOB API. Authentication requires:
- API key + secret from Polymarket account
- Request signing (similar to Binance HMAC pattern)

### 3.4 mapper.py
Maps between Polymarket and system canonical types:
- Symbol format: `POLY:{slug}:{outcome}` (e.g., `POLY:btc-above-100k-march:YES`)
- Polymarket `token_id` → system `symbol`
- Shares → `qty` (Decimal)
- Probability price (0.01-0.99) → `price` (Decimal)
- Buy YES = long event probability, Buy NO = short event probability

### 3.5 market_discovery.py
Periodic scanner (hourly) for crypto-related markets:
- Keyword filter: BTC, Bitcoin, ETH, Ethereum, crypto, Solana, SOL
- Minimum liquidity: $10K 24h volume
- Minimum time to expiry: >24h
- Outputs active market list with metadata

---

## 4. Data Pipeline & Feature Engine

### 4.1 data.py — MarketScanner
- Hourly scan of Polymarket API for new crypto markets
- Store active market list + historical probability/orderbook snapshots (SQLite)
- Read corresponding crypto asset prices from Binance data pipeline (existing infrastructure)

### 4.2 features.py — ~30 Features

| Category | Features | Description |
|----------|----------|-------------|
| Probability momentum | `prob_ret_1h/4h/12h/24h` | Probability change rate |
| Mean reversion | `prob_zscore_24h/72h` | Probability deviation from rolling mean |
| Volatility | `prob_vol_12h/24h` | Probability volatility |
| Orderbook | `bid_ask_spread`, `mid_price`, `depth_imbalance` | Liquidity + directional signal |
| Trade flow | `trade_intensity`, `buy_sell_ratio`, `large_trade_flag` | Trade behavior patterns |
| Expiry | `hours_to_expiry`, `time_decay_rate` | Time decay dynamics |
| Cross-market | `btc_price_vs_strike`, `btc_momentum_12h` | Binance BTC price vs market strike |

Bar period: 1h (consistent with existing system).

### 4.3 signals.py
- Input: feature DataFrame per market
- Output: `signal in [-1, +1]`, positive = buy YES (probability underpriced), negative = buy NO
- V1: rule-based (btc_price_vs_strike + prob_zscore agreement)
- V2: LightGBM (after accumulating sufficient data)

---

## 5. Decision Module & Sizing

### 5.1 decision.py — PolymarketDecisionModule
Interface: `decide(snapshot) -> Iterable[IntentEvent]` (same as existing modules)

Logic:
1. For each active market, read features
2. Generate signal (V1 rules, V2 ML)
3. Signal strength > threshold (0.15) → generate IntentEvent
4. Skip if probability < 0.15 or > 0.85 (extreme prices, low edge)
5. Skip if hours_to_expiry < 6 (time decay risk for new positions)

Exit conditions:
- Probability moves against position by > stop_loss (15%)
- Signal reversal
- Auto-close if hours_to_expiry < 1 (unless deep in-the-money)

### 5.2 sizing.py — Kelly Criterion
Binary outcomes use Kelly formula: `f = (p * b - q) / b`
- `p` = estimated true probability
- `b` = odds = (1 - market_price) / market_price
- Conservative: half-Kelly (f * 0.5)
- Per-market cap: 10% of account
- Total Polymarket cap: 30% of account

---

## 6. Runner

### 6.1 PolymarketRunner
Independent runner, does NOT share coordinator with LiveRunner.

**Reuses from existing system**:
- KillSwitch, DrawdownCircuitBreaker (risk protection)
- OrderStateMachine (order lifecycle tracking)
- EventRecorder (replay/recovery)
- Prometheus metrics export
- Structured JSON logging

**Does NOT reuse**:
- FeatureComputeHook / RustFeatureEngine (105 price features not applicable)
- InferenceBridge / constraint_pipeline (z-score/deadzone/trend-hold not applicable)
- GateChain (8-gate chain designed for perpetual futures)

**Main loop** (1h cycle):
1. MarketScanner discovers/updates active crypto markets
2. Fetch orderbook + trades for each active market
3. Compute ~30 features per market
4. Run decision module → signals → IntentEvents
5. Size positions (half-Kelly) → OrderEvents
6. Execute via Polymarket CLOB API
7. Monitor fills, reconcile positions

**Entry point**: `python3 -m polymarket.runner --config config/polymarket.yaml`

### 6.2 PolymarketConfig
```python
@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    scan_interval_sec: int = 3600
    min_liquidity_usd: float = 10_000
    max_position_pct: float = 0.10
    max_total_pct: float = 0.30
    signal_threshold: float = 0.15
    stop_loss_pct: float = 0.15
    kelly_fraction: float = 0.5
    crypto_keywords: tuple = ("BTC", "Bitcoin", "ETH", "Ethereum",
                              "crypto", "Solana", "SOL")
```

---

## 7. Reuse Matrix

| Component | Reuse? | Notes |
|-----------|--------|-------|
| event/types.py (MarketEvent, OrderEvent, FillEvent) | Yes | Universal event semantics |
| execution/state_machine/ | Yes | 8-state order lifecycle universal |
| risk/kill_switch.py | Yes | Global emergency stop |
| risk/drawdown_breaker.py | Yes | Equity protection |
| execution/safety/duplicate_guard.py | Yes | Fill dedup |
| execution/reconcile/ | Yes | Position reconciliation |
| monitoring/ (Prometheus, alerts) | Yes | Observability |
| runner/recovery.py | Partial | Checkpoint/restore pattern |
| features/enriched_computer.py | No | Price technical features not applicable |
| ext/rust/constraint_pipeline.rs | No | z-score/deadzone/trend-hold not applicable |
| engine/coordinator.py | No | Independent event loop |
| runner/gate_chain.py | No | Futures-specific gates |

---

## 8. Non-Goals (V1)

- No chain-level monitoring (Polygon CTF contracts)
- No NLP/news-based signals
- No non-crypto markets
- No market making
- No cross-market arbitrage
- No ML model (rule-based V1; ML as V2 after data accumulation)

---

## 9. Testing Plan

- Unit tests for API client (mock responses)
- Unit tests for mapper (Polymarket JSON → canonical types)
- Unit tests for feature computation
- Unit tests for Kelly sizing
- Integration test for market discovery → signal → order flow
- Backtest on historical Polymarket data (if available via API)

---

## 10. Risk Controls

- Half-Kelly sizing (conservative)
- Per-market 10% cap, total 30% cap
- Stop-loss at 15% probability move
- No trading near expiry (<6h for entry, <1h auto-exit)
- Skip extreme probability markets (<0.15 or >0.85)
- Shared KillSwitch with main trading system
- Independent DrawdownCircuitBreaker for Polymarket equity
