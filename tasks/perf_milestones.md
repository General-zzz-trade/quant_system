# 性能压榨里程碑规划 (ALL COMPLETE 2026-03-09)

> **Status**: COMPLETED (2026-03-24) — 性能优化里程碑全部完成 (2026-03-09).
> 更新时间: 2026-03-12. 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).

## 当前基线

每 tick 热路径延迟分布（3 symbol 场景）：

| 组件 | 延迟 | 占比 |
|------|------|------|
| FeatureComputeHook (unified) | ~60 μs | 18% |
| 外部数据源解析 (_resolve_bar_sources) | ~80 μs | 24% |
| Monitoring (Prometheus gauges) | ~30 μs | 9% |
| Pipeline state export | ~15 μs | 4% |
| Decision engine (signal+hash+alloc) | ~80 μs | 24% |
| Dispatcher + dedup + guards | ~15 μs | 4% |
| WebSocket recv→parse→dispatch | ~20 μs | 6% |
| Python 胶水 (coordinator, loop, locks) | ~30 μs | 9% |
| **合计** | **~330 μs** | 100% |

---

## M1: Monitoring 降频 + Decision 哈希加速（1-2天）

**目标**: 省 ~60 μs/tick，零架构风险

### M1.1: Monitoring 按需写入（省 ~20 μs）

**问题**: `EngineMonitoringHook.__call__()` 每 tick 执行 20-30 次 `set_gauge()`。
3 symbol × (2 market + 3 position) = 15 次 labelled gauge set，每次 ~1-2 μs。

**方案**: 降频到每 N 个 event 才写 Prometheus，关键指标（balance/equity）保持每 tick。

```python
# engine_hook.py __call__():
self._event_count += 1
# 关键指标: 每 tick
if self.metrics is not None:
    self.metrics.inc_counter("pipeline_events_total")
    if equity is not None:
        self.metrics.set_gauge("equity_usdt", float(equity))

# 非关键指标: 每 10 tick（1m bar = 每 10 分钟, 足够 Grafana 面板）
if self._event_count % 10 == 0:
    # market data age, prices, positions, kill switch, counters, bridge hold
    ...
```

**风险**: 零。Prometheus scrape 间隔通常 15-30s，10-tick 降频完全无损。

### M1.2: Decision stable_hash → Rust（省 ~40 μs）

**问题**: `decision/utils.py:stable_hash()` 用 Python hashlib.sha1，每 tick 2 次调用。
SHA1 本身 ~5 μs/call Python，加上 `.encode()` + 循环。

**方案 A**: 迁移到已有的 `rust_stable_hash()` (SHA256，rust/src/digest.rs)。

```python
# decision/utils.py
from _quant_hotpath import rust_stable_hash

def stable_hash(parts: list[str], *, prefix: str) -> str:
    text = "\x1f".join(parts)
    return f"{prefix}-{rust_stable_hash(text, 16)}"
```

**方案 B**: 新增 `rust_stable_hash_sha1()` 保持 SHA1 兼容（避免 order_id 变化影响去重）。

**建议**: 方案 A，但需要一次性清理历史 dedup 缓存。SHA256 比 SHA1 更安全，Rust 实现 <1 μs。

**风险**: 低。order_id 格式变化不影响运行时（每次启动重新生成）。

### 验证
```bash
python3 -m pytest tests/ -x -q
# 手动: 对比 monitoring Grafana 面板，确认指标连续性
```

---

## M2: 外部数据源批量推送（2-3天）

**目标**: 省 ~50 μs/tick

### M2.1: _resolve_bar_sources → Rust 侧缓存

**问题**: `feature_hook.py:_resolve_bar_sources()` 每 tick 调 10+ Python callable，
逐个解析 funding_rate/OI/LS_ratio/spot_close/FGI/IV/PCR/onchain/liquidation/mempool/macro/sentiment。
每个 source 是 `lambda: latest_value` 模式，Python 函数调用开销 ~3-5 μs/call。

**方案**: 在 RustUnifiedPredictor 侧缓存外部数据，Python 只在**数据变化时**推送。

```python
# 新方法: push_external_data() — 仅在 source 值变化时调用
predictor.push_external_data(symbol,
    funding_rate=0.0001,  # 只推非 NaN 的
    open_interest=15000000.0,
    # ... 其他变化的字段
)

# push_bar_and_predict() 使用缓存的 external data
predictor.push_bar_and_predict(symbol, close=..., volume=..., high=..., low=..., open=..., hour_key=...)
```

**Rust 侧**:
```rust
struct ExternalData {
    funding_rate: f64,
    open_interest: f64,
    ls_ratio: f64,
    // ... 30 fields, all f64::NAN by default
}
// push_bar 时直接从缓存读，零 Python↔Rust 边界穿越
```

**优化效果**:
- 数据源 poll 频率远低于 bar 频率（funding = 8h, OI = 5m, macro = 1d）
- 大部分 tick 零 source 调用，只在数据更新时推一次

### M2.2: 数据源 pull → push 反转

**当前**: feature_hook 每 tick pull 所有 source（`_funding_src()`, `_oi_src()`, ...）
**目标**: source 自己 push 到缓存，feature_hook 只读缓存

```python
# 新增: ExternalDataCache (Python dict, source push 时写入)
class ExternalDataCache:
    _data: Dict[str, Dict[str, float]]  # {symbol: {field: value}}

    def update(self, symbol: str, **kwargs): ...
    def get_for_symbol(self, symbol: str) -> Dict[str, float]: ...
```

Runner 初始化时把 cache 注入 source callback + feature_hook。
Source callback 周期性写入 cache（已有 polling thread），feature_hook 直接读。

**风险**: 中。需改动 runner 初始化逻辑和 source callback 模式。

### 验证
```bash
python3 -m pytest tests/ -x -q
# benchmark: 对比 push_bar_and_predict latency with/without external data
```

---

## M3: Pipeline State 延迟导出（1-2天）

**目标**: 省 ~10 μs/tick

### M3.1: Lazy State Export

**问题**: `pipeline.py:_export_store_state()` 每次 state change 做 6 次 Rust→Python dict copy：
- `dict(store.get_markets())` — 全 market dict
- `dict(store.get_positions())` — 全 position dict
- `account_from_rust()`, `portfolio_from_rust()`, `risk_from_rust()`

大部分下游只读 1-2 个 symbol 的 market/position。

**方案**: LazyStateSnapshot — 只在访问时才跨 Rust 边界取数据。

```python
class LazyStateSnapshot:
    __slots__ = ("_store", "_markets", "_positions", "_account", ...)

    @property
    def markets(self):
        if self._markets is None:
            self._markets = dict(self._store.get_markets())
        return self._markets
```

**风险**: 低。PipelineOutput 接口不变，只是延迟实际化。
但需确认所有下游消费者的访问模式（monitoring hook 访问全部 markets/positions）。

**实际收益**: 如果 monitoring 降频（M1.1），则 9/10 tick 不需要 full export → 直接省 ~12 μs。

### 验证
```bash
python3 -m pytest tests/unit/engine/ tests/integration/ -x -q
```

---

## M4: WebSocket Rust 客户端（1-2周，架构级）

**目标**: 省 ~20 μs/tick + 消除 GIL 竞争 + 降低 tail latency

### 现状分析

| 组件 | 实现 | 延迟 |
|------|------|------|
| WS recv | Python websocket-client (blocking) | ~100-500 μs（网络 I/O，非 CPU） |
| JSON parse (kline) | Rust rust_parse_kline | ~1-2 μs |
| JSON parse (depth) | Rust rust_parse_depth | ~1-2 μs |
| JSON parse (user_stream) | **Python json.loads** | ~5-15 μs |
| JSON parse (trade) | **Python json.loads** | ~5-15 μs |
| MarketEvent 构造 | Python dataclass | ~2-3 μs |
| 线程间传递 | queue.put_nowait → queue.get | ~1-2 μs |

**关键发现**: WS recv 本身是网络 I/O bound，CPU 开销不大。
真正的问题是 **GIL 竞争**：WS recv thread 持 GIL 做 json.loads 时，engine loop thread 被阻塞。

### Phase 1: UserStream + Trade JSON → Rust（3天）

**文件**: `user_stream_processor_um.py`, `ws_trade_stream.py`
**新 Rust**: `rust/src/json_parse.rs` 扩展

```rust
// 已有: rust_parse_kline, rust_parse_depth
// 新增:
pub fn rust_parse_user_stream(raw: &str) -> Option<PyDict>  // ORDER_TRADE_UPDATE → fields
pub fn rust_parse_agg_trade(raw: &str) -> Option<PyDict>    // aggTrade → fields
```

**效果**: json.loads (5-15 μs) → Rust serde (1-2 μs)，省 ~8-12 μs/tick。

### Phase 2: Rust WS Transport（1周）

**方案**: `tokio-tungstenite` + `crossbeam-channel` → Python consumer。

```
[Binance WS]
    ↓ (TCP)
[Rust tokio runtime — 独立线程，无 GIL]
    ├─ tungstenite recv frame
    ├─ serde_json parse
    ├─ 构造 RustMarketEvent / RustTradeEvent
    └─ crossbeam::channel::send(event)
        ↓
[Python thread — channel.recv()]
    └─ 直接拿到 Rust event object → dispatch
```

**新 Rust 模块**: `rust/src/ws_client.rs` (~500 LOC)

**PyO3 API**:
```rust
#[pyclass]
pub struct RustWsClient {
    tx: crossbeam::Sender<WsCommand>,  // connect/subscribe/close
    rx: crossbeam::Receiver<WsEvent>,  // parsed events
    runtime: tokio::runtime::Handle,
}

#[pymethods]
impl RustWsClient {
    fn connect(&self, url: &str) -> PyResult<()>;
    fn subscribe(&self, streams: Vec<String>) -> PyResult<()>;
    fn recv(&self, timeout_ms: u64) -> PyResult<Option<PyObject>>;  // 阻塞，但不持 GIL
    fn close(&self) -> PyResult<()>;
}
```

**关键**: `recv()` 调用 `py.allow_threads(|| channel.recv_timeout(dur))` — 释放 GIL 等待。

**新 Cargo deps**: `tokio`, `tokio-tungstenite`, `crossbeam-channel`, `futures-util`

### Phase 3: 集成到 Runtime（2-3天）

替换 `BinanceMarketDataRuntime` 中的 transport：

```python
# market_data_runtime.py
if rust_ws_available:
    self._rust_client = RustWsClient()
    self._rust_client.connect(url)
    self._rust_client.subscribe(streams)
else:
    # fallback to websocket-client
    ...
```

**recv loop 改造**:
```python
def _run_loop(self):
    while self._running:
        event = self._rust_client.recv(timeout_ms=5000)  # GIL released during wait
        if event is not None:
            self._dispatch(event)
```

### Phase 4: Reconnection + ListenKey 管理（2天）

Rust 侧实现 reconnect state machine（mirror `reconnecting_ws_transport.py`）:
- Exponential backoff (1s → 60s)
- Auto re-subscribe on reconnect
- ListenKey renewal notification

### 预期收益

| 指标 | Before | After |
|------|--------|-------|
| JSON parse (user_stream) | 5-15 μs (Python) | 1-2 μs (Rust) |
| JSON parse (trade) | 5-15 μs (Python) | 1-2 μs (Rust) |
| GIL contention per WS recv | ~50-200 μs (tail) | ~0 μs (GIL released) |
| Reconnect downtime | ~2-5s (Python) | ~1-2s (Rust async) |

**主要价值不在平均延迟，在 tail latency**: GIL 竞争导致的 P99 抖动可从 ~500 μs 降到 ~50 μs。

### 风险

- **高工程量**: ~800-1000 LOC Rust，需要 tokio runtime 管理
- **Cargo 依赖膨胀**: tokio + tungstenite + crossbeam 增加编译时间和 binary size
- **测试复杂度**: 需要 mock WS server 做集成测试
- **运维**: Rust panic 在 WS thread 需要优雅降级到 Python fallback

---

## M5: Python 胶水精简（3-5天）

**目标**: 省 ~15 μs/tick，减少 Python 对象分配

### M5.1: Coordinator emit() 精简

**问题**: `coordinator.py:emit()` 每 tick 有 RLock + getattr 链 + guard 调用。

**方案**: 对 market event（最频繁路径）做 fast-path bypass：

```python
def emit(self, event, actor=None):
    # Fast path: market event → feature_hook → pipeline → done
    if self._is_market_event(event):
        features = self._cfg.feature_hook.on_event(event)
        inp = PipelineInput(event=event, features=features, actor=actor)
        out = self._pipeline.apply(inp)
        if self._cfg.on_pipeline_output and out.advanced:
            self._cfg.on_pipeline_output(out)
        return
    # Slow path: other events (fills, funding, etc.)
    ...
```

### M5.2: Event 类型用 int enum

**问题**: `dispatcher.py` 用 4 层 getattr 查找 event_type（1-3 μs）。

**方案**: 所有 event 类加 `__event_type_id__: int` class variable，dispatcher 直接读。

### M5.3: PipelineInput 轻量化

**问题**: 每 tick 创建 PipelineInput dataclass，有 __init__/__post_init__ 开销。

**方案**: 改用 `__slots__` 普通类或 namedtuple。

### 预期收益

| 优化 | 省 |
|------|---|
| Fast-path bypass | ~5 μs |
| Int enum routing | ~2 μs |
| Lightweight PipelineInput | ~3 μs |
| 减少 getattr 链 | ~3 μs |

---

## 执行路线图

```
Week 1:  M1 (monitoring + hash)     → 省 ~60 μs，低风险
Week 1:  M3 (lazy state export)     → 省 ~10 μs，低风险
Week 2:  M2 (source 批量推送)       → 省 ~50 μs，中风险
Week 2:  M5.1-M5.3 (胶水精简)      → 省 ~15 μs，低风险
Week 3-4: M4 Phase 1-2 (Rust WS)   → 省 ~15 μs 均值，P99 大幅改善
Week 4-5: M4 Phase 3-4 (集成+重连) → 生产就绪
```

## 预期总收益

| 场景 | Before | After | 改善 |
|------|--------|-------|------|
| 平均 tick 延迟 | ~330 μs | ~180 μs | -45% |
| P99 tick 延迟 | ~800 μs | ~300 μs | -63% |
| GIL contention | 每 WS recv | 接近零 | 质变 |

## 不做的事

| 方向 | 理由 |
|------|------|
| Coordinator 全迁 Rust | 编排逻辑，ROI 极低 |
| Risk evaluator 优化 | 已是 Rust，~2 μs，无空间 |
| Feature engine 优化 | 已是 Rust incremental，~33 μs 含 105 特征 |
| 替换 Prometheus 库 | Pull-based，非热路径瓶颈 |
