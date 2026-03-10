# 极限优化方案 — 基于阿里云东京 ECS 实际硬件

## 硬件约束

| 项目 | 值 | 影响 |
|------|------|------|
| CPU | Xeon Platinum (Skylake) × 2 vCPU | CPU0=OS/IO, CPU1=交易热路径 |
| RAM | 3.4 GiB (可用 ~800 MiB) | 模型常驻内存，无余量做大缓存 |
| 内核 | 6.11.0-lowlatency, PREEMPT_DYNAMIC | 已最优 |
| 隔离 | isolcpus=1, nohz_full=1, idle=poll | CPU1 已完全隔离 |
| 网络 | 到 Binance 4.4ms (东京同区域) | 网络已是极限 |
| 虚拟化 | KVM/QEMU | 有 ~5-10μs jitter，无法消除 |
| AVX | AVX-512 可用 | 可用于 SIMD 特征计算 |

## 当前延迟分解

| 阶段 | 延迟 | 占比 |
|------|------|------|
| WS 接收 (网络) | 4,400 μs | 81% |
| 引擎处理 (CPU) | 1,000 μs | 18% |
| REST 下单 (网络) | 30,000-200,000 μs | — |
| **信号→下单总计** | **~35-205 ms** | — |

## 极限目标

| 阶段 | 目标 | 手段 |
|------|------|------|
| WS 接收 | 4.4 ms (不变) | 已在东京同区域 |
| 引擎处理 | **200 μs** | Phase D: 全 Rust 热路径 |
| 下单 | **4.4 ms** | Phase E: REST→WS 协议切换 |
| **信号→下单总计** | **~9 ms** | 4x 提速 |

---

## Phase D: 全 Rust 热路径 (引擎 1000→200 μs)

### 当前 Python 热路径开销分解

| 步骤 | 当前 | 目标 | 节省 | 方法 |
|------|------|------|------|------|
| Event float 提取 | 40 μs | 10 μs | 30 μs | Rust 预处理 |
| 外部数据解析 | 100 μs | 10 μs | 90 μs | Rust 缓存+批处理 |
| 特征 dict 构造 | 35 μs | 5 μs | 30 μs | Rust 预构建 |
| Guard + 路由 | 25 μs | 10 μs | 15 μs | 单 Rust 调用 |
| Snapshot 构造 | 20 μs | 10 μs | 10 μs | Rust 构建 |
| Order event 封装 | 20 μs | 5 μs | 15 μs | Rust 返回 |
| Python 解释器开销 | 340 μs | 50 μs | 290 μs | 减少 boundary crossings |
| Rust 计算 (不变) | 180 μs | 100 μs | 80 μs | SIMD + 内联优化 |
| **总计** | **1000 μs** | **200 μs** | **800 μs** | |

### D.1: RustHotLoop — 统一入口 (~2 天)

**目标**: 将 coordinator._handle_market_tick_fast() 的 Python 包装消除

当前调用链 (每 tick 10+ 次 Python→Rust 边界交叉):
```
Python: coordinator.emit()
  → Python: dispatcher.dispatch()
    → Rust: rust_route_event()
    → Python: _handle_pipeline_event()
      → Python: _handle_market_tick_fast()
        → Python: 提取 float (5x getattr)
        → Python: 外部数据解析 (12x callable)
        → Rust: tp.process_tick()           ← 核心 Rust 调用
        → Python: 构造 features dict
        → Python: 构造 snapshot
        → Python: decision bridge
```

目标调用链 (1 次 Python→Rust 边界交叉):
```
Python: coordinator.emit()
  → Rust: rust_hot_loop(raw_event, external_cache)  ← 单次调用
    → 提取 float
    → push_bar + features + predict + state
    → 构造 features dict (在 Rust 内)
    → 返回 RustTickBundle {tick_result, features_dict, snapshot_fields}
  → Python: decision bridge (轻量包装)
```

**新 Rust 模块**: `hot_loop.rs` (~300 LOC)
- `RustHotLoop` 持有 `RustTickProcessor` + 外部数据缓存
- `process(event_dict) -> RustTickBundle` 单次调用
- 内部合并 float 提取 + process_tick + features dict 构造

**文件改动**:
- `ext/rust/src/hot_loop.rs` (新建, ~300 LOC)
- `ext/rust/src/lib.rs` (注册新模块)
- `engine/coordinator.py` (简化 _handle_market_tick_fast)

### D.2: 外部数据 Rust 缓存 (~1 天)

**目标**: 12 个 Python source callable → 1 个 Rust 缓存查询

当前: 每 5 bar 调用 12 个 Python callable，每次 ~10μs = 120μs
目标: Python 后台线程定期填充 Rust 缓存，热路径只读 Rust 缓存 (~2μs)

```rust
struct ExternalDataCache {
    funding_rate: f64,
    oi: f64,
    ls_ratio: f64,
    spot_close: f64,
    // ... 30+ fields
    last_update_bar: u64,
}

impl ExternalDataCache {
    fn update_from_python(&mut self, data: &PyDict) { ... }
    fn get_snapshot(&self) -> ExternalData { ... }  // zero-copy
}
```

**文件改动**:
- `ext/rust/src/external_cache.rs` (新建, ~150 LOC)
- `engine/feature_hook.py` (后台线程填充)

### D.3: Snapshot 消除 (~1 天)

**目标**: 不再构造 Python StateSnapshot 对象

当前: `_build_snapshot()` 分配 frozen dataclass (11 个字段)
目标: `RustTickBundle` 直接携带 snapshot 字段，decision bridge 直接读取

```rust
#[pyclass]
struct RustTickBundle {
    // tick_processor 结果
    advanced: bool,
    changed: bool,
    ml_score: f64,
    ml_short_score: f64,
    // 状态快照 (直接引用 Rust 堆)
    markets: PyObject,
    positions: PyObject,
    account: PyObject,
    portfolio: PyObject,
    risk: PyObject,
    // 特征
    features: PyObject,  // 预构建的 PyDict
    // 元数据
    symbol: String,
    event_index: i64,
    ts: i64,
}
```

**文件改动**:
- `ext/rust/src/hot_loop.rs` (扩展 RustTickBundle)
- `engine/pipeline.py` (删除 _build_snapshot 调用)
- `decision/ml_decision.py` (适配 RustTickBundle)

### D.4: SIMD 特征计算 (~1 天)

**目标**: 利用 AVX-512 加速 105 特征的 rolling window 计算

当前: feature_engine.rs 逐个标量计算
目标: 批量 SIMD 计算 (8×f64 并行)

```rust
#[cfg(target_feature = "avx512f")]
fn compute_rolling_means_avx512(windows: &[&[f64]; 8], out: &mut [f64; 8]) {
    // 8 个 rolling mean 同时计算
}
```

**关键优化点**:
- rolling mean/std: 8 路并行 → ~4x 提速
- z-score 标准化: SIMD 批量
- 特征归一化: 全 105 特征一次 SIMD pass

**预计节省**: 30→10 μs (特征计算部分)

**文件改动**:
- `ext/rust/src/feature_engine.rs` (SIMD 路径)
- `ext/rust/Cargo.toml` (target-feature flag)

---

## Phase E: 网络层极限优化 (下单 30ms→5ms)

### E.1: WS 下单替代 REST (~2 天)

**目标**: REST API 下单 (30-200ms) → WebSocket 下单 (4-5ms)

Binance 支持 WebSocket API 下单 (wss://ws-api.binance.com/ws-api/v3):
```json
{
  "id": "uuid",
  "method": "order.place",
  "params": {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "MARKET",
    "quantity": "0.001",
    "timestamp": 1234567890,
    "signature": "hmac_sha256..."
  }
}
```

**优势**: 复用已有 WS 连接，省去 TCP 握手 + TLS 协商 (~25ms)

**实现**:
- 扩展 `RustWsClient` 支持双向通信 (recv + send)
- 新增 `RustOrderSubmitter` — WS 下单 + HMAC 签名 (在 Rust 完成)
- 签名在 Rust 中计算 (SHA256 HMAC, 已有 sha2/hmac crate)

**文件改动**:
- `ext/rust/src/ws_client.rs` (扩展 send 功能)
- `ext/rust/src/order_submit.rs` (新建, ~200 LOC)
- `execution/adapters/binance/ws_order_adapter.py` (新建, ~100 LOC)
- `execution/order_router.py` (WS 优先路由)

### E.2: 预签名优化 (~0.5 天)

**目标**: 消除下单时的签名计算延迟

当前: 下单时计算 HMAC-SHA256 (~5μs，但需要 Python str 拼接参数 ~20μs)
目标: Rust 中预构建参数 + 签名

```rust
impl RustOrderSubmitter {
    fn submit_market_order(&self, symbol: &str, side: &str, qty: f64) -> PyResult<String> {
        let ts = SystemTime::now()...;
        let params = format!("symbol={}&side={}&type=MARKET&quantity={:.8}&timestamp={}",
            symbol, side, qty, ts);
        let sig = hmac_sha256(&self.secret_key, params.as_bytes());
        let msg = json!({
            "id": uuid_v4(),
            "method": "order.place",
            "params": { ... "signature": hex(sig) }
        });
        self.ws_send(msg.to_string())
    }
}
```

### E.3: TCP Tuning 微调 (~0.5 天)

当前已有 busy_poll=50, tcp_low_latency=1，进一步:

```bash
# 禁用 TCP 延迟确认 (减少 40ms ACK 延迟)
net.ipv4.tcp_quickack = 1

# 减小 TCP 初始拥塞窗口 (小包快发)
ip route change default via <gw> initcwnd 10 initrwnd 10

# SO_PRIORITY 标记交易流量
# 在 Rust WS client 中设置 socket 优先级
setsockopt(fd, SOL_SOCKET, SO_PRIORITY, 6)
```

---

## Phase F: 内存与 GC 优化 (减少 jitter)

### F.1: GC 冻结 (~0.5 天)

**目标**: 消除 GC pause 导致的延迟尖峰

当前: Python GC 在任意时刻可能触发 stop-the-world (~100-500μs)

```python
# 在交易循环启动时:
import gc
gc.disable()  # 完全禁用 GC

# 在非交易时段 (每 5 分钟) 手动触发:
def _gc_maintenance():
    gc.collect(generation=0)  # 只收集 gen0，快速
    gc.collect(generation=1)  # gen1 较快
    # gen2 仅在每小时整点触发
```

**文件改动**:
- `engine/loop.py` (gc.disable at startup)
- `engine/coordinator.py` (定时 GC 维护)

### F.2: 对象池 (~1 天)

**目标**: 消除热路径上的对象分配

当前: 每 tick 分配 ~5 个临时 Python 对象 (SimpleNamespace, dict, etc.)
目标: 预分配对象池，复用

```python
class _TickObjectPool:
    """预分配对象，避免热路径 malloc"""
    def __init__(self):
        self._snapshot_buf = [StateSnapshot.__new__(StateSnapshot) for _ in range(4)]
        self._idx = 0

    def get_snapshot(self) -> StateSnapshot:
        obj = self._snapshot_buf[self._idx % 4]
        self._idx += 1
        return obj
```

### F.3: 内存预热 (~0.5 天)

**目标**: 启动时预热所有关键数据结构

```python
def _warmup():
    """在 trading loop 启动前，预热所有内存页"""
    # 1. 预热模型权重 (触发 page fault)
    for model in models:
        model.predict(dummy_features)

    # 2. 预热 Rust 状态
    tp.process_tick(dummy_symbol, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    # 3. 锁定内存页 (防止 swap)
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
```

---

## Phase G: CPU 极限 (仅限 2 核场景)

### G.1: 线程亲和性 (~0.5 天)

当前 CPU 分配 (2 核):
```
CPU0: OS + 网络中断 + Docker + 监控
CPU1: 交易热路径 (isolcpus)
```

优化:
```python
import os
# 交易主线程绑定 CPU1
os.sched_setaffinity(0, {1})

# WS 接收线程绑定 CPU0 (与网络中断同核，L1 cache 命中)
ws_thread.set_affinity({0})
```

### G.2: 优先级提升 (~0.5 天)

```bash
# 交易进程设为实时优先级
chrt -f 50 python3 runner/live_runner.py

# 或在代码中:
import os
os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
```

**注意**: SCHED_FIFO 在 2 核机器上要小心，一个核被占满会影响 OS 响应

### G.3: CPU 缓存优化 (~1 天)

Xeon Platinum L3: 33 MiB — 足够放下所有模型 + 特征数据

```
当前内存布局估算:
- LightGBM 模型 (3 symbol × ~2MB) = ~6 MB
- 特征引擎 rolling windows (105 × 500 bar × 8B) = ~420 KB
- 状态存储 (~10 KB)
- 总计热数据: ~7 MB → 完全放入 L3
```

优化: 确保热数据结构在 Rust 中连续分配 (Vec 而非 HashMap)
- feature_engine 的 window 数据改为 `Vec<[f64; 105]>` 连续内存
- 模型树节点改为 `Vec<TreeNode>` 平坦数组

---

## 实施路线图

| 阶段 | 内容 | 预计引擎延迟 | 全链路延迟 | 工作量 |
|------|------|-------------|-----------|--------|
| **现状** | 已完成 P0-P13 | 1000 μs | ~35 ms | — |
| **D.1** | RustHotLoop 统一入口 | 600 μs | ~35 ms | 2 天 |
| **D.2** | 外部数据 Rust 缓存 | 500 μs | ~35 ms | 1 天 |
| **D.3** | Snapshot 消除 | 450 μs | ~35 ms | 1 天 |
| **D.4** | SIMD 特征计算 | 400 μs | ~35 ms | 1 天 |
| **E.1** | WS 下单 | 400 μs | **~9 ms** | 2 天 |
| **E.2** | 预签名 | 400 μs | ~9 ms | 0.5 天 |
| **F.1** | GC 冻结 | 350 μs (p99↓) | ~9 ms | 0.5 天 |
| **F.2** | 对象池 | 300 μs | ~9 ms | 1 天 |
| **F.3** | 内存预热+锁定 | 280 μs (p99↓) | ~9 ms | 0.5 天 |
| **G.1-3** | CPU 亲和+优先级+缓存 | **200 μs** | **~9 ms** | 2 天 |
| **总计** | | **200 μs** | **~9 ms** | **~12 天** |

## 极限目标总结

```
信号→下单全链路:
  WS 接收:     4.4 ms  (网络，不可压缩)
  引擎处理:    0.2 ms  (Phase D: 全 Rust, 当前 5x)
  WS 下单:     4.4 ms  (Phase E: REST→WS, 当前 6-40x)
  ─────────────────────
  总计:       ~9 ms   (当前 ~35ms 的 4x 提速)

延迟分布:
  p50:  ~200 μs  (引擎)
  p95:  ~350 μs  (偶发 Python GC)
  p99:  ~500 μs  (hypervisor jitter)
  p999: ~2 ms    (极端: GC + 虚拟化 + 中断)
```

## 不可突破的硬限制

| 限制 | 原因 | 突破方法 (超出本机范围) |
|------|------|----------------------|
| 4.4ms 网络延迟 | 阿里云东京→Binance 物理距离 | 交易所共置 (<0.1ms) |
| ~5μs hypervisor jitter | KVM 虚拟化 | 裸金属服务器 |
| Python GIL | CPython 限制 | 完全 Rust (无 Python) |
| 2 核 CPU | ECS 规格 | 升级 4-8 核 |
| 3.4G 内存 | ECS 规格 | 升级 8-16G |

## 可行的下一步硬件升级 (成本收益)

| 升级 | 月费增加 | 收益 |
|------|---------|------|
| 4 核 8G ECS | +$30 | 专用 WS 核 + 更大缓存 |
| 裸金属 (阿里云 ecs.ebmc7) | +$200 | 消除虚拟化 jitter |
| AWS Tokyo c7i.xlarge | +$100 | 更高单核频率 (3.6GHz) |
| 交易所共置 | +$500-2000 | 网络 <0.1ms |

## 优先级建议

**ROI 最高的 3 步** (5 天工作，80% 收益):
1. **E.1: WS 下单** — 30ms→5ms，全链路最大瓶颈 (2 天)
2. **D.1: RustHotLoop** — 引擎 1000→600μs (2 天)
3. **F.1: GC 冻结** — p99 从 2ms→500μs (0.5 天)

做完这 3 步，全链路 ~35ms → ~10ms，足以支撑 **秒级 scalping**。
