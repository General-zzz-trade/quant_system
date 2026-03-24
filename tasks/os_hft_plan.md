# OS HFT 改造计划（基于当前硬件）

> **Status**: COMPLETED (2026-03-24) — OS/kernel tuning 专项计划. HFT 策略已验证不可行 (Sharpe -5 to -25).
> 更新时间: 2026-03-12. 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).
> 生产策略: Strategy H (4h primary + 1h scaler), 非 HFT.

## 当前系统摸底

| 项目 | 当前值 | 评估 |
|------|--------|------|
| CPU | Xeon Platinum, 1 core + HT = 2 vCPU | 虚拟化，无 cpufreq 控制 |
| RAM | 3.4GB, swap 1GB used | 内存紧张 |
| Kernel | 6.8.0-63-generic, PREEMPT_VOLUNTARY | 非 RT，非 lowlatency |
| NIC | virtio_net (虚拟) | 无 kernel bypass 可能 |
| NUMA | 单节点 | N/A |
| Disk | virtio (vda), mq-deadline | 可优化为 none |

### 已完成的调优（✓ 不需重复）

| 优化 | GRUB/sysctl | 状态 |
|------|-------------|------|
| `nohz_full=1` | GRUB ✓ | CPU1 无时钟中断 |
| `rcu_nocbs=1` | GRUB ✓ | RCU 回调不在 CPU1 |
| `idle=poll` | GRUB ✓ | 无 C-state 唤醒延迟 |
| `processor.max_cstate=1` | GRUB ✓ | 限制低功耗状态 |
| `tsc=reliable` | GRUB ✓ | 稳定 TSC |
| `nosoftlockup` | GRUB ✓ | 无 watchdog 开销 |
| `mitigations=off` | GRUB ✓ | 无 Spectre/Meltdown 开销 |
| `iommu=pt` | GRUB ✓ | IOMMU 透传 |
| `vm.swappiness=1` | sysctl ✓ | 最小化交换 |
| `busy_read/busy_poll=50` | sysctl ✓ | 忙轮询网络 |
| `tcp_fastopen=3` | sysctl ✓ | TFO 开启 |
| IRQ affinity → CPU0 | runtime ✓ | 网络中断不打扰 CPU1 |
| THP=always | runtime ✓ | 透明大页 |
| ethtool rx-usecs=0 | hardware ✓ | 零合并延迟 |
| Ring buffer=4096 | hardware ✓ | 最大 |

### 可改进项（本计划目标）

| 项目 | 当前 | 目标 | 预期收益 |
|------|------|------|----------|
| Kernel | generic (PREEMPT_VOLUNTARY) | lowlatency (PREEMPT) | 中断延迟 ~50μs→~15μs |
| isolcpus | 未设置 | isolcpus=1 | CPU1 完全专属交易 |
| Hugepages | THP only (0 static) | 预分配 512MB static | 消除 THP 分裂延迟 |
| qdisc | fq (fair queue) | noqueue | 消除排队延迟 |
| Disk scheduler | mq-deadline | none | virtio 不需要调度 |
| Swap 占用 | 1GB used | 清理释放 | 减少 page fault |
| skew_tick | 未设置 | skew_tick=1 | 减少 CPU 间 tick 同步 |
| GRO | on | off | 减少 batching 延迟 |
| Lock-free queue | Python queue.Queue | Rust SPSC ring buffer | 消除 mutex ~10μs |
| Thread pinning | cpuset="1" (docker) | taskset + isolcpus | 完全绑核 |
| mlock | 未锁定 | mlock 关键内存 | 防止 page fault |

---

## Phase 1: Lowlatency 内核（需要 reboot）

### 1.1 安装 lowlatency 内核
```bash
apt install -y linux-image-lowlatency-hwe-24.04 linux-headers-lowlatency-hwe-24.04
```

lowlatency 内核特性（vs generic）：
- CONFIG_PREEMPT=y（vs PREEMPT_VOLUNTARY）— 全抢占式
- CONFIG_HZ=1000（vs 250）— 更高精度定时
- 更短的调度延迟

### 1.2 更新 GRUB 参数
```
# /etc/default/grub GRUB_CMDLINE_LINUX 追加：
isolcpus=1 skew_tick=1 hugepagesz=2M hugepages=256
```

完整 GRUB_CMDLINE_LINUX 目标：
```
mitigations=off processor.max_cstate=1 idle=poll tsc=reliable nosoftlockup
nohz_full=1 rcu_nocbs=1 iommu=pt isolcpus=1 skew_tick=1
hugepagesz=2M hugepages=256
```

### 1.3 验证（reboot 后）
```bash
uname -r                                    # 确认 lowlatency
cat /proc/cmdline                           # 确认 isolcpus=1
cat /sys/devices/system/cpu/isolated        # 应显示 "1"
grep HugePages_Total /proc/meminfo          # 应显示 256
cyclictest -t1 -p90 -i1000 -l10000 -a1     # 延迟测试（目标 p99 < 20μs）
```

- [ ] 安装 lowlatency 内核
- [ ] 更新 GRUB 参数
- [ ] reboot + 验证

---

## Phase 2: 网络 + 磁盘调优（运行时，可立即执行）

### 2.1 网络 qdisc
```bash
# noqueue 消除排队延迟（直接发送）
tc qdisc replace dev eth0 root pfifo_fast
```

### 2.2 关闭 GRO（减少 batching）
```bash
ethtool -K eth0 gro off
```

### 2.3 磁盘调度器
```bash
# virtio 不需要 IO 调度，none = 最低延迟
echo none > /sys/block/vda/queue/scheduler
```

### 2.4 Swap 清理
```bash
# 把 swap 内容刷回内存（如果空间够）
swapoff /swapfile2              # 释放几乎没用的 2GB swapfile2
# 保留 /swapfile 作为安全网
sysctl -w vm.swappiness=0       # 运行时完全不 swap（但保留文件）
```

### 2.5 TCP 极致调优
```bash
sysctl -w net.ipv4.tcp_nodelay=1
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.core.optmem_max=16777216
```

### 2.6 持久化
所有改动写入 `deploy/sysctl-trading.conf` + 启动脚本。

- [ ] qdisc → pfifo_fast
- [ ] GRO off
- [ ] disk scheduler → none
- [ ] swap 清理
- [ ] TCP 调优
- [ ] 持久化配置

---

## Phase 3: Rust SPSC Ring Buffer（替换 queue.Queue）

当前 EngineLoop 用 `queue.Queue`（内部 threading.Lock + Condition）。
每次 put/get 有 ~5-10μs 的 mutex 开销。

### 3.1 新建 `rust/src/spsc_ring.rs`
```rust
// 无锁单生产者单消费者环形缓冲区
// PyO3 导出：RustSpscRing
// - push(event: PyObject) -> bool
// - pop() -> Option<PyObject>
// - len() -> usize
// - capacity() -> usize
//
// 内部：CacheLine-padded AtomicUsize head/tail
// 固定容量，预分配，零 malloc
```

### 3.2 修改 `engine/loop.py`
```python
# 替换 queue.Queue 为 RustSpscRing
# submit() → ring.push()
# step() → ring.pop()
# 移除 threading.Lock 依赖
```

### 3.3 验证
```bash
pytest tests/unit/engine/ -x -q
# + 微基准测试：put/get 延迟 < 200ns（vs Queue ~5μs）
```

- [ ] 实现 RustSpscRing
- [ ] 集成到 EngineLoop
- [ ] 测试 + 基准

---

## Phase 4: 线程绑核 + mlock

### 4.1 交易线程绑核
```python
# runner 启动后，主线程绑 CPU1
import os
os.sched_setaffinity(0, {1})
```

配合 `isolcpus=1`，CPU1 上只有交易进程，零上下文切换。

### 4.2 mlock 关键内存
```python
import ctypes
libc = ctypes.CDLL("libc.so.6")
# MCL_CURRENT | MCL_FUTURE = 3
libc.mlockall(3)  # 锁定所有当前和未来分配的内存
```

防止交易热路径的内存被 swap out 导致 page fault (~10-100μs)。

### 4.3 Docker 配置
docker-compose.yml 已有 `ulimits.memlock: -1` 和 `SYS_NICE`。
需确认 `isolcpus` 后 `cpuset: "1"` 的正确交互。

- [ ] 绑核代码
- [ ] mlock 代码
- [ ] Docker 验证

---

## Phase 5: 启动脚本统一（持久化所有运行时调优）

### 5.1 创建 `deploy/tune-os.sh`
```bash
#!/bin/bash
# 运行时 OS 调优（每次启动执行）
sysctl -p /etc/sysctl.d/99-trading.conf
echo none > /sys/block/vda/queue/scheduler
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo 1 > /proc/irq/29/smp_affinity
echo 1 > /proc/irq/30/smp_affinity
ethtool -K eth0 gro off
tc qdisc replace dev eth0 root pfifo_fast
```

### 5.2 Systemd service
```ini
# /etc/systemd/system/trading-tune.service
[Unit]
Description=Trading OS Tuning
After=network.target

[Service]
Type=oneshot
ExecStart=/opt/quant_system/deploy/tune-os.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

- [ ] 创建 tune-os.sh
- [ ] 创建 systemd service
- [ ] enable + 验证

---

## 预期效果

| 指标 | 改造前 | 改造后（预估） |
|------|--------|---------------|
| 内核中断延迟 (p99) | ~50-100μs | ~10-20μs (lowlatency) |
| Queue put/get | ~5-10μs | <0.2μs (SPSC ring) |
| 上下文切换 (CPU1) | 偶发 | 零（isolcpus） |
| Page fault | 偶发 | 零（mlock + hugepages） |
| 网络 qdisc 延迟 | ~5-10μs (fq) | ~0μs (pfifo_fast) |
| tick-to-trade 估计 | ~95-150μs | ~50-80μs |

## 执行顺序

Phase 2（立即，无需重启）→ Phase 4.2/4.1（代码改动）→ Phase 3（SPSC ring）→ Phase 5（持久化）→ Phase 1（需重启，最后做）
