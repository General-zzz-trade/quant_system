# 小资金增长改造方案

> 状态: 专题策略/资金管理改造方案，不代表当前默认生产配置
> 更新时间: 2026-03-12
> 当前系统现状请优先参考: [`research.md`](/quant_system/research.md) 与 [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

## 问题诊断

| 资金量 | 最保守仓位 (5%×2x) | 能否开仓 | 问题 |
|--------|-------------------|---------|------|
| $140 | $14 notional | ❌ 低于$100最低限 | 无法交易 |
| $500 | $50 notional | ❌ 低于$100最低限 | 无法交易 |
| $1,000 | $100 notional | ✅ 刚好达标 | 无冗余 |
| $2,000 | $200 notional | ✅ 安全 | 正常运作 |

当前系统假设资金 >= $2,000。$140需要以下改造。

---

## 第一层: 阶梯式风控 (Staged Risk Management)

核心思路: 不同资金阶段用不同的风险参数，随着资金增长自动降低风险。

### 设计

```python
class StagedRiskManager:
    """根据当前权益自动调整 risk_fraction 和 leverage。

    阶段设计原则:
    - 小资金阶段: 激进但有止损保护，目标快速脱离危险区
    - 中等资金: 中等风险，开始稳定复利
    - 大资金: 保守，保护利润
    """

    STAGES = [
        # (min_equity, max_equity, risk_fraction, leverage, max_dd_pct)
        Stage(0,     300,   0.50, 3.0, 0.25),   # 生存期: 激进但严格止损
        Stage(300,   800,   0.25, 3.0, 0.20),   # 成长期: 中等风险
        Stage(800,  2000,   0.12, 3.0, 0.15),   # 稳定期: Half-Kelly
        Stage(2000, 5000,   0.05, 2.0, 0.10),   # 安全期: 保守复利
        Stage(5000, float('inf'), 0.03, 2.0, 0.08),  # 机构级
    ]
```

**关键**: 阶段切换必须有**滞后** (hysteresis) — 从Stage 2降到Stage 1需要权益降到280 (不是300)，防止在边界反复切换。

### 实现文件
- NEW: `risk/staged_risk.py`
- MODIFY: `decision/backtest_module.py` (集成StagedRiskManager)
- MODIFY: `ext/rust/src/bin/main.rs` (Rust二进制集成)

---

## 第二层: 多品种组合 (Portfolio Diversification)

当前: 单品种 (ETH或BTC)，月波动 ±30-50%
目标: 3-5品种，月波动降到 ±10-15%

### 数学原理
- N个相关系数ρ的品种，组合波动率 ≈ σ × √((1 + (N-1)ρ) / N)
- 3个品种 (ρ≈0.6): 波动率降低 ~25%
- 5个品种 (ρ≈0.5): 波动率降低 ~35%

### 候选品种
| 品种 | 与BTC相关性 | 日均波动率 | 流动性 |
|------|-----------|----------|--------|
| BTCUSDT | 1.00 | 2.5% | 极高 |
| ETHUSDT | 0.85 | 3.5% | 极高 |
| SOLUSDT | 0.75 | 5.0% | 高 |
| BNBUSDT | 0.70 | 3.0% | 高 |
| DOGEUSDT | 0.60 | 5.5% | 高 |

### 资金分配
- **$140起步**: 一次只开1个品种 (谁的z-score最强开谁)
- **$300+**: 最多同时持有2个品种
- **$800+**: 最多3个品种
- **$2000+**: 全部品种

### 实现
- NEW: `portfolio/small_cap_allocator.py` (资金分配器)
- MODIFY: `decision/backtest_module.py` (多品种协调)
- 现有 `runner/backtest_runner.py:run_multi_backtest()` 已支持多品种

---

## 第三层: 反脆弱回撤控制 (Anti-Fragile Drawdown)

当前问题: 一个-37%月份直接毁掉半年利润。

### 设计

```python
class DrawdownController:
    """分级回撤响应 — 回撤越深，仓位越小。"""

    def position_scale(self, current_dd: float) -> float:
        """
        DD < 5%:   scale = 1.0  (正常)
        DD 5-10%:  scale = 0.7  (缩仓30%)
        DD 10-15%: scale = 0.4  (缩仓60%)
        DD 15-20%: scale = 0.2  (最小仓位)
        DD > 20%:  scale = 0.0  (停止交易，等待恢复)
        """

    def on_recovery(self, recovered_pct: float) -> None:
        """回撤恢复50%后，逐步恢复正常仓位。"""
```

### 效果预估
原来 M6 的 -37% 回撤:
- 5% DD后缩到70% → 实际约 -22%
- 10% DD后缩到40% → 实际约 -15%
- 15% DD停止 → 最差 -15%

**$140 最差变 $119 (而不是 $88)，还能继续交易。**

### 实现
- NEW: `risk/drawdown_controller.py`
- MODIFY: `decision/backtest_module.py`
- MODIFY: `ext/rust/src/bin/main.rs` (已有max_drawdown_pct, 需要分级)

---

## 第四层: 提高交易频率 (Higher Frequency)

当前: 1h bars, ~1笔/天, 18个月489笔
问题: 太少的交易 → 月收益不稳定 → 复利效果差

### 方案: 15分钟K线
- 交易频率 ×4 → ~4笔/天, 18个月 ~2000笔
- 单笔盈亏更小 → 月度更稳定
- 成本问题: 8bps/笔 → 需要 IC > 0.01 在15m上

### 实现路径
1. 先验证15m的alpha: 用现有模型在15m数据上回测
2. 如果IC > 0.01: 训练专门的15m模型
3. 如果IC不够: 用1h信号但在15m级别执行 (更好的entry timing)

### 文件
- NEW: `scripts/train_15m_alpha.py`
- MODIFY: `features/batch_feature_engine.py` (支持15m特征)
- 现有Rust binary已支持任意interval

---

## 实施优先级

```
Phase 1 (1-2天): 阶梯风控 + 回撤控制
  → 防止$140爆仓，让系统"活下来"
  → 回测验证: $140能否安全度过最差月份

Phase 2 (2-3天): 多品种组合
  → 降低月波动，让复利更稳定
  → 训练 SOL/BNB/DOGE 模型，多品种回测

Phase 3 (3-5天): 15分钟alpha
  → 提高交易频率，平滑收益曲线
  → 需要下载15m数据 + 训练新模型

Phase 4 (后续): 高级优化
  → Kelly自适应仓位 (已有portfolio/allocator_kelly.py)
  → 信号强度 → 仓位大小映射 (已有dynamic_leverage)
  → 跨品种信号确认 (BTC趋势确认ETH方向)
```

---

## 预期效果

| 指标 | 当前 (单品种5%×2x) | Phase 1 | Phase 1+2 | Full |
|------|-------------------|---------|-----------|------|
| 月波动 | ±30-50% | ±15-25% | ±10-15% | ±5-10% |
| 最差月 | -37% | -15% | -10% | -7% |
| 年化收益 | ~11% | ~8% | ~15% | ~20% |
| $140→$280 | 不可能 | ~12个月 | ~8个月 | ~6个月 |
| 爆仓概率 | 高 | 低 | 极低 | 极低 |

注意: Phase 1 年化会降低 (更保守)，但**存活率大幅提升**。
Phase 2 加上多品种后，年化反而高于单品种 (分散化红利)。

---

## 最低可行方案 (如果只做一件事)

如果只能选一个改造，选 **阶梯风控 + 回撤控制** (Phase 1):

1. $140-$300: risk=50%, leverage=3x, DD>15%停止
2. $300-$800: risk=25%, leverage=3x, DD>15%停止
3. $800+: risk=12%, leverage=3x, DD>10%缩仓

这样:
- 最好情况: 5个月翻倍 ($140→$300)
- 最差情况: 亏到$119停止，等下个月恢复
- 不会爆仓，不会跌破$100开仓门槛
