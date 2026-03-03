# 特征优化：提升稳定 + 扩大候选池

## 完成状态: DONE

### Step 1: WF 验证新配置 — DONE
```
10 fixed: basis, ret_24, fgi_normalized, fgi_extreme, parkinson_vol,
          atr_norm_14, rsi_14, tf4h_atr_norm_14, basis_zscore_24, cvd_20
6 candidate: funding_zscore_24, basis_momentum, vol_ma_ratio_5_20,
             mean_reversion_20, funding_sign_persist, hour_sin
4 flexible slots
```

结果对比:
| 指标 | 旧配置 (8+4) | 新配置 (10+4) |
|------|-------------|---------------|
| Positive Sharpe | 15/21 | **17/21** |
| Avg Sharpe | 2.81 | **4.12** |
| Total Return | +117% | **+119%** |

弱点期改善: fold 12 和 14 翻正

Feature stability (21/21): 13个特征100%稳定
- 10 fixed + funding_sign_persist + funding_zscore_24 + basis_momentum
- 第4个flex slot在 mean_reversion_20 (10/21) 和 vol_ma_ratio_5_20 (11/21) 间交替

### Step 2: 重训练生产模型 — DONE
- train_v8_production.py 配置已更新 (N_FLEXIBLE=4, 新CANDIDATE_POOL)
- OOS 18个月: Sharpe=3.79, Return=+63%, IC=0.0665
- Bootstrap P(S>0)=99.1%, 12/18正月
- 4/4 production gates PASS
- 模型注册: alpha_v8_BTCUSDT v2 (promoted)

### Step 3: 回测验证 — DONE
- Sharpe: 3.82, Return: +35.60%, Annual: +22.64%
- Max DD: -12.09%, Profit factor: 1.11
- H1 Sharpe: 4.04, H2 Sharpe: 3.48 (无衰减)
- 11/18 positive months
- Monthly gate: active 26.3% → 14.3%

### Step 4: 更新 final_results.json — DONE
- BTCUSDT features 更新为14特征 (10 fixed + 4 greedy selected)
- 添加 walkforward / oos / monthly_gate 配置

## 下一步
- Paper trading 阶段
