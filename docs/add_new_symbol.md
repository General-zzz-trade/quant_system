## 添加新交易对

### 现状总结 (2026-03-27)

**已激活 (SYMBOL_CONFIG)**: BTCUSDT, ETHUSDT (1h + 4h)
**已有模型但未激活**: SOLUSDT, SUIUSDT (模型已训练, 通过质量门控)
**有数据但无模型**: DOGEUSDT, XRPUSDT, LINKUSDT, ADAUSDT, AVAXUSDT, DOTUSDT, BNBUSDT, NEARUSDT, AAVEUSDT
**15m 已禁用**: BTCUSDT_15m, ETHUSDT_15m (Sharpe 不达标)

### 候选交易对评估

| 交易对 | 1h bars | 补充数据 | 模型 | 模型 Sharpe | 备注 |
|---------|---------|----------|------|-------------|------|
| SOLUSDT | 48,262 | funding + OI | gate_v2 (v11) | 2.09 | **最佳候选**, 数据+模型齐全 |
| SUIUSDT | 25,188 | OI + LS + Liq | gate_v2 (v11) | 2.51 | 高 Sharpe 但仅 49 笔交易, 历史较短 |
| DOGEUSDT | 49,774 | 无 | 无 | - | 数据充足, 需下载补充数据+训练 |
| XRPUSDT | 54,239 | 无 | 无 | - | 数据最多, 需补充数据 |
| LINKUSDT | 53,975 | 无 | 无 | - | 数据充足 |
| BNBUSDT | 53,407 | 无 | 无 | - | 数据充足 |
| ADAUSDT | 53,639 | 无 | 无 | - | 数据充足 |
| AVAXUSDT | 47,976 | 无 | 无 | - | |
| DOTUSDT | 48,753 | 无 | 无 | - | |
| NEARUSDT | 47,456 | 无 | 无 | - | |
| AAVEUSDT | 39,001 | 无 | 无 | - | 数据较少 |

### 推荐优先级

1. **SOLUSDT** -- 模型已训练 (Sharpe 2.09, IC 0.049, 299 笔交易), 有 funding + OI 数据, 流动性好
2. **SUIUSDT** -- 模型已训练 (Sharpe 2.51), 但仅 49 笔交易 + 历史 2.8 年, 需观察
3. **XRPUSDT / DOGEUSDT** -- 数据量大 (>49k bars), 高流动性, 但需先下载补充数据再训练

### 步骤: 激活已有模型的交易对 (SOL/SUI)

```bash
# 1. 验证模型仍然有效 (检查 IC 衰减)
python3 -m monitoring.ic_decay_monitor --symbol SOLUSDT

# 2. 如需重新训练
python3 -m alpha.retrain.cli --symbol SOLUSDT --force

# 3. 添加到 SYMBOL_CONFIG (strategy/config.py)
# 参考现有 SOL 模型 config: dz=0.5, mh=12, maxh=96
```

在 `strategy/config.py` 的 SYMBOL_CONFIG 中添加:

```python
"SOLUSDT": {"size": 0.01, "model_dir": "SOLUSDT_gate_v2", "max_qty": 5000, "step": 0.01},
```

```bash
# 4. 启动时加入 --symbols
python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h SOLUSDT --ws

# 5. 重启服务
sudo systemctl restart bybit-alpha.service
```

### 步骤: 从零添加新交易对

```bash
# 1. 下载 OHLCV 数据
python3 -m data.downloads.data_refresh --symbol XXXUSDT

# 2. 下载补充数据 (funding, OI, ls_ratio)
#    检查 data/downloads/ 中对应的下载脚本
#    至少需要: funding rate + open interest

# 3. 验证数据量
python3 -c "import pandas as pd; df=pd.read_csv('data_files/XXXUSDT_1h.csv'); print(f'{len(df)} bars, {df.iloc[0][0]} -> {df.iloc[-1][0]}')"

# 4. 训练模型
python3 -m alpha.retrain.cli --symbol XXXUSDT --force

# 5. 检查模型质量
cat models_v8/XXXUSDT_gate_v2/config.json | python3 -m json.tool

# 6. 添加到 SYMBOL_CONFIG + 重启 (同上)
```

### 质量门控要求

- 至少 **5,000 bars** 历史数据 (约 7 个月 1h)
- Walk-forward **Sharpe > 1.0**
- 平均 **IC > 0.02**
- 至少 **15 笔交易** (防止过拟合)
- Bootstrap **p5 Sharpe > 0** (95% 置信区间正)
- Funding rate 和 OI 数据可用 (模型特征依赖)

### SYMBOL_CONFIG 参数说明

| 参数 | 说明 | 如何确定 |
|------|------|----------|
| `size` | 最小下单量 | Bybit 合约规格, 如 BTC=0.001, ETH=0.01, SOL=0.1 |
| `model_dir` | 模型目录名 | `models_v8/` 下的目录 |
| `max_qty` | 最大下单量 | Bybit 合约规格 |
| `step` | 下单精度 | Bybit 合约规格, `_round_to_step()` 依赖此值 |
| `symbol` | 实际交易对 | 仅多时间框架需要 (如 `BTCUSDT_4h` -> `BTCUSDT`) |
| `interval` | K线周期 | 默认 60 (1h), 可设 15/240 |
| `warmup` | 预热 bars 数 | 默认 800, 4h 用 300 |
| `use_composite_regime` | 复合 regime | 仅 BTC 启用 |

### 注意事项

- 新交易对需要确认 Bybit 上的合约规格 (size/step/max_qty)
- Direction alignment: ETH 目前跟随 BTC 方向, 新币种需决定是否也跟随
- CrossAssetComputer: 新交易对自动使用 BTCUSDT 作为 benchmark
- 每增加一个交易对, WS 连接数 +1, 注意 Bybit WS 连接数限制
- 多时间框架 (4h) 可在 1h 验证通过后再添加
- 模型签名: 生产环境需要 HMAC-SHA256 签名, 训练后自动生成
