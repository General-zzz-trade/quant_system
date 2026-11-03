# BTCUSDT 2025 历史研究样本

来源：Binance 公开 USDⓈ-M 永续合约市场数据接口，经只读 Binance 连接器获取。
取得日期：2026-11-03（北京时间）。没有访问账户或调用交易接口。

- K 线：`GET https://fapi.binance.com/fapi/v1/klines`，symbol=BTCUSDT，interval=1h，按 1000 行分页。
  区间为 2025-01-01 00:00:00 UTC 至 2026-01-01 00:00:00 UTC（不含右端），共 8760 根。
  仅保存开盘时间、OHLC 与 BTC 基础币成交量；成交量不是 USDT 成交额。
- 资金费：`GET https://fapi.binance.com/fapi/v1/fundingRate`，同品种同区间，1000＋95 行分页。
  共 1095 个结算事件，保留实际返回的 markPrice 与 fundingRate。
- 原始资金费时间相对 UTC 8 小时边界存在 0—16 毫秒偏移，保存于 `observed_funding_time`。
  `funding_time` 对齐到该次名义结算边界；模拟约定先结算边界前持仓的资金费，再处理下一开盘成交。
  此小时级约定不能精确复现结算前后毫秒级成交，导入器拒绝超过 1 秒的偏移。
- K 线标准化内容 SHA-256：`968aa803ab7d7983ebea0021422349799b2ab969ce974e9044877161a9020de6`。
  运行报告另记录文件数据指纹、资金费指纹、模型来源和源码指纹。

运行：

```bash
python -m runner.offline study --bars research/datasets/btcusdt_2025/bars.csv --funding research/datasets/btcusdt_2025/funding.csv --out .cache/btc-2025-study
```

这是公开历史样本，不证明此前研究者从未查看过这一时期。
单次流程训练与测试严格按时间隔离；原有策略参数和历史模型仍可能包含此前研究产生的选择偏差。
