# 规格草稿：E3_vol_target_trend（2026-11-17）

**未验证。不构成实盘批准。** `passed` 是 false；签名发布和标记通过只由技术负责人做。

- 来源：`results.json` 的最终选择和 `universe.json` 最后一份清单。
- 上实盘之前：前向跟踪 4 周，每周一次登记的查看，4 次都还成立才进入小资金试运行。

## 回测假设的实盘设置

- `STOP_PCT=0.08`
- 交易所杠杆最多 6 倍（止损要在逐仓强平距离的一半以内）
- `BINANCE_MARGIN_TYPE=ISOLATED`
- `STRATEGY_CAPITAL_USDT=300`
- `STRATEGY_LEVERAGE=1`

## 实盘程序还不支持的（要先改程序）

- signal reset after a stop: the runtime pauses entries for a fixed time
