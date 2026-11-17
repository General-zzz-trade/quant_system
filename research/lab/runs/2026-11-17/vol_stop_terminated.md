# 按波动率收紧止损（vol_stop.md）：终止（范围改为只做 BTC 和 ETH）

- 2026-11-17 约 07:00 UTC，技术负责人把范围改回只做 BTCUSDT 和 ETHUSDT，并选择"现在停掉"这一轮全市场的回测。
- 停的时候 1274 个新配置回放了约 688 个。`vol_screen.json` 没有生成，`vol_stop.md` 里的比较没有做，**没有任何结果被看过**。
- 已回放的配置登记在 `research/lab/trials.jsonl`（备注 "2026-11-17 run on the 2020+ snapshot"），以后照样算进这份快照的试验数。
- 按波动率收紧止损的代码（`lab.um_lab.vol_stop_pct`、`StopVariant(vol_k=...)`）已合并，可以在 BTC/ETH 上重新登记后再测。
- 同目录的 `alpha.json` 是 2026-11-17 给技术负责人看过的分析：今晚 4 个幸存者和 A6 的每日收益对 BTC 的回归（alpha、beta、t 值），只作记录。
