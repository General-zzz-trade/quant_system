# 校准试运行用的规格副本（2026-11-16）

这是 `research/results/overlay_2026-11-13/final_spec.json` 的逐字节副本，`passed` 仍是 `false`。

为什么要副本：原文件的 SHA-256 登记在 `research/forward_looks.jsonl` 里，2027-02-13 和 2027-05-13 的两次前向查看会核对它，
所以原文件不能改。校准试运行（技术负责人 2026-11-16 晚的决定）用这份副本。

**这个策略没有被验证过的优势。** 它在 2026-11-13 只通过了开发窗口的筛选。试运行的目的是拿真实成交来核对回测和检验程序本身，
不期望盈利。原规格里"主网：拒绝，需要一次被人记录为通过的前向模拟"这句话仍然是研究上的结论；技术负责人决定为校准而运行它。

## 只能由技术负责人本人做的两步

代码和 Claude 都不改 `passed`。

1. 把这份副本里的 `"passed": false` 改成 `"passed": true`（只改这一处）。
2. 在实盘用的那份代码里，用自己的签名密钥发布：

   ```bash
   cd /quant_system
   sudo -u ubuntu env QUANT_MODEL_SIGN_KEY='你的签名密钥' python3 -m scripts.publish_strategy_spec \
       --study research/results/overlay_pilot_2026-11-16 --out models_binance/strategy
   ```

   同一个密钥要写在服务的环境变量文件里（`QUANT_MODEL_SIGN_KEY`），程序启动时用它核对签名。

## 150 USDT 下它会怎么交易

- 两个品种各分到约 75 USDT，仓位权重最高 0.9。
- BTC 一手约 71 USDT（0.001 BTC），最多只能买得起一手的边缘，基本不会开仓。
- ETH 的最小下单额是 20 USDT，所以只有目标仓位变化超过约 0.27（20 ÷ 75）时才会下单。成交会很少，一个月可能只有几笔。
- 手续费、滑点和资金费已经用账户的真实成交核对过（`research/lab/runs/2026-11-16/real_fills.md`）。这次试运行主要检验程序自己：
  挂止损、对账、告警、亏损停止线。
