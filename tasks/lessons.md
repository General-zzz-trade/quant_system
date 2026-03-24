# Lessons Learned

> 状态: 研究与迭代过程中的经验记录 (仍然有效，持续更新)
> 更新时间: 2026-03-24
> 该文档保留时间上下文，不作为当前运行时或生产契约真相源；当前系统现状请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md)

## Alpha & Signal Quality

### OOS衰减是最大风险
- **发现**: IS回测(全样本) BTC Sharpe=1.88, SOL Sharpe=2.81, 但OOS(样本外) BTC Sharpe=-2.05, SOL Sharpe=0.54
- **根因**: LightGBM在IS上过拟合，特征importance高但OOS泛化差
- **规则**: 任何策略改进必须在OOS上验证，IS提升不代表真实alpha
- **行动**: 永远先跑OOS再下结论；考虑walk-forward交叉验证替代固定split

### 不对称阈值有效但治标不治本
- **发现**: threshold_short=0.01确实减少低质量做空（SOL交易-42%），但OOS做空依然亏损
- **规则**: 做空信号质量差不是阈值问题，是模型对下跌预测能力本身不足
- **行动**: 考虑long-only策略或做空用独立模型

### 止损在IS上效果不明显
- **发现**: atr_stop=2.0单独使用对BTC/ETH几乎无影响，SOL反而略差
- **根因**: 止损在趋势型alpha中可能截断正确方向的波动
- **规则**: 止损参数需要和信号质量匹配，信号弱时止损重要，信号强时止损可能有害

### 多时间框架重采样的前瞻泄露
- **发现**: 4h特征 forward-fill 到 1h 行时，当前 4h group 内早期 1h 行拿到了用 group 最后 close 计算的特征
- **影响**: IC 从真实 0.01~0.02 虚高到 0.35~0.40（95%+ 是泄露）
- **修复**: 每根 1h bar 只使用**上一个已完成** 4h bar 的特征 (`group_key - 1`)
- **规则**: 任何时间框架重采样，必须用 lag-1 映射（当前 bar 只能看到已完结的更高频 bar）
- **验证**: 用极端跳价数据构造测试，确认泄露前后差异

### Ensemble不能弥补弱信号
- **发现**: 5个基础模型 OOS IC 都在 [-0.03, +0.03]，ensemble 只达到 +0.016~0.018
- **根因**: Ridge meta-learner 只是加权平均，不能创造新信号
- **规则**: 信号多样性不够时，堆模型复杂度无意义；应从数据源/特征层突破
- **行动**: ETH P(Sharpe>0)=85% 是唯一有潜力的方向

## Feature Engineering

### C++ RollingWindow没有_data属性
- 不能直接访问raw data，用deque代替
- `_USING_CPP` flag模式在Python/C++间切换

### CrossAssetComputer on_bar顺序敏感
- benchmark必须先推、altcoin后推
- pair state在altcoin push时更新
- 违反此顺序会导致静默错误（NaN特征）

### feature_hook.py的inspect.signature模式
- 用参数签名检测on_bar接口版本，自动适配不同computer
- 比isinstance检查更灵活但也更脆弱

## Backtest Infrastructure

### 回测/实盘一致性是关键
- **发现**: RegimeGate只在live生效，backtest绕过 → 回测高估收益
- **规则**: 所有影响信号的组件必须同时在backtest和live中启用
- **行动**: 已在backtest_runner加入enable_regime_gate参数

### Embargo机制防止前视偏差
- backtest中用EmbargoExecutionAdapter延迟1个bar执行
- 没有embargo会导致回测结果显著优于实盘

### 费用模型必须真实
- fee_bps=4 + slippage_bps=2 匹配Binance taker费率
- SOL高频交易(1.25笔/天)时fee占比高达39%
- min_hold_bars=3降低交易频率，费用从97k降到50k

## Architecture & Code

### Decimal vs float选择
- 价格和余额用Decimal保证精度
- 特征计算和ML推理用float（性能优先）
- 边界处转换：Decimal(str(float_val))

### 向后兼容的参数升级模式
- 所有新参数默认=当前行为（0或None）
- 测试第一个用例永远是"默认参数=旧行为"
- CLI flags用default=0.0而非required=True

## Workflow

### 分批提交优于大提交
- 按功能分3批：alpha研究 → 生产管道 → 策略升级
- 每批有清晰的变更主题和commit message
- data_files/models/output/ 应在gitignore中（大文件不入仓库）

### gitignore要精确
- `models/` 会匹配 `alpha/models/` → 用 `/models/` 只匹配顶层
- 同理 `output/` vs `outputs/`（项目中两个都存在）
