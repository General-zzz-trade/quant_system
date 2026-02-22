# 项目开发路线图

> 创建日期: 2026-02-22
> 基于: PROJECT_EVALUATION.md + architecture_assessment.md + trading_capabilities_analysis.md

---

## 总体策略：四阶段渐进式开发

```
阶段一 (1-2周)        阶段二 (2-3周)         阶段三 (3-4周)         阶段四 (2-3周)
┌──────────────┐    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  止血加固     │ →  │  核心补全      │  →  │  策略扩展      │  →  │  生产上线      │
│              │    │               │     │               │     │               │
│ · 安全修复   │    │ · 测试覆盖    │     │ · 更多策略     │     │ · 监控告警     │
│ · 并发修复   │    │ · 空壳清理    │     │ · 更多交易所   │     │ · 影子交易     │
│ · 内存泄漏   │    │ · 长函数拆分  │     │ · ML 信号      │     │ · 渐进上线     │
│ · 错误处理   │    │ · 端到端测试  │     │ · 回测验证     │     │ · 运维手册     │
└──────────────┘    └───────────────┘     └───────────────┘     └───────────────┘
```

**核心原则**: 先修后建，先测后扩，先纸后实。

---

## 阶段一：止血加固（1-2 周）

> 目标: 修复所有可能导致资金损失或系统崩溃的问题，不新增功能。

### 1.1 [P0] 安全修复（第 1 天）

- [ ] **轮换 API 密钥**
  - 立即在 Binance 后台废弃泄露的密钥
  - 修复 `execution/live/run_binance_um_user_stream.py:29-30`
  ```python
  # 修改前 (错误)
  api_key = os.getenv("w0i6Nm1hhV0Jqy3E49VftG0f74A3B9pPBdTGX2Dyqui8Sdf64pUnK7zGhHf9fc3P")
  # 修改后 (正确)
  api_key = os.getenv("BINANCE_API_KEY")
  api_secret = os.getenv("BINANCE_API_SECRET")
  ```
  - 添加 `.env.example` 文件说明需要的环境变量
  - 确保 `.gitignore` 包含 `.env`

- [ ] **清理 Git 历史中的密钥**
  - 使用 `git filter-repo` 或 `BFG Repo-Cleaner` 清除历史记录中的密钥字符串

### 1.2 [P0] 并发安全修复（第 1-2 天）

- [ ] **修复 RiskAggregator 竞态条件** (`risk/aggregator.py`)
  ```python
  # 将统计更新放入已有的 self._lock 保护范围内
  with self._lock:
      st.calls += 1
      # ... rule evaluation ...
      st.allow += 1
  ```

- [ ] **修复 TokenBucket 线程安全** (`execution/bridge/execution_bridge.py:106-128`)
  - 给 `allow()` 方法加 `threading.Lock()`

- [ ] **修复 EventBus 订阅竞态** (`event/bus.py`)
  - 在 `subscribe()`/`publish()` 中加锁，或在 `publish()` 时复制列表再迭代

### 1.3 [P0] 内存泄漏修复（第 2-3 天）

- [ ] **事件去重集合加 TTL/容量上限** (`engine/dispatcher.py`)
  ```python
  # 方案: 带 TTL 的 dict 替代无限增长的 set
  class TTLDedup:
      def __init__(self, ttl_sec: int = 86400, max_size: int = 1_000_000):
          self._seen: dict[str, float] = {}
          self._ttl = ttl_sec
          self._max = max_size

      def is_new(self, event_id: str) -> bool:
          now = time.monotonic()
          if event_id in self._seen:
              return False
          # 定期清理过期条目
          if len(self._seen) >= self._max:
              cutoff = now - self._ttl
              self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
          self._seen[event_id] = now
          return True
  ```

- [ ] **EventBus 添加 unsubscribe 机制**
  - 实现 `unsubscribe(handler, event_type)` 方法
  - 返回的订阅对象支持 `dispose()` 模式

### 1.4 [P0] 错误处理加固（第 3-5 天）

- [ ] **修复静默异常** (`engine/pipeline.py:172-178`)
  ```python
  def _to_float(x: Any, *, default: float = 0.0) -> float:
      if x is None:
          return default
      try:
          return float(x)
      except Exception:
          logger.warning("Float conversion failed for %r, using default %s", x, default)
          return default
  ```

- [ ] **修复决策引擎 None 崩溃** (`decision/engine.py:119`)
  ```python
  close = getattr(snapshot.market, "close", None)
  last_price = getattr(snapshot.market, "last_price", None)
  price = close or last_price
  if price is None:
      raise ValueError("No price available in market snapshot")
  price_hint = Decimal(str(price))
  ```

- [ ] **REST 客户端参数校验** (`execution/adapters/binance/rest.py`)
  - 下单前验证: qty > 0, price > 0 (限价单), symbol 非空
  - 捕获 `json.JSONDecodeError`

- [ ] **修复 Guard 语义矛盾** (`engine/loop.py:218`)
  - ALLOW 动作应继续处理而非 DROP

### 1.5 阶段一验收标准

- [ ] 所有已知的 CRITICAL 和 HIGH 问题修复并通过 code review
- [ ] API 密钥已轮换，Git 历史已清理
- [ ] 系统可以运行 24 小时不崩溃（空跑，不下单）
- [ ] 已有的 5 个测试全部通过

---

## 阶段二：核心补全（2-3 周）

> 目标: 让核心模块达到 70%+ 测试覆盖率，清理技术债务。

### 2.1 架构桩文件与接口补全（第 1-2 天）

> 注意: 空壳文件是项目架构规划的一部分，保留不删除。

- [ ] **为核心桩文件补充接口定义**
  - 给空壳文件添加明确的类/函数签名和 docstring
  - 标记 `raise NotImplementedError("TODO")` 而非空 `pass`
  - 让每个桩文件清楚表达"这里将来要做什么"

- [ ] **补齐主要包的 `__init__.py` 导出**
  - context, engine, event, execution 四个核心包缺少 `__all__` 定义

### 2.2 核心模块测试（第 3-10 天）

按优先级补测试，目标覆盖率 70%+：

- [ ] **风控模块测试** (P0 — 资金安全)
  - `test_leverage_cap_rule.py` — 杠杆超限 → REJECT/REDUCE
  - `test_max_drawdown_rule.py` — 回撤触发 → KILL + 减仓放行
  - `test_max_position_rule.py` — 仓位超限 → 自动缩减
  - `test_kill_switch.py` — 多层级熔断 + TTL 过期 + 减仓模式
  - `test_risk_aggregator.py` — 多规则聚合 + 优先级合并 + 并发安全

- [ ] **状态管理测试** (P0 — 数据正确性)
  - `test_market_reducer.py` — OHLCV 更新
  - `test_account_reducer.py` — 余额/保证金计算
  - `test_position_reducer.py` — 开仓/加仓/减仓/平仓
  - `test_state_pipeline.py` — 端到端 reducer 链

- [ ] **引擎模块测试** (P1 — 核心编排)
  - `test_coordinator.py` — 事件驱动编排
  - `test_pipeline.py` — 状态管道正确性
  - `test_loop.py` — 事件循环 + 守卫决策
  - `test_guards.py` — ALLOW/DROP/RETRY/STOP 语义

- [ ] **决策引擎测试** (P1 — 交易逻辑)
  - `test_signal_models.py` — 各信号模型输出正确性
  - `test_allocators.py` — 等权/单资产分配
  - `test_sizer.py` — 固定比例仓位计算
  - `test_decision_engine.py` — 完整决策链路

- [ ] **执行层测试** (P1 — 订单安全)
  - `test_order_mapper.py` — 内部格式 ↔ Binance 格式
  - `test_fill_mapper.py` — 成交映射
  - `test_execution_bridge.py` — 幂等性 + 限速 + 熔断器

### 2.3 长函数重构（第 8-12 天）

- [ ] **拆分 `context/constraints.py`** (634 行 → 多个 <100 行函数)
  - `validate_tick_size()` — 价格精度校验
  - `validate_step_size()` — 数量精度校验
  - `validate_notional()` — 名义价值校验
  - `validate_position_limit()` — 仓位限制校验

- [ ] **拆分 `execution/bridge/execution_bridge.py._send()`** (193 行)
  - `_check_rate_limit()` — 限速检查
  - `_check_circuit_breaker()` — 熔断检查
  - `_check_idempotency()` — 幂等去重
  - `_attempt_send()` — 单次发送尝试
  - `_handle_retry()` — 重试逻辑

- [ ] **拆分 `runner/backtest_runner.py`** (1,193 行)
  - `_load_csv_data()` — 数据加载
  - `_build_trades_from_fills()` — 交易重建
  - `_build_summary()` — 统计计算

### 2.4 Decimal/float 统一（第 10-12 天）

- [ ] **确定策略**: 内部计算一律使用 `Decimal`，仅在 I/O 边界转换
- [ ] **修复 `runner/backtest_runner.py`** — 消除 `float(starting_balance)` 等隐式转换
- [ ] **修复 `engine/pipeline.py._to_float()`** — 改为 `_to_decimal()` 或保留但加日志

### 2.5 端到端集成测试（第 11-14 天）

- [ ] **完整交易链路测试**
  ```
  MarketEvent → Signal → Decision → Intent → RiskCheck → Order → Fill → StateUpdate
  ```
  - 正常流程: 信号触发 → 下单 → 成交 → 仓位更新
  - 风控拦截: 信号触发 → 下单 → 风控拒绝 → 无仓位变化
  - 熔断触发: 回撤超限 → Kill Switch 激活 → 仅允许减仓
  - 幂等性: 重复事件 → 状态不变

### 2.6 阶段二验收标准

- [ ] 核心模块 (risk, state, engine, decision) 测试覆盖率 ≥ 70%
- [ ] 空壳文件清理完成，项目文件数减少 40%+
- [ ] 无函数超过 100 行
- [ ] 端到端测试通过
- [ ] `pytest` 全部通过，零 warning

---

## 阶段三：策略扩展（3-4 周）

> 目标: 扩展交易策略，完善回测，为实盘做准备。

### 3.1 策略开发（第 1-10 天）

- [ ] **布林带策略 (Bollinger Band)**
  - 突破上轨做空，突破下轨做多
  - 参数: window=20, std_dev=2.0

- [ ] **RSI 超买超卖策略**
  - RSI > 70 做空，RSI < 30 做多
  - 结合 ATR 过滤低波动期

- [ ] **动量策略 (Momentum)**
  - N 期收益率排名，做多强势标的
  - 支持多标的横截面排序

- [ ] **网格交易策略**
  - 固定间隔挂单，震荡行情适用
  - 新增 GridSignal 信号模型

- [ ] **MACD 策略**
  - 新增 MACD 技术指标到 features/technical.py
  - MACD 金叉/死叉 + 柱状图背离

### 3.2 ML 信号集成（第 8-14 天）

- [ ] **完善 `decision/signals/ml/model_runner.py`**
  - 定义特征输入格式 (DataFrame or dict)
  - 定义模型输出格式 (SignalResult)
  - 支持 sklearn/xgboost/lightgbm 模型加载

- [ ] **实现简单 ML 策略**
  - 特征: SMA, EMA, RSI, ATR, 波动率, 收益率
  - 模型: XGBoost 分类器 (多/空/平)
  - 训练: 历史数据滚动训练
  - 回测: 与规则策略对比

### 3.3 回测增强（第 10-18 天）

- [ ] **多标的回测**
  - 当前只支持单标的，扩展为多标的同时回测
  - 组合层面的统计指标

- [ ] **交易成本模型**
  - 手续费: maker/taker 费率
  - 滑点: 基于 ATR 的滑点模型
  - 资金费率: 永续合约 funding rate

- [ ] **回测报告增强**
  - 添加: 年化收益、Calmar 比率、胜率、盈亏比
  - 添加: 月度收益热力图
  - 添加: 回撤曲线
  - 输出: JSON + CSV 格式

- [ ] **Walk-forward 验证**
  - 滚动窗口: 训练期 + 验证期
  - 防止过拟合

### 3.4 多交易所支持（第 15-21 天）

- [ ] **OKX 适配器**
  - REST API: 下单/撤单/查询
  - WebSocket: 用户流
  - 订单/成交映射

- [ ] **Bybit 适配器** (可选)
  - 类似 OKX 的实现
  - 复用现有的 Adapter 接口

### 3.5 SimVenue 完善（第 18-24 天）

- [ ] **延迟模拟**
  - 可配置的下单延迟 (ms)
  - 网络抖动模拟

- [ ] **滑点模型**
  - 基于订单大小的线性滑点
  - 基于波动率的动态滑点

- [ ] **部分成交模拟**
  - 基于可配置的成交概率
  - 限价单的被动成交逻辑

### 3.6 阶段三验收标准

- [ ] 至少 5 个策略可独立运行回测
- [ ] 集成策略的回测报告包含完整统计
- [ ] ML 策略框架可加载外部模型
- [ ] SimVenue 支持延迟和滑点模拟
- [ ] 至少 2 个交易所适配器可用

---

## 阶段四：生产上线（2-3 周）

> 目标: 安全、可观测、渐进式上线。

### 4.1 监控与告警（第 1-5 天）

- [ ] **完善 monitoring 模块**
  - 指标采集: 持仓、P&L、风控状态、延迟
  - 告警规则: 回撤告警、连接断开、异常订单
  - 输出: 结构化日志 (JSON)

- [ ] **健康检查**
  - WebSocket 连接状态
  - 行情数据新鲜度 (stale data detection)
  - 资金余额阈值

- [ ] **结构化日志**
  - 统一日志格式: timestamp, level, module, trace_id, message, data
  - 关键操作全链路 trace_id 传递

### 4.2 密钥与配置管理（第 3-5 天）

- [ ] **配置外部化**
  - 所有可调参数从代码中抽取到配置文件
  - 支持 YAML/JSON 配置
  - 环境变量覆盖机制

- [ ] **密钥管理**
  - 生产环境: 密钥通过环境变量注入
  - 密钥不落盘、不入日志
  - 支持密钥轮换（不停机）

### 4.3 影子交易（第 5-10 天）

- [ ] **Paper Trading 模式**
  - 连接真实行情，但不发送真实订单
  - 用 SimVenue 模拟成交
  - 记录"如果真实交易会怎样"

- [ ] **对比验证**
  - 影子交易结果 vs 回测结果
  - 检查信号一致性、延迟影响
  - 至少运行 1 周

### 4.4 渐进式上线（第 10-15 天）

- [ ] **阶段 A: 最小仓位实盘**
  - 单标的 (BTCUSDT)
  - 最小仓位 (0.001 BTC)
  - 单策略 (MA Cross)
  - 观察 3 天

- [ ] **阶段 B: 扩大规模**
  - 多标的 (BTC + ETH)
  - 正常仓位比例
  - 多策略 (集成信号)
  - 观察 1 周

- [ ] **阶段 C: 全量运行**
  - 全部目标标的
  - 全部策略
  - 完整风控规则
  - 持续监控

### 4.5 运维文档（第 12-15 天）

- [ ] **部署文档**
  - 环境准备步骤
  - 配置参数说明
  - 启动/停止命令

- [ ] **应急手册**
  - 手动触发 Kill Switch
  - API 连接中断处理
  - 仓位手动平仓流程

- [ ] **日常运维 Checklist**
  - 每日检查项
  - 周度检查项
  - 月度检查项

### 4.6 阶段四验收标准

- [ ] 影子交易运行 1 周无异常
- [ ] 监控告警覆盖核心指标
- [ ] 应急手册编写完成
- [ ] 最小仓位实盘运行 3 天无异常
- [ ] 运维文档齐全

---

## 里程碑时间线

```
Week 1-2:   ████ 阶段一: 止血加固
Week 3-5:   ████████ 阶段二: 核心补全
Week 6-9:   ████████████ 阶段三: 策略扩展
Week 10-12: ████████ 阶段四: 生产上线
```

**总预计时间: 10-12 周**

---

## 开发优先级决策树

遇到不确定该做什么时，按此顺序判断：

```
1. 会导致资金损失吗？     → 是 → 立即修复 (P0)
2. 会导致系统崩溃吗？     → 是 → 本周修复 (P0)
3. 影响交易正确性吗？     → 是 → 下周修复 (P1)
4. 影响开发效率吗？       → 是 → 安排到阶段二 (P1)
5. 是新功能/新策略吗？    → 是 → 安排到阶段三 (P2)
6. 是优化/锦上添花吗？    → 是 → 安排到阶段四或之后 (P3)
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 修复并发问题引入新 bug | 高 | 每个修复配套测试，修改前后对比 |
| 删除空壳文件破坏导入 | 中 | 删除前检查所有 import 引用 |
| 新策略过拟合历史数据 | 高 | Walk-forward 验证 + 样本外测试 |
| 实盘环境与回测差异大 | 高 | 影子交易 ≥ 1 周，对比回测结果 |
| 交易所 API 变更 | 中 | 适配器层隔离，定期检查 API 文档 |
| 单人开发精力有限 | 高 | 严格按优先级推进，不跳阶段 |
