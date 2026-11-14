# ADR-0003：端口的粒度与定义方式

- 状态：提议，待批准
- 日期：2026-11-14

## 背景

资金路径上的协作者没有接口定义：

- `CompositeRiskGate` 的 7 个构造参数全是 `Any`（`shared/entry_risk.py:16-18`）。
- `ConfirmedExecutor` 的 `adapter` 是 `Any`（`shared/confirmed_execution.py:53`）。
- `BinanceAdapter` 一个类承担行情、账户快照、下单、止损四种职责（`execution/adapters/binance/adapter.py`，490 行）。
- 测试用的假交易所（`tests/unit/binance/fake_venue.py`）与真实适配器是否一致，只能靠人工比对。

组件传 `None` 时入场门静默跳过对应检查（`shared/entry_risk.py:42,46,50,52`）。

## 决策一：粒度

### A. 一个大的 `Exchange` 端口

- 把行情、账户、下单、止损约 20 个方法放进一个接口。
- 优点：接线简单，一个对象传到底。
- 缺点：每个假实现都要实现全部方法；研究用的模拟交易所没有条件单，只能留空方法；入场门只需要读权益和持仓，却能拿到下单方法。

### B. 按能力拆成小端口（建议）

- `MarketData`、`ExchangeAccount`、`OrderGateway`、`StopOrderGateway` 四个，再加存储、时钟、通知等（清单见 `docs/arch/target.md` 第 3 节）。
- 优点：每个用例只拿到它需要的能力，入场门拿不到下单方法；假实现可以只实现用到的端口；模拟交易所不必假装支持条件单。
- 缺点：组装根里要传的对象变多。一个适配器类可以同时实现多个端口来缓解。

### C. 保持鸭子类型，只补测试

- 缺点：契约仍然只存在于调用点，改适配器时没有任何静态检查。

### 建议

B。

## 决策二：用 `Protocol` 还是抽象基类

### A. `typing.Protocol`（建议）

- 结构化类型：现有的 `BinanceAdapter` 和测试里的 `FakeBinance` 不用改继承关系就能被标注为端口的实现。
- 这对逐步替换有利：可以先把端口写出来，给现有类加类型标注，再迁移实现。
- 缺点：不符合接口的实现要等类型检查或契约测试才暴露，不会在实例化时报错。

### B. `abc.ABC`

- 优点：缺方法的子类实例化时立即报错。
- 缺点：现有实现和测试替身都要改继承；与逐步替换的顺序相冲突。

### 建议

A，并为每个端口写一套"端口契约测试"：同一组测试分别跑在真实适配器（对假 HTTP）、假交易所和模拟交易所上，保证三者行为一致。仓库里已有 `mypy` 配置，可以对 `trading/` 开启 `disallow_untyped_defs`。

## 约束

- 端口的方法集合取自现有调用点，不预先设计用不到的方法。
- 执行模式下必需的组件在构造时检查，缺失抛错，不再静默跳过。
- 端口方法出错时抛异常，不返回"空"。现有 Bybit 适配器把查询失败返回成空元组，与"无持仓"不可区分；Binance 适配器的 `get_balances` 也是这样（`execution/adapters/binance/adapter.py:216-218`，出错时返回空快照）。新端口不保留这种语义。

## 后果

- 新增 `trading/application/ports.py`。
- 现有的 `FakeBinance` 成为端口契约测试的被测对象之一，它与真实适配器的差异会被暴露出来。已知的一处：它的成交查询忽略 `limit` 参数（`tests/unit/binance/fake_venue.py:199-201`），所以测不到成交截断。

## 需要您确认

无。这是纯结构决策，不改变行为。
