# ADR-0002：分层方式与强制手段

- 状态：**已批准**（2026-11-14）：import-linter 为主、AST 测试为辅；组装根在 `infrastructure`。实施中的调整：`application` 允许使用 `logging`（见 `target.md` 2.1）。
- 日期：2026-11-14

## 背景

仓库里现在没有任何架构规则检查。结果是：12 个顶层包互相成环；库代码 import 入口脚本；领域规则里直接开 SQLite、取系统时钟、读环境变量（`docs/arch/current-state.md` 2.2 的 V1–V10）。

要求是四层（domain / application / adapters / infrastructure），依赖只能向内。需要决定两件事：规则靠什么强制；组装根和环境变量读取放在哪一层。

## 决策一：靠什么强制

### A. import-linter（建议）

- 在 `.importlinter` 里声明契约，CI 里运行 `lint-imports`。
- 优点：声明式，契约本身就是文档；支持分层、禁止依赖、互相独立三类契约，正好覆盖需要的规则；失败信息直接给出违规的 import 链。
- 缺点：多一个开发依赖；查不了"某个函数是否被调用"这类规则（例如 `os.environ` 的使用）。

### B. 自己写 AST 测试

- 在 `tests/architecture/` 里用 `ast` 模块写规则。本次诊断用的就是这类脚本。
- 优点：零新依赖；什么规则都能写。
- 缺点：要自己维护解析相对 import、包 `__init__` 传递依赖等细节；规则散在测试代码里，不如契约文件一目了然。

### C. 只靠约定和代码评审

- 缺点：现状就是这种方式的结果。

### 建议

A 为主，B 补 A 查不了的两条规则（环境变量只在 `infrastructure` 读；`domain` 不 import `logging` 和 `os`）。import-linter 只装在开发与 CI 环境，不进运行时镜像。

## 决策二：组装根和配置读取放在哪

### A. 放在 `infrastructure`，它是最外层（建议）

- `infrastructure` 负责读环境变量、校验、构造适配器、把它们注入应用层、提供命令行入口。
- 四层的顺序就是 `infrastructure → adapters → application → domain`，一条分层契约即可表达。
- 适配器需要的共用技术代码（例如 SQLite 连接的开关方式）放在各自的适配器子包里，不放 `infrastructure`。

### B. 再加第五层 `entrypoints`

- `infrastructure` 只放技术基础件（时钟、日志、连接工厂），允许适配器依赖它；组装根单独放最外层。
- 优点：适配器可以共用基础件。
- 缺点：多一层；与要求的四层命名不一致；"适配器依赖 infrastructure"容易变成新的杂物间。

### 建议

A。

## 规则清单

1. `trading` 内依赖只能向内。
2. `trading` 不 import 任何遗留顶层包。
3. `domain` 和 `application` 不 import `sqlite3`、`urllib`、`http`、`socket`、`subprocess`、`time`、`_quant_hotpath`。
4. `adapters` 的子包之间互不 import。
5. 只有 `infrastructure` 读环境变量。配置对象是不可变的，校验在加载时完成，硬上限（例如敞口 3 倍）在 `domain` 常量里，配置只能调低。
6. 应用层的协作者参数必须用端口类型，不允许 `Any`。

契约草案见 `docs/arch/target.md` 2.3。

## 后果

- CI 增加一个 `lint-imports` 步骤和 `tests/architecture/`。
- 规则只约束 `trading/`。遗留包不受约束，也不会因为这些规则而需要修改。
- 契约的具体写法要在迁移计划第 2 步实际运行后定稿：我没有在这个仓库里运行过 import-linter。

## 需要您确认

是否接受 import-linter 作为开发依赖。
