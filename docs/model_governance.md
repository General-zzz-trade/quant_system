# Model Governance

> 更新时间: 2026-03-19
> 目标: 固定 registry / loader / shadow compare / promotion / rollback 的职责边界，并明确哪些能力已经接到运行时、哪些还只是候选路径
> 适用范围: 当前 registry / loader / scripts shadow compare / CLI 的真实治理流程

---

## 1. 当前真相源

当前模型治理相关源码入口：

- [`research/model_registry/registry.py`](/quant_system/research/model_registry/registry.py)
- [`research/model_registry/artifact.py`](/quant_system/research/model_registry/artifact.py)
- [`alpha/model_loader.py`](/quant_system/alpha/model_loader.py)
- [`scripts/ops/model_loader.py`](/quant_system/scripts/ops/model_loader.py)
- [`scripts/ops/shadow_compare.py`](/quant_system/scripts/ops/shadow_compare.py)
- [`scripts/shared/cli.py`](/quant_system/scripts/shared/cli.py)（由 [`scripts/cli.py`](/quant_system/scripts/cli.py) 转发）

---

## 2. 当前实际存在两条模型装配路径

### 2.1 活跃 directional alpha 服务路径

当前 `bybit-alpha.service` 使用的是文件系统模型装配：

- 入口: [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py)
- 模型加载: [`scripts/ops/model_loader.py`](/quant_system/scripts/ops/model_loader.py)
- artifact 来源: `models_v8/*/config.json` + 同目录下的 `.pkl`

这条路径当前的特点：

- 不依赖 `ModelRegistry`
- 不依赖 `ArtifactStore`
- 不支持 registry-based promotion / rollback / autoload
- 以各模型目录内的 `config.json` 为当前事实

### 2.2 Framework / LiveRunner 路径

`runner/live_runner.py` 只有在同时提供：

- `LiveRunnerConfig.model_registry_db`
- `LiveRunnerConfig.model_names`

时，才会启用 registry-based loader：

- registry: `ModelRegistry`
- artifact store: `ArtifactStore`
- loader: `ProductionModelLoader`

这条路径当前支持：

- load production models
- `reload_if_changed()`
- `inspect_production_model(s)`
- ops audit / model status 观察

结论：

- registry / promote / rollback / autoload 的正式 contract 当前只对 framework path 成立
- 当前活跃的 `bybit-alpha.service` 并没有接上这条治理链

---

## 3. 职责边界

### 3.1 ModelRegistry

[`ModelRegistry`](/quant_system/research/model_registry/registry.py) 当前负责：

- 注册模型版本
- 维护 `name -> version -> model_id`
- 记录 `params / features / metrics / tags`
- 标记 production 版本
- 记录 promotion / rollback 审计动作
- 查询版本与审计历史

它当前不负责：

- 直接加载权重进 runtime
- 验证 artifact 内容是否可反序列化
- 自动决定候选模型是否应 promotion

### 3.2 ProductionModelLoader

[`ProductionModelLoader`](/quant_system/alpha/model_loader.py) 当前负责：

- 查询 registry 中的当前 production version
- 从 artifact store 读取 `weights` / `weights.sig`
- 推断模型类型并实例化
- 在 production `model_id` 变化时触发 reload
- 输出 `available / has_weights / has_signature / loaded_model_id / autoload_pending`

它当前的真实校验边界：

- 若 `ArtifactStore.verify_digest()` 存在，则执行 digest 校验；失败时拒绝加载
- 若 registry feature schema 与模型 feature schema 都存在且不相等，则拒绝加载
- 若只有一侧缺失 feature schema，则只告警，不阻塞加载
- `features.feature_catalog.validate_model_features()` 当前是 advisory warning，不阻塞加载

它当前不负责：

- promotion 决策
- shadow compare 胜负裁定
- 事务化保证 registry metadata 与 artifact 一致提交

### 3.3 `scripts/ops/model_loader.py`

这个 loader 只服务于当前活跃 directional alpha 服务：

- 读取 `models_v8/*/config.json`
- 加载 horizon ensemble 的 LGBM / XGB / Ridge `.pkl`
- 解析 `deadzone / min_hold / max_hold / long_only / ic_weighted`
- 直接构造 AlphaRunner 所需的 `model_info`

它不是 registry loader，也不应在文档中被误写成 production governance contract 的一部分。

### 3.4 Shadow Compare

[`scripts/ops/shadow_compare.py`](/quant_system/scripts/ops/shadow_compare.py) 当前负责：

- 在同一 OOS 数据上对比 production model 与 candidate model
- 计算 IC / Sharpe / stability
- 给出 `should_promote`
- 可选 `auto_promote`

当前 promotion 判据确实存在于脚本中，而不是统一 policy schema 中。

因此：

- 它是当前默认比较工具
- 但还不是 repo 级强制治理引擎

---

## 4. 当前 promotion / rollback 真相

### 4.1 Promotion

当前真实流程：

1. 训练脚本产出 candidate artifact 与 metrics
2. `ModelRegistry.register()` 记录 metadata
3. `scripts/ops/shadow_compare.py` 对比 candidate 与 production
4. 满足条件后调用 `registry.promote(model_id, ...)`
5. 若 framework runtime 已启用 registry loader，则 `reload_if_changed()` 检测到 `model_id` 变化并重载

说明：

- `promote()` 会记录 `reason / actor / metadata`
- promotion precondition 目前只做 warning，不是强制审批流

### 4.2 Rollback

当前 rollback 已具备正式 API：

- `ModelRegistry.rollback_to_previous(name)`
- `ModelRegistry.rollback_to_previous(name, to_model_id=...)`
- `ModelRegistry.rollback_to_previous(name, to_version=...)`

当前 CLI 入口：

- `quant model-promote --model-id ...`
- `quant model-rollback --model ...`
- `quant model-rollback --model ... --to-version ...`
- `quant model-rollback --model ... --to-model-id ...`
- `quant model-history --model ...`
- `quant model-inspect --model ...`

当前也已具备相应单测与集成验证；“rollback 缺少 API / 测试”已不是当前事实。

---

## 5. Live Autoload Boundary

当前 live autoload 只对 framework path 成立，边界如下：

- 只响应 registry 中 production `model_id` 的变化
- 不负责决定是否 promotion
- 不负责自动比较 candidate 与 production
- digest 校验失败时拒绝加载
- 双边 feature schema 明确不一致时拒绝加载
- 单边 schema 缺失时只告警，不会自动 fail fast

当前最小观察面：

- `inspect_production_model()`
- `inspect_production_models()`
- `ops-audit` 中的 `model_status / model_reload / timeline`

---

## 6. 当前仍未完全收口

- 当前活跃 `bybit-alpha.service` 仍未接上 registry / artifact / autoload 治理链
- registry metadata 与 artifact 完整性还没有统一事务化保证
- shadow compare 的 promotion policy 仍然散落在脚本，而不是单一 schema
- 单边 feature schema 缺失时当前仍允许加载，属于兼容优先而非严格治理

---

## 7. 当前建议

当前最安全的口径是：

- 如果讨论 host 上正在交易的 directional alpha，模型事实以 `models_v8/*/config.json` 为准
- 如果讨论 framework runtime 的 model lifecycle，治理事实以 `ModelRegistry + ArtifactStore + ProductionModelLoader + scripts/shared/cli.py` 为准
- 不要把这两套路径写成已经统一
