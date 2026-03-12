# Model Governance

> 更新时间: 2026-03-12
> 目标: 固定 registry / loader / shadow compare / promotion / rollback 的职责边界，减少模型上线流程的隐式约定
> 适用范围: 当前 registry / loader / scripts shadow compare 的生产治理流程

---

## 1. 当前真相源

当前模型治理相关的源码入口：

- [`research/model_registry/registry.py`](/quant_system/research/model_registry/registry.py)
- [`alpha/model_loader.py`](/quant_system/alpha/model_loader.py)
- [`scripts/shadow_compare.py`](/quant_system/scripts/shadow_compare.py)

---

## 2. 职责边界

### 2.1 ModelRegistry

[`ModelRegistry`](/quant_system/research/model_registry/registry.py) 当前负责：

- 注册模型版本
- 维护 `name -> version -> model_id`
- 记录 params / features / metrics / tags
- 标记 production 版本
- 比较两个版本的 metadata / metrics / features

它当前不负责：

- 加载模型权重到运行时
- 验证 artifact 是否完整
- 直接驱动 live runtime 热更新

### 2.2 ProductionModelLoader

[`ProductionModelLoader`](/quant_system/alpha/model_loader.py) 当前负责：

- 按模型名查询当前 production version
- 从 artifact store 读取 weights / signature
- 推断模型类型并实例化
- 校验 registry feature schema 与权重中声明的 feature schema 是否一致
- 记录已加载的 `name -> model_id`
- 在 production model_id 变化时触发 reload

它当前不负责：

- 决定哪个版本应该 promotion
- 写入 registry 元数据
- 决定 shadow compare 的胜负逻辑

### 2.3 Shadow Compare

[`scripts/shadow_compare.py`](/quant_system/scripts/shadow_compare.py) 当前负责：

- 对 production model 和 candidate model 做同一 OOS 数据下的对比
- 计算 IC / Sharpe / stability 等比较指标
- 给出 `should_promote` 判断
- 可选地执行 `auto_promote`

它当前不负责：

- 作为 live runtime 的默认热更新机制
- 替代 registry 的元数据记录

---

## 3. 当前 Promotion 流程

当前实际流程可总结为：

1. 训练脚本产出 candidate model 和 metrics
2. `ModelRegistry.register()` 记录 metadata
3. `shadow_compare.py` 对比 candidate 与 current production
4. 满足条件时调用 `registry.promote(model_id)`
5. `ProductionModelLoader.reload_if_changed()` 在 model_id 变化时重新加载

当前 promotion 判据主要出现在 [`scripts/shadow_compare.py`](/quant_system/scripts/shadow_compare.py)：

- candidate `passed`
- candidate `overall.ic > production overall.ic`
- candidate `h2.ic > 0`
- candidate `deflated_sharpe > 0`

说明：

- 这是一套存在于脚本中的当前默认制度
- 还不是全仓统一 schema / policy 层面的正式 contract

---

## 4. 当前 Rollback 流程

当前 rollback 的实际机制仍然偏手动：

1. 查询 `registry.list_versions(name)`
2. 选择旧版本 `model_id`
3. 调用 `registry.promote(old_model_id)`
4. 由 `ProductionModelLoader.reload_if_changed()` 检测并重载

当前缺口：

- 没有单独的 `rollback()` API
- 没有正式的 rollback runbook
- 没有统一的 “promotion reason / rollback reason” 审计字段

---

## 5. Shadow Compare 的制度位置

当前建议把 shadow compare 视为：

- promotion 前的比较门槛
- 生产切换前的准入工具

不建议把它视为：

- live runtime 内部的自动模型选择器
- registry 的替代品

---

## 6. 现阶段建议制度

建议把模型治理固定成以下流程：

1. `register`
   - 记录 params / features / metrics / tags
2. `shadow compare`
   - candidate vs production on a fixed OOS slice
3. `promote`
   - 仅在比较结果和准入门槛通过后执行
4. `reload`
   - loader 检测 `model_id` 变化后重载
5. `rollback`
   - promotion 到上一个稳定版本

---

## 7. 当前仍未完全收口

- model artifact 完整性和 registry metadata 还没有统一事务化保证
- schema mismatch / feature mismatch 已开始形成 loader 级硬约束，但还未扩展到完整 artifact / config / runtime schema
- rollback 缺少一等公民 API 和测试
- live autoload 的行为边界还没有正式 runbook

---

## 8. 下一步

后续应优先补：

1. registry promote / rollback tests
2. schema mismatch / feature mismatch tests
3. model lifecycle runbook
