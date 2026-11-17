# QoreRL 设计原则与当前架构

本文记录当前代码的职责边界与设计取舍，取代早期围绕自动敏感层修复的草案。后续工作、触发条件和验收标准见 [todo.md](todo.md)，运行说明见 [开发指南](docs/ray-grpo.zh-CN.md)。

## 1. 项目要解决什么问题

QoreRL 的首期价值是：把已有及新出现的量化、量化误差缓解技术变成有适用条件和验证证据的 recipe，让用户可以便捷选择、组合、观测，并在异常后保存状态、调整配置、继续训练。

长期研究自动选择方案与自适应精度，但自动修复不是首期可用的前提。缩小 outlier、降低张量误差、接近 BF16 梯度，都不能单独证明训练收益；最终需要同时考察训练质量、数值行为和完整执行成本。

底层复用 Ray、Megatron / Transformer Engine、vLLM 的能力。通用名称和成熟执行机制尽量沿用，把创新集中在 recipe、证据和恢复工作流上。

## 2. 核心分层：谁安排工作，谁承接调用，谁完成计算

| 层次 | 当前代码 | 应负责的事情 |
| --- | --- | --- |
| 训练编排 | `qorerl/runtime/ray_runtime.py` 的 `run_sync_grpo()` | 安排资源，决定什么时候采样、训练、发布权重和保存状态 |
| Ray worker | `qorerl/runtime/ray_workers.py` 的 `RayLearnerWorker`、`RayRolloutWorker` | 持有 engine，承接远程调用与进程内初始化/关闭 |
| 后端引擎 | `qorerl/engines/` 的角色接口、工厂、Megatron/vLLM 实现 | 完成模型加载、前后向、参数更新、生成、权重导出与安装 |
| 算法 | `qorerl/training/objective.py` | 组内优势、策略目标与 loss 聚合语义 |
| 任务与奖励 | `qorerl/training/tasks.py` | 产生任务输入、按任务规则评分 |
| 数据协议与批处理 | `contracts.py`、`batching.py` | 前者定义字段和身份/版本/完整性检查；后者构造设备上的 padding、label、mask |

训练编排应当能够直接读成：

```text
让 rollout 生成样本
        ↓
校验身份、版本和完整组，按固定顺序整理样本
        ↓
按任务规则计算奖励，再按 GRPO 计算组内优势
        ↓
让 learner 更新参数
        ↓
导出新权重，所有 rollout 安装完成并确认版本
        ↓
在没有在途请求的边界保存 checkpoint，开始下一轮
```

编排代码不应知道 Megatron 的层名、vLLM 的量化 layout；vLLM 的加载逻辑不应决定下一轮 GRPO 何时开始。数据协议不应实现某一道任务的判题规则，也不应硬编码 GRPO 优势计算。

当前 GRPO 数学定义仍由单 rank learner 调用，任务选择仍与首个原生场景/CPU fixture 绑定；这不是已完成的通用算法或任务插件系统。出现第二个真实需求时，再扩展明确的接口，避免先建设无实际用途的抽象。

## 3. 名称表达什么

- `Policy`：策略模型或动作分布的算法概念。
- `Learner`：消费训练数据并更新模型参数的角色。
- `Rollout`：用已发布策略采集回答的角色。
- `Ray…Worker`：由 Ray Actor 托管的有状态执行进程。这里的 Ray Actor 不等于 actor–critic 中的 actor。
- `Engine`：封装具体训练或推理后端，不决定全局训练时序。
- `PaddedMicrobatch`：每条回答占一行并补齐长度；当前没有 sequence packing。
- `behavior_log_probs`：实际生成这些 token 的行为策略给出的 log-prob，不默认等于训练侧旧策略重算值。
- `policy_version`：策略发布身份，包含 `weight_version` 和 `recipe_id`，不是策略模型对象。

当前 `weight_version` 表示每个训练 batch 处理后的发布代次，不是权重内容 hash，也不是 optimizer step 数。一个 batch 可以执行多个 optimizer step；全零优势也可以不更新参数但推进发布代次。日志中的 `optimizer_steps` 单独记录真正的更新次数。

`recipe_id` 目前只是受配置白名单约束的固定标识，不代表已有可组合 recipe、动态量化状态或热切换能力。将来的量化状态变化必须纳入执行身份和恢复协议，不能只改一个字符串。

## 4. 发布与恢复的边界

保留三个不同动作：

1. `export_weights`：learner 形成可同步的权重表示。
2. `install_weights`：rollout 完成加载，必要时包含重排、派生状态重建；成功后才确认版本。
3. `publish`：编排侧等待所有目标 rollout 的确认，随后允许新一轮采样。

当前只实现同步执行，更新时没有在途生成请求。vLLM refit 开始前清除有效版本，失败后不能继续带着未确认状态采样。optimizer update 和 refit 不自动重试，因为中途失败并不代表完全没有修改状态。

checkpoint 的 payload 和 metadata 写完后才通过目录 rename 发布。异常证据另存 `incident.json`，不把故障现场冒充可恢复 checkpoint。恢复从最近一个完整 checkpoint 重新创建引擎、加载 learner 状态、发布权重，再采新样本。

当前只支持同训练语义恢复，修改 recipe、学习率等会被拒绝。未来“暂停 → 用户改配置 → 继续训练”需要显式验证状态兼容性与重建范围，并记录实验延续关系，不通过取消所有配置检查来实现。

当前写出 schema 2 checkpoint，继续读取 schema 1。新字段为 `policy_version.{weight_version,recipe_id}`、`master_parameters`；旧字段仅在读取边界转换。历史记录不改写成新字段。该兼容仅涉及结构改名，不意味着支持任意历史算法或精度方案之间迁移。

## 5. 单 rank 基线与当前限制

`SingleRankGRPOLearner` 独立于角色接口，明确承载当前的 GRPO、FP32 master 参数、梯度累积和 PyTorch AdamW。Megatron adapter 当前使用它作为可核对的单 rank 更新实现；它没有 Megatron 原生分布式 optimizer 或 pipeline schedule。

模型梯度先以模型 dtype 产生，再转为 FP32 跨 microbatch 累积。这里没有宣称整个 backward 使用 FP32。未来多卡 learner 应使用原生 Megatron 调度、梯度同步、optimizer 和分片 checkpoint，不在当前循环上简单增加 DP/TP 参数。

当前 loss 接收完整 logits；词表并行和 fused loss 需要引擎提供合适的可微分 token log-prob/loss 路径。当前权重快照是完整 CPU tensor 字典，checkpoint payload 是 bytes；支持分桶或 GPU 传输时还需要调整 payload 接口，不能认为已有通用高性能传输抽象。

为首条链路可核对，当前使用完整组同步、普通 padding、完整 CPU 权重传输，关闭 prefix cache/CUDA graph。性能优化必须保留样本、版本、loss 归一化和恢复语义，并用实测结果验证。

## 6. recipe 应当扩展哪里

recipe 需要描述 Fprop、Dgrad、Wgrad、权重导出和 rollout 各自的数值方案、后端/硬件要求、可组合条件及状态保存需求。后端负责把这些要求落实到实际执行；编排侧负责选择、验证、记录、协调生效与恢复边界。

同一个 recipe 可以为训练与 rollout 指定不同操作。例如只调整训练侧 Wgrad 的操作数精度，不要求 rollout 执行反向。影响前向策略的变更需要协调两端并直接检查训推概率差异；相同配置名称不保证相同数值结果。

每项方法先作为明确范围内的实验选项，经过验证后再标记为适用于某类工作负载的 recipe。组合的收益需要重新验证。自动推荐和自动应用均要保留触发证据、改动和后续效果。

## 7. 已实现、待验证与历史材料

- 已实现：Ray 同步 GRPO，Megatron/vLLM adapter，明确区分的 CPU 测试引擎，版本发布、完整组校验、同配置 checkpoint/resume。
- 本地测试范围：真实 CPU Ray 进程及优化器更新、算法语义、失败时不确认 refit、恢复一致性。adapter mock 测试仅验证调用约定。
- 待验证：原生 NVIDIA GPU 执行、两端 log-prob 与 refit 数值结果、原生恢复行为。
- 未实现：原生低精度 recipe、告警暂停控制器、改变 recipe 后恢复、多 rank、异步训练、自动自适应。

运行汇总区分 `native_gpu_executed`（执行事实）与 `native_gpu_acceptance`（当前不自动判定验收）。选择 native backend 并结束循环，不代表质量、有效更新或 refit 的验收全部通过。

早期 tiny-MoE 校准源码、独立 CLI 与旧测试已退出当前代码树。[历史负面结果](docs/experiments/2026-11-08/README.md) 保留为归档，能从固定 Git 提交恢复；当前 CPU GRU 工程测试和 Ray 验证记录继续保留。后续研究重新接入时，需要按当前 recipe 与证据标准组织。
