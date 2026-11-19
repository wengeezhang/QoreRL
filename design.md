# QoreRL 设计原则与当前架构

QoreRL 把量化和量化误差缓解方法做成可复现的 **recipe**。一份 recipe 定义哪些层使用哪种数值计算、训练和采样两端各自怎样执行，以及哪些状态需要一起保存。用户选择方案，查看它在自己任务上的表现，再决定是否调整。框架负责让这个选择在指定范围内生效，并让每批样本都能追溯到实际使用的策略版本。

## 1. 用户能控制什么

用动态 SmoothQuant 举例，配置里有两份层列表，层号从 0 开始。**`recipe_options.layers` 固定哪些 decoder 层的 `mlp.down_proj` 自第一批起执行低精度计算**。**`smoothquant_layers` 在这份名单内选出一部分，给它们额外做通道缩放 S**。其余已选层直接做低精度矩阵乘法。名单外的层始终走 BF16。一次校准可以更新 S，也可以在授权范围内增加使用 S 的层。

当前低精度格式默认是 `mxfp8`。原生训练省略 recipe 时使用直接 MXFP8。需要全 BF16 对照时显式选择 `bf16-baseline-v1`。要换用 NVFP4，可在通用 recipe 名（如 `smoothquant-mlp-down-v1`）下设置 `recipe_options.precision: "nvfp4"`。无论哪种低精度，基础层范围和 S 的子集规则相同。原生 GPU 验收均待完成。[训练配置](qorerl/training/config.py) · [格式与选层](qorerl/recipes/spec.py)

| 用户场景 | 用户设置 | 实际行为及实现入口 |
| --- | --- | --- |
| 固定量化方案 | 直接 MXFP8 选 `layers`。固定 Hadamard 再设分块和种子。固定 SmoothQuant 提供 `scale_artifact` 并设 `refresh_every_optimizer_steps: 0` | 训练和采样从第一批执行所选方案，外部 S 在首次采样前导入。[变换状态](qorerl/recipes/state.py) |
| 只找 outlier | `discovery_only: true` | 写出发现分数后继续训练。原生 BF16 对照可单独观测各层；SmoothQuant 的观测限于获准校准的层。[输入采集](qorerl/engines/single_rank_learner.py) · [发现报告](qorerl/recipes/calibration.py) |
| 在固定范围刷新 S | 指定 `layers` 和初始 `smoothquant_layers`，沿用默认的 `allow_layer_expansion: false` | 只评估已使用 S 的层。已知目标时可设 `discover_outliers: false`，直接拟合与验证。省略 `smoothquant_layers` 时默认在全部低精度层上使用 S。[校准](qorerl/recipes/calibration.py) |
| 先给少数低精度层加 S，再按需扩大 | 初始 `smoothquant_layers` 选低精度范围的一个子集，另设 `allow_layer_expansion: true` | 校准覆盖 `layers`。已使用 S 的层可以刷新，其余直接 MXFP8 层可以提议首次加 S。两类变更均须通过候选验证。[选层](qorerl/recipes/spec.py) · [训练端](qorerl/recipes/megatron_mxfp8.py) · [采样端](qorerl/recipes/vllm_mxfp8.py) |
| 人工审核或无人值守 | 默认 `approval: "manual"`，可选 `"auto"` | 两种方式共用相同的证据、检查点和发布过程，批准权限独立于可调整的层范围。[确认与续训](qorerl/runtime/calibration.py) |

周期默认每 20 次**实际完成的优化器更新**触发一次，在跨过间隔后的第一个同步批边界评估。省略 `layers` 表示选择所有支持的 decoder `mlp.down_proj`。指定较小名单可以限制量化范围。[校准配置与使用示例](docs/recipes/dynamic-smoothquant.zh-CN.md)

<a id="smoothquant-expansion"></a>

### 动态扩展时，第 25 层什么时候用上 SmoothQuant？

采用[原生示例配置](configs/grpo_dynamic_smoothquant.json)。用户选 `layers: [7,15,20,25]`、`smoothquant_layers: [7,15,20]`，并打开 `allow_layer_expansion`。四层从启动起就运行 MXFP8，只有第 25 层还可以首次加上 S。

| 时刻 | 第 7、15、20 层 | 第 25 层 | 第 23 层等名单外层 |
| --- | --- | --- | --- |
| 第一次采样 | MXFP8 + SmoothQuant | 直接 MXFP8 | BF16 |
| 到达校准周期 | 在留出激活上比较旧 S 与候选 S | 若发现 outlier，拟合 S，用相同权重和留出激活比较直接 MXFP8 与 MXFP8 + 候选 S | BF16，排除在本轮 SmoothQuant 校准外 |
| 候选待审 | 继续用已发布的 S | 继续直接 MXFP8。候选必须降低相对 BF16 参考输出的误差，并满足配置的改善率和首次启用误差上限 | BF16 |
| 批准并发布 | 应用本次获批的刷新 | 若该层获批，全部 rollout 确认后，从下一批开始使用 MXFP8 + SmoothQuant。候选未通过或被拒绝时继续直接 MXFP8 | BF16 |
| 后续校准 | 可以继续刷新 S | 已启用后可继续刷新 S | BF16 |

即使第 23 层有明显 outlier，这次运行也不会给它生成 SmoothQuant 候选。首次启用的误差上限 `max_activation_relative_l2` 可选。所比较的误差是局部线性层输出误差，端到端效果需单独验证。

## 2. 一批训练由谁负责

**一个同步 GRPO 训练批**从 rollout 为每道题生成一组回答开始。编排端核对样本版本与分组完整性，计算奖励和组内优势，再把训练批交给 learner。learner 完成训练后导出权重，全部 rollout 副本安装并确认版本，下批采样才能开始。[训练主循环](qorerl/runtime/ray_runtime.py)

| 职责 | 负责的决定或计算 | 代码入口 |
| --- | --- | --- |
| 训练编排 | 安排采样、更新和发布，在批次之间触发校准、保存检查点或暂停 | [run_sync_grpo](qorerl/runtime/ray_runtime.py) |
| Ray worker | 各自持有 learner 或 rollout 引擎，接收远程调用，管理进程内资源 | [ray_workers.py](qorerl/runtime/ray_workers.py) |
| 后端引擎 | Megatron / Transformer Engine 执行训练并导出权重，vLLM 安装权重并生成回答 | [Megatron learner](qorerl/engines/megatron.py) · [vLLM rollout](qorerl/engines/vllm.py) |
| 精度 recipe | 选择目标层，定义数值变换和需要保存的状态，在后端实现对应算子 | [规格](qorerl/recipes/spec.py) · [状态](qorerl/recipes/state.py) · [后端适配](qorerl/recipes/) |
| 观测 | Recipe Insights 在采集到的操作数上重放固定方案，记录逐层误差和耗时，供用户比较 | [Recipe Insights](docs/recipe-insights.zh-CN.md) |
| 任务、算法与数据 | 定义任务和奖励，计算优势与 loss，校验轨迹并构造模型输入 | [任务](qorerl/training/tasks.py) · [目标函数](qorerl/training/objective.py) · [轨迹协议](qorerl/training/contracts.py) · [批处理](qorerl/training/batching.py) |

例如，给 SmoothQuant 增加一种估计 S 的方法，拟合与验证逻辑属于 recipe。何时暂停采样、恢复后向哪些副本发布，仍由编排端决定。这样可以分别修改数值方法和训练时序。

## 3. 只换 S，为什么还要重新发布？

假设第 7 层的权重 W 刚刚同步。此时用户接受了新的 S。learner 下次计算这层时，会用新的 S 对输入做 `X / S`，对权重的对应列做 `W * S`，然后量化。vLLM 则已经缓存了按**旧 S** 缩放、量化和重排的权重。要让下批采样也执行用户选定的方案，vLLM 必须同步新的 S 并重建缓存，即使 W 的数值完全相同。

![动态 SmoothQuant 的校准与发布。到达校准周期时，可以只做发现，也可以为指定层准备候选。启动时已有 S 则可以直接导入。接受候选后，两端应用新 S，所有 rollout 确认版本才开始下一批采样。](docs/assets/smoothquant-calibration-flow.zh-CN.svg)

**候选需要什么证据？** 默认每跨过 20 次实际优化器更新，learner 在同步批边界用更新后的权重重放刚训练的轨迹前缀。校准证据写入 `calibration.jsonl`。图中的固定范围 CPU 示例有两层都检出了 outlier。在相同的留出输入上，换 S 后，第 0 层相对同层 BF16 输出的误差从 7.82% 降至 4.79%，第 1 层却从 3.31% 升至 3.86%。因此只有第 0 层进入候选。这些证据回答的是该层的前向数值问题，训练奖励和端到端成本需要另外实测。[候选比较与 CPU 示例](docs/recipes/dynamic-smoothquant.zh-CN.md)

**批准的是哪一次变更？** 产生候选后，编排端先保存变更前的 learner、优化器和 recipe 状态，并把候选及证据与这个检查点绑定。人工模式暂停训练，用户恢复时按候选 ID 接受或拒绝这份变更。拒绝后沿用原 S。自动模式应用满足预设条件的候选。[确认与恢复实现](qorerl/runtime/calibration.py)

**如何知道下一批用的是新方案？** 每批轨迹记录 `PolicyVersion(weight_version, recipe_id)`。前者记录训练批的权重发布代次。对于动态 SmoothQuant，后者还标识 recipe 规格和当前各层 S 的内容摘要。只换 S 时，权重代次保持原值，策略版本仍会改变。learner 保留原坐标系的模型参数和优化器状态，发布权重与新的 recipe 状态。所有 rollout 重建缓存并确认预期版本后，编排端才放行采样。安装失败则停止本次运行，保留故障记录和最近的完整恢复点。早期版本允许把 BF16 层升级为 MXFP8，当时保存的扩层检查点与现在的固定低精度范围语义不同。恢复会检查 recipe 身份；运行新设计时需显式声明低精度范围并开始新的运行。[版本标识](qorerl/recipes/state.py) · [加载与确认](qorerl/engines/vllm.py)

### 一次人工确认会写下哪些文件？

下面缩成三层 CPU 模拟。第 0、1 层启用 MXFP8，初始只给第 1 层加 SmoothQuant。第 2 层在低精度范围之外。为演示审核，测试把校准周期设为一次优化器更新。第一批训练结束后，框架提议让第 0 层也启用 SmoothQuant，此时训练暂停。打开这次运行的输出目录，按下面的顺序读就能还原经过。[生成这些文件的测试](tests/test_calibration_workflow.py)

| 想知道什么 | 去哪里看 | 本例读到什么 |
| --- | --- | --- |
| 训练为什么停了，从哪里继续？ | 运行目录的 `summary.json` | `status` 是 `awaiting_review`。`checkpoint` 指向 `checkpoints/step-000001-before-calibration/`，`candidate_id` 是接受或拒绝时要传的完整 ID。 |
| 保存时实际生效的是哪个版本？ | 上述检查点的 `metadata.json` | 第 1 批的权重已经发布，第 0、1 层执行低精度计算，此时只有第 1 层启用 SmoothQuant。`policy_version` 标识当时生效的策略，`pending_candidate` 指向待审提案。普通检查点也有这份元数据，只有带候选的检查点才有这个字段和同目录的 `candidate.json`。 |
| 这次准备改哪一层，凭什么改？ | 同一检查点的 `candidate.json` | `evidence` 记录所有被评估的层。第 0 层为 `proposed`，第 1 层为 `no_gain`，第 2 层在范围外，证据中没有它的记录。真正要发布的新 S 只出现在 `scale_artifact.scales` 的第 0 层条目里。 |
| 用户的决定生效了吗？ | 运行目录的 `calibration.jsonl`，每行一个事件 | 原目录依次记录 `evaluated`、`awaiting_review`。人工接受并续训后，**新输出目录**记录 `applied`，选择拒绝则记录 `rejected`。接受分支的第 2 批在 `trajectories.jsonl` 中，每条轨迹的 `policy_version` 都与 `applied.policy` 一致。 |

<details>
<summary>展开查看这次暂停写入的实际 JSON</summary>

以下节选保留实际字段和值。`metadata.json` 还保存了用于续训兼容性检查的完整 `config`。这里的 `boundary` 表示所有 rollout 已确认当前版本，且没有在途样本。

```json
{
  "schema": 2,
  "step": 1,
  "policy_version": {
    "weight_version": 1,
    "recipe_id": "mxfp8-smoothquant-mlp-down-v1:52badf3f31e93eab:b6e1880eba09c4845b562a93519ba3b449033125469094a655ac9506de039c25"
  },
  "pending_candidate": "4c65a5ecaf946494d4823a5b482433126f8616b1dd52acb75b765d27f6931428",
  "boundary": "all_rollouts_acknowledged_no_inflight_samples"
}
```

`candidate.json` 下面只展示第 0 层的证据和该层 32 个通道中前 3 个 S。相对 L2 误差都以同一份 BF16 输出为参照。第 0 层当前是直接 MXFP8，误差为 7.82%。加上候选 S 后降至 4.79%，相对降幅为 38.74%。

```json
{
  "optimizer_steps": 1,
  "scale_artifact": {
    "kind": "smoothquant_scales",
    "scales": {
      "model.layers.0.mlp.down_proj.weight": [
        72647.875,
        1.6489697694778442,
        1.4567848443984985
      ]
    }
  },
  "evidence": [
    {
      "target": "model.layers.0.mlp.down_proj.weight",
      "action": "activate",
      "status": "proposed",
      "current_relative_l2": 0.07822830391426117,
      "direct_low_precision_relative_l2": 0.07822830391426117,
      "candidate_relative_l2": 0.04792346157040247,
      "relative_improvement": 0.3873897403818577
    }
  ],
  "checkpoint_sha256": "1f67499afd73ca03b8db4e3e008233e649d2e71aeea9a30047fc305aa0a53215"
}
```

运行目录的 `summary.json` 给出这次要审核的 ID，可直接传给 `--accept-candidate` 或 `--reject-candidate`。

```json
{
  "status": "awaiting_review",
  "last_step": 1,
  "candidate_id": "4c65a5ecaf946494d4823a5b482433126f8616b1dd52acb75b765d27f6931428"
}
```

续训时，`--resume` 接收整个检查点目录。框架校验候选内容算出的 ID 是否等于 `metadata.pending_candidate`、候选的 `source_policy` 是否等于 `metadata.policy_version`，以及同目录的 `learner.pt` 内容摘要是否等于 `candidate.checkpoint_sha256`。这样确认的就是当时看到的提案，恢复的也是与它配套的模型和优化器状态。[检查点写入](qorerl/runtime/checkpoint.py) · [续训校验](qorerl/runtime/calibration.py) · [接受或拒绝的命令示例](docs/recipes/dynamic-smoothquant.zh-CN.md#在-cpu-上走完一次人工确认)

</details>

其余文件按用途查阅即可。[运行文件写入位置](qorerl/runtime/ray_runtime.py)

| 文件 | 什么时候看 |
| --- | --- |
| `config.json`、`recipe.json`、`workers.json` | 复核启动条件时，分别查看解析后的训练配置、recipe 的规格及初始选层、实际启动的进程。本例 `recipe.json` 中的 `layers: [0,1]` 是固定的低精度范围，`smoothquant_layers: [1]` 是最初使用 S 的层。后续 S 的覆盖范围可从发布记录中核对。 |
| `metrics.jsonl`、`trajectories.jsonl` | 排查某个训练批时，前者每批一行，包含 `reward_mean`、`clip_fraction` 等。后者保留该批实际采样的 token、行为策略 log-prob、版本，以及对应的奖励与优势。 |
| `recipe_metrics.jsonl`、`incident.json` | 启用 Recipe Insights 且产生逐层观测时，前者记录局部比较。训练运行异常退出时，后者记录异常类型、消息和最近的恢复点 `last_recovery_checkpoint`。正常等待审核属于 `summary.json` 的状态。 |

## 4. 今天可以依赖到什么程度

| 状态 | 范围 |
| --- | --- |
| 本地已验证 | CPU Ray 多进程测试覆盖同步 GRPO、策略版本发布、SmoothQuant 的首次启用与刷新、接受与拒绝，以及从检查点继续训练。示例中的量化数字来自 CPU 模拟。[流程测试](tests/test_calibration_workflow.py) |
| 原生代码已实现，GPU 待验收 | 首个后端组合是单 rank dense Qwen3、Megatron / TE 和 vLLM，覆盖 BF16 基线及选定 MLP 输出投影的直接低精度、固定 Hadamard 和动态 SmoothQuant。MXFP8 为默认格式，NVFP4 在相同层范围内可选。GPU 上的实际执行、refit 后同一 token 的训推概率差异、学习效果和成本均待测量。[运行与验收指南](docs/ray-grpo.zh-CN.md) |
| 后续工作 | 当前可从检查点确认的变更限于 SmoothQuant 候选。通用告警后的跨 recipe 调整、多 rank、MoE、异步训练，以及根据训练反馈自动选择修复方案，列入[开发计划](todo.md)。 |
