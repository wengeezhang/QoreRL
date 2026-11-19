# 在训练中更新 SmoothQuant 的通道缩放

QoreRL 允许用户指定在哪些 MLP down projection 上使用 MXFP8，以及何时校准、由谁批准和何时启用新的通道缩放 S。第一版覆盖单 rank dense Qwen3 的 Megatron/TE learner 和 vLLM rollout。本地 CPU 流程已经运行，原生 GPU 的执行、开销和训练效果待验证。原始 SmoothQuant 研究的是离线校准的 INT8 推理；这里将通道缩放用于训练期的 MXFP8。[原文](https://arxiv.org/abs/2211.10438)

## 选择从哪一步开始

涉及修改 S 的入口使用 `mxfp8-smoothquant-mlp-down-v1`。单独发现也可以在原生 BF16、直接 MXFP8、固定 Hadamard 上运行，保持原计算方式。BF16 检查全部 down projection。固定范围的 MXFP8 recipe 检查 `recipe_options.layers` 选定的层（层号从 0 开始）。动态 SmoothQuant 开启扩层后，校准时检查所有 decoder 的 MLP down projection。阶段开关填在 `recipe_calibration` 中。

| 我手里有什么 | 设置 | 到期时发生什么 |
| --- | --- | --- |
| 想先找出哪个目标层有 outlier | `discovery_only: true` | 输出各层分数和观测行数，继续用当前配置训练。 |
| 知道要检查哪些层，还不知道哪些值得调整 | `discover_outliers: true` | 检查范围内各层，仅为超过发现阈值的层拟合并比较 S。 |
| 已经知道要校准第 7、19 层 | `recipe_options.layers: [7,19]`，`discover_outliers: false` | 直接为指定层拟合和比较 S，仍记录分数供复核。 |
| 已有校准好的 S | `scale_artifact: "scales.json"`，`refresh_every_optimizer_steps: 0` | 启动时完成兼容性检查，在第一次采样前发布这些 S。把间隔设为正数可在此基础上继续刷新。 |

`refresh_every_optimizer_steps` 默认 20。一次 rollout 批可能有多个更新 epoch，也可能由于全零优势跳过更新。程序按实际完成的 optimizer 更新数计数，在跨过指定间隔后的第一个同步批边界校准一次。恢复后的计数从 checkpoint 延续。周期校准默认 `approval: "manual"`；`"auto"` 表示事先同意应用在下述留出数据上满足条件的候选。显式指定外部文件表示用户已选择这组 scale。

例如，使用 BF16 训练时，在配置中加入 `"recipe_calibration": {"discovery_only": true, "refresh_every_optimizer_steps": 20}`，即可在更新数达到周期时将报告写入 `calibration.jsonl`。已有 BF16 激活张量时，也可以独立调用 `qorerl.recipes.calibration.outlier_report(samples, CalibrationConfig(discovery_only=True))`。其中 `samples` 把层名映射到形如 [token 数, 输入通道数] 的 BF16 张量。单独发现使用全部采集行。

## 它拿什么来决定换 S

1. learner 在刚完成更新的权重上，重放刚训练的轨迹前缀，收集所选层在变换前的输入 X。按前缀去重，排除 padding，每层最多保留 `max_rows` 行（默认 128）。输入行的前半用于拟合，后半留作比较。两个集合仍可能来自同一条回答。
2. 每个输入通道 j 取拟合行上的最大绝对值 `x_max[j]`。发现分数为这些通道最大值中的最大值除以它们的平均值。默认阈值为 4。这个分数描述同一层内通道幅度的不均衡。
3. 对入选层取权重第 j 列的绝对值最大值 `w_max[j]`，计算 `S[j] = x_max[j]^alpha / w_max[j]^(1-alpha)`，默认 alpha 为 0.5。前向用 `X/S` 和 `W*S` 做配套变换，W 的存储形状为 [out, in]。未观测到有效信号的通道沿用原 scale。新 S 是相对原始参数的绝对缩放，刷新时替换指定层的旧值。
4. 对已经启用的层，在相同的留出行和权重上，分别用当前 S 和候选 S 做 MXFP8 前向，与该层原始 BF16 线性投影比较相对 L2 误差。只有候选误差**严格下降**，且相对降幅达到 `min_relative_improvement`（默认 0），该层才进入候选。各层分别比较，文件中保留未入选的原因。多层同时应用的误差、训推 log-prob、奖励与端到端成本还需要另行评估。

这个比较让使用者分清两件事。一层可能具有明显 outlier，同时当前 MXFP8 的输出比候选缩放更接近 BF16。发现阶段可以选中它，准备阶段仍可保留原配置。

## 只在发现 outlier 后扩大范围

以下是可合并到[原生训练配置](../../configs/grpo_dynamic_smoothquant.json)的字段。首批只有第 7、19 层的 `down_proj` 使用 MXFP8 + SmoothQuant。每跨过 20 次优化器更新，learner 在同一轮校准中也采集其他 BF16 `down_proj` 的输入。

```json
{
  "recipe_id": "mxfp8-smoothquant-mlp-down-v1",
  "recipe_options": {"layers": [7, 19], "allow_layer_expansion": true},
  "recipe_calibration": {
    "refresh_every_optimizer_steps": 20,
    "approval": "manual",
    "max_activation_relative_l2": 0.05
  }
}
```

在 `calibration.jsonl` 的逐层证据中，`action` 区分两类评估。

| action | 这层当前怎么执行 | 怎样读证据 |
| --- | --- | --- |
| `refresh` | MXFP8 + 当前 S | 比较当前与候选 S 相对同一 BF16 输出的误差；只有达到降幅条件才提议刷新。 |
| `activate` | BF16 | 新层始终要先发现 outlier，再比较直接 MXFP8（`direct_mxfp8_relative_l2`）及配上候选 S 后的误差（`candidate_relative_l2`）。上例要求后者 ≤ 0.05，即相对 L2 误差不超过 5%。 |

5% 是这个示例选用的容许值，默认配置不设上限。两种试算和参考值使用同一份留出输入和同一份权重。实际训练与采样仍沿用上一版本，直到批准并向所有 rollout 发布新版本。选择 `auto` 会自动应用合格候选。完整的第 23 层例子见[设计文档](../../design.md#扩层特例第-23-层什么时候离开-bf16)。

## 在 CPU 上走完一次人工确认

在仓库根目录运行：

```bash
uv run qorerl train --config configs/grpo_cpu_smoothquant.json --out runs/sq-review
```

这个例子使用显式的块量化模拟和一个带已知通道不均衡的小模型，含一个 learner 和两个 rollout 进程。为了在短运行中看到决策，间隔设为一次更新。一次本地运行第一批有 4 行用于拟合、5 行用于比较，得到：

| 层号 | 发现分数 | 当前 S 的相对误差 | 新 S 的相对误差 | 决定 |
| --- | ---: | ---: | ---: | --- |
| 0 | 31.98 | 0.07823 | 0.04792 | 提议更换 S |
| 1 | 31.62 | 0.03312 | 0.03862 | 保留当前 S |
| 2 | 1.78 | — | — | 低于发现阈值 |

`runs/sq-review/calibration.jsonl` 记录证据；`summary.json` 提供 `candidate_id` 和 `checkpoint`。待确认时训练退出、释放 worker。查看 checkpoint 内的 `candidate.json` 后，用新的输出目录继续：

```bash
CP=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["checkpoint"])')
ID=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["candidate_id"])')
uv run qorerl train --config configs/grpo_cpu_smoothquant.json \
  --resume "$CP" --accept-candidate "$ID" --out runs/sq-accepted
```

拒绝时把参数改为 `--reject-candidate "$ID"`。接受时保留模型参数、optimizer 和更新计数，发布新的 scale 版本。所有 rollout 确认后，才用新版本生成下一批。拒绝时继续使用原来的 S。可在 `trajectories.jsonl` 的 `policy_version` 中核对实际采样版本。每次应用前都有恢复点，失败时查看 `incident.json` 中的路径。需要重做决策时，可从变更前的 checkpoint 开始另一个输出目录。

## 导入已有 scale

外部工具需提供每个目标层一个有限、严格为正的 FP32 向量，长度等于该层的输入通道数。启动导入只允许覆盖初始选定层，可只提供其中一部分。之前按相同模型版本和相同初始层做的固定范围校准，也可以作为扩层运行的起点。初始层中未提供 S 的默认全 1，范围外仍走 BF16。下例将校准工具已经生成的 `scales` 字典封装为可导入 JSON：

```python
import json
from pathlib import Path

from qorerl.recipes.calibration import make_scale_artifact
from qorerl.training.config import TrainConfig

cfg = TrainConfig.load("configs/grpo_dynamic_smoothquant.json")
# scales["model.layers.7.mlp.down_proj.weight"] 是校准得到的一维 CPU float32 Tensor
artifact = make_scale_artifact(
    scales,
    model=cfg.model,
    model_revision=resolved_model_revision,  # 此次启动会使用的模型 commit
    recipe_id=cfg.policy_recipe_id,
)
Path("configs/scales.json").write_text(json.dumps(artifact))
```

设置 `recipe_calibration.scale_artifact` 为 `"scales.json"`。相对路径从配置文件所在目录解析。导入检查模型、版本、recipe 选层、目标名称、通道数和内容摘要。使用本地模型目录时，请保留校准对应的模型快照；目录名本身无法标识其内容。checkpoint 独立保存所有生效向量，正常续训时无需再次读取外部文件。

## 原生运行入口与验证范围

```bash
uv run qorerl train --config configs/grpo_dynamic_smoothquant.json --out runs/sq-native
```

[原生配置](../../configs/grpo_dynamic_smoothquant.json) 以第 7、19 层为例，需要支持原生 MXFP8 的 Megatron/TE 与 vLLM 环境。安装前提沿用 [MXFP8 指南](mxfp8-hadamard.zh-CN.md)。该 recipe 的校准证据写入 `calibration.jsonl`，包含当前 S 和候选 S 的实际比较。GPU 验收还需要检查两端相同 token 的 log-prob、refit 后输出、学习曲线、校准额外显存及耗时。vLLM 启动时只为初始启用层建立量化权重缓存，后续扩层和重建缓存的峰值显存需要单独预留和测量。
