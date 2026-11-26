# 在训练中更新 SmoothQuant 的通道缩放

[English](dynamic-smoothquant.md) · **简体中文**

用户先选定哪些层的 `mlp.down_proj` 用低精度计算，再决定其中哪些层启用 SmoothQuant、何时重新校准、由谁批准新的逐通道平滑因子 S。本文以默认的 MXFP8 为例，实验性的 NVFP4 路径沿用相同的选层和校准规则。第一版覆盖单 rank dense Qwen3 的 Megatron/TE learner 和 vLLM rollout。原始 SmoothQuant 研究离线校准的 INT8 推理，QoreRL 将通道缩放用于训练期间。本地 CPU 已验证校准与发布流程，原生 GPU 的执行、开销和训练效果待验证。[原文](https://arxiv.org/abs/2211.10438)

## 查看校准并准备 GPU 验收

校准展示直接复用已有比较记录，关联不同运行段的审批和发布；每次查看一个窗口，保留未通过的候选。[查看和导出校准证据](../recipe-insights.zh-CN.md#smoothquant复用已有校准证据)。

单张 B200 可以从 [`grpo_single_gpu_smoothquant.json`](../../configs/grpo_single_gpu_smoothquant.json) 开始：Qwen3-1.7B、一个共置 rollout worker、microbatch=1，每 20 次实际更新进行人工校准。加载前运行 `gpu-preflight`，完整训练段结束后运行 `validate-run`。[GPU 验收指南](../gpu-validation.zh-CN.md) 说明命令、证据、阈值和仍需验证的内容。原生 GPU 尚未验证；H200 无法执行当前原生 MXFP8 路径。

## 选择从哪一步开始

选 `smoothquant-mlp-down-v1` 后，先用 `recipe_options.layers` 指定固定的低精度范围，再用 `smoothquant_layers` 指定哪些低精度层初始启用 SmoothQuant。层号从 0 开始，两份名单都针对 `mlp.down_proj`。省略前者会选全部支持的层，省略后者会在全部已选层叠加 S。显式传入 `smoothquant_layers: []` 可从直接 MXFP8 起步，配合 `allow_layer_expansion: true` 等待候选。原生 BF16、直接 MXFP8 和固定 Hadamard MXFP8 也可以单独采集 outlier 证据。下面的校准开关均位于 `recipe_calibration` 中。

| 我手里有什么 | 设置 | 到期时发生什么 |
| --- | --- | --- |
| 想先找出哪个目标层有 outlier | `discovery_only: true` | 输出各层分数和观测行数，继续用当前配置训练。 |
| 知道要检查哪些层，还不知道哪些值得调整 | `discover_outliers: true` | 检查范围内各层，仅为超过发现阈值的层拟合并比较 S。 |
| 已经知道要校准第 7、19 层 | `recipe_options.layers: [7,19]`，`discover_outliers: false` | 直接为指定层拟合和比较 S，仍记录分数供复核。 |
| 已有校准好的 S | `initial_artifact: "smoothing-factors.json"`，`refresh_every_optimizer_steps: 0` | 启动时完成兼容性检查，在第一次采样前发布这些 S。把间隔设为正数可在此基础上继续刷新。 |

`refresh_every_optimizer_steps` 默认 20。一次 rollout 批可能有多个更新 epoch，也可能由于全零优势跳过更新。程序按实际完成的 optimizer 更新数计数，在跨过指定间隔后的第一个同步批边界校准一次。恢复后的计数从 checkpoint 延续。周期校准默认 `approval: "manual"`；`"auto"` 表示事先同意应用在下述留出数据上满足条件的候选。显式指定外部文件表示用户已选择这组初始平滑因子。

例如，显式选择 `recipe_id: "bf16-baseline-v1"` 时，在配置中加入 `"recipe_calibration": {"discovery_only": true, "refresh_every_optimizer_steps": 20}`，即可在更新数达到周期时将报告写入 `calibration.jsonl`。已有激活时，把层名映射到形如 [token 数, 输入通道数] 的 BF16 张量。将这个映射和 `CalibrationConfig(discovery_only=True)` 传给 `qorerl.recipes.calibration.discover`，即可得到达到阈值的层名和分数。同模块的 `outlier_report` 返回所有层的证据，含采集行数和筛选原因。单独发现使用全部采集行。

## 它拿什么来决定换 S

1. learner 在刚完成更新的权重上，重放刚训练的轨迹前缀，收集所选层在变换前的输入 X。按前缀去重，排除 padding，每层最多保留 `max_rows` 行（默认 128）。输入行的前半用于拟合，后半留作比较。两个集合仍可能来自同一条回答。
2. 每个输入通道 j 取拟合行上的最大绝对值 `x_max[j]`。发现分数为这些通道最大值中的最大值除以它们的平均值。默认阈值为 4。这个分数描述同一层内通道幅度的不均衡。
3. 对入选层取权重第 j 列的绝对值最大值 `w_max[j]`，计算 `S[j] = x_max[j]^alpha / w_max[j]^(1-alpha)`，默认 alpha 为 0.5。前向用 `X/S` 和 `W*S` 做配套变换，W 的存储形状为 [out, in]。未观测到有效信号的通道沿用原平滑因子。新 S 是相对原始参数的绝对缩放，刷新时替换指定层的旧值。
4. 对每层取同一份留出激活和权重，计算 BF16 参考输出，再分别运行当前配置和候选 S 的 MXFP8 前向。相对 L2 误差是两份输出之差的 L2 范数除以 BF16 输出的 L2 范数。只有候选误差严格下降，且相对降幅达到 `min_relative_improvement`（默认 0），才提议变更。一次 outlier 命中可以走到拟合，也可能在留出比较时被淘汰。

## 只在低精度范围内扩大 SmoothQuant 覆盖

[原生训练配置](../../configs/grpo_dynamic_smoothquant.json) 的关键字段如下。四层 `down_proj` 自首批起都运行 MXFP8，其中三层已有 S。每跨过 20 次优化器更新，learner 采集这四层的输入。

```json
{
  "recipe_id": "smoothquant-mlp-down-v1",
  "recipe_options": {
    "precision": "mxfp8",
    "layers": [7, 15, 20, 25],
    "smoothquant_layers": [7, 15, 20],
    "allow_layer_expansion": true
  },
  "recipe_calibration": {
    "refresh_every_optimizer_steps": 20,
    "discover_outliers": true,
    "approval": "manual",
    "max_activation_relative_l2": 0.05
  }
}
```

`calibration.jsonl` 中的 `action` 给出每层的比较对象。

| action | 例子 | 当前执行方式 | 候选通过条件 |
| --- | --- | --- | --- |
| `refresh` | 第 7、15、20 层 | MXFP8 + 旧 S | 新 S 比旧 S 的误差低，且达到要求的相对降幅。 |
| `activate` | 第 25 层 | 直接 MXFP8 | 先发现 outlier。候选 S 必须比直接 MXFP8 误差低，达到要求的相对降幅，且通过首次启用误差上限（如果设置）。 |

这里的 0.05 将**首次启用**后的相对 L2 误差限制在 5% 以内。默认不设上限。第 25 层获批并发布后，从下一批开始使用 MXFP8 + SmoothQuant。其余 BF16 层从观察到发布都不在此次变更范围内。[逐批行为](../../design.zh-CN.md#smoothquant-expansion)

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

`runs/sq-review/calibration.jsonl` 记录证据；`summary.json` 提供 `candidate_id` 和 `checkpoint`。待确认时训练退出、释放 worker。在 checkpoint 的 `candidate.json` 中查看 `evidence` 和 `candidate`。后者是 SmoothQuant 提案，`candidate.scales` 按层列出拟应用的平滑因子。用新的输出目录继续。

```bash
CP=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["checkpoint"])')
ID=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["candidate_id"])')
uv run qorerl train --config configs/grpo_cpu_smoothquant.json \
  --resume "$CP" --accept-candidate "$ID" --out runs/sq-accepted
```

拒绝时把参数改为 `--reject-candidate "$ID"`。接受时保留模型参数、optimizer 和更新计数，发布新的平滑因子。所有 rollout 确认后，才用新版本生成下一批。拒绝时继续使用原来的 S。可在 `trajectories.jsonl` 的 `policy_version` 中核对实际采样版本。每次应用前都有恢复点，失败时查看 `incident.json` 中的路径。需要重做决策时，可从变更前的 checkpoint 开始另一个输出目录。

## 导入已有的平滑因子

外部工具为每个提供的目标层生成一个有限、严格为正的 FP32 向量，长度等于该层的输入通道数。启动导入仅接收初始 `smoothquant_layers` 内的条目，可以只提供其中一部分。缺省的初始 S 填 1；低精度范围内其余层直接运行 MXFP8，范围外走 BF16。下例将已校准的 `smoothing_factors` 封装为可导入 JSON。

```python
import json
from pathlib import Path

from qorerl.recipes.smoothquant import make_smoothing_artifact
from qorerl.training.config import TrainConfig

cfg = TrainConfig.load("configs/grpo_dynamic_smoothquant.json")
# smoothing_factors["model.layers.7.mlp.down_proj.weight"] 是校准得到的一维 CPU float32 Tensor
artifact = make_smoothing_artifact(
    smoothing_factors,
    model=cfg.model,
    model_revision=resolved_model_revision,  # 此次启动会使用的模型 commit
    recipe_id=cfg.policy_recipe_id,
)
Path("configs/smoothing-factors.json").write_text(json.dumps(artifact))
```

设置 `recipe_calibration.initial_artifact` 为 `"smoothing-factors.json"`。旧配置也可沿用 `scale_artifact`。相对路径从配置文件所在目录解析。SmoothQuant 文件沿用 schema 1，`scales` 和 `scale_digest` 分别保存逐通道平滑因子及其内容摘要。导入检查模型、版本、recipe 选层、目标名称、通道数和内容摘要。使用本地模型目录时，请保留校准对应的模型快照；目录名本身无法标识其内容。checkpoint 独立保存所有生效向量，正常续训时无需再次读取外部文件。

## 原生运行入口与验证范围

```bash
uv run qorerl train --config configs/grpo_dynamic_smoothquant.json --out runs/sq-native
```

[原生配置](../../configs/grpo_dynamic_smoothquant.json) 需要支持原生 MXFP8 的 Megatron/TE 与 vLLM 环境。安装前提见 [MXFP8 指南](mxfp8-hadamard.zh-CN.md)。vLLM 从启动起为整个低精度范围建立量化权重缓存，发布 S 时重新打包 `W*S`，量化器在打包时计算各量化块的 scale。GPU 验收要检查两端相同 token 的 log-prob、refit 后输出、学习曲线和校准开销。如需 NVFP4，在同一通用 recipe 下将 `precision` 改为 `"nvfp4"`。NVFP4 路径要求两端都安装 Transformer Engine 2.18.x，并使用支持该内核的 SM10.x GPU。learner 和 vLLM worker 都用 TE 的 NVFP4 量化与 GEMM；vLLM 在权重发布后缓存已打包的权重。候选比较也使用 NVFP4 对照。两种格式的原生 GPU 执行与端到端效果均待验证，CPU 流程示例只模拟 MXFP8。[训练与打包实现](../../qorerl/recipes/te_nvfp4.py)
