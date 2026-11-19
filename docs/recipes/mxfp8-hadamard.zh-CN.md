# 第一个实验 recipe：MXFP8 MLP down projection + 固定 Hadamard

本 recipe 要回答一个具体问题：**在相同的 MLP `down_proj` 量化范围下，加入固定 Hadamard，是否比直接 MXFP8 更适合 RL 训练？**

当前已经实现原生执行适配、配置、状态保存和本地测试；**尚未在 NVIDIA GPU 上运行验证，也没有训练收益或加速结论**。先用 BF16 对照验证整个训练链路，再验证原生 MXFP8，最后比较 Hadamard 的收益与成本。

## 1. 三组配置，只改变一个因素

| 配置 | MLP down projection | 其余计算 |
| --- | --- | --- |
| [`grpo_megatron_vllm.json`](../../configs/grpo_megatron_vllm.json) | BF16 | BF16 基线 |
| [`grpo_mxfp8_mlp_down.json`](../../configs/grpo_mxfp8_mlp_down.json) | MXFP8 | 同 BF16 基线 |
| [`grpo_mxfp8_hadamard_mlp_down.json`](../../configs/grpo_mxfp8_hadamard_mlp_down.json) | 配套旋转输入和权重，再做 MXFP8 | 同 BF16 基线 |

这三份配置的模型、训练任务、seed、batch、学习率及训练步数相同。默认选择每个 Transformer block 的 `mlp.down_proj`，不是全模型 MXFP8。这里的 MLP 输入具体是 SwiGLU 之后、down projection 之前的激活；不旋转 gate/up projection，不穿过 SiLU，也不改 attention、router 或 `lm_head`。

首版仅接受未量化的 **dense Qwen3**，使用一张 learner GPU 和一张 rollout GPU，TP/PP/CP/EP 为 1。MoE、模型并行、CUDA graph、prefix cache、预量化 checkpoint 均不在当前验证范围内。CPU 测试引擎不能启用这两个 recipe；不支持的配置直接报错，没有模拟量化或 BF16 静默兜底。

配置示例：

```json
{
  "recipe_id": "mxfp8-hadamard-mlp-down-v1",
  "recipe_options": {
    "layers": [0, 1, 2],
    "block_size": 128,
    "seed": 0
  }
}
```

`layers` 从 0 开始，省略则选全部；空列表、重复索引、越界索引都会拒绝。通用 recipe 名 `mlp-down-v1` 和 `hadamard-mlp-down-v1` 可通过 `recipe_options.precision` 选择底层格式，默认是 `mxfp8`，也可选 `nvfp4`。下面的分块、kernel 与验证命令针对 MXFP8。Hadamard 的 `block_size` 为不小于 2 的 2 的幂，必须整除输入通道数；两种 recipe 都要求权重的输入和输出维度为 32 的倍数。**Hadamard 的 block size 与 MXFP8 每 32 个元素共享 scale 是两个不同参数。** `seed` 决定每层固定的 ±1 符号，训练过程中不重新抽样。

## 2. 实际执行的数学变换

沿用 PyTorch 权重布局：输入 `X[M,K]`、权重 `W[N,K]`，普通线性层为 `Y = X @ W.T`。

设 H 是分块归一化 Hadamard，D 是固定 ±1 对角矩阵，R = D H。两端都执行：

```text
X_rot = X @ R
W_rot = W @ R

Y = X_rot @ W_rot.T
  = X @ R @ R.T @ W.T
  = X @ W.T
```

等式描述实数运算中的配套变换。实际实现的蝶形加减用 FP32 累积，结果转回 BF16，再交给原生量化器，因此仍有舍入误差，不能承诺逐位相同或量化误差一定减小。

训练侧保存的 Parameter 和 optimizer 的 FP32 master 都处于**原始坐标系**。旋转是参与 autograd 的计算：`dX_rot`、`dW_rot` 要乘 `R.T` 才返回原始参数坐标系。随机符号与 H 的次序不能混用；`D H` 的逆为 `H D`。

这里没有把参数永久改写成旋转后的 checkpoint，也不把旋转离线穿过 SwiGLU。每次 forward 对当前权重构造旋转表示，避免更新后复用旧权重缓存。

## 3. 哪些计算真正使用低精度

| 阶段 | 当前执行 |
| --- | --- |
| canonical 参数、层输入/输出 | BF16；optimizer master、状态和跨 microbatch 累积为 FP32 |
| 选定层 Fprop | TE 原生 MXFP8 GEMM：E4M3 元素、每 32 元素 E8M0 scale |
| 选定层 Dgrad / Wgrad | TE 原生 MXFP8 GEMM；返回 BF16 梯度，Hadamard backward 映射回原坐标系 |
| TE 行/列表示 | 从 BF16 分别量化，不能用行表示简单转置代替列表示 |
| 权重导出 | Bridge 导出原始坐标系 BF16 权重，另带 recipe 状态 |
| rollout 选定层 | vLLM 的 `FlashInferCutlassMxfp8LinearKernel`；原生输入量化和 MXFP8 GEMM |
| rollout 权重缓存 | 从 canonical BF16 配套旋转、量化，再由 native kernel 完成后处理 |

本实现使用[两个格式共用的自定义 autograd](../../qorerl/recipes/te_block_linear.py) 调用 TE 的 Fprop/Dgrad/Wgrad；它是低精度训练的反向估计规则，不是对离散舍入函数求出了非零的精确导数。它没有加入 BF16 Wgrad 修复、每 20 步校准或自动选择敏感层。

训练侧 token 行数 M 可以不是 32 的倍数：GEMM 内部把 X 和对应的上游梯度补零，输出和 Dgrad 去掉补齐行。这些内部行不成为轨迹、不进入 reward 或 loss。通道维度不偷偷补齐；不满足要求时拒绝执行。

TE 与 vLLM 使用各自的原生量化实现。相同格式、相同旋转和相同 canonical 权重，**不等于两端量化字节、scale 或 log-prob 已经对齐**。训推差异必须测量，不能由 recipe 名称推断。

## 4. 同步、恢复与报告

`recipe_id` 在配置里是可读的 preset 名称；实际 `PolicyVersion.recipe_id` 包含规格的 hash，覆盖方法版本、作用域、选层、block size、seed。改变旋转配置会改变策略身份。

checkpoint 和 WeightSnapshot 另存实际 canonical target 名称、shape 和固定 signs。接收端先核对这些状态，再加载权重；恢复时也先核对，再修改模型。只存 seed、不核对实际变换是不够的。

每次 refit 的顺序为：

```text
校验模型、recipe 身份、目标形状和固定 signs
    → 使当前 rollout 版本及派生缓存不可服务
    → 使用原生 load_weights 加载 canonical BF16
    → 检查参数加载覆盖
    → 重建旋转/量化/kernel layout 缓存
    → CUDA synchronize
    → 确认新版本，允许下一批采样
```

派生缓存不是模型的注册参数，不会被 Bridge 当成原始权重再次导出，也不会在下一次同步时重复旋转。vLLM 在加载模型后、显存 profiling 和 KV 分配前安装 adapter，以便 profiling 看见该执行路径。

同配置 checkpoint/resume 已接入 recipe 状态检查。**恢复时切换 recipe、改 block size 或 seed 仍不支持**；这需要未来的显式迁移流程。

运行会生成：

- `recipe.json`：解析后的规格及身份。
- `workers.json`：启动时各引擎的 backend、目标层、版本和验证状态。
- `metrics.jsonl`：原有 reward/loss/耗时，加上首次 optimizer update 前的 `pre_update_ratio_mean`、`pre_update_abs_log_ratio_mean`、`pre_update_clip_fraction`。它们使用本轮同一权重快照和已采样 token，不混入后续 epoch 的参数更新，但只是均值与比例，不能代替尾部误差检查。
- `summary.json` 的 `execution`：最终引擎执行报告。vLLM 的 apply counter 从最近一次 refresh 计数；最后一次发布后尚未采样时为 0，不代表整次训练没执行该算子。

## 5. 运行方式与 GPU 验证顺序

接口按下列上游版本编写，**不是已经 GPU 实测通过的环境组合**：

- Megatron Bridge 0.6.2 / Core 0.19.2。
- Transformer Engine **2.18.x**；适配器检查版本及 `general_gemm` ABI。
- vLLM **0.32.0**；适配器检查版本及 FlashInfer CUTLASS MXFP8 的支持条件。
- 两端共同范围收窄到支持该 kernel 的 **SM10 系列 NVIDIA GPU**，例如 B200。不能把所有名字带 Blackwell 的 GPU 都视为支持。

先在配置好 CUDA 依赖的环境中 `uv pip install -e .`。CPU 开发用的 `uv.lock` 不锁定 CUDA 栈；`uv sync` 也不是 native 环境安装器。允许通过两个 Ray runtime env 分开 learner/rollout 环境，方法见[开发指南](../ray-grpo.zh-CN.md)。

先测单算子：

```bash
python -m pytest -q tests/test_te_mxfp8.py tests/test_hadamard.py
python -m qorerl.recipes.compare --out results/mxfp8-operator.json
```

probe 固定输入、canonical 权重与 MSE 目标，比较 BF16、MXFP8、MXFP8+Hadamard 的输出/Dgrad/Wgrad 误差、Wgrad 余弦、前后向耗时及额外已分配显存。它不会退回 CPU，也不会自动判断哪组通过；这只是算子实验，不是 GRPO 收益证据。

probe 的显存值是预热后每轮新增的 PyTorch allocated 峰值，不包含预热已保留的 signs/workspace、allocator reserved 空间或非 PyTorch CUDA 分配，不能当作完整显存占用比较。

再按顺序运行三份端到端配置：

```bash
qorerl train --config configs/grpo_megatron_vllm.json --out results/grpo-bf16
qorerl train --config configs/grpo_mxfp8_mlp_down.json --out results/grpo-mxfp8
qorerl train --config configs/grpo_mxfp8_hadamard_mlp_down.json --out results/grpo-mxfp8-hadamard
```

先确认有效更新、两次以上 refit 和恢复行为，再做学习质量比较。短的 10 步加法任务只用于 smoke test；相同 seed 下不同策略也可能生成不同轨迹，不能把三次 reward 的一次差值当成精度效果。

验收还需要同 snapshot、同 token 的配对 log-prob 检查、非有限值与误差尾部、原生训练梯度/参数检查、多个 seed 的训练质量，以及达到同等质量所需的完整时间与成本。`native_gpu_executed=true` 只记录执行事实；不会自动设为质量验收通过。

## 6. 开销与后续工作

当前优先让执行路径可核对：Hadamard 使用 PyTorch 蝶形运算，没有融合 CUDA kernel；learner 每次 forward 旋转并量化当前权重。vLLM 同时保留 BF16 canonical 权重及派生量化缓存，refresh 构建新缓存期间还可能同时持有旧缓存。因此，这一版**不宣称减少权重显存，也不宣称比 BF16 更快**。

vLLM 的推理 profiling 能看见已安装的持久缓存，但未必覆盖 refit 时的 BF16 传输、旋转临时空间及新旧缓存重叠峰值。示例保留 `gpu_memory_utilization=0.7`；实际模型需测同步峰值并预留空间，不能仅根据推理 profiling 把显存用满。

若验证有数值/学习收益，再优化 FWHT kernel、权重缓存生命周期、同步成本，并扩展模型/作用域。不能先增加 MoE、模型并行或自动修复再把所有差异混在一起评估。

方法来源：[QuaRot](https://arxiv.org/abs/2404.00456) 提供正交旋转的参考，但本 recipe 不是 QuaRot 全模型 PTQ 复现，也没有将其推理实验外推为 RL 训练保证。执行依据为 [TE 的 GEMM 接口](https://github.com/NVIDIA/TransformerEngine/blob/15288857644368b8b7bdddd9bead7d6d3d04de80/transformer_engine/pytorch/cpp_extensions/gemm.py)、[TE Linear 的前后向](https://github.com/NVIDIA/TransformerEngine/blob/15288857644368b8b7bdddd9bead7d6d3d04de80/transformer_engine/pytorch/module/linear.py) 及 [vLLM 0.32 的 FlashInfer MXFP8 kernel](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py)。

## 7. 查看训练内的局部配对观测

可显式启用 [Recipe Insights](../recipe-insights.zh-CN.md)：默认每 20 步轮转观测两层，截取实际输入和 canonical 权重，比较 BF16、MXFP8、MXFP8 + Hadamard 的局部前向误差及 CUDA event 耗时。它不校准、不切换 recipe，也不测完整训练质量；原生 GPU 验证仍待完成。配置、实时页面和 HTML 导出命令见该指南。

![Recipe Insights 合成数据预览，非 GPU 实测](../assets/recipe-insights-demo.png)
