# QoreRL 第一版：每 20 步校准，只恢复 Wgrad 的 BF16 激活

本实验实现讨论中的方案，不以理论上的“有益无害”为前提。触发后实际应用干预，并如实记录改善、无变化或恶化。

## 本版做了什么

单进程、单设备运行一个真正包含稀疏专家路由的 tiny Transformer：2 个 block，hidden=32，4 个 attention head，每层 4 个 SwiGLU 专家、top-2 路由。任务是读入 `[BOS, a, b, SEP]`，生成两个 token `[a, b]`。每个位置答对奖励 0.5；两个都答对奖励 1。

默认先用 BF16 做 5 个 SFT warmup step，构造共享的起始 checkpoint；然后各实验组独立做 60 个 GRPO 风格的 RL update。Warmup 不计入 RL 的 20 步校准周期。没有下载外部 checkpoint，也没有引入 Ray、Megatron、vLLM 或 SGLang。

这是机制验证用的小模型，不是已有能力的大模型；复制任务不能证明方法对数学、代码或长链 RL 的收益。

## 最重要的语义：只替换一个 Wgrad 操作数

设 `X` 是进入 Linear 量化器前的 BF16 激活，形状 `[M,K]`，权重 `W` 为 `[N,K]`，传回的梯度 `G` 为 `[M,N]`。

```text
前向：    Q_K(X) @ Q_K(W).T
Dgrad：   Q_N(G) @ Q_N(W)
Wgrad：   Q_M(G).T @ Q_M(X)

触发恢复后，只有最后一行变成：
Wgrad：   Q_M(G).T @ X
```

`Q_K`、`Q_N`、`Q_M` 表示沿该次 GEMM 的归约轴，以 32 个元素为一组量化。前向行量化与反向列量化都从原始 BF16 操作数独立产生，不把前向已经量化的激活转置后再次量化。

特别注意：

- `X` 是**当前 MXFP8 前向路径中，该 Linear 在自身量化前保存的激活**。它包含上游低精度执行造成的影响。
- 不把全 BF16 参考路径中的该层激活移植过来。
- 不把量化后的值简单 `.to(bfloat16)` 当作恢复了丢失的信息。
- G 保持当前路径及 Wgrad 的原有量化值；不将 G 也恢复到高精度。
- 原型中，Wgrad 的 GEMM 用 FP32 承载这些操作数；G 的数值仍然是量化后的值。这不意味着硬件能直接做 MXFP8 × BF16 的混合输入 Tensor Core GEMM；原生实现需要另行选择或实现对应路径。
- 该层前向、Dgrad、路由和 loss 的定义保持不变。
- 用自定义 autograd Function 明确控制三个 GEMM；不是对普通 fake-quant 的 STE 进行猜测。

当前干预粒度是一个 `PrecisionLinear.weight`，例如 `blocks.0.moe.experts.2.down.weight`，不是整个 Transformer block。

## 每 20 步怎么执行

第 20 次 RL optimizer update **之前**，权重是完成第 19 次 update 后的权重：

1. 按当前 rollout 前向方案采样，固定本批 token、old log probabilities、rewards 和组内 advantages。
2. 不更新参数，运行 A：全部可量化 Linear 使用 BF16，得到参考梯度。
3. 同一快照、同一批数据运行 B：纯 MXFP8 前向和反向。
4. 对每个可量化 Linear 的整个 weight 梯度展平，计算余弦相似度。默认 `cosine < 0` 且两个梯度范数均大于 `1e-8` 时触发。
5. 运行 C：仍是 B 的前向与 Dgrad，只有命中层的 Wgrad 使用保存的 BF16 激活。
6. 记录 C 与 A 的梯度夹角；检查 B/C logits 必须逐值相同，未选中参数的梯度必须逐值相同。
7. 用 C 的梯度执行本次唯一一次真实 optimizer update。

第 40、60 步重复。已经命中的层在本次 run 后续步骤中持续使用 BF16 Wgrad 激活；不会自动撤销。后续校准中的 B 仍然是纯 MXFP8，以避免上次的干预掩盖原始差异。

采用 AdamW、FP32 master parameters/optimizer states 和梯度裁剪。两次探测都不修改 optimizer 状态；也不会把 BF16 参考梯度直接写回实际训练。

零优势批次、梯度近零、专家未被路由到、两条路径专家活动不一致，会单独记录；不把缺失梯度随意当成方向相反。

## 同时做两种验证

### 数值与单步更新

每个校准点保存三个路径的逐层梯度统计，包括 norm、cosine 和触发/持续恢复名单。

默认还在私有模型和**同一份当前 AdamW 状态的副本**上分别尝试 B/C 的一步更新，测量：

- 更新后，原 MXFP8 前向路径的 policy loss；
- 更新后，BF16 参考路径的 policy loss；
- 参考梯度与实际 AdamW 参数更新的点积；
- 更新幅度与裁剪前梯度范数。

这些试更新不影响真实模型、真实 optimizer 或 RNG。它们是固定数据上的 surrogate loss 诊断，不等同于后续 reward 或最终收敛保证。可用 `--no-trial-updates` 关闭。

### 完整训练对照

每个 seed 都从同一个 warmup checkpoint 启动：

| 实验组 | 前向 | 反向 |
|---|---|---|
| `bf16` | BF16 基线 | BF16 基线 |
| `mxfp8` | MXFP8 数值仿真 | 固定 MXFP8 数值仿真 |
| `adaptive` | 与 MXFP8 组相同的方案 | 每 20 步检查并干预选中层的 Wgrad 激活 |

各组共享 prompt 随机序列、起始采样随机种子和固定评测题；各自随着权重更新从自己的 policy 采样，因此后期生成序列可以不同。只有校准点内部的 A/B/C 是严格固定同一份采样数据的配对对照。

日志同时包含 sampled reward、exact match、非零优势比例、entropy、ratio、clip fraction 和固定评测题的 greedy 正确率。当前没有 reference KL penalty、router load-balancing auxiliary loss、EOS 提前停止、KV cache 或多轮 PPO epoch；每个采样 batch 更新一次。组内中心化优势在 ratio=1 时会使训练前 policy loss 接近零，这是正常的，不能用这个零值判断是否在学习。

## 本机能验证什么，不能验证什么

后端明确标为 `mxfp8_emulated_fp32_gemm`：

- E4M3FN 最近偶数舍入，含非规格化数；每 32 元素共用一个 E8M0 幂次 scale。
- scale 使用 `2 ** ceil(log2(amax / 448))`，指数限制为 `[-127,127]`；全零组保持零。
- 将模拟量化的值还原为 FP32 来做 GEMM；BF16 操作数和输出通过 BF16 round-trip 表示。
- FP32 累加、FP32 optimizer states；norm、softmax 和 attention 内部采用高精度运算。
- QKV/out projection 和专家 gate_up/down 纳入 MXFP8；router 和 lm_head 保持 BF16 Linear。
- 专家的 token 数可变化；量化函数对不满 32 的尾块补零再裁掉，不把 padding 当作实际 token。
- 不支持原生 MXFP8 kernel，也不声称仿真与 TE 全部边界数值 bitwise 一致。CUDA 设备上运行这个后端，仍然是仿真。
- 为了验证干预，prototype 保存了 BF16 激活，未优化显存；运行时间不是原生低精度加速比。

方向与缩放设计参考 [NVIDIA MXFP8 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)。原生 Blackwell/TE 集成必须另外验证量化字节、归约轴、保存张量、累加与转换行为，然后重跑训练对照。

## 运行

```bash
cd /Users/bytedance/projects/QoreRL
uv sync --group dev
uv run pytest -q

# 三个随机种子、三个实验组；校准发生在 RL step 20/40/60。
uv run qorerl --steps 60 --seeds 0,1,2 --out results/my-first-study

# 快速检查接口；这不是默认 20 步周期的正式对照。
uv run qorerl --steps 2 --warmup-steps 1 --calibration-interval 2 \
  --prompts 2 --group-size 4 --arms adaptive --out results/my-smoke-test
```

输出目录不能已存在，防止无意覆盖旧结果。`--cosine-threshold` 默认 0，可设为负数收紧触发范围；不能设正数冒充“方向相反”。

## 结果文件

```text
results/my-first-study/
  config.json             环境、模型和全部运行参数
  summary.json            所有 seed/arm 汇总
  report.md               可读汇总
  seed_0/
    initial.pt            各组共享的起点
    warmup.jsonl          独立记录的 SFT warmup
    adaptive/
      steps.jsonl         每个 RL update 的指标
      calibrations.jsonl  A/B/C 逐层指标、命中层和试更新结果
      summary.json
      final.pt            模型、optimizer、override 名单、配置与 RNG 状态
```

当前 CLI 没有自动 resume 命令；checkpoint 是实验复查和后续开发的材料。

## 如何读结论

- 先确认 20/40/60 的生命周期真的发生了，且前向与 Dgrad 未被偷偷改动。
- 再看自然训练是否出现反向梯度。若没有，结论是“这次没有触发”，不能声称自动恢复已经提高训练效果。
- 如果出现，分别报告：夹角有没有恢复、单步 loss 怎么变、最后 reward 怎么变。三者不能互相代替。
- 若回退没有改善，也保留记录。当前策略按方向检测触发直接应用，不事后挑选看起来成功的干预。
- 单测有显式的故障注入来覆盖触发分支；故障注入不会进入任何正式训练 run。

源码入口：`qorerl/precision.py` 控制算子数值；`qorerl/calibration.py` 做 A/B/C 对照；`qorerl/experiment.py` 控制每 20 步的生命周期。
