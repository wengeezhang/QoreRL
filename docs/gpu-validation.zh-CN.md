# GPU 实验准备与冒烟验收

[English](gpu-validation.md) · **简体中文**

这些工具用于准备实验、检查已经保存的证据。代码经过本地测试，**原生 GPU 验证仍待进行**。冒烟验收通过只说明限定范围内的执行检查通过，不代表恢复正确、训练收敛或 SmoothQuant 带来了收益。

## 硬件与配置

| 实验 | 配置 | 当前适配器要求 |
| --- | --- | --- |
| 单卡 BF16，learner 与 rollout 共置 | [`grpo_single_gpu.json`](../configs/grpo_single_gpu.json) | H200 或 B200；实际显存共存仍需测量 |
| 单卡 MXFP8 + 动态 SmoothQuant | [`grpo_single_gpu_smoothquant.json`](../configs/grpo_single_gpu_smoothquant.json) | 受支持的 Blackwell SM10/SM11、TE 2.18.x；首选验证目标为 B200 |
| 实验性 NVFP4 + SmoothQuant | 将 `recipe_options.precision` 改为 `nvfp4` | 受支持的 Blackwell SM10.x、TE 2.18.x；需要独立验收 |

H200 属于 Hopper。支持 FP8 不等于支持当前适配器的原生 MXFP8/NVFP4 路径。[NVIDIA 支持矩阵](https://docs.nvidia.com/deeplearning/transformer-engine/support_matrix.html)。依赖和后端环境隔离见[执行指南](ray-grpo.md)。

两份单卡配置均使用 Qwen3-1.7B、一个 rollout worker、microbatch=1。共置仅安排两个进程使用同一张卡，不划分显存。learner 常驻，vLLM 显存比例为 0.2。SmoothQuant 配置中 `[7,15,20,25]` 从开始就使用 MXFP8，`[7,15,20]` 先叠加 S，每 20 次**实际 optimizer 更新**评估一次，人工批准。扩展仅限已启用低精度的范围；其他 BF16 层保持原状。

先完成 1.7B BF16，再做低精度实验；实测显存与正确性后再考虑 4B。配置文件本身不是显存足够或能够加速的证据。

## 运行前检查

在执行后端的环境中运行：

```bash
uv run qorerl gpu-preflight --config configs/grpo_single_gpu.json \
  --out results/preflight-bf16
uv run qorerl gpu-preflight --config configs/grpo_single_gpu_smoothquant.json \
  --out results/preflight-smoothquant
```

两个后端环境不同时，分别在对应环境使用 `--role learner`、`--role rollout`。默认 `--role all` 检查本机可见 GPU 是否满足整个布局，不会远程检查 Ray 节点。工具记录依赖版本，低精度模式还检查适配器的硬件和 TE API 条件；不加载模型，不运行前反向，也不创建云实例。

输出到一个**新目录**中的 `report.json` 和 `report.md`。退出码：`0` 为 passed，`1` 为 failed，`2` 为 incomplete。依赖存在不等于二进制兼容。

## 训练、查看与确认

```bash
uv run qorerl train --config configs/grpo_single_gpu_smoothquant.json --out results/sq-review
uv run qorerl inspect --run results/sq-review
```

出现人工候选时，保存检查点和候选后结束本段训练，状态为 `awaiting_review`。阅读证据后，使用 `summary.json` 中的 checkpoint 和 candidate_id：

```bash
uv run qorerl train --config configs/grpo_single_gpu_smoothquant.json \
  --resume CHECKPOINT_PATH --accept-candidate CANDIDATE_ID --out results/sq-accepted
uv run qorerl inspect --run results/sq-accepted --related-run results/sq-review
uv run qorerl report --run results/sq-accepted --related-run results/sq-review \
  --out results/sq-accepted/calibration.html
```

展示页直接读取 `calibration.jsonl`，不会再次采集或回放模型。每次评估单独展示同一权重版本下的成对误差和决策记录。`--related-run` 显式关联之前的训练段，不会自动跟随日志里的检查点路径。旧记录没有 evaluation ID 时，可通过本段相同边界和 candidate ID 关联。可以导出当前窗口的 SVG、PNG，以及包含数据和选择状态的 HTML。

没有候选也是有效实验结果。某层存在 outlier，并不意味着新拟合的 S 一定更好。`no_gain`、未选中和未比较的层都会保留。校准耗时包含采集、拟合与比较，不是算子延迟；不能据此宣称 FLOPs 减少或训练加速。

## 验收已完成的原生训练段

在评判结果之前，根据匹配的 BF16 参考实验和预定数值预算，选定最大 token log-prob 差异阈值。工具没有通用默认值。下面的 `MAX_LOGPROB_GAP` 需要先按实验要求设置：

```bash
uv run qorerl validate-run --run results/sq-accepted \
  --related-run results/sq-review --out results/sq-acceptance \
  --min-updates 2 --max-logprob-gap "$MAX_LOGPROB_GAP" \
  --check-checkpoints --require-calibration
```

BF16 基线去掉 `--related-run` 和 `--require-calibration`。

| 检查 | 所需证据 |
| --- | --- |
| 执行和部署 | 原生后端、训练段完成、worker 的节点/GPU 分配、无 incident |
| 环境和显存 | 实际 learner 与 vLLM 执行进程的 CUDA、设备、依赖版本和 allocator 峰值 |
| 有效训练 | 有限的 loss/梯度诊断，非零组内优势，**本段内**至少指定次数的实际 optimizer 更新 |
| 同快照概率差异 | 第一个更新 epoch 的采样 token log-prob 平均及最大绝对差；最大值满足用户阈值 |
| 发布和采样 | 每次发布得到所有 rollout 副本一致确认；后续 batch 使用之前最后确认的策略；保存的 log-prob 有限 |
| 检查点内容 | 初末模型张量有限、确实发生变化，更新计数增长且存在 optimizer 状态 |
| 可选校准链路 | 关联到成对候选证据，在权重版本不变时应用新 recipe 身份，后续采样使用该身份 |

`--check-checkpoints` 用 `weights_only=True` 在 CPU 上加载两个本地检查点，主机内存需要容纳两份及反序列化开销。不提供该选项或 log-prob 阈值，结果为 incomplete。CPU 模拟不能通过原生验收。全零优势可能增加 batch/weight version，却不会更新 Adam，不能算有效更新。若算术任务没有奖励差异，需要调整任务/数据实验重跑，不能仅凭 completed 算通过。

再次暂停等待人工审核时，这一段尚未完成。每段恢复后的运行都需要自己达到更新要求；关联历史段仅提供候选来源，不合并更新次数。完整验收拒绝超过 16 MiB 的 JSONL 和未结束的最后一行；可拆成更短的实验段。实时展示则读取有界尾部，并提示历史可能缺失。

## 后续仍需人工实验验证

- GPU 上的中断恢复与连续执行对照；注入发布故障，确认部分更新的副本停止服务。
- refit 后独立检查实际输出、加载的张量内容；协议确认并非独立测量。
- EOS、截断、不同长度和 microbatch 下的梯度归一化；采样 token 概率差不等于全词表 KL。
- 负载下整卡的显存峰值。已记录的是各进程的 **PyTorch allocator** 峰值，不包含所有原生分配，也不能相加当作同时刻整卡峰值。
- 相同工作负载下的 BF16、直接 MXFP8、固定 S 和动态 S 对比，多随机种子的留出任务评估，以及校准/发布开销和达到同等质量的成本。

验收报告会列明这些边界，也不会修改 `summary.json` 中的 `native_gpu_acceptance`。完整 GPU 验证和 recipe 收益需要分别确认。
