# 2026-11-08：tiny MoE Wgrad 激活恢复实验记录

## 已验证的范围

在 Apple Silicon CPU 上，完成了单进程 tiny MoE 的 GRPO 风格训练以及周期校准闭环。模型 59,296 个参数，2 层、每层 4 专家、top-2。全部 MXFP8 结果来自数值仿真，**不是 Blackwell/TE 原生训练结果**。

两轮实验均使用 seeds 0/1/2，每个 seed 有 BF16、固定 MXFP8、adaptive 三组。没有向正式训练注入异常，没有为触发干预而调成正余弦阈值；触发条件始终是 cosine < 0 且两侧 norm > 1e-8。

## 第一轮：40 步 SFT warmup，60 步 RL

```bash
uv run qorerl --steps 60 --warmup-steps 40 --seeds 0,1,2 \
  --out results/wgrad-cpu-20261108
```

每个 adaptive run 在 RL step 20/40/60 校准，合计 9 次。

这轮 warmup 后，固定题集的 greedy token accuracy 已经是 100%，不适合充分观察学习曲线；采样奖励仍有提升空间。

自然训练中只出现一次触发：seed=0，step=40，`blocks.1.moe.experts.2.gate_up`。

| 指标 | 干预前 | 只恢复 Wgrad BF16 激活后 |
|---|---:|---:|
| 与 BF16 参考的逐层梯度 cosine | -0.11675442 | -0.11680047 |
| 该层梯度 norm | 1.02580129e-5 | 1.02569306e-5 |
| 全模型梯度 cosine | 0.99565213 | 0.99565213 |
| 私有 AdamW 一步更新后的 MXFP8 policy loss | -0.004993015 | -0.004993015 |
| 私有 AdamW 一步更新后的 BF16 policy loss | -0.006765507 | -0.006765507 |

该层 BF16 参考梯度 norm 为 1.37021896e-6。这里不能把负余弦直接解释为整个训练发生严重错误：该层梯度很小，全模型方向仍高度接近。干预没有把局部方向转正，也没有在打印精度下改变试更新后的 loss。

保存的 [完整触发事件](saturated-trigger-events.json) 包含全部逐层统计与试更新结果；[汇总](saturated-report.md)、[配置](saturated-config.json)、[原始汇总 JSON](saturated-summary.json) 同时保留。

## 第二轮：5 步 SFT warmup，100 步 RL

为观察实际学习，从较少 warmup 的同一类 checkpoint 起跑，没有更换模型或修改阈值：

```bash
uv run qorerl --steps 100 --warmup-steps 5 --seeds 0,1,2 \
  --out results/wgrad-learning-20261108
```

每个 adaptive run 在 step 20/40/60/80/100 校准，合计 15 次。

| Seed | BF16 初始 → 最终 greedy token accuracy | MXFP8 初始 → 最终 | Adaptive 最终 | 自然触发 |
|---|---|---|---:|---:|
| 0 | 0.2383 → 0.9102 | 0.2266 → 0.8789 | 0.8789 | 0 |
| 1 | 0.1289 → 0.6602 | 0.1328 → 0.6484 | 0.6484 | 0 |
| 2 | 0.1367 → 0.9648 | 0.1641 → 0.9570 | 0.9570 | 0 |

各组使用相同的起点权重；初始分数略有差异来自前向精度不同。评测题固定但不是独立 held-out 泛化任务。

三个 seed 在全部校准点上的最小有效逐层 cosine 分别为 0.66350、0.93921、0.92670，因此没有命中“方向相反”条件。没有为了演示自动恢复而强制回退。

另外逐张量核对了这轮固定 MXFP8 与 adaptive 的最终 checkpoint：三个 seed 的所有参数完全相同。这验证了**没有触发时，额外 BF16 校准与私有试更新不会偷偷改变实际训练**。

见 [完整汇总](learning-report.md)、[配置](learning-config.json)、[原始汇总 JSON](learning-summary.json)。完整逐步日志与 checkpoint 位于本机 `results/wgrad-learning-20261108/`。

## 实现检查

`uv run pytest -q`：11 项通过。覆盖 E4M3 舍入和 subnormal、行/列量化区别、非 32 整除的尾块、Wgrad-only 对 Fprop/Dgrad 的隔离、校准期间模型/Adam/RNG 不变、缺失和近零梯度处理，以及实际第 20/40 步的触发周期。

触发控制逻辑另有明确标注的故障注入单测。该单测只证明程序分支可运行，不作为训练效果证据。

## 当前结论

已经验证训练、周期对照、自动触发和 Wgrad-only 干预的实现闭环。尚未证明这种修复能提高学习质量：第一轮那次自然触发未改善局部梯度方向；第二轮没有触发，无法评价修复收益。

这些结果也不能据此否定方法在更大模型或原生 kernel 中的价值。下一阶段应先让原生 MXFP8 路径与此处的操作数、保存张量语义对齐，再做更有区分度的学习任务。
