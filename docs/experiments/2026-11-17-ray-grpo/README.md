# Ray GRPO 本地链路验证

2026-11-17，在 macOS ARM64 / Python 3.12.13 / PyTorch 2.15.0 / Ray 2.60.0 上运行：

```bash
uv run qorerl train --config configs/grpo_cpu_test.json --out results/ray-grpo-local-v1
```

这是 `torch_test` CPU 工程测试，不是 Megatron/vLLM 的 GPU 验证，也不是低精度收益实验。

| 检查项 | 结果 |
| --- | --- |
| 执行进程 | driver + 1 learner + 2 rollout actors，PID 互不相同 |
| 训练进度 | 40 个完整采样/更新/同步循环，40 次真实 AdamW 更新 |
| 每步样本 | 8 组 × 8 个回答，组内计算 advantage |
| 权重发布 | 初始化版本 0，随后发布到版本 40 |
| 前 10 步平均采样 reward | 0.0890625 |
| 后 10 步平均采样 reward | 0.23125 |
| 各步 `abs_log_ratio_mean` 的最大值 | 约 1.15e-7 |
| checkpoint | step 0、20、40 |

reward 来自每步训练采样，不是独立测试集，不据此声称泛化或性能优势。任务只生成一个数字 token，因此 `finish_reason=length` 很常见；它不表示模型推理被截断导致训练失败。单 epoch、训推模型相同时，更新前 ratio 接近 1，组内 advantage 中心化使 loss 数值接近 0，但梯度并不为 0。

本目录保留当次运行的 [配置](config.json)、[逐步指标](metrics.jsonl)、[进程信息](workers.json)、[汇总](summary.json)。汇总中的 checkpoint 绝对路径是运行机器上的位置，大型 checkpoint 与完整轨迹位于本地 `results/ray-grpo-local-v1/`，不纳入 Git。

另外，`tests/test_ray_training.py` 使用真实 Ray 多进程比较连续 3 步训练与“训练 2 步、保存、重启并训练到第 3 步”：最终模型参数和 Adam 状态逐值一致。独立测试还覆盖缺失/重复/陈旧轨迹拒绝、response/EOS mask、不同大小 microbatch 的聚合、全零优势跳过动量更新，以及 adapter glue 的失败不确认版本逻辑。adapter glue 测试使用 mock，并不代表 native kernels 已运行。

本次验证命令：`uv run pytest -q`，32 passed；`uv run ruff check qorerl tests`，通过；`git diff --check`，通过。旧研究实验的测试包含在这 32 项中。

下一轮验证应在双 NVIDIA GPU 环境执行 `configs/grpo_megatron_vllm.json`，尤其核对 BF16 log-prob 差异、Bridge 导出与 vLLM refit、有效非零优势组和 checkpoint 恢复。细节见 [执行说明](../../ray-grpo.zh-CN.md)。
