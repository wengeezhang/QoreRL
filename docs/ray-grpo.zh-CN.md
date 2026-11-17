# Ray + Megatron + vLLM：第一条 GRPO 训练链路

本阶段先建立可验证的 BF16 基线。量化 recipe、自动告警后调整 recipe、自动修复都不属于这次的完成项。

**验证状态：本地 CPU 的真实 Ray 多进程训练已运行；Megatron + vLLM 适配器已实现，但尚未在 NVIDIA GPU 上验证。** CPU 测试不会假装调用这两个 GPU 引擎，也不能证明它们的数值一致性、性能或 API 组合在真实环境中全部可用。

## 1. 一次训练怎么走

默认原生场景是 `Qwen/Qwen3-0.6B` 的整数加法。每题生成 4 个答案，判断是否与整数答案完全匹配，正确为 1，否则为 0。模型使用非 thinking 模板，要求只回答整数。默认一个 batch 有 4 道题、16 个回答。

```text
driver / coordinator：不执行模型 forward，不直接取得完整权重
  │
  ├─ RayLearnerWorker → Megatron Core + Transformer Engine，GPU 0
  │    load HF → Bridge 转为 Megatron 模型 → BF16 forward/backward
  │    FP32 master parameters / FP32 跨 microbatch 梯度累积 / AdamW states
  │
  └─ RayRolloutWorker → vLLM，GPU 1
       接收 learner 导出的权重 → load_weights → 确认版本 → 采样
```

启动先加载同一个模型 revision；远程 HF 的 `main` 会解析为具体 commit，再传给两端并记录。learner 导出第 0 版权重，所有 rollout worker 确认后，才开始第一批采样。后续循环为：

1. 用版本 v 的 rollout 引擎采样，记录实际 token、对应的 behavior log-prob、组 ID、样本 ID、权重版本和 recipe ID。
2. coordinator 收齐完整的组，校验版本和输入身份，计算 reward 和组内优势。缺失、重复、版本错误的轨迹直接拒绝。
3. learner 对同一批 prompt + 已采样回答做 teacher forcing，计算当前策略的 log-prob、GRPO loss，反向更新。
4. learner 导出版本 v+1 的权重，所有 rollout worker 完成 refit 并确认版本。
5. 此时没有在途请求，可以保存 checkpoint，随后开始下一批。

这里的权重版本是一次训练 batch 后的发布版本，不等于 optimizer step 数。`update_epochs=2` 会更新两次 optimizer、发布一次；全零优势 batch 跳过 optimizer，但仍发布一个有明确进度的新版本。

## 2. GRPO 的具体定义

每组 rewards 用总体标准差归一化：

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
ratio_t     = exp(current_log_prob_t - behavior_log_prob_t)
term_t      = -min(ratio_t * advantage_i,
                  clip(ratio_t, 1-epsilon, 1+epsilon) * advantage_i)
loss        = mean_over_responses(mean_over_response_tokens(term_t))
```

当前没有 critic，也没有 reference KL penalty，等价于 beta=0 的组内基线 PPO surrogate。一个回答的 advantage 作用于该回答的有效 token。prompt 和 padding 不进 loss，真实生成的 EOS 进入 loss。达到长度限制的回答保留已生成 token，按实际文本判题；没有引入 DAPO 的超长奖励整形。

**分母来自实际 rollout**，没有额外重算训练侧 `old_log_probs`，也没有独立的 TIS 校正。本阶段采样固定为 temperature=1、top_p=1、top_k 禁用，并禁用 HF generation config 带入额外采样设置，避免引入尚未实现的分布修正语义。

两端即使都是 BF16，也可能因算子和批处理差异得到不同 log-prob。日志的 `ratio_mean`、`abs_log_ratio_mean` 用来观察这种差异；它们不保证训练稳定。如果 `update_epochs > 1`，这些日志还包含当前策略更新造成的偏离，不能全归因于训推数值差异。

loss 按回答等权。microbatch 的梯度乘以 `该 microbatch 的回答数 / 整批回答数`，再累积；不能把大小不同的 microbatch loss 不加区分地平均。模型梯度先在模型 dtype 中产生，再转为 FP32 累积到 master gradient；这不等于 backward 的全部计算与梯度存储都是 FP32。FP32 master gradient 在整个 batch 累积结束后裁剪并交给 AdamW。全零优势 batch 不推进 Adam 动量，避免无学习信号时仍因历史动量改变参数。

## 3. 三个边界为什么这样划分

| 边界 | 本次实现 | 后续扩展位置 |
| --- | --- | --- |
| 算法与引擎 | 算法接收轨迹、logits、mask、advantage；不依赖 Megatron 层名称 | 可更换算法或量化 recipe，不重写调度循环 |
| 训练与采样 | `PolicyVersion(weight_version, recipe_id)`，发布后必须收到所有 ack | 量化状态、异步陈旧度、恢复都要明确对应的版本 |
| 权重导出与传输 | Bridge 导出 HF 命名 CPU 张量，经 Ray ObjectRef 交给 rollout | 后续换成 NCCL/IPC/分桶传输，不改变轨迹协议 |

当前只有整批同步，没有 Ray actor 自动重试。更新 optimizer、refit 都不是可以随意重放的幂等操作；中途失败时不会把不明状态当作成功继续采样。

vLLM 的 worker extension 在实际执行模型上调用 `load_weights`，检查返回的已加载参数名是否覆盖模型参数。它是 BF16 首期加载完整性检查，**不是完整的 weight checker**，不证明每个 fused shard 的字节或执行结果都正确。GPU 验证仍须检查实际 refit 后的行为。

## 4. 本地运行

Python 3.12+，使用 uv：

```bash
uv sync --group dev
uv run qorerl train --config configs/grpo_cpu_test.json --out results/ray-local
```

CPU 配置启动 1 个 learner 和 2 个 rollout actor，使用小型 GRU 模型做数字复制。模型是工程测试 fixture，采样、log-prob、反向、AdamW 和 Ray IPC 都是真的，但不是 Megatron/vLLM。

输出目录必须不存在。这个示例默认 40 步，每 20 步存 checkpoint。检查：

```text
results/ray-local/
  config.json                 解析后的配置和模型 revision
  workers.json                driver / actors 的 PID 与 Ray node ID
  metrics.jsonl               loss、reward、ratio、clip、梯度范数和各阶段耗时
  trajectories.jsonl          每条回答的 token / behavior log-prob / reward / advantage / 版本
  checkpoints/step-000000/    采样前的初始安全边界
  checkpoints/step-000020/
  checkpoints/step-000040/
    metadata.json
    learner.pt                模型、FP32 masters、AdamW state、进度、Torch RNG
  summary.json                完成情况；区分 native GPU 路径执行事实与验收结论
  incident.json               仅失败时产生，记录错误和最近完整 checkpoint
```

当前任务的 prompt 和 sampling seed 由 seed、step、group ID 确定，因此无需外部 dataloader 游标。训练禁用 dropout，未引入 scheduler。以后增加数据流、调度器或随机 recipe 时，必须补齐相应 checkpoint 状态，不能继续声称只保存这些字段就能完整恢复。

### 续训

复制配置，把 `steps` 改成 60，作为 `configs/grpo_cpu_continue.json`。`steps` 是训练结束时的总步数，不是额外步数。

```bash
uv run qorerl train \
  --config configs/grpo_cpu_continue.json \
  --resume results/ray-local/checkpoints/step-000040 \
  --out results/ray-continued
```

恢复时新建 actor，加载 learner 的完整训练状态，重新导出并同步权重，再生成新轨迹。续训输出写到新目录，原实验不覆盖。

目前只允许改变总步数、checkpoint 频率、RPC timeout 和 actor runtime environment；学习率、batch、recipe、模型版本等语义变化会拒绝。**这里实现的是同配置恢复；用户修改量化 recipe 后恢复的工作流尚未实现。** CPU 测试检查连续训练与中断续训的最终参数和 optimizer states 逐值一致；GPU 是否能逐值复现需另外测。

当前写出 schema 2 checkpoint：版本字段为 `policy_version.weight_version` / `recipe_id`，master 参数为 `master_parameters`。读取仍兼容旧 schema 1 的字段；只转换字段名称，不改变模型、optimizer 或训练语义。新轨迹日志也写出 schema 2；指标中的 `policy_version` 整数字段改名为 `weight_version`。历史归档保留原始字段。

`summary.json` 的 `native_gpu_executed` 仅表示 native 路径完成执行；`native_gpu_acceptance` 为 `not_evaluated` 或 CPU 情况下的 `not_applicable`。它不会因为选用了 GPU backend 就声称完成了下面列出的验收。

失败不会继续尝试更新，不会把异常现场覆盖成新的恢复点。可能丢失距离最近 checkpoint 的若干步；这是明确的恢复语义。

## 5. GPU 环境与下一轮验证

原生 adapter 按以下上游 API 编写，版本兼容性需要 GPU 实测，不是已认证的安装组合：

- Megatron Bridge `0.6.2`：`AutoBridge`、`apply_overrides_and_finalize`、`provide_distributed_model`、`load_hf_weights`、`export_hf_weights`。
- Megatron Core `0.19.2`：单 rank GPT forward，输出 `[batch, sequence, vocabulary]`。
- vLLM `0.32.0`：`LLM`、`SamplingParams`、worker extension 和 `collective_rpc`。
- Transformer Engine、PyTorch、CUDA、FlashAttention 需要按所用 GPU 镜像配套安装；本地 CPU 的 `uv.lock` 不充当 CUDA 镜像锁文件。

默认一个 learner GPU、一个 rollout GPU，TP/PP/CP/EP 全为 1。模型检查只接受未量化的 dense Qwen3。不是所有 HF 模型都能直接使用这条路径。

在已安装好原生依赖的环境中安装 QoreRL：

```bash
uv pip install -e .
qorerl train --config configs/grpo_megatron_vllm.json --out results/native-baseline
```

不要把 CPU 快速开始的 `uv sync` 当成 Megatron/vLLM CUDA 栈安装器。它不会替你安装 native engines，且同步环境可能移除未在项目锁文件声明的已有包。

如果两端依赖冲突，可分别创建环境，并通过 `learner_runtime_env` / `rollout_runtime_env` 传入 Ray 支持的 runtime-env 设置（例如各自的 `py_executable`）。各节点需有相同的 QoreRL 代码与兼容的 Ray/Python 版本；不同 Python 环境的 native 接入同样属于待验证项。首期建议先在一台双 GPU 机器验证，再扩大集群。

GPU 验收要实际检查：

1. 两端加载同一 snapshot 后，在同一组 token 上比较 behavior / training log-prob，检查有限值和差异分布。
2. 至少完成两次 optimizer update 和后续 refit；确认 optimizer 更新计数大于 0、权重内容改变、新版本产生的 log-prob 对应新权重。只有发布版本增加还不足以证明 refit 生效。
3. 用包含正确和错误回答的组确认非零 advantage 和有效梯度，避免“全部答对/全部答错”的空训练通过验收。
4. 检查 EOS、长度截断、可变长度、多个 group、不同 microbatch 大小。
5. 中断后从 checkpoint 恢复，核对 optimizer/progress/版本并继续采样；执行故障注入确认不会带着半更新权重继续服务。

## 6. 代码位置与当前取舍

| 文件 | 职责 |
| --- | --- |
| `qorerl/runtime/ray_runtime.py` | `run_sync_grpo()`：资源安排、采样/训练/发布顺序与失败处理 |
| `qorerl/runtime/ray_workers.py` | 薄 Ray worker：持有 engine，承接远程调用 |
| `qorerl/runtime/checkpoint.py` | checkpoint 原子发布、schema 兼容与恢复配置检查 |
| `qorerl/training/contracts.py` | 数据与版本协议、样本身份和组完整性检查 |
| `qorerl/training/batching.py` | `PaddedMicrobatch`、右侧 padding、shift 和 response mask |
| `qorerl/training/objective.py` | GRPO 组内优势、surrogate 和聚合 |
| `qorerl/training/tasks.py` | 原生整数加法、CPU 复制任务与精确匹配评分 |
| `qorerl/engines/megatron.py` | Bridge 加载/导出、Megatron forward |
| `qorerl/engines/vllm.py` | vLLM 初始化、worker 内 refit、带 log-prob 的采样 |
| `qorerl/engines/base.py` | learner/rollout 角色接口 |
| `qorerl/engines/factory.py` | 延迟加载和选择具体 backend |
| `qorerl/engines/single_rank_learner.py` | 单 rank GRPO、梯度累积和 FP32 master optimizer |

首期为可核对性选择同步执行、完整 CPU 权重传输、关闭 prefix cache/CUDA graph、保存完整轨迹、普通 padding。它们会带来开销，当前实现不宣称吞吐领先。

后续并行训练应作为 learner 内部 worker group 实现，不能在当前单 rank optimizer 上直接加一个 DP 配置就宣称支持。高性能传输、sequence packing、dynamic batch、异步调度也需要自己的正确性验证。当前协议保留了角色、样本和版本边界，但不代表这些功能已经实现。

早期独立 tiny-MoE 校准实现已从当前源码移除；[历史结果](experiments/2026-11-08/README.md) 保留为归档。默认流程没有每 20 步梯度校准或自动修复。下一步先验证原生 BF16 基线，再为量化 recipe 增加前向/反向/导出/rollout 的能力描述、状态保存和兼容性规则。

上游接口参考：[Bridge RL integration](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-rl-integration.html)、[Bridge provider 0.6.2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.6.2/src/megatron/bridge/models/model_provider.py)、[vLLM LLM 0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/llm.py)。这些是接口依据，不替代本项目的端到端测试。
