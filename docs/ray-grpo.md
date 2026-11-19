# Synchronous GRPO with Ray, Megatron, and vLLM

**English** · [简体中文](ray-grpo.zh-CN.md)

The native path defaults to direct MXFP8 on selected projections. The repository also provides an explicit BF16 reference, a fixed Hadamard recipe, and dynamic SmoothQuant. Users can authorize SmoothQuant to extend scaling within a declared low-precision scope. [Direct MXFP8 and fixed Hadamard](recipes/mxfp8-hadamard.md) · [Dynamic SmoothQuant](recipes/dynamic-smoothquant.md)

**Validation status.** The CPU configuration has run with real Ray processes on a local machine. The Megatron and vLLM adapters are implemented. Native API compatibility, numerical agreement, and performance still require NVIDIA GPU validation.

## 1. Follow one training batch

The default native scenario trains `Qwen/Qwen3-0.6B` on integer addition. The prompt uses the non-thinking template and asks for an integer. Each prompt produces four responses. An exact integer answer earns 1, and other answers earn 0. The default batch has four prompts and 16 responses.

```text
Driver / coordinator (orchestration only)
  │
  ├─ RayLearnerWorker → Megatron Core + Transformer Engine, GPU 0
  │    Load HF checkpoint via Bridge → Megatron model → recipe forward/backward
  │    Default low-precision scope: mlp.down_proj
  │    FP32 master parameters, cross-microbatch gradient accumulation, and AdamW state
  │
  └─ RayRolloutWorker → vLLM, GPU 1
       Receive exported weights → load_weights → acknowledge policy version → sample
```

At startup, the driver resolves a remote HF `main` revision to a specific commit, records it, and passes it to both engines. The learner exports version 0. The first rollout waits for every worker to acknowledge it. Each batch then follows the same sequence.

1. Rollout samples with version v. Each response records the generated tokens, their behavior log-probabilities, group and sample IDs, weight version, and recipe ID.
2. The coordinator verifies input identity, group completeness, and policy version. It rejects missing, duplicate, or wrong-version trajectories, then scores the responses and computes advantages within each group.
3. The learner uses teacher forcing on those same prompts and sampled responses to compute current log-probabilities, the GRPO loss, and gradients.
4. The learner exports weights for version v+1. Every rollout worker refits its model and acknowledges the new version.
5. With no requests in flight, the system can save a checkpoint before starting the next batch.

A weight version labels a publication after a training batch. For example, `update_epochs=2` takes two optimizer steps on a batch with learning signal and publishes weights once. An all-zero-advantage batch skips the optimizer and still publishes a new version to record progress.

## 2. Compute the GRPO objective

Within each group, QoreRL normalizes rewards using the population standard deviation:

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
ratio_t     = exp(current_log_prob_t - behavior_log_prob_t)
term_t      = -min(ratio_t * advantage_i,
                  clip(ratio_t, 1-epsilon, 1+epsilon) * advantage_i)
loss        = mean_over_responses(mean_over_response_tokens(term_t))
```

This is a group-baseline PPO surrogate with no critic and zero reference KL penalty. Each response's advantage weights its generated tokens. The mask excludes prompt and padding positions and includes a generated EOS. A length-capped response contributes its actual generated tokens and receives the task score for its text. The task has no DAPO-style overlength shaping.

**The denominator comes from the rollout that sampled the token.** The learner does not separately compute `old_log_probs` or apply a TIS correction. Sampling fixes temperature and top-p to 1 and disables top-k. It also ignores HF generation-config defaults that would otherwise change the behavior distribution.

Even in BF16, differences in kernels and batching can change learner and rollout log-probabilities. `ratio_mean` and `abs_log_ratio_mean` summarize this discrepancy. With multiple update epochs, those summaries also include changes caused by optimizer updates. To isolate the comparison before the first update, inspect `pre_update_ratio_mean`, `pre_update_abs_log_ratio_mean`, and `pre_update_clip_fraction`. They cover the first epoch's microbatches, while the starting weights are still active. Means and fractions leave the shape of the tails unobserved.

Each response receives equal weight. The learner multiplies each microbatch gradient by `responses_in_microbatch / responses_in_batch` before accumulating it. Gradients are produced in model dtype and converted to FP32 for accumulation on the master weights. After accumulation, the learner clips the master gradients and passes them to AdamW. It skips AdamW entirely on an all-zero-advantage batch so the optimizer's existing momentum cannot move the weights without a learning signal.

## 3. Identify the interfaces

| Interface | Current implementation | What the interface can accommodate |
| --- | --- | --- |
| Algorithm and engine | The algorithm consumes trajectories, logits, masks, and advantages, independently of Megatron layer names. | A new algorithm or precision recipe can reuse the scheduling loop. |
| Training and rollout | Each publication carries `PolicyVersion(weight_version, recipe_id)` and requires acknowledgements from every replica. | Precision state, future async staleness, and recovery can refer to a specific version. |
| Export and transport | Bridge exports CPU tensors with HF names. The driver forwards a Ray object reference without fetching the full weights. | NCCL, IPC, or bucketed transfer could implement a different transport using the same trajectory contract. |

Execution currently synchronizes full batches. The runtime disables automatic Ray actor retries because a partially completed optimizer step or refit cannot be safely replayed without examining its state. A failure stops training so the next run can start at a complete checkpoint.

The vLLM worker extension calls `load_weights` on the live model and verifies parameter coverage from the returned names. This catches missing loads at parameter-name granularity. Correct bytes within every fused shard and correct post-refit outputs require additional checks on hardware.

## 4. Run locally

Use Python 3.12 or later and uv:

```bash
uv sync --group dev
uv run qorerl train --config configs/grpo_cpu_test.json --out results/ray-local
```

The CPU fixture uses a small GRU for digit copying, one learner process, and two rollout processes. It runs real sampling, log-probability calculation, backward, AdamW, and Ray IPC using the test engines. Use a new output directory. This example takes 40 training steps and saves a checkpoint every 20.

```text
results/ray-local/
  config.json                 Resolved configuration and model revision
  recipe.json                 Resolved numerical specification and identity
  workers.json                Driver/worker process and Ray node IDs
  metrics.jsonl               Loss, reward, ratios, clipping, gradient norm, stage timings
  trajectories.jsonl          Tokens, behavior log-probs, rewards, advantages, and versions
  checkpoints/step-000000/    Initial boundary before sampling
  checkpoints/step-000020/
  checkpoints/step-000040/
    metadata.json
    learner.pt                Model, FP32 masters, AdamW state, progress, Torch RNG, recipe state
  summary.json                Completion and native execution/acceptance fields
  incident.json               Written on failure with the latest complete recovery point
```

Prompts and sampling seeds in this task are determined by seed, step, and group ID. No external dataloader cursor is required. Training disables dropout and uses no scheduler. Future data streams, schedulers, or randomized recipes need to checkpoint their own state to preserve recovery semantics.

### Resume from a checkpoint

Copy the CPU configuration to `configs/grpo_cpu_continue.json` and change `steps` to 60. It specifies the **total** number of steps at completion, so the run resumes from 40 and continues to 60.

```bash
uv run qorerl train \
  --config configs/grpo_cpu_continue.json \
  --resume results/ray-local/checkpoints/step-000040 \
  --out results/ray-continued
```

Resume creates fresh workers, restores the learner's full state, re-exports and synchronizes weights, and then generates new trajectories. It writes to a new output directory and preserves the original run.

You can change total steps, checkpoint frequency, RPC timeout, worker runtime environments, and observation settings at resume. Training semantics, including learning rate, batch configuration, model revision, and recipe identity, must match. A pending SmoothQuant proposal can be accepted or rejected when resuming its bound checkpoint. CPU tests compare final model parameters and optimizer state element by element between continuous and resumed runs. GPU reproducibility still needs separate measurement. [Manual candidate review](recipes/dynamic-smoothquant.md#manual-review-on-cpu)

MXFP8 synchronization and recovery also check the actual target shapes and fixed signs before rebuilding rollout caches. Parameters and optimizer state stay in the original coordinates. [Recipe setup and comparison commands](recipes/mxfp8-hadamard.md)

New checkpoints use schema 2, with `policy_version.weight_version`, `recipe_id` under `policy_version`, and `master_parameters` for the master weights. The reader accepts older schema 1 names with the same training semantics. Trajectory logs now use schema 2, and the integer metric formerly called `policy_version` is `weight_version`. Historical reports preserve their original fields.

In `summary.json`, `native_gpu_executed` records whether the native path completed. `native_gpu_acceptance` is `not_evaluated` for native execution or `not_applicable` on CPU. On failure the process stops and preserves the most recent complete checkpoint. Progress since that checkpoint may need to be replayed.

## 5. Set up and validate the GPU path

The native adapters target these upstream interfaces. The exact combination still needs testing on GPU.

- Megatron Bridge `0.6.2` for `AutoBridge`, `apply_overrides_and_finalize`, `provide_distributed_model`, `load_hf_weights`, and `export_hf_weights`.
- Megatron Core `0.19.2` for a single-rank GPT forward returning `[batch, sequence, vocabulary]`.
- vLLM `0.32.0` for `LLM`, `SamplingParams`, worker extensions, and `collective_rpc`.
- Install compatible Transformer Engine, PyTorch, CUDA, and FlashAttention versions for the GPU image. The local development lockfile covers CPU development.

The initial scope is an unquantized dense Qwen3, one learner GPU and one rollout GPU, with TP/PP/CP/EP all equal to 1. After installing the native dependencies in the target environment, install QoreRL and run the BF16 reference:

```bash
uv pip install -e .
qorerl train --config configs/grpo_megatron_vllm.json --out results/native-baseline
```

The CPU quick start's `uv sync` handles development dependencies. It does not provision the CUDA stack and can remove preinstalled packages missing from the project lockfile.

If learner and rollout dependencies conflict, configure separate environments using Ray runtime-env settings in `learner_runtime_env` and `rollout_runtime_env`, for example distinct `py_executable` values. Each node needs the same QoreRL code and compatible Ray and Python versions. Start native validation on a single dual-GPU machine. Cross-environment execution still needs verification there.

<a id="native-nvfp4-requirements"></a>

### Native NVFP4 requirements and execution

Choose `mlp-down-v1` for direct quantization, `hadamard-mlp-down-v1` for fixed Hadamard, or `smoothquant-mlp-down-v1` for dynamic SmoothQuant. Set `"precision": "nvfp4"` in `recipe_options`. The layer scope, initial S subset, and approvals follow the same rules as MXFP8. For example, changing precision in the [dynamic SmoothQuant configuration](../configs/grpo_dynamic_smoothquant.json) starts layer 25 with direct NVFP4.

Both engines need an SM10.x GPU supported by the kernel and Transformer Engine 2.18.x. On selected projections the learner uses native TE GEMMs for forward, Dgrad, and Wgrad and exports BF16 weights in the canonical coordinates. The vLLM worker packs each published weight into FP4 E2M1 values, FP8 E4M3 block scales, and a tensor-level scale. It quantizes inputs at inference. Both weight dimensions must be multiples of 32. Local tests exercise configuration and adapter calls. Native numerics, memory, and learning behavior need GPU measurements. [Training GEMM](../qorerl/recipes/te_block_linear.py) · [NVFP4 packing and inference](../qorerl/recipes/te_nvfp4.py)

At GPU validation, use identical weights to compare learner and rollout log-probabilities for the same token across different batch compositions. TE computes a second-level scale across the input tensor. Changing the batch maximum can change that token's quantization.

### GPU acceptance checks

1. Load one snapshot on both engines. Compare behavior and training log-probabilities on matched tokens, check for nonfinite values, and inspect the error distribution.
2. Run at least two optimizer updates and refits. Confirm that the update count advanced, the weight contents changed, and log-probabilities after publication correspond to the new weights.
3. Use groups containing both correct and incorrect responses. Verify nonzero advantages and useful gradients.
4. Exercise EOS, truncated responses, variable lengths, multiple groups, and different microbatch sizes.
5. Resume an interrupted run and verify optimizer state, progress, version, and subsequent sampling. Inject a publication failure and check that partially installed weights never serve a new batch.

## 6. Find the implementation

| Source | Responsibility |
| --- | --- |
| [Ray runtime](../qorerl/runtime/ray_runtime.py) | `run_sync_grpo()`, resource placement, the sampling/training/publication order, and failure handling |
| [Ray workers](../qorerl/runtime/ray_workers.py) | Own the engines and receive remote calls |
| [Checkpointing](../qorerl/runtime/checkpoint.py) | Atomic checkpoint publication, schema compatibility, and resume validation |
| [Contracts](../qorerl/training/contracts.py) | Data and version contracts, sample identity, and group integrity |
| [Batching](../qorerl/training/batching.py) | `PaddedMicrobatch`, right padding, shifting, and response masks |
| [Objective](../qorerl/training/objective.py) | Group advantages, clipped policy surrogate, and loss reduction |
| [Tasks](../qorerl/training/tasks.py) | Native integer addition, CPU copying fixture, and exact-match scoring |
| [Megatron](../qorerl/engines/megatron.py) | Bridge loading and export, Megatron forward |
| [vLLM](../qorerl/engines/vllm.py) | Initialization, in-worker refit, and sampling with log-probabilities |
| [Engine interfaces](../qorerl/engines/base.py) | Learner and rollout role contracts |
| [Engine factory](../qorerl/engines/factory.py) | Backend selection and lazy loading |
| [Single-rank learner](../qorerl/engines/single_rank_learner.py) | GRPO optimization, gradient accumulation, and FP32 master optimizer |
| [Recipes](../qorerl/recipes/) | Precision specifications and state, Hadamard and SmoothQuant, TE/vLLM adapters, GPU operator probe |

The first implementation favors auditability: synchronous execution, full CPU weight transfer, full trajectory logs, ordinary padding, and disabled prefix caching and CUDA graphs. Those choices carry measurable costs. Future parallel training needs a worker group inside the learner. Faster transport, sequence packing, dynamic batches, and async scheduling each require separate correctness validation.

The earlier standalone tiny-MoE calibration experiment remains available as [archived results](experiments/2026-11-08/README.md). The default workflow has no periodic gradient calibration or automatic gradient repair. Native validation starts with the BF16 reference, then compares direct MXFP8 with fixed Hadamard under matched conditions.

Upstream interfaces: [Bridge RL integration](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-rl-integration.html), [Bridge provider 0.6.2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.6.2/src/megatron/bridge/models/model_provider.py), and [vLLM 0.32.0 LLM](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/llm.py).
