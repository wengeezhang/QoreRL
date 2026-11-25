# Fixed Hadamard rotations for MXFP8 MLP down projections

**English** · [简体中文](mxfp8-hadamard.zh-CN.md)

With the same MLP down projections quantized, does adding a fixed Hadamard rotation improve RL training over direct MXFP8? The recipe includes native engine adapters, configurations, checkpoint state, and local tests. **NVIDIA GPU validation, training-quality measurements, and speed measurements are pending.** The comparisons below establish the BF16 reference, exercise direct MXFP8, and then isolate the cost and effect of the rotation.

## 1. Compare three matched configurations

| Configuration | MLP down projection | Other operations |
| --- | --- | --- |
| [BF16](../../configs/grpo_megatron_vllm.json) | BF16 | BF16 baseline |
| [Direct MXFP8](../../configs/grpo_mxfp8_mlp_down.json) | MXFP8 | Same baseline |
| [MXFP8 with fixed Hadamard](../../configs/grpo_mxfp8_hadamard_mlp_down.json) | Rotate input and weight, then apply MXFP8 | Same baseline |

All three use the same model, task, seed, batch, learning rate, and number of steps. By default the recipe targets `mlp.down_proj` in every Transformer block. It rotates the activations **after SwiGLU and immediately before the down projection**. The gate and up projections, SiLU, attention, router, and `lm_head` stay on their original paths.

The first native implementation accepts an unquantized **dense Qwen3** checkpoint, TP/PP/CP/EP all set to 1, and either separate GPUs or [single-GPU collocation](../ray-grpo.md#single-gpu-collocation) for one learner and one rollout worker. MoE, model parallelism, CUDA graphs, prefix caching, and prequantized checkpoints are outside the current validation scope. The CPU test engine rejects these two native recipe configurations. Unsupported combinations raise an error.

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

`layers` uses zero-based decoder indices. Omitting it selects all supported layers. Empty lists, duplicates, and out-of-range indices fail validation. The generic recipe IDs `mlp-down-v1` and `hadamard-mlp-down-v1` let `recipe_options.precision` select the base format, with default `mxfp8` and experimental `nvfp4`. The kernels and validation commands below cover MXFP8.

Hadamard `block_size` must be a power of two of at least 2 and divide the number of input channels. Both native recipes require input and output channels divisible by 32. **Hadamard block size controls the width of the rotation. MXFP8 separately shares a scale across 32 elements.** The seed fixes each layer's ±1 signs for the run.

## 2. Rotate the inputs and weights together

With PyTorch's layout, `X[M,K]` and `W[N,K]` give `Y = X @ W.T`. Let H be a normalized block Hadamard transform, D the diagonal matrix of fixed ±1 signs, and R = D H. Both engines compute:

```text
X_rot = X @ R
W_rot = W @ R

Y = X_rot @ W_rot.T
  = X @ R @ R.T @ W.T
  = X @ W.T
```

The identity holds in real arithmetic. The implementation accumulates butterfly additions and subtractions in FP32, then casts back to BF16 for the native quantizer. Rounding can still affect the result. Measure the error for the actual inputs and weights.

The training `Parameter` and FP32 optimizer master stay in the **original coordinate system**. Autograd maps `dX_rot` and `dW_rot` back by multiplying by `R.T`. The sign and Hadamard factors have a specific order: the inverse of `D H` is `H D`.

Each training forward pass builds a rotated representation from the current weight. The checkpoint keeps the canonical parameter, and the transform operates downstream of SwiGLU.

## 3. Trace precision through the layer

| Operation or representation | Execution |
| --- | --- |
| Canonical parameters and layer inputs/outputs | BF16, with optimizer master weights, optimizer state, and cross-microbatch gradient accumulation in FP32 |
| Forward (Fprop) at selected layers | Native TE MXFP8 GEMM with E4M3 values and an E8M0 scale per 32 elements |
| Input and weight gradients (Dgrad/Wgrad) | Native TE MXFP8 GEMM returns BF16 gradients. Hadamard backward maps gradients to the original coordinates. |
| TE row-wise and column-wise representations | Each is quantized directly from BF16. Transposing one representation would reuse scales from the wrong groups. |
| Weight export | Bridge exports canonical BF16 weights together with recipe state. |
| Selected rollout layers | vLLM's `FlashInferCutlassMxfp8LinearKernel` quantizes inputs and runs native MXFP8 GEMM. |
| Rollout weight cache | Rotate canonical BF16 weights for the Hadamard branch, quantize them, and let the native kernel prepare its layout. |

The shared custom autograd implementation calls TE for Fprop, Dgrad, and Wgrad. Its backward pass defines the gradient estimator used in low-precision training. A BF16 Wgrad replacement, periodic gradient calibration, and automatic sensitive-layer selection are outside this recipe. [Block linear autograd](../../qorerl/recipes/te_block_linear.py)

Training allows M, the number of token rows, to vary. GEMM zero-pads the input and upstream gradient internally, then strips the added rows from the output and Dgrad. The added rows never enter a trajectory, reward, or loss. Unsupported channel dimensions fail validation.

TE and vLLM use their respective native quantizers. Even with identical formats, rotations, and canonical weights, measuring matched-token log-probabilities is necessary to establish agreement. Matching quantized bytes and scales is a separate check.

## 4. Synchronization, recovery, and run records

The configuration's `recipe_id` is a readable preset name. The published `PolicyVersion.recipe_id` incorporates a hash of the specification, including the method version, scope, selected layers, block size, and seed. Changing the rotation configuration changes the policy identity.

The checkpoint and weight snapshot also carry actual canonical target names, shapes, and fixed signs. The receiver verifies these before loading weights. Resume performs the same checks before changing the model.

```text
Verify model, recipe identity, target shapes, and fixed signs
    → mark the active rollout version and derived caches unavailable
    → load canonical BF16 weights through native load_weights
    → verify parameter coverage
    → rebuild the rotated, quantized, kernel-ready weight caches
    → synchronize CUDA
    → acknowledge the new version and allow the next rollout batch
```

Derived caches live outside the registered model parameters. Bridge therefore exports canonical weights, and each publication rotates them once. vLLM installs the adapter after model loading and before memory profiling and KV allocation, so the profiling run sees the installed execution path.

Same-configuration resume verifies recipe state. Changing recipe, block size, or seed at resume would require an explicit migration.

| File | What to inspect |
| --- | --- |
| `recipe.json` | The resolved specification and identity. |
| `workers.json` | The backend, target layers, versions, and validation state reported at startup. |
| `metrics.jsonl` | Reward, loss, timing, and the first epoch's `pre_update_ratio_mean`, `pre_update_abs_log_ratio_mean`, and `pre_update_clip_fraction`. The pre-update summaries compare the same snapshot and sampled tokens before the first optimizer update. Investigate tails separately. |
| `summary.json` | The final engine `execution` report. vLLM counts applications since the most recent refresh. Zero immediately after the final publication can mean that sampling has finished. |

## 5. Run the GPU comparisons

The adapters target the following interfaces. This combination still needs validation on hardware.

- Megatron Bridge 0.6.2 and Megatron Core 0.19.2.
- Transformer Engine **2.18.x**. The adapter checks the version and `general_gemm` ABI.
- vLLM **0.32.0**. The adapter checks the version and FlashInfer CUTLASS MXFP8 support.
- Both backends need an **SM10-series NVIDIA GPU** supported by the kernel, such as B200. Check the kernel's requirements for the particular GPU.

Install QoreRL with `uv pip install -e .` in an environment that already has the CUDA dependencies. The development lockfile covers CPU dependencies. For separate learner and rollout environments, use Ray runtime environments as described in the [developer guide](../ray-grpo.md).

Start with the operator:

```bash
python -m pytest -q tests/test_te_mxfp8.py tests/test_hadamard.py
python -m qorerl.recipes.compare --out results/mxfp8-operator.json
```

The probe fixes the input, canonical weight, and MSE target. It compares BF16, MXFP8, and MXFP8 with Hadamard on output, Dgrad and Wgrad error, Wgrad cosine similarity, forward/backward latency, and incremental allocated GPU memory. It requires the native GPU path. Interpret the recorded values against the experiment's acceptance criteria.

Memory values are the incremental peak **PyTorch allocated bytes** for each warmed-up iteration. Persistent signs and workspaces from warm-up, allocator reserves, and allocations outside PyTorch contribute separately to process memory.

Then run the three end-to-end configurations:

```bash
qorerl train --config configs/grpo_megatron_vllm.json --out results/grpo-bf16
qorerl train --config configs/grpo_mxfp8_mlp_down.json --out results/grpo-mxfp8
qorerl train --config configs/grpo_mxfp8_hadamard_mlp_down.json --out results/grpo-mxfp8-hadamard
```

Verify actual optimizer updates, at least two weight refits, and recovery before comparing learning quality. The ten-step addition task is a smoke test. A shared seed can still yield different trajectories under different policies. Compare matched-token log-probabilities on the same snapshot, nonfinite values, error tails, and native gradients and parameters. Training-quality and cost claims need multiple seeds and full time and cost to comparable quality. `native_gpu_executed=true` records native execution only.

## 6. Costs and open work

The current Hadamard transform uses PyTorch butterfly operations. The learner rotates and quantizes the current weight on each forward pass. vLLM retains both canonical BF16 weights and derived quantized weights. During refresh it can hold old and new caches concurrently. The memory cost and relative speed need measurement.

Inference profiling sees installed persistent caches. Refit can add BF16 transport buffers, rotation temporaries, and overlapping caches beyond that profile. The example sets `gpu_memory_utilization=0.7`. Measure peak memory during synchronization before selecting the headroom for another model.

If numerical and learning gains hold up, the next candidates are a fused fast Walsh–Hadamard transform kernel, a shorter weight-cache lifetime, and lower synchronization cost. Validate those changes before expanding to MoE, model parallelism, or automated intervention.

[QuaRot](https://arxiv.org/abs/2404.00456) provides the orthogonal rotation reference. QoreRL implements the scoped down-projection experiment and measures its RL effects independently. For the native execution interfaces, see the [TE GEMM wrapper](https://github.com/NVIDIA/TransformerEngine/blob/15288857644368b8b7bdddd9bead7d6d3d04de80/transformer_engine/pytorch/cpp_extensions/gemm.py), [TE Linear forward and backward](https://github.com/NVIDIA/TransformerEngine/blob/15288857644368b8b7bdddd9bead7d6d3d04de80/transformer_engine/pytorch/module/linear.py), and [vLLM 0.32 FlashInfer MXFP8 kernel](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py).

## 7. Observe paired measurements during training

Recipe Insights optionally rotates through two layers every 20 training batches by default. It captures actual inputs and canonical weights, then compares local forward error and CUDA-event latency for the three variants. It runs independently of calibration or recipe changes. Native GPU validation remains pending. See [Recipe Insights](../recipe-insights.md) for configuration, live viewing, and HTML export.

![Recipe Insights preview using synthetic data](../assets/recipe-insights-demo.png)
