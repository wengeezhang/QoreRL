# Recipe Insights during training

**English** · [简体中文](recipe-insights.zh-CN.md)

Recipe Insights captures a small sample of real learner inputs and the current weights. On each sampled layer, it compares the forward output and latency of **BF16, MXFP8, and MXFP8 with fixed Hadamard**. Open the local viewer during training or export a standalone HTML report afterward. Probes are opt-in and currently support the direct MXFP8 and fixed Hadamard MXFP8 recipes on native Megatron. Native GPU execution and measurements still await validation.

![Recipe Insights dashboard showing synthetic data](assets/recipe-insights-demo.png)

The preview uses synthetic data to illustrate the interface.

## Collect measurements

In an environment with the native CUDA dependencies installed, save the following as `configs/grpo_mxfp8_observed.json`. Omitted training settings take their `TrainConfig` defaults. See the [recipe and hardware requirements](recipes/mxfp8-hadamard.md).

```json
{
  "steps": 40,
  "recipe_id": "mxfp8-hadamard-mlp-down-v1",
  "recipe_options": {"block_size": 128, "seed": 0},
  "recipe_observation": {
    "enabled": true,
    "every_steps": 20,
    "layers_per_window": 2,
    "max_rows": 128,
    "repeats": 5,
    "max_snapshot_bytes": 134217728
  }
}
```

```bash
qorerl train --config configs/grpo_mxfp8_observed.json --out results/grpo-observed
```

Use a second terminal for the live viewer, or export a snapshot after training.

```bash
qorerl inspect --run results/grpo-observed --port 8765
qorerl report --run results/grpo-observed --out results/grpo-observed/report.html
```

The viewer listens on localhost at [127.0.0.1:8765](http://127.0.0.1:8765). It reads `config.json` and `recipe_metrics.jsonl` from the run directory and shows layer coverage, recent observations, and comparisons between the three configurations. The live view refreshes every two seconds by default and can be paused. The HTML report embeds the data available at export time. Use the SVG or PNG button to export the data source, summary, and both layer charts in one image. **Save HTML** keeps the current data source and comparison selection.

## What a probe captures

- The first default window opens at training batch **20**, followed by batches 40, 60, and so on. A ten-batch smoke run finishes before that window. Each window selects up to two installed target layers in rotation.
- For each selected layer, the probe captures its first down projection in the first microbatch of the first update epoch. It flattens the leading input dimensions into rows, copies up to the first 128 rows, and copies the full canonical BF16 weight. The sample reflects the prompt, response, and padding layout encountered at that point. It is a prefix sample.
- Capture takes place before the training-side rotation. Once the first microbatch has finished backpropagation and released its graph, the probe replays private copies under `no_grad` before the optimizer update. Training keeps its original output, parameters, gradients, optimizer, and recipe state. The probe restores RNG state afterward. A batch with all-zero advantages can still produce a measurement while its optimizer step count stays fixed.
- `step` labels the training batch. `weight_version` and `optimizer_steps` record progress **at capture, before the update**. In a run starting at zero with one update per preceding batch, batch 20 captures weight version 19. Resume derives the next window from training progress. Observation settings can change across a same-semantics resume.

`max_snapshot_bytes` bounds the **total bytes of retained input and canonical weight copies per window**, with a default of 128 MiB. Peak GPU memory can be higher due to FP32 Hadamard intermediates, quantized representations, outputs, and CUDA workspaces. A skipped layer or a layer that never ran produces a `skipped` record. Recoverable capture or replay failures produce `error`. Missing measurements carry no error estimate. An unrecoverable CUDA error stops training.

## Read the measurements

| Field | Interpretation |
| --- | --- |
| `shape_mnk` | Sampled rows M, output channels N, and input channels K. `original_rows` counts rows before sampling. |
| `output_relative_l2` | `‖Y_variant − Y_bf16‖₂ / ‖Y_bf16‖₂`. All three variants receive the same X and W. The BF16 reference has zero error against itself. |
| `latency_ms` | Median CUDA-event latency across five interleaved repeats by default, after a warm-up for each variant. Each invocation includes its rotation, quantization, padding, and GEMM as applicable. Rotated and quantized weights are rebuilt for the measurement. |
| `probe_seconds` | Host capture time plus replay wall time. Synchronization can wait on previously queued training work, and asynchronous copy time is not isolated. Subtracting this value from run time does not yield a clean estimate of training time without probes. |

The BF16 reference calls `torch.nn.functional.linear` on the same X and W. X already contains the effects of upstream training layers. The result measures the local operator on that particular input. The Hadamard variant includes both BF16 transform rounding and MXFP8 quantization error.

CUDA events measure the sampled forward call in milliseconds. Dgrad, Wgrad, rollout, synchronization, checkpointing, and time to reach a training-quality target are outside that measurement. Layers in a latest-value chart may come from different training batches. Use full-run logs and matched GPU experiments to evaluate throughput and training quality.

## Preview the viewer on CPU

Install the project in the local development environment, then run:

```bash
qorerl demo-report --out results/insights-demo
qorerl inspect --run results/insights-demo --port 8765
qorerl report --run results/insights-demo --out results/insights-demo/report.html
```

`demo-report` creates a new directory and refuses to overwrite an existing one. Demo records carry `source=demo`, and the viewer labels the display accordingly. Real GPU records carry `source=measured`. The demo supports UI exploration only.
