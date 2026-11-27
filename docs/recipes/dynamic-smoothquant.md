# Updating SmoothQuant smoothing factors during training

**English** · [简体中文](dynamic-smoothquant.zh-CN.md)

Choose which `mlp.down_proj` layers run in low precision, then choose which of those layers start with SmoothQuant, when to recalibrate, and who may approve new per-channel smoothing factors (`S`). This guide uses the default MXFP8 format. Experimental NVFP4 follows the same layer-selection and calibration rules. The native implementation targets a single-rank dense Qwen3 with a Megatron/Transformer Engine learner and vLLM for rollout. The original SmoothQuant work calibrates INT8 inference offline. QoreRL applies channel scaling during training. Local CPU tests exercise calibration, decisions, and publication. Native GPU execution, overhead, and training quality still await validation. [SmoothQuant paper](https://arxiv.org/abs/2211.10438)

## Inspect calibration and prepare GPU acceptance

For the code-level path from activation capture to persisted evidence and charts, see [SmoothQuant calibration: from training inputs to a visible decision](../smoothquant-calibration.md).

The calibration viewer reuses these saved comparisons and links approvals across run segments. It shows one evaluation window at a time, including candidates that did not qualify. [View and export calibration evidence](../recipe-insights.md#smoothquant-reuse-calibration-evidence).

For a single B200, start with [`grpo_single_gpu_smoothquant.json`](../../configs/grpo_single_gpu_smoothquant.json): Qwen3-1.7B, one collocated rollout worker, microbatch 1 and manual calibration every 20 actual updates. Use `gpu-preflight` before loading and `validate-run` after a completed segment. The [GPU validation guide](../gpu-validation.md) covers commands, evidence, explicit thresholds and remaining checks. Hardware execution remains unvalidated; H200 cannot run this native MXFP8 path.

## Choose where to start

With `recipe_id: "smoothquant-mlp-down-v1"`, `recipe_options.layers` fixes the low-precision layer set. `recipe_options.smoothquant_layers` selects the subset that starts with SmoothQuant. Both use zero-based decoder layer indices for `mlp.down_proj`. Omitting `layers` selects every supported layer. Omitting `smoothquant_layers` enables S on every selected layer. To start all selected layers in direct MXFP8 and allow later proposals, set `smoothquant_layers: []` and `allow_layer_expansion: true`. Native BF16, direct MXFP8 and fixed Hadamard MXFP8 runs can also collect outlier evidence.

| Starting point | Settings | What happens at calibration time |
| --- | --- | --- |
| Find which target layers show outliers | `discovery_only: true` | Report each layer's score and row count. Continue training with the active configuration. |
| Inspect a set of layers before deciding which to adjust | `discover_outliers: true` | Score the observed layers. Fit and compare S only for layers that meet the discovery criterion. |
| Calibrate known targets, such as layers 7 and 19 | `recipe_options.layers: [7,19]` and `discover_outliers: false` | Fit and compare S for those layers and record their scores for inspection. |
| Start with calibrated smoothing factors | `initial_artifact: "smoothing-factors.json"` and `refresh_every_optimizer_steps: 0` | Verify compatibility and publish the factors before the first rollout. A positive interval allows subsequent refreshes. |

The calibration switches in this table belong in `recipe_calibration`. The default `refresh_every_optimizer_steps` is 20. A rollout batch can run multiple update epochs or skip optimizer updates when all advantages are zero. Calibration runs once at the first synchronized batch boundary after the update counter crosses an interval. The count carries over on resume. Periodic calibration defaults to `approval: "manual"`. Choosing `"auto"` preauthorizes candidates that pass the held-out comparison below. Supplying an external artifact explicitly selects the initial smoothing factors.

For example, select `recipe_id: "bf16-baseline-v1"` and add `"recipe_calibration": {"discovery_only": true, "refresh_every_optimizer_steps": 20}` to write a periodic report to `calibration.jsonl`. If activations are already available, map each layer name to a BF16 tensor of shape `[tokens, input_channels]`. Pass the mapping and `CalibrationConfig(discovery_only=True)` to `qorerl.recipes.calibration.discover` to get the names and scores of layers that meet the threshold. `outlier_report` in the same module returns every layer’s evidence, including the row count and selection reason. Standalone discovery uses all collected rows.

## How the learner evaluates new smoothing factors

Two `recipe_calibration.evaluation_scope` values are supported. Default `local` is the projection-level tool described below. Opt-in `model_logits` discovers outliers and fits factors from full-model BF16 activations, then compares current and complete candidate logits against BF16. It applies no local-error prefilter: only a jointly qualifying combination enters review. Use the [CPU config](../../configs/grpo_cpu_smoothquant_logits.json) or [single-GPU config](../../configs/grpo_single_gpu_smoothquant_logits.json); see the [workflow, evidence and limitations](../smoothquant-calibration.md). Native GPU validation is pending.


1. After an update, the learner replays the trajectory prefixes from the batch it just trained on. It deduplicates prefixes, excludes padding, and captures the input X before transformation, up to `max_rows` per selected layer (default 128). The first half of the rows is used for fitting and the rest for evaluation. Both halves may contain tokens from the same response.
2. For each input channel `j`, take the largest absolute value over fitting rows, `x_max[j]`. The discovery score is the largest channel maximum divided by the mean of the channel maxima. Its default threshold is 4. The score describes variation in channel magnitudes within a layer.
3. For selected layers, take the maximum absolute weight in column `j`, called `w_max[j]`. Fit `S[j] = x_max[j]^alpha / w_max[j]^(1-alpha)` with default `alpha = 0.5`. The forward pass pairs `X/S` with `W*S` for W stored as `[out, in]`. Channels with no usable signal keep their existing factor. Refresh replaces the layer's factors with absolute values relative to its canonical weights.
4. Use the same held-out X and W for the BF16 reference, the current configuration, and the candidate's low-precision forward pass. Relative L2 error is the norm of the output difference divided by the norm of the BF16 output. To propose a change the error must strictly improve and the relative reduction must meet `min_relative_improvement` (default 0). A layer can show outliers and still do better with its existing factors.

## Expand SmoothQuant within the low-precision layer set

In the [native configuration](../../configs/grpo_dynamic_smoothquant.json), all four selected down projections run MXFP8 from the first batch. Three start with S. At each 20-update boundary the learner observes all four.

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

The `action` field in `calibration.jsonl` identifies the baseline for each comparison.

| Action | Layers in this example | Current baseline | Acceptance criteria |
| --- | --- | --- | --- |
| `refresh` | 7, 15, 20 | MXFP8 with existing S | New S reduces error relative to existing S and meets the requested relative improvement. |
| `activate` | 25 | Direct MXFP8 | The layer shows an outlier. The candidate improves on direct MXFP8, meets the requested relative improvement, and meets the optional first-activation error bound. |

Here `0.05` caps the relative L2 error at **5% on first activation**. The default has no cap. After approval and publication, layer 25 runs MXFP8 with SmoothQuant starting with the next batch. The recipe excludes the other BF16 layers from both observation and publication. [Layer 25 through a complete cycle](../../design.md#smoothquant-expansion)

<a id="manual-review-on-cpu"></a>

## Walk through manual review on CPU

Run from the repository root:

```bash
uv run qorerl train --config configs/grpo_cpu_smoothquant.json --out runs/sq-review
```

This example uses explicit block-quantization simulation, a small model with known channel imbalance, one learner process, and two rollout processes. The interval is one optimizer update so the decision appears in a short run. In one local run, the first batch supplied four rows for fitting and five for comparison.

| Layer | Discovery score | Relative error with current S | Relative error with candidate S | Decision |
| --- | ---: | ---: | ---: | --- |
| 0 | 31.98 | 0.07823 | 0.04792 | Propose a refresh |
| 1 | 31.62 | 0.03312 | 0.03862 | Keep the active S |
| 2 | 1.78 | — | — | Below discovery threshold |

Inspect `runs/sq-review/calibration.jsonl` for the evidence and `summary.json` for the `candidate_id` and `checkpoint`. While awaiting review, the training process exits and releases its workers. In the checkpoint’s `candidate.json`, inspect `evidence` and the SmoothQuant artifact under `candidate`. Its `scales` map contains the proposed smoothing factors for each qualifying layer. Resume into a fresh output directory.

```bash
CP=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["checkpoint"])')
ID=$(python3 -c 'import json; print(json.load(open("runs/sq-review/summary.json"))["candidate_id"])')
uv run qorerl train --config configs/grpo_cpu_smoothquant.json \
  --resume "$CP" --accept-candidate "$ID" --out runs/sq-accepted
```

To reject the proposal, use `--reject-candidate "$ID"`. Acceptance preserves model parameters, optimizer state, and the update count, then publishes the new smoothing factors. The next batch starts after every rollout worker acknowledges the new version. Rejection retains the active S. Check `policy_version` in `trajectories.jsonl` to see what actually generated each response. Each application has a recovery point. On failure, `incident.json` records its path. To revisit a decision, resume the pre-change checkpoint into another fresh directory.

## Import smoothing factors from an external tool

For each supplied target layer, produce a finite, strictly positive FP32 vector with one entry per input channel. Startup import accepts a subset of the initial `smoothquant_layers`. Unspecified initial factors default to 1. Other layers inside the low-precision scope start in direct MXFP8, and layers outside it use BF16. The following example packages the calibrated vectors in `smoothing_factors`.

```python
import json
from pathlib import Path

from qorerl.recipes.smoothquant import make_smoothing_artifact
from qorerl.training.config import TrainConfig

cfg = TrainConfig.load("configs/grpo_dynamic_smoothquant.json")
# smoothing_factors["model.layers.7.mlp.down_proj.weight"] is a calibrated 1D CPU float32 tensor.
artifact = make_smoothing_artifact(
    smoothing_factors,
    model=cfg.model,
    model_revision=resolved_model_revision,  # Model commit for this run.
    recipe_id=cfg.policy_recipe_id,
)
Path("configs/smoothing-factors.json").write_text(json.dumps(artifact))
```

Set `recipe_calibration.initial_artifact` to `"smoothing-factors.json"`. Older configurations can still use `scale_artifact`. Relative paths resolve from the config file's directory. The SmoothQuant artifact uses the existing schema-1 fields `scales` and `scale_digest` for its per-channel factors and their digest. Import checks the model, revision, recipe scope, target names, channel counts, and content digest. For a local model directory, retain the snapshot used for calibration. The directory name alone does not identify its contents. Checkpoints carry all active vectors, so ordinary resume can proceed without re-reading the external file.

## Run on the native backends

```bash
uv run qorerl train --config configs/grpo_dynamic_smoothquant.json --out runs/sq-native
```

The [native configuration](../../configs/grpo_dynamic_smoothquant.json) requires a Megatron/TE and vLLM setup with native MXFP8. vLLM builds the quantized weight cache for the entire low-precision scope at startup. Publishing S rebuilds the packed weights from `W*S`. The quantizer computes its own block scales when packing these transformed weights. GPU validation covers matched-token log-probabilities on both engines, outputs after refit, learning curves, and calibration overhead. [MXFP8 setup](mxfp8-hadamard.md)

For NVFP4, set `precision` to `"nvfp4"` on the same generic recipe. Both engines require Transformer Engine 2.18.x and a supported SM10.x GPU. Both use TE's NVFP4 quantization and GEMM. The vLLM worker caches packed weights after publication, and candidate evaluation uses an NVFP4 baseline. Native GPU execution and end-to-end effects remain unvalidated for both formats. The CPU walkthrough simulates MXFP8 only. [NVFP4 training and packing implementation](../../qorerl/recipes/te_nvfp4.py)
