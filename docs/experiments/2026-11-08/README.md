# Tiny MoE experiment · Restoring BF16 activations for Wgrad

**Archived November 8, 2026.** This directory preserves the configurations, results and reports from an early standalone numerical experiment. It ran on an Apple Silicon CPU using MXFP8 emulation. Native Blackwell/TE validation remains separate.

The commands below require a checkout of [`b7df310`](https://github.com/wengeezhang/QoreRL/tree/b7df310a8f5d04b4c1b5bea388c4068ef366d01c), the revision that provided `qorerl research`. That revision contains the [entry point](https://github.com/wengeezhang/QoreRL/blob/b7df310a8f5d04b4c1b5bea388c4068ef366d01c/qorerl/experiment.py), [quantization and backward pass](https://github.com/wengeezhang/QoreRL/blob/b7df310a8f5d04b4c1b5bea388c4068ef366d01c/qorerl/precision.py), [calibration logic](https://github.com/wengeezhang/QoreRL/blob/b7df310a8f5d04b4c1b5bea388c4068ef366d01c/qorerl/calibration.py) and [original run instructions (Chinese)](https://github.com/wengeezhang/QoreRL/blob/b7df310a8f5d04b4c1b5bea388c4068ef366d01c/docs/wgrad-experiment.zh-CN.md). The JSON, generated reports and test counts describe that revision.

## Experiment setup

One process trained a tiny MoE with a GRPO-style objective. The model had 59,296 parameters, two layers, four experts per layer and top-2 routing. In both rounds we ran three seeds (0, 1 and 2), each with three arms: BF16, fixed MXFP8 and adaptive MXFP8.

In the adaptive arm a layer triggered only when its MXFP8 gradient had a negative cosine similarity to its BF16 reference and both gradient norms exceeded 1e-8. Training used this same threshold throughout. Fault injection was confined to a separate unit test.

## Round 1 · 40 SFT warmup steps and 60 RL steps

```bash
uv run qorerl research --steps 60 --warmup-steps 40 --seeds 0,1,2 \
  --out results/wgrad-cpu-20261108
```

Each adaptive run calibrated at RL steps 20, 40 and 60 (nine checks total). Greedy token accuracy on the fixed evaluation prompts had already reached 100% after warmup, although sampled reward still had room to improve.

The trigger fired once, at step 40 of seed 0 for `blocks.1.moe.experts.2.gate_up`. The intervention used BF16 activations only for that layer's weight-gradient calculation.

| Measurement | Before intervention | After intervention |
| --- | ---: | ---: |
| Layer gradient cosine vs. BF16 | -0.11675442 | -0.11680047 |
| Layer gradient norm | 1.02580129e-5 | 1.02569306e-5 |
| Whole-model gradient cosine vs. BF16 | 0.99565213 | 0.99565213 |
| MXFP8 policy loss after one shadow AdamW step | -0.004993015 | -0.004993015 |
| BF16 policy loss after one shadow AdamW step | -0.006765507 | -0.006765507 |

The layer's BF16 reference gradient norm was 1.37021896e-6. The local gradients pointed in opposite directions, but both were small and the whole-model directions remained close. The intervention left the local cosine negative. The two trial updates produced the same losses at the precision shown. [Full trigger event](saturated-trigger-events.json) · [Report](saturated-report.md) · [Configuration](saturated-config.json) · [Raw summary](saturated-summary.json)

## Round 2 · 5 SFT warmup steps and 100 RL steps

We shortened warmup to leave room for learning on the evaluation task. We kept the model and trigger thresholds from round 1.

```bash
uv run qorerl research --steps 100 --warmup-steps 5 --seeds 0,1,2 \
  --out results/wgrad-learning-20261108
```

Each adaptive run calibrated at steps 20, 40, 60, 80 and 100 (15 checks total).

| Seed | BF16 greedy token accuracy, initial → final | Fixed MXFP8, initial → final | Adaptive, final | Triggers |
| --- | --- | --- | ---: | ---: |
| 0 | 0.2383 → 0.9102 | 0.2266 → 0.8789 | 0.8789 | 0 |
| 1 | 0.1289 → 0.6602 | 0.1328 → 0.6484 | 0.6484 | 0 |
| 2 | 0.1367 → 0.9648 | 0.1641 → 0.9570 | 0.9570 | 0 |

All three arms for each seed started from the same weights. Forward precision accounts for the small differences in initial scores. The fixed evaluation prompts measure progress on this task. They do not measure generalization to an independent held-out set.

The minimum valid layer cosine across calibration steps was 0.66350, 0.93921 and 0.92670 for seeds 0, 1 and 2, respectively. No layer met the trigger. We compared every parameter in the final fixed and adaptive MXFP8 checkpoints and found exact equality for all three seeds. This checks that the additional BF16 probes and shadow updates left the live training parameters untouched when no intervention fired. The per-step logs and checkpoints were stored locally under `results/wgrad-learning-20261108/`. [Report](learning-report.md) · [Configuration](learning-config.json) · [Raw summary](learning-summary.json)

## Implementation checks

At the time of the experiment, `uv run pytest -q` passed 11 tests. Coverage included E4M3 rounding and subnormals, row and column quantization, tail blocks shorter than 32, and isolation of Wgrad-only changes from forward and Dgrad computation. Tests also checked model, Adam and RNG state during calibration, missing or near-zero gradients, and triggers at steps 20 and 40. A separate fault-injection test exercised the trigger branch.

## Interpretation

The experiment exercised periodic comparisons, an automatic trigger and a Wgrad-only intervention. The one natural intervention failed to improve the local gradient direction. In the run with measurable learning, no intervention fired, so its effect on learning quality remains open. Testing on larger models requires matching the native kernel's operands and saved tensors to the emulation, then evaluating on a task that can distinguish the alternatives.
