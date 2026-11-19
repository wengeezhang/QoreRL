# Local Ray GRPO integration run

On November 17, 2026, the `torch_test` engine ran a CPU integration test on macOS ARM64 with Python 3.12.13, PyTorch 2.15.0 and Ray 2.60.0.

```bash
uv run qorerl train --config configs/grpo_cpu_test.json --out results/ray-grpo-local-v1
```

| Measurement | Result |
| --- | --- |
| Processes | Driver, one learner and two rollout actors, each with a distinct PID |
| Training | 40 complete rollout, update and publication cycles, with 40 AdamW updates |
| Samples per step | Eight groups of eight responses, with advantages computed within each group |
| Weight publication | Initial version 0 through final version 40 |
| Mean sampled reward, first 10 steps | 0.0890625 |
| Mean sampled reward, last 10 steps | 0.23125 |
| Maximum per-step `abs_log_ratio_mean` | Approximately 1.15e-7 |
| Checkpoints | Steps 0, 20 and 40 |

These rewards came from training rollouts. Evaluation on an independent test set would be needed to assess generalization. The task generates a single digit token, so `finish_reason=length` is common. With one training epoch and matching rollout and learner models, the pre-update ratio is close to one. Centering the within-group advantages gives a loss value near zero while still allowing a nonzero gradient.

This directory contains the [configuration](config.json), [per-step metrics](metrics.jsonl), [worker records](workers.json) and [run summary](summary.json). Absolute checkpoint paths in the summary refer to the run host. The large checkpoints and full trajectories were kept under `results/ray-grpo-local-v1/` on that host.

A multi-process Ray test in [test_ray_training.py](../../../tests/test_ray_training.py) compared three uninterrupted steps with two steps followed by a checkpoint, restart and third step. The final model parameters and Adam states matched element for element. Other tests covered rejection of missing, duplicate or stale trajectories, response and EOS masks, aggregation across uneven microbatches, skipping momentum updates for all-zero advantages, and withholding version acknowledgments when an adapter fails. The adapter tests used mocks.

For this run, `uv run pytest -q` passed 32 tests, including tests of the early research prototype. Both `uv run ruff check qorerl tests` and `git diff --check` passed.

## Regression check after the worker and engine refactor

Later that day, after separating worker and engine responsibilities, renaming fields and removing the early research prototype, `uv run pytest -q` passed 37 tests. Both Ruff and the diff check passed again. New coverage included schema 1 and 2 checkpoint recovery, the subsequent Adam update, and version and recipe checks.

The refactored code resumed directly from the original run's schema 1 checkpoint:

```bash
uv run qorerl train \
  --config configs/grpo_cpu_test.json \
  --resume results/ray-grpo-local-v1/checkpoints/step-000020 \
  --out results/ray-grpo-refactor-legacy-resume
```

After accounting for renamed fields and excluding timing, trajectories, behavior log probabilities, rewards, advantages and all remaining per-step metrics matched steps 21–40 of the original run. All 36 tensors covering model parameters, FP32 master parameters, Adam state and Torch RNG state matched element for element at step 40. Both runs recorded 40 optimizer updates. The resumed run wrote a schema 2 checkpoint and stored its outputs locally under `results/ray-grpo-refactor-legacy-resume/`.

This establishes CPU recovery compatibility across the refactor. The next validation is a native Megatron/vLLM run on two NVIDIA GPUs using `configs/grpo_megatron_vllm.json`, checking BF16 log-probability differences, Bridge export and vLLM refit, groups with nonzero advantages, and checkpoint recovery. [Native execution guide](../../ray-grpo.md)
