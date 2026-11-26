# GPU experiment preparation and smoke acceptance

**English** · [简体中文](gpu-validation.zh-CN.md)

The tools below prepare an experiment and audit its saved evidence. Their implementation is locally tested; **native GPU execution remains pending**. A passed smoke audit establishes a limited execution check. It does not certify recovery correctness, convergence, or a SmoothQuant benefit.

## Choose the hardware and configuration

| Experiment | Configuration | Hardware for the current adapters |
| --- | --- | --- |
| BF16 baseline, one learner + one rollout on one card | [`grpo_single_gpu.json`](../configs/grpo_single_gpu.json) | H200 or B200; actual memory coexistence still needs measurement |
| MXFP8 + dynamic SmoothQuant, the same single-card layout | [`grpo_single_gpu_smoothquant.json`](../configs/grpo_single_gpu_smoothquant.json) | Supported Blackwell SM10/SM11 with TE 2.18.x; B200 is the intended first target |
| Experimental NVFP4 + SmoothQuant | Change `recipe_options.precision` to `nvfp4` | Supported Blackwell SM10.x with TE 2.18.x; separate acceptance required |

H200 is Hopper. Its FP8 support does **not** provide the native MXFP8/NVFP4 path used by these adapters. [NVIDIA Transformer Engine support matrix](https://docs.nvidia.com/deeplearning/transformer-engine/support_matrix.html). See the [native setup guide](ray-grpo.md) for dependencies and backend environment isolation.

Both single-card configurations use Qwen3-1.7B, one rollout worker and microbatches of one. Collocation reserves one logical GPU and assigns both workers to it; it does not partition VRAM. The learner remains resident, and vLLM's memory fraction is 0.2. The SmoothQuant configuration selects low-precision layers `[7, 15, 20, 25]`, starts S on `[7, 15, 20]`, and evaluates candidates every 20 **actual optimizer updates** with manual approval. Layers outside that low-precision set remain BF16.

Start with the 1.7B BF16 baseline, then the low-precision experiment. Move to 4B only after measuring memory and correctness at 1.7B. No published memory-fit or speedup claim follows from the configurations alone.

## Preflight before loading the model

Run in the environment that will execute the engines:

```bash
uv run qorerl gpu-preflight --config configs/grpo_single_gpu.json \
  --out results/preflight-bf16
uv run qorerl gpu-preflight --config configs/grpo_single_gpu_smoothquant.json \
  --out results/preflight-smoothquant
```

If learner and rollout use different environments, execute once inside each with `--role learner` and `--role rollout`. `--role all` checks the locally visible resources for the whole requested layout, not remote Ray nodes. Package presence is recorded, not treated as proof of binary compatibility. Low-precision preflight also calls the adapter's hardware and TE API guard. It does not load the model or run a forward/backward pass.

Each invocation writes `report.json` and `report.md` to a **new** directory. Exit codes are `0` for `passed`, `1` for `failed`, and `2` for `incomplete`. None of these commands allocates a cloud instance.

## Run, inspect and review

```bash
uv run qorerl train --config configs/grpo_single_gpu_smoothquant.json --out results/sq-review
uv run qorerl inspect --run results/sq-review
```

A manual proposal ends the run segment with `awaiting_review`, after saving the checkpoint and candidate. Review the evidence, then use the `checkpoint` and `candidate_id` in `summary.json`:

```bash
uv run qorerl train --config configs/grpo_single_gpu_smoothquant.json \
  --resume CHECKPOINT_PATH --accept-candidate CANDIDATE_ID --out results/sq-accepted
uv run qorerl inspect --run results/sq-accepted --related-run results/sq-review
uv run qorerl report --run results/sq-accepted --related-run results/sq-review \
  --out results/sq-accepted/calibration.html
```

The viewer reuses `calibration.jsonl`; it adds no capture or replay. Each selected window keeps one weight version, its own paired errors and its decision history. `--related-run` explicitly supplies earlier segments; the viewer never follows checkpoint paths found in logs. It supports older records without evaluation IDs through same-run boundaries and matching candidate IDs. SVG/PNG exports show the selected window. **Save HTML** embeds its data and selection.

No candidate is a valid experimental outcome. An outlier need not benefit from a newly fitted S. `no_gain`, unselected targets and missing comparisons remain visible. Calibration timing measures capture, fitting and comparison, not the low-precision operator's latency. The plot cannot establish reduced FLOPs or faster end-to-end training.

## Audit a completed native run segment

Select a maximum absolute token log-probability gap **before** judging the run, based on a matched BF16 reference and the intended numerical budget. There is no universal default. The command below leaves `MAX_LOGPROB_GAP` to that experimental decision:

```bash
uv run qorerl validate-run --run results/sq-accepted \
  --related-run results/sq-review --out results/sq-acceptance \
  --min-updates 2 --max-logprob-gap "$MAX_LOGPROB_GAP" \
  --check-checkpoints --require-calibration
```

For a BF16 baseline, omit `--related-run` and `--require-calibration`. The auditor checks:

| Evidence | What must be recorded |
| --- | --- |
| Execution and placement | Native backends, a completed segment, worker node/GPU assignments, no incident |
| Environment and memory | CUDA, device, package versions and allocator peaks from the actual learner and vLLM executing processes |
| Real training | Finite loss/gradient diagnostics, nonzero group advantages and at least the requested actual optimizer updates **within this segment** |
| Same-snapshot policy comparison | First-update-epoch mean and maximum absolute sampled-token log-prob gaps; the maximum meets the user-supplied bound |
| Publication and sampling | Every publication has matching rollout acknowledgments; each batch uses the last acknowledged preceding policy; saved log-probs are finite |
| Checkpoint contents | Initial/final local model tensors are finite, some changed, update counts advanced and optimizer state exists |
| Optional calibration path | A linked paired candidate comparison, successful application at unchanged weight version with a new recipe identity, and subsequent sampling with that identity |

`--check-checkpoints` loads two local checkpoints on CPU with `weights_only=True`; budget host RAM for both plus deserialization overhead. Without this option or a log-prob bound, the audit is incomplete. A CPU fixture cannot pass native acceptance. All-zero group advantages advance batch versions without updating Adam; they cannot satisfy the update requirement. If a native arithmetic run yields no useful reward variation, adjust the task/data experiment and rerun—do not use its completed status as training evidence.

A run paused for another manual review is incomplete until continued. Each resumed segment must itself contain enough updates for this audit; related segments provide calibration provenance, not extra update counts. For bounded memory use, full validation refuses JSONL files larger than 16 MiB and incomplete final lines; use shorter experimental segments. Live visualization instead reads a bounded recent tail and warns about missing history.

## What remains outside this audit

- Compare uninterrupted and resumed execution, including optimizer and recipe state, on GPU. Inject publication failures and verify that partially updated replicas stop serving.
- Independently compare outputs and loaded tensor contents after refit. An acknowledgment records the software protocol, not an independent tensor measurement.
- Verify gradient normalization across EOS, truncation, sequence lengths and microbatch sizes. The mean/max log-prob checks cover sampled response tokens, not full-vocabulary KL.
- Measure device-wide memory under load. Recorded peaks are per-process **PyTorch allocator** peaks; they exclude some native allocations and cannot be added to obtain a simultaneous device peak.
- Compare BF16, direct MXFP8, fixed-S and dynamic-S runs on matched workloads and held-out task metrics, with repeated seeds. Report calibration/publication overhead and time to quality alongside local error.

The generated report lists these limits and never changes `summary.json`'s `native_gpu_acceptance` field. Full GPU validation and evidence of a recipe's benefit remain separate reviewed milestones.
