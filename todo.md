# QoreRL development roadmap

The checkboxes track implementation. Native GPU validation and evidence of a benefit are separate milestones. See the [design document](design.md) for the architecture.

## P0 · Validate the native BF16 baseline

- [x] Add GPU environment preflight and a separate JSON/Markdown smoke audit for real updates, checkpoint tensor changes, publication acknowledgments and sampled-token log-prob gaps. Record executing-process versions and allocator peaks. Local tests cover acceptance logic; native execution, recovery and task quality remain unvalidated. [Runbook](docs/gpu-validation.md)

- [x] Add explicit separate/collocated Ray placement, serialized collocated startup, a configurable vLLM memory budget, and placement changes on checkpoint resume. CPU workflow and logical-GPU scheduling tests cover the control path; CUDA coexistence remains unvalidated.
- [ ] Run Megatron and vLLM on one NVIDIA host, using separate GPUs or single-GPU collocation. Measure both processes' peak memory after optimizer updates, weight refits and calibration. Record the GPU, CUDA, PyTorch, Transformer Engine (TE), Bridge, Megatron Core and vLLM versions, and the exact commands.
- [ ] Score the same token sequences on both engines at the same snapshot. Check for non-finite log probabilities and inspect the tails of the differences. Measure policy changes due to an optimizer update separately.
- [ ] Complete at least two real optimizer updates and refits with nonzero within-group advantages. Check that weight contents change, every target receives its update, and inference uses the new weights.
- [ ] Check loss and gradient normalization with EOS, length truncation, variable-length samples and different microbatch sizes.
- [ ] Interrupt and resume training, then inject a publication failure. Verify optimizer, progress and version recovery, and that a partially updated replica stops serving. Measure reproducibility on GPU.
- [ ] Publish a validation report. Record a completed native run as `native_gpu_executed`. Review the numerical and recovery checks separately before marking the baseline validated.

The [execution guide](docs/ray-grpo.md) gives the entry points and acceptance criteria. The native GPU run is pending.

## P1 · Validate the first native low-precision recipe

The [MXFP8 and Hadamard guide](docs/recipes/mxfp8-hadamard.md) describes the implementation. Measuring its GPU benefit requires the reproducible BF16 baseline from P0.

- [x] Implement native MXFP8 training, export and rollout for selected MLP down projections, with a fixed Hadamard comparison. Specify the precision of the forward pass, input gradient (Dgrad) and weight gradient (Wgrad). Keep other layers in BF16.
- [x] Define recipe specifications, include options in their identifiers, save fixed signs and targets, and check compatibility on resume. Dynamic SmoothQuant also versions its smoothing factors by content.
- [x] Validate model, shape, hardware and backend support at startup. Report unsupported combinations explicitly.
- [ ] Validate NVFP4 forward, Dgrad, Wgrad, candidate calibration and publication on native GPUs with matched weights and inputs. Record numerical results on both sides of refit, recovery behavior, gains and overhead.
- [ ] Run operator probes and all three GRPO configurations on GPU. Verify nonzero updates, refit and resume, and compare the native quantizers.
- [ ] For each recipe, publish its source, reproducible configuration, workload, numerical results, training quality, memory and end-to-end cost. Include failures and neutral results.
- [x] Keep canonical weights, rebuild vLLM's rotated and quantized weights and kernel-layout caches after refit, and acknowledge a version only after successful installation. GPU execution checks are pending.
- [ ] Reuse SmoothQuant's transformed and quantized training weights across microbatches within one optimizer step. Implementation is deferred until native correctness and profiling are established.
  - Keep canonical model weights and FP32 optimizer master weights unchanged in meaning. Build `W_eff = W * S` and its quantized representations once while both `W` and `S` remain unchanged, then reuse them for subsequent microbatches. Inputs still require their own transform and quantization.
  - Preserve autograd connectivity to canonical weights and the mapping `dW = dW_eff * S`. Cache numerical representations without reusing a backward graph that has already been freed.
  - Invalidate on every actual optimizer update, smoothing-factor change, checkpoint restore or weight reload, and incompatible device, dtype or layout changes. Do not use only the published `weight_version`: one learner update can contain multiple optimizer steps.
  - Compare cached and uncached outputs, input/weight gradients and parameter updates across gradient accumulation, multiple update epochs, factor refresh and resume. Measure weight transform/quantization calls, GPU time, peak memory and end-to-end training cost; fewer repeated operations alone do not establish a speedup.
- [ ] Once measurements establish a benefit, optimize the FWHT CUDA kernel, repeated weight transforms, cache memory use and synchronization cost. Then expand to MoE and parallel execution.

Acceptance requires reproducible BF16 and MXFP8 comparisons on the same workload and a way for users to confirm the configuration that ran. Each additional method and combination needs its own evaluation.

## P1 · Observe recipes during training

[Recipe Insights](docs/recipe-insights.md) describes the current probes and what they measure. Native GPU measurements are pending.

- [x] Add opt-in probes that rotate through two layers every 20 steps by default. Bound the input rows and bytes copied from canonical weights. Replay the three forward configurations with paired inputs and weights captured before the update, and record versions and failures.
- [x] Reuse SmoothQuant calibration logs for per-window current/candidate charts, outlier evidence, and review/publication history across explicitly linked run segments. Export HTML/SVG/PNG and label CPU simulation separately. No additional model replay; native measurements pending.
- [x] Provide a live layer view, standalone HTML reports and a clearly labeled synthetic demo. Allow diagnostics to change on resume independently of the training recipe.
- [ ] Validate sampling, CUDA event timing and error measurements on native GPUs. Check RNG, parameters, gradients and training outcomes with and without probes. Measure total wall time and peak memory directly in addition to `probe_seconds` and the copy budget.
- [ ] Add Dgrad, Wgrad and gradient-direction probes when a concrete question warrants them. Retain paired references and account for the cost of saving upstream gradients and activations. Measure training quality separately.
- [ ] Compare train and rollout log-probability tails on the same snapshot and tokens, alongside end-to-end quality and cost. Define how to compare observation windows and report their coverage when layers are sampled at different steps.

## P1 · Observe, pause, adjust and resume

[Dynamic SmoothQuant](docs/recipes/dynamic-smoothquant.md) exercises this workflow.

- [x] Support known targets or automatic discovery, periodic candidate preparation, comparisons on held-out rows, manual or rule-based approval, checkpointing and coordinated publication. Allow direct import of fitted smoothing factors.
- [ ] Check checkpoint recovery and both engines' actual outputs on native hardware. Report local error, train–rollout log-probability differences, task quality, memory and time separately.
- [ ] Use independent prompts across batches to evaluate refresh intervals, candidate filters and joint application to several layers before expanding automatic approval.
- [ ] Add full-model candidate validation after local screening and before approving/publishing a change. **Deferred; document the design now, do not implement this stage yet.** Current per-layer comparisons use inputs captured under the active recipe and do not measure the complete proposed combination. See the [calibration rationale and limits](docs/smoothquant-calibration.md#what-happens-when-several-layers-qualify-together).
  - At fixed canonical weights, replay the same held-out token sequences with the current configuration and with all proposed layer changes together. Each execution must generate its own downstream activations, so upstream/downstream interactions are included. Use prompts independent of fitting data; record output or token log-probability changes, non-finite values and the acceptance criteria.
  - Keep an optional full-model BF16 diagnostic reference separate from the existing local `F.linear(X, W)` reference. BF16 replay uses current weights and is not a counterfactual BF16-trained model. Numerical comparisons do not establish task reward or convergence gains; those need separate evaluation.
  - Keep candidate evaluation provisional, restore active factors and affected caches on rejection or failure, and preserve optimizer and training state. Bind evidence to the candidate and source policy; measure additional replay time and peak memory. Test cases where layers pass individually but their combined change fails the full-model criterion.
- [ ] Distinguish invalid numerical states, statistical anomalies and loss of training quality. Record evidence, possible causes and supported adjustments. Treat outliers and negative gradient cosines as observations that need investigation.
- [ ] Make warning and pause conditions configurable and define safe stopping points. Save incidents separately from the most recent complete recovery checkpoint.
- [ ] Check compatibility after configuration changes and implement explicit migrations. Specify which model, optimizer and recipe states to preserve, reset or rebuild.
- [ ] Extend checkpoints when adding a data cursor, scheduler or randomized recipe state. Record the schema, configuration and source checkpoint.
- [ ] Coordinate learner and rollout for changes to the forward policy. Clear stale in-flight data, republish and sample again. Keep compatibility checks for existing runs until each migration is defined.

Acceptance means a user can select a supported adjustment from the evidence, resume, verify what took effect, and compare quality and cost before and after. General automatic selection and hot switching are later work.

## P2 · Extend execution and interfaces as workloads require

| Work item | When to start | Acceptance criteria |
| --- | --- | --- |
| Multi-rank native learner | A model needs model or data parallelism after single-rank GPU validation | Use Megatron's native optimizer, schedule, gradient synchronization and sharded checkpointing. Allow a multi-rank update loop. |
| Pluggable policy objective | A second algorithm or correction recipe needs it | Separate objective selection from backend execution. Preserve gradient, mask and loss aggregation semantics. |
| Token log-probability or fused-loss interface | Vocabulary parallelism or logit memory becomes a bottleneck | Avoid gathering full `[B,S,V]` logits and verify parallel loss and gradient equivalence. |
| Sharded, bucketed or GPU weight transfer | Full CPU export becomes a memory or synchronization bottleneck | Separate version metadata from the transfer payload or handle. Acknowledge only a complete installation and stop serving after a failed install. |
| Independent task and reward selection | A second task or dataset arrives | Decouple the task from the backend. Support independent scorers and recovery of dataset state. |
| Dynamic batching and sequence packing | Padding or variable-length workloads measurably raise cost | Preserve per-response and valid-token reductions. Check masks at boundaries, reordering and gradient accumulation. |
| Cache, CUDA graph and performance work | Native correctness is established and a profile identifies the cost | Verify cache and graph references after refit. Measure actual execution, numerical behavior and cost. |
| Asynchronous rollout and training | Measurements identify synchronous waiting as a major bottleneck | Decouple published versions from batch steps. Define staleness, correction and discard policies. Retain the behavior log probabilities and version used to collect each sample. |
| Extend Ray resource and lifecycle management | Worker groups, offload or another scheduling scheme needs it | Build on the separate/collocated placement boundary; keep scheduling, memory lifecycle and training orchestration responsibilities explicit. |

Execution can evolve while sample identity, behavior log probabilities, recipe identity, publication acknowledgments and recovery boundaries stay explicit. Use versioned schemas and migrations for protocol changes.

## Long-term research · From candidate comparisons to adaptation

- [ ] Revisit BF16 activations for Wgrad, selective higher precision, scaling, rotations and their combinations on native kernels. Include the costs of retaining activations, calibration, transforms and synchronization.
- [ ] Compare interventions on the same snapshot and samples to isolate them from distribution changes. Use separate data to select and evaluate candidates.
- [ ] Test whether gradient diagnostics predict learning gains. Establish evidence for the scope of any proposed error threshold.
- [ ] Use reproducible evidence to recommend supported adjustments for user approval, then study automatic actions with explicit budgets, acceptance conditions, follow-up measurements and recovery.

In the [archived tiny-MoE experiment](docs/experiments/2026-11-08/README.md), a negative gradient cosine triggered an intervention that failed to improve the local direction. The archive records the evidence behind that finding. Further validation is needed before adopting the intervention as a recipe or default calibration rule.
