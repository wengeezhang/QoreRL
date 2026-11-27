# SmoothQuant calibration: from training inputs to a visible decision

**English** · [简体中文](smoothquant-calibration.zh-CN.md)

QoreRL separates three responsibilities: **the learner and recipe produce calibration evidence; the driver records, reviews and publishes a change; the viewer reads and presents that evidence.** Opening a page, refreshing a chart or exporting a report does not load a model or run another GPU calibration. Calibration itself does perform additional forward computation.

This document follows the current implementation through those three responsibilities. For recipe choices and hardware setup, see the [SmoothQuant guide](recipes/dynamic-smoothquant.md). For native experiment acceptance, see [GPU validation](gpu-validation.md). The walkthrough below uses real CPU training and quantization simulation, not native GPU measurements.

Two evaluation scopes share the same review and publication workflow:

| `recipe_calibration.evaluation_scope` | Fitting inputs | Acceptance measurement |
| --- | --- | --- |
| `local` (default) | Captured under the active recipe | Each projection compared with its own local BF16 output; a diagnostic tool |
| `model_logits` (opt-in) | Captured during full-model BF16 replay | The complete fitted combination compared with the current recipe, both against full-model BF16 logits |

## 1. The complete path

```mermaid
flowchart TD
    A[Training batch finishes<br/>Updated weights published and acknowledged] --> B{Optimizer counter<br/>crossed the calibration interval?}
    B -->|No| NEXT[Next training batch]
    B -->|Yes| C
    subgraph COMPUTE[1 · Produce evidence — learner and recipe]
        C[Replay distinct token prefixes<br/>Current weights + selected evaluation scope<br/>Capture inputs before the local transform]
        C --> D[Discover outliers on fitting rows<br/>Fit candidate smoothing factors S]
        D --> E[Compare current and candidate<br/>on held-out rows<br/>Local outputs OR full-model logits]
    end
    E --> F
    subgraph RECORD[2 · Record and decide — driver]
        F[Append evaluated + comparison evidence<br/>to calibration.jsonl]
        F --> G{Any qualifying candidate?}
        G -->|Yes| H[Save pre-change checkpoint<br/>candidate.json + metadata.json + learner.pt]
        H --> I{Manual or automatic decision}
        I -->|Approve| J[Apply recipe state · publish to rollout<br/>Wait for all acknowledgements · append applied]
        I -->|Reject| K[Keep current state<br/>Append rejected on resume]
    end
    G -->|No| NEXT
    J --> NEXT
    K --> NEXT
    F -. Read recorded evidence .-> V
    J -. Read publication outcome .-> V
    K -. Read rejection outcome .-> V
    subgraph VIEW[3 · Present evidence — no model execution]
        V[Read config.json and calibration.jsonl<br/>Join explicitly supplied run segments]
        V --> W[Select one evaluation window<br/>Errors · outliers · candidate and publication states]
        W --> X[Live view · standalone HTML · SVG / PNG]
    end
    classDef compute fill:#edf4fb,stroke:#7192ae,color:#18323b
    classDef record fill:#e7f4f0,stroke:#4b9a8b,color:#18323b
    classDef view fill:#faf4e8,stroke:#b7a073,color:#18323b
    class C,D,E compute
    class F,G,H,I,J,K record
    class V,W,X view
```

Manual approval pauses **by exiting the training process and releasing workers**. The user reviews the saved proposal, then resumes into a new output directory with an explicit acceptance or rejection. Automatic approval uses the same checkpoint, application and publication path in the current run. A discovery-only run stops after recording outlier evidence and produces no candidate.

## 2. How calibration produces evidence

### Configure the scope, schedule and decision

The following fragment belongs in a training config. Layer numbers are zero-based. It defines an immutable low-precision scope and a smaller initial SmoothQuant overlay:

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
    "discovery_only": false,
    "outlier_threshold": 4.0,
    "alpha": 0.5,
    "max_rows": 128,
    "min_relative_improvement": 0.0,
    "max_activation_relative_l2": 0.05,
    "approval": "manual"
  }
}
```

Layers 7, 15, 20 and 25 all use MXFP8 from the start; only 7, 15 and 20 start with S. Expansion observes all four and may propose S for layer 25. It **cannot turn another BF16 layer into a low-precision layer, or switch to Hadamard**. With expansion disabled, observation is limited to layers already using SmoothQuant. The optional first-activation error cap above is 5%; its default is `null`, meaning no additional cap.

The driver calls `crosses_interval(before, after, interval)` after the normal training update and weight publication. It tests `before // interval < after // interval`, with a positive interval. The counter is **actual optimizer updates**, not rollout batches, microbatches or viewer refreshes. Moving from 19 to 22 updates causes one calibration at that batch boundary; a batch that makes no optimizer update does not advance this schedule. Restore preserves the counter.

Relevant code: [configuration and shared discovery](../qorerl/recipes/calibration.py), [scope in RecipeState](../qorerl/recipes/state.py), [interval test](../qorerl/runtime/calibration.py), [driver](../qorerl/runtime/ray_runtime.py).

### Local mode: collect inputs from the just-consumed batch

`SingleRankGRPOLearner.prepare_candidate(batch)` validates the sample policy, collects inputs and dispatches to `RecipeSpec.candidate_handler`. SmoothQuant owns the fitting and application functions; the driver does not calculate S.

The learner replays **recorded tokens** using its updated model. It does not ask rollout to sample another set of responses. `_calibration_prefixes` builds contexts from `prompt_ids + response_ids[:-1]`, deduplicates shared causal prefixes and bounds the total number of new token rows. `_sample_calibration_inputs` replays one unpadded sequence at a time under `no_grad` and captures up to `max_rows` per observed projection.

Replay keeps the **currently installed execution configuration**. In the example above, the down projections in layers 7, 15 and 20 execute MXFP8 with their active smoothing factors `S_old`; layer 25's down projection executes direct MXFP8 until SmoothQuant is approved for it. Projections outside the low-precision scope keep their BF16 path. Collection does not switch the whole model to BF16 or install candidate factors.

On the native learner, `MegatronRecipeInstallation.forward` calls `calibrator.capture(name, x)` **before** `transform_input` and `transform_weight`. It copies the target `down_proj` input, then continues forward with the active recipe. This is not a collection of whole Transformer-block outputs: attention and other MLP operations also contribute to the captured input. The capture routine detaches and clones the retained rows, so later transformations do not overwrite them.

Thus X precedes **this projection's own** SmoothQuant transformation, but already contains the effects of upstream layers, including their quantization. Its BF16 storage dtype does not make it an activation from an all-BF16 model. Canonical W is the current model's BF16 weight before the local recipe transform, stored as `[output_channels, input_channels]`.

`activation_windows` splits the bounded rows in order: the first `floor(n / 2)` rows fit the candidate, and the remainder evaluate it. Identical token prefixes are not repeated across this split, but both halves can still come from the same response; this is not an independent task-level validation set. Discovery-only mode uses all rows for discovery.

The driver records both `sample_policy` (the policy that generated the tokens) and `policy` (the updated policy used for evaluation). Raw activation matrices and replay outputs remain temporary in-memory data; **`calibration.jsonl` is an evidence log, not an activation dataset**.

Relevant code: [prefix replay and capture](../qorerl/engines/single_rank_learner.py), [native projection adapter](../qorerl/recipes/megatron_mxfp8.py), [recipe dispatch](../qorerl/recipes/spec.py).

### Local mode: discover, fit and compare

For each input channel j, fitting computes:

```text
x_max[j] = max over fitting rows of abs(X[:, j])
w_max[j] = max over output channels of abs(W[:, j])
outlier_score = max(x_max) / mean(x_max)
S_new[j] = x_max[j]^alpha / w_max[j]^(1 - alpha)
```

The outlier score measures channel-magnitude imbalance within one layer; it is not the percentage of outlier tokens. Channels without usable nonzero activation and weight maxima retain their current factor; a newly activated layer starts from ones. Fitting uses FP64 arithmetic and stores the resulting positive factors in FP32. `S_new` is relative to canonical W and replaces the layer's old S, rather than multiplying it again.

**S is the SmoothQuant smoothing factor, not the quantizer's scale.** The transform pairs `X / S` with `W * S` along input channels. Quantization then derives its own block scales from those transformed operands. In exact arithmetic the paired transform preserves the linear output; finite precision and quantization are why we measure it.

For a selected layer, all three evaluations share the **same held-out X and canonical W**:

| Evaluation | Computation |
| --- | --- |
| Local reference | `Y_ref = F.linear(X_heldout, W)` in BF16 |
| Current configuration, `refresh` | Low-precision forward with the existing S |
| Current configuration, `activate` | Direct low-precision forward, without SmoothQuant |
| Candidate | The same low-precision kernel with `X_heldout / S_new` and `W * S_new` |

The measured error is `||Y - Y_ref||₂ / ||Y_ref||₂`. A proposal requires strictly smaller candidate error, relative improvement `(current_relative_l2 - candidate_relative_l2) / current_relative_l2 >= min_relative_improvement`, and a factor vector that actually changes. First activation also requires outlier evidence and satisfies `max_activation_relative_l2` when configured. With `discover_outliers: false`, existing SmoothQuant targets bypass the outlier threshold, but newly activated targets still need to meet it.

`prepare_smoothquant_candidate` returns **only qualifying factor vectors**, plus evidence for **every observed layer**. It does not mutate installed state. A high outlier score can still lead to `no_gain`. The CPU fixture injects a simulated kernel; the native adapter supplies its selected low-precision kernel.

Relevant code: [`activation_windows`](../qorerl/recipes/calibration.py), [`prepare_smoothquant_candidate`, `_fit_smoothing_factor`, `_measure_output_relative_l2`](../qorerl/recipes/smoothquant.py).

### Why the BF16 reference is local

The local comparison asks: **for inputs the active training configuration actually produces, does changing this projection's S reduce its forward error?** It holds the captured input, current canonical weight and low-precision kernel fixed, and varies the smoothing factors. This isolates the local effect of the change on those inputs; it is not a full-model ablation or a guarantee of minimal impact on training.

| Reference | Input and execution | What it establishes |
| --- | --- | --- |
| **Local BF16 comparison — implemented** | Reuse X captured from the active recipe, then compute only this projection as `F.linear(X, W)` in BF16. | A common reference for comparing the old and proposed S on the same input. Upstream low-precision effects remain in X. |
| **Full-model BF16 replay — used by `model_logits`** | Run the recorded tokens through the whole model in BF16, at the same current weights, and collect a separate set of inputs `X_bf16`. | A diagnostic reference that removes upstream low-precision quantization from that replay. It answers how the current weights behave under BF16 execution. |

Full-model BF16 inputs still depend on all preceding weights, attention and nonlinear operations, as well as BF16 rounding. An outlier may already exist in that computation, or upstream quantization may introduce or amplify it in the active path. Full-model BF16 replay can help investigate that distinction; it does not remove upstream influence altogether or identify a training failure's cause by itself. The weights also retain their training history: this replay would not reproduce a model trained entirely in BF16 from the beginning.

Fitting on `X_bf16` targets a different input distribution from the one the low-precision model currently encounters, so it is not automatically a better calibration source. Local mode fits on actual execution inputs and uses a shared local BF16 reference for comparison. It runs the old, reference and candidate **projection calculations** on held-out rows; it does not run three complete models or propagate those comparison outputs through later layers.

### What happens when several layers qualify together

In local mode, candidates are evaluated per layer, but one artifact can contain **several qualifying layers**, which are merged into the active state together after approval. Consider layers 7 and 20:

1. Replay with the active recipe collects `X7_old` and `X20_old`. Layer 7's proposed S is evaluated on `X7_old`; layer 20's proposed S is evaluated on `X20_old`. Preparation leaves the installed factors unchanged throughout these comparisons.
2. If both qualify, the candidate contains both updates. After application, layer 7's new S can change its quantized output. That change propagates through the intervening computations, so layer 20 may now receive `X20_new` rather than the `X20_old` used to qualify its candidate.

Layer 20's improvement on `X20_old` therefore does not establish an improvement on `X20_new`. **Individually better local candidates do not guarantee a better combined model.** Local mode neither recalibrates downstream layers after each upstream proposal nor replays the full candidate combination before publication. A later calibration observes the updated execution path, but it does not validate the combination before the intervening batches run.

The viewer's per-layer bars and averages must be read with that limitation: they are comparisons on previously captured inputs, not measurements of the published combination's end-to-end benefit. Coordinated publication ensures that learner and rollout acknowledge the same recipe state; it does not establish that the combination improves numerical or task quality.

The `model_logits` mode below evaluates the complete combination before review. It provides a model-level numerical check, while local mode remains available for projection-level diagnosis.

### Model-logits mode: fit on BF16 inputs, evaluate the complete combination

Replace the local error cap with an explicit model-level scope:

```json
"recipe_calibration": {
  "evaluation_scope": "model_logits",
  "refresh_every_optimizer_steps": 20,
  "max_rows": 128,
  "discover_outliers": true,
  "min_relative_improvement": 0.0,
  "max_logits_relative_l2": null,
  "approval": "manual"
}
```

`max_logits_relative_l2` optionally caps the candidate's joint logits error. It cannot be combined with the local-only `max_activation_relative_l2`. Scope, overlay expansion and approval permissions remain unchanged. Discovery-only uses `local`; joint evaluation requires a candidate-capable recipe.

```mermaid
flowchart TD
    A[Freeze current canonical weights and recorded token contexts] --> B[Full-model BF16 replay]
    B --> C[Capture authorized down_proj inputs<br/>Discover outliers and fit one set of factors]
    B --> R[Retain BF16 logits at held-out token positions]
    A --> D[Replay current low-precision recipe<br/>with current factors]
    C --> E[Replay low-precision model<br/>with all proposed factors together]
    R --> F{Does the whole candidate reduce<br/>logits relative L2 and satisfy limits?}
    D --> F
    E --> F
    F -->|Yes| G[Record joint evidence · review · publish]
    F -->|No| H[Record failure · keep current recipe]
    classDef replay fill:#edf4fb,stroke:#7192ae,color:#18323b
    classDef decision fill:#e7f4f0,stroke:#4b9a8b,color:#18323b
    class A,B,C,D,E,R replay
    class F,G,H decision
```

1. **Reference and discovery.** `CalibrationReplay` replays the bounded, deduplicated contexts with all installed low-precision projections bypassed. The native adapter calls the original BF16 forwards using canonical W. This also bypasses low-precision projections outside the editable SmoothQuant overlay; only authorized targets are captured. Fitting uses the first half of the distinct input rows, while the remaining positions supply BF16 reference logits.
2. **Fit the whole proposal.** `smoothquant_model._fit_joint_factors` discovers outliers on those BF16-path fitting rows and fits S for selected targets. It retains existing factors elsewhere. It does **not** discard a layer based on local projection error before testing the combination. Factors that do not change are omitted.
3. **Replay current and candidate.** At the same weights and on the same recorded tokens, run the current low-precision configuration, then a private copy of its state containing **all** proposed changes. Each run produces its own downstream activations. Thus a change to layer 7 can change the input of layer 20 during candidate evaluation. None of these runs samples new responses.
4. **Evaluate final logits.** For held-out positions and the entire vocabulary, compute `current_relative_l2 = ||Z_current - Z_bf16||₂ / ||Z_bf16||₂` and `candidate_relative_l2 = ||Z_candidate - Z_bf16||₂ / ||Z_bf16||₂`. The norm spans all retained entries; it is not a mean of layer errors. A candidate must strictly improve, meet `min_relative_improvement`, and satisfy the optional logits cap. Nonfinite candidate logits or an invalid fitted factor reject the complete proposal. A zero reference norm produces no proposal. Backend execution errors propagate as failures rather than being classified as numerical rejection.
5. **Review the combination as one change.** Only a jointly qualifying set becomes an artifact. Its model comparison is recorded and bound to the same review digest as its factors, source policy and checkpoint. Rejection leaves the current state intact. There is no automatic subset search. Approval still uses checkpointing, recipe application and coordinated publication before the next batch.

Preparation never updates canonical parameters or optimizer state. Provisional executions isolate recipe state and its device caches, disable probes, run in evaluation/no-grad mode, and restore incoming state references, module modes, call counters and Torch CPU/CUDA RNG even on failure. Native support remains the existing single-model-rank adapter. Local CPU tests include explicit native adapter stubs, not GPU execution.

The CPU fixture identifies its reference as `reference_execution: "cpu_fixture"`: its RNN/head remain FP32 and its low-precision arithmetic is simulated. A native reference is labeled `"bf16"`. Neither a CPU chart nor a passing CPU workflow is native BF16/MXFP8 validation.

**Limits.** BF16 replay removes low-precision execution effects from this reference, not the model's training history. The candidate is fitted to BF16 activations but evaluated with its own low-precision activations, so this distribution difference is measured rather than assumed harmless. The holdout excludes fitting token positions, but can share response history; independent prompt validation is still needed. Lower held-out logits error is not proof of RL convergence or reward improvement. This mode adds up to three full-model replays of the bounded contexts, stores reference/candidate logits on CPU, and temporarily retains captured activations. Native wall time and peak memory must be measured.

Entry points: [replay execution](../qorerl/engines/calibration_replay.py), [joint fitting and decision](../qorerl/recipes/smoothquant_model.py), [CPU example](../configs/grpo_cpu_smoothquant_logits.json), [single-GPU example](../configs/grpo_single_gpu_smoothquant_logits.json), [isolation and regression tests](../tests/test_model_calibration.py).

## 3. How evidence, candidates and decisions are written

### Each file has a different job

```text
results/sq-review/
├── config.json
├── calibration.jsonl
├── publications.jsonl
├── trajectories.jsonl
├── summary.json
└── checkpoints/step-000001-before-calibration/
    ├── metadata.json
    ├── candidate.json
    └── learner.pt
results/sq-accepted/                 # New directory on manual resume
├── config.json
├── calibration.jsonl               # applied/rejected; later evaluations if due
├── publications.jsonl
└── trajectories.jsonl
```

This is the relevant subset of files from a manual-review run, not an exhaustive directory listing.

| File | Producer and purpose | Read by the calibration viewer? |
| --- | --- | --- |
| `config.json` | Driver: resolved training configuration; identifies configured backend and recipe. | Yes |
| `calibration.jsonl` | Driver: append-only evaluations, per-layer evidence and decision events. One complete JSON object per line. | Yes |
| `candidate.json` | Checkpoint writer: proposed recipe artifact, evidence, source policy, evaluation ID and learner payload digest. | No; used for review and resume |
| `metadata.json` | Checkpoint writer: training progress, saved policy, config, boundary and pending candidate ID. | No; used for resume |
| `learner.pt` | Learner: model, master parameters, optimizer, counters, recipe state and other recovery state. | No |
| `publications.jsonl` | Driver: published policy and each rollout acknowledgement. | No; inspect for publication audit |
| `trajectories.jsonl` | Driver: sampled tokens, rewards, advantages and generating policy. | No; inspect to verify the next batch used the published policy |
| `summary.json` | Driver: completion/pause status, checkpoint path and pending candidate ID. | No; used to operate the run |

An `incident.json` file records a failure and recovery point. The viewer checks its presence and displays a warning; it does not recover training.

### Example: one layer's evidence

The following **illustrative** event is pretty-printed for reading; the actual JSONL writer emits one line. IDs are placeholders and the numbers are explanatory, not GPU results. A real `evidence` array contains every observed layer.

```json
{
  "event": "evaluated",
  "evaluation_id": "example-evaluation-001",
  "step": 20,
  "policy": {"weight_version": 20, "recipe_id": "<active-recipe-before>"},
  "sample_policy": {"weight_version": 19, "recipe_id": "<active-recipe-before>"},
  "optimizer_steps": 20,
  "evaluation_seconds": 0.25,
  "evidence": [{
    "target": "model.layers.7.mlp.down_proj.weight",
    "action": "refresh",
    "selection": "outlier",
    "outlier_score": 12.0,
    "threshold": 4.0,
    "alpha": 0.5,
    "fit_rows": 64,
    "heldout_rows": 64,
    "status": "proposed",
    "reason": null,
    "current_relative_l2": 0.08,
    "candidate_relative_l2": 0.05,
    "relative_improvement": 0.375
  }]
}
```

The 8% and 5% values are errors relative to the local BF16 output. The improvement is 37.5%, not 3%. First activation additionally records `direct_low_precision_relative_l2` and `max_activation_relative_l2` in the raw evidence.

Per-layer statuses and run-level events answer different questions:

| Per-layer status | Meaning |
| --- | --- |
| `discovered` | Selected by discovery; may be the final result in discovery-only mode. |
| `not_selected` | Did not meet the discovery threshold; no candidate comparison. |
| `proposed` | Passed the configured local or joint comparison; not yet proof of application. |
| `model_rejected` | A fitted layer was part of a rejected joint proposal; see `comparison.reason`. |
| `unchanged` | Joint fitting produced the already active factor vector. |
| `no_gain` | No strict improvement, insufficient requested gain, or unchanged factors. Both measured errors remain visible when available. |
| `activation_cost_exceeded` | First activation exceeds the configured output-error cap, despite an improvement. “Cost” here is error, not runtime or FLOPs. |
| `insufficient_rows`, `insufficient_signal`, `invalid_candidate` | Evidence or candidate is unusable; `reason` explains why. |

| Run-level event | When the driver writes it |
| --- | --- |
| `evaluated` | After learner preparation returns, even if there is no qualifying candidate. |
| `awaiting_review` | After saving a qualifying proposal and pre-change checkpoint, before exiting for manual review. |
| `applied` | After applying the approved proposal and completing required publication/acknowledgements. |
| `rejected` | On explicit rejection during resume; the saved active state remains in use. |
| `imported` | After applying an offline initial artifact and publishing it before first sampling. This has no newly measured comparison. |

`append_json` writes with `allow_nan=False` and ends each event with a newline. The driver is the writer; the learner returns the artifact and evidence over Ray. These logs are not a transactional database. Checkpoints use a temporary directory and rename it into place only after their files exist.

### Where the joint comparison is recorded

Model mode adds `comparison` to both the `evaluated` event and `candidate.json`. Per-layer `evidence` retains outliers, row counts and the joint decision's layer statuses, with local error fields set to `null`. This example is from the first batch of the CPU simulation config, **not a GPU measurement**:

```json
{
  "comparison": {
    "scope": "model_logits",
    "reference": "full_model_bf16",
    "activation_source": "full_model_bf16",
    "reference_execution": "cpu_fixture",
    "fit_rows": 4,
    "heldout_rows": 5,
    "current_relative_l2": 0.04761516112516803,
    "candidate_relative_l2": 0.03801908469524742,
    "relative_improvement": 0.20153405350650802,
    "min_relative_improvement": 0.0,
    "max_logits_relative_l2": null,
    "targets": [
      "model.layers.0.mlp.down_proj.weight",
      "model.layers.1.mlp.down_proj.weight"
    ],
    "status": "proposed",
    "reason": null
  }
}
```

This is one model-level result; it does not claim that each layer improved by 20.15%. All proposed factors enter review together, and the joint evidence is part of the review digest. A rejected evaluation still records its comparison and reason, but creates no applicable artifact.

### Bind approval to the exact saved proposal

The schema-2 `candidate.json` envelope contains `source_policy`, `optimizer_steps`, `candidate`, `evidence`, `evaluation_id` and `checkpoint_sha256`. Its nested SmoothQuant artifact contains the model/revision, base recipe identity, per-target factor vectors and their digest. Only qualifying layers are included in this candidate mapping; the checkpoint retains all active recipe state.

The artifact still uses the schema-1 names `kind: "smoothquant_scales"`, `scales` and `scale_digest` for compatibility. They contain **SmoothQuant factors**, not low-precision quantizer scales. Python code uses `smoothing_factors`; the JSON names are a preserved wire format.

`candidate_id` is the SHA-256 of the serialized review envelope, including the learner checkpoint digest. It is stored as `metadata.pending_candidate` and exposed in `summary.json` and event records. Resume checks the review ID, source policy, supplied decision ID and `learner.pt` digest. Do not hand-edit a saved candidate after reviewing its ID. Schema-1 reviews remain readable under their former `scale_artifact` field.

After acceptance, the recipe validates and merges the factors into `RecipeState`. S can change without an optimizer update: `weight_version` stays the same while the factor digest changes `recipe_id`. Rollout must receive that new state and rebuild transformed/packed weights. All replicas acknowledge the expected policy before the next sampling batch. On publication failure the driver stops and records a recovery point rather than continuing with mixed versions.

Relevant code: [review envelope and validation](../qorerl/runtime/calibration.py), [checkpoint publication](../qorerl/runtime/checkpoint.py), [application and logging](../qorerl/runtime/ray_runtime.py), [recipe application](../qorerl/recipes/smoothquant.py), [rollout packing](../qorerl/recipes/vllm_mxfp8.py).

## 4. How the viewer presents existing evidence

![Calibration view generated from a CPU simulation, not native GPU evidence](assets/smoothquant-calibration-cpu.png)

`qorerl inspect` starts the local HTTP server in `observability/viewer.py`. `load_report` calls `load_calibration` for the main directory and any explicit `--related-run` directories. It renders `calibration.html`; the live page polls `/api/report` every two seconds. Reading a new event updates the charts. This is **live log viewing**, not continuous model recalibration.

### One window, paired comparisons and separate outcomes

Each `evaluated` event becomes a selectable window. The page shows its run, batch, optimizer counter, evaluation policy, sample policy and evidence. Per-layer paired error bars compare current/candidate values; the second chart displays outlier scores and their thresholds. The layer table includes `activate`/`refresh`, fitting/held-out row counts, status and reason. The event table separately displays linked approvals/publications.

In local mode, the summary computes equal-weight means of current and candidate errors over **all complete pairs in the selected window**, including `no_gain` candidates. It then displays `(mean_candidate - mean_current) / mean_current * 100`. A reduction is negative on the dashboard, whereas `relative_improvement` in evidence is positive. This summary is neither the effect of the applied combination nor an average of per-layer percentage improvements. “Proposed / observed” counts `status == "proposed"` against the number of evidence rows, not all model layers.

Missing values stay `null` and appear as `—` or `×`, never zero. Different weight versions are not merged into one layer comparison. The viewer recalculates relative improvement from each complete error pair. `evaluation_seconds` is preparation wall time, including the driver/Ray wait; `apply_publish_seconds` is application and publication wall time. Neither is operator latency or compute saved.

In `model_logits` mode the error chart contains **one current/candidate logits pair**, and the summary is its model-level error change. The outlier chart and layer table still show every observed layer. Per-layer output errors are `null`, because this mode does not perform local comparisons. The context shows joint status/reason, reference execution and held-out row count; HTML/SVG/PNG exports preserve the scope. The local-mode means above are never substituted for joint logits evidence.

![Joint logits view exported from the CPU simulation, not GPU validation](assets/smoothquant-logits-cpu.png)

### Join a decision across resume directories

The original run writes `evaluated` and `awaiting_review`; a resumed run writes `applied` or `rejected` in its own directory. Supply both explicitly:

```bash
uv run qorerl inspect --run results/sq-accepted \
  --related-run results/sq-review --view calibration --port 8765
```

`evaluation_id` links the evaluation to its review and decision; `candidate_id` also links outcomes for the same saved proposal. For older records without evaluation IDs, the reader associates an event with the preceding same-step evaluation **within that run**, then uses a matching candidate ID to connect explicit run segments. It does not guess that equal step numbers in different runs mean the same evaluation, or open paths mentioned inside logs. Several supplied decision branches produce `multiple_branches` instead of overwriting one outcome with another.

For live reads, `read_events` defers a final line without its terminating newline. Completed malformed records produce warnings. Reads are bounded to the latest 16 MiB per JSONL file with a visible warning if earlier history was omitted. The execution-source badge comes from each run's backend configuration (`cpu_simulation`, `native_gpu`, or `unknown`); it is not a GPU acceptance certificate.

### Export without rerunning calibration

| Output | Implementation and contents |
| --- | --- |
| CLI HTML report | `export_report` embeds the loaded evidence, page code and logo with live polling disabled. It needs no model, checkpoint tensors or network connection to view. |
| **Save HTML** in the page | Saves the current loaded report and selected evaluation window as an offline snapshot. |
| **SVG** in the page | `exportSvg` draws the selected window's source label, summary, error/outlier charts and measurement notes into a 1200 × 800 image. |
| **PNG** in the page | Rasterizes that SVG using a browser canvas at 2400 × 1600. |

Images summarize the selected window; they are not a full event archive. Keep the JSONL files and HTML when readers need to inspect other windows or individual outcomes. No viewer operation calls `prepare_candidate` or `apply_candidate`; decisions remain explicit training commands.

Relevant code: [file parsing and joins](../qorerl/observability/calibration.py), [HTTP server and HTML export](../qorerl/observability/viewer.py), [charts, polling and image export](../qorerl/observability/calibration.html), [CLI](../qorerl/cli.py).

## 5. Reproduce the complete workflow on CPU

The commands below demonstrate `local`. To exercise joint logits evaluation, use `configs/grpo_cpu_smoothquant_logits.json` in both train commands and new output directories. Review, related-run linking and export commands are unchanged. A later jointly qualifying candidate can pause again; follow that run's `summary.json`.


Run from the repository root with the project environment installed. Each training output directory must be new. The [CPU config](../configs/grpo_cpu_smoothquant.json) uses a tiny fixture, simulated low precision, three initial SmoothQuant layers and calibration every actual update.

```bash
uv run qorerl train --config configs/grpo_cpu_smoothquant.json --out results/sq-review
uv run qorerl inspect --run results/sq-review --view calibration --port 8765
```

Open `http://127.0.0.1:8765`. Inspect the first window and the saved `candidate.json`. In another terminal, read the exact recovery point and decision ID:

```bash
CP=$(uv run python -c 'import json; print(json.load(open("results/sq-review/summary.json"))["checkpoint"])')
ID=$(uv run python -c 'import json; print(json.load(open("results/sq-review/summary.json"))["candidate_id"])')
uv run qorerl train --config configs/grpo_cpu_smoothquant.json \
  --resume "$CP" --accept-candidate "$ID" --out results/sq-accepted
uv run qorerl report --run results/sq-accepted --related-run results/sq-review \
  --view calibration --out results/sq-calibration.html
```

Select the **original evaluation window** in the report to see its linked `Applied` decision. Later windows can have different outcomes. Use the SVG/PNG buttons to export that window. To inspect both directories live, stop the first viewer and run the command in section 4. To reject instead, resume the same pre-change checkpoint into another new directory with `--reject-candidate "$ID"`.

Rerunning the CPU config while writing this document produced the following first window, with four fitting rows and five held-out rows per layer. These numbers belong only to this simulated fixture:

| Layer | Outlier score | Current relative L2 | Candidate relative L2 | Outcome |
| --- | ---: | ---: | ---: | --- |
| 0 | 31.98 | 0.078228 | 0.047923 | `proposed`: propose a refresh |
| 1 | 31.62 | 0.033125 | 0.038617 | `no_gain`: keep the active S |
| 2 | 1.78 | — | — | `not_selected`: no candidate comparison |

After acceptance, the first window linked to `applied`, the resumed run completed batch 3, and HTML export succeeded. **Discovering an outlier, qualifying a candidate and publishing a change are three different outcomes.**

Verify three different facts: (1) evidence explains which layer qualified; (2) `applied` and `publications.jsonl` show the new policy was published; (3) the next batch's trajectories carry that policy. The existing [workflow tests](../tests/test_calibration_workflow.py) also compare manual acceptance with automatic approval and cover first activation within the low-precision scope. [Viewer tests](../tests/test_calibration_viewer.py) cover joins, branches, incomplete log lines and standalone export.

## 6. Interpret the result at its measured scope

Local mode measures projection error on inputs produced by the active recipe. Model-logits mode measures the complete candidate's output against full-model BF16 at fixed current weights. The latter includes interactions between proposed changes but only on its bounded held-out token positions. Neither mode establishes the cause of an end-to-end failure, measures gradient error, or proves reward, convergence, FLOP or throughput gains. Training-level quality and cost still need separate evaluation.

SmoothQuant's viewer consumes `calibration.jsonl`; it does **not** require the separate `recipe_observation` probe. That probe compares fixed BF16/MXFP8/Hadamard methods and writes `recipe_metrics.jsonl`. Enabling it for SmoothQuant is not the way to obtain these charts. See [Recipe Insights](recipe-insights.md) for that separate path.

For native validation, use the [single-GPU configuration](../configs/grpo_single_gpu_smoothquant.json) and [GPU acceptance guide](gpu-validation.md). Preserve configuration, calibration evidence, publication records, trajectories and training metrics together. A chart produced from a CPU simulation or a native-configured directory must not be presented as validated GPU training benefit.
