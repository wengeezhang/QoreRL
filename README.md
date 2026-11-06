# QoreRL

**Quantization-native reinforcement learning.**

QoreRL is an early-stage project building a reinforcement learning framework with automatic sensitivity analysis, adaptive precision optimization, and coordinated low-precision training and rollouts.

Its goal is to make low-precision RL a managed training workflow: identify sensitive computations, diagnose numerical errors, evaluate repairs, and apply validated precision recipes across training and rollout engines.

> **Status: design and initial development.** This repository currently contains the project overview. The capabilities and integrations below are planned; there is no runnable training implementation or published benchmark yet.

## Why QoreRL?

Lower precision can reduce memory use and computation cost, but the benefits depend on where quantization is applied and how errors propagate through training. A large activation outlier does not necessarily imply a sensitive layer, and reducing tensor error does not automatically improve training quality.

Developers need answers to practical questions:

- Which operators actually need additional precision?
- Can a small transformation preserve low-precision execution at those positions?
- Does a recipe remain effective as weights and rollout distributions change?
- Can training and rollout engines adopt the recipe without worsening their numerical mismatch?
- Do the gains survive the overhead of observation, transformation, validation, and weight synchronization?

QoreRL aims to answer these questions and turn the answers into automatic, verifiable actions.

## Core capabilities

### 1. Sensitivity analysis

Identify sensitive computations at the module and GEMM level, distinguishing forward propagation (**Fprop**), activation gradients (**Dgrad**), and weight gradients (**Wgrad**).

Use inexpensive measurements to shortlist candidates, then validate their importance through targeted precision interventions. Sensitivity is evaluated for a particular model state, workload, and quantization recipe; it is not a permanent property inferred from outlier magnitude alone.

### 2. Numerical diagnosis

Inspect activation, weight, and gradient distributions alongside actual quantization errors. Relevant signals include persistent outlier channels, transient spikes, underflow, clipping, and the interaction between tensor distributions and quantization groups.

These signals guide candidate generation. They do not determine the repair by themselves.

### 3. Automatic repair

Evaluate supported alternatives for sensitive computations:

- Keep the current recipe.
- Apply channel-wise scaling.
- Apply a supported rotation or a combination of transformations.
- Increase precision selectively.

Select candidates according to measured numerical behavior, training quality, and execution cost. Every candidate must respect the model's computation graph and the backend's kernel, layout, and fusion constraints.

The intended user experience is automatic diagnosis, validation, and repair during a run, with a report explaining each decision. If no candidate meets the acceptance criteria, retain the current recipe or use a configured fallback.

### 4. Training and rollout coordination

Manage a shared, versioned precision recipe covering relevant formats, scaling parameters, transformations, and execution constraints. Forward-path changes must be coordinated across training and rollout engines; backward-only choices remain training-specific.

Matching configuration is necessary but does not guarantee matching numerics. Candidate recipes must be checked using actual engine execution, including relevant batching and prefill/decode behavior.

## Intended workflow

```text
Run with the current accepted recipe
                  |
Observe multiple rollout batches and corresponding training computations
                  |
Identify sensitive computations and propose supported repairs
                  |
Validate candidates on separate observations and paired engine checks
                  |
Accept a candidate that meets quality, consistency, and cost criteria
                  |
Activate it at a coordinated training/weight-sync boundary
                  |
Monitor, refresh when justified, or restore the previous recipe
```

Observation should reuse representative training data where practical. Multiple batches improve coverage and reduce sampling noise; they do not guarantee an unbiased view of future workloads. Candidate selection and acceptance evaluation should use separate observations.

Paired comparisons use the same model snapshot and token sequences to separate numerical mismatch from policy changes. Trajectories are associated with the weight and recipe versions that produced them.

Recipe changes are committed only at safe boundaries, with matching forward/backward state and consistent activation across engines. Restoring a recipe changes subsequent execution; it does not undo optimizer updates. Recovery of training state requires a separate checkpoint policy.

## First research question

> Can channel-wise scaling at a small number of sensitive computations improve low-precision training beyond existing block quantization and rotations? If so, when does the scaling need to be updated?

The initial study will compare:

| Experiment | Question |
| --- | --- |
| Existing low-precision recipe | What errors and costs remain in the baseline? |
| Selective, fixed channel scaling | Does scaling provide an additional benefit? |
| Periodically refreshed scaling | Does refreshing improve on a fixed recipe? |
| Observation-triggered refresh | Can useful updates be selected with less overhead? |
| Selective higher precision | How does scaling compare with a straightforward precision fallback? |

Existing rotations remain part of the baseline where applicable. Additional rotation and combined-transform candidates will be evaluated at compatible computation sites.

If fixed scaling is sufficient, the framework should keep it fixed. Dynamic updates must justify their additional cost.

## Evaluation principles

- **Measure more than tensor error.** Include operator outputs, gradient behavior, and end-to-end training quality.
- **Check train-rollout mismatch directly.** Compare log probabilities on matched inputs and model snapshots, with numerical finiteness checks and attention to tail errors.
- **Validate combined changes.** Individually acceptable repairs can interact when enabled together.
- **Use real execution paths.** Simulated quantization is useful for screening; final validation must cover the intended kernels and precision semantics.
- **Count the full cost.** Include observation, candidate evaluation, transformations, quantized-weight rebuilding, synchronization, memory, and wall-clock time.
- **Keep a baseline and a fallback.** No transformation is assumed beneficial merely because it produces smaller outliers.

## Initial scope and roadmap

The first implementation will focus on one model and one low-precision recipe through **Megatron Core and Transformer Engine**. Rollout integration will expand to **vLLM** and **SGLang** after the training-side behavior is validated. These are integration targets, not current compatibility claims.

- [ ] Capture representative tensors and numerical statistics across training stages.
- [ ] Build a sensitivity-analysis and targeted-replay workflow for Fprop, Dgrad, and Wgrad.
- [ ] Compare selective channel scaling with the baseline and higher-precision fallbacks.
- [ ] Validate accepted recipes in complete training runs and measure their net cost.
- [ ] Automate multi-batch observation, candidate evaluation, and recipe selection.
- [ ] Coordinate recipe activation and verification across training and rollout engines.
- [ ] Add drift-triggered refresh, fallback policies, and reproducible benchmarks.

## Development

The current focus is establishing reproducible evidence for selective scaling before expanding the automatic repair search space. Installation instructions and runnable examples will be added with the implementation.

Design discussions and experiment proposals are welcome through [GitHub Issues](https://github.com/wengeezhang/QoreRL/issues).
