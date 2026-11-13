# QoreRL

**Practical low-precision reinforcement learning.**

QoreRL is an early-stage, quantization-native reinforcement learning framework being built to make low-precision methods easier to reproduce, compare, and use in training.

The first milestone focuses on configurable precision recipes, numerical diagnostics, and a guided recovery workflow: observe a run, preserve recoverable state when problems arise, present evidence and supported options, and let the user adjust the configuration before resuming. Automatic precision adaptation remains an ongoing research goal, not a prerequisite for a useful first release.

> **Available today:** a runnable, single-process tiny-MoE research prototype with paired BF16 gradient probes and an experimental Wgrad intervention. MXFP8 is numerically emulated, **not executed on native Tensor Cores**. Native engine integrations, the recipe catalog, alert-driven pause, and configurable checkpoint recovery are planned. The current CLI saves experiment checkpoints but does not implement resume.

## What QoreRL is building

### 1. Reproducible methods and configurable recipes

Implement selected methods from existing and emerging research, reproduce their behavior, and expose them as individual options or tested combinations. A **precision recipe** specifies which computations use which numerical methods, together with any transformations and execution requirements.

Each method or recipe should document:

- **Scope:** forward propagation (Fprop), activation gradients (Dgrad), weight gradients (Wgrad), weight export, rollout, or an associated RL correction.
- **Evidence:** source paper or implementation, reproducible configuration, tested models and workloads, and whether results use emulation or native kernels.
- **Compatibility:** supported hardware and engine versions, tested combinations, and unsupported or conflicting settings.
- **Tradeoffs:** training quality, numerical behavior, memory use, and end-to-end execution cost.

Experimental options must remain distinguishable from recipes validated for a stated workload. Two individually successful methods are not automatically a validated combination. Unsupported combinations should fail configuration validation rather than silently fall back to different behavior.

### 2. Numerical diagnostics and guided recovery

Help users understand what happened, preserve useful training progress, and try supported adjustments without rebuilding their experiment by hand.

Observations may include non-finite values, quantization statistics, gradient behavior, train-rollout probability mismatch, and training-quality trends. Reports should separate **measured symptoms**, **possible causes**, and **candidate actions**. An outlier or a negative gradient cosine alone does not establish a failure or prove that a particular intervention will help.

Pause rules should be explicit and configurable. Clear invalid states can block an optimizer update; weaker signals can produce warnings or request additional observations. When a pause is required, retain diagnostic artifacts and stop at a safe boundary.

The recovery workflow will distinguish:

- **Recovery checkpoints:** complete training state from a boundary that passed the configured checks, including model and optimizer state, scheduler and progress state, RNG/data state, and the active precision recipe and its state.
- **Incident artifacts:** observations and, where useful, tensor or state snapshots from the anomaly. An incident snapshot does not replace the previous recovery checkpoint.

For the first milestone, recovery means editing the configuration and restarting from a compatible checkpoint. The framework should validate the change, rebuild required engine state, and record the new recipe version. Forward-policy changes require coordinated training and rollout configuration; the initial recovery path will discard pending rollout data and collect fresh trajectories. Backward-only choices remain training-specific.

### 3. Continuous research toward automatic adaptation

Continue studying sensitivity analysis, selective higher precision, channel scaling, rotations, and other numerical interventions alongside the practical training workflow.

Automation will progress from **human-selected adjustments**, to **controlled candidate comparisons and recommendations**, and eventually to **automatic actions within explicitly validated conditions and budgets**. Each adjustment should preserve its triggering evidence, configuration change, and measured outcome so that manual experiments can inform future automation.

No intervention is assumed harmless. The ability to reproduce and evaluate a method is useful even before an automatic selection rule has been established.

## Why QoreRL?

Lower precision can reduce memory use and computation cost, but the benefits depend on where quantization is applied and how errors propagate through training. A large activation outlier does not necessarily imply a sensitive layer, and reducing tensor error does not automatically improve training quality.

Developers need answers to practical questions:

- Can I reproduce a published low-precision method on my training stack?
- Which methods can I combine, and what evidence supports that combination?
- When a run behaves abnormally, what is known and what can I change?
- Can I preserve progress, adjust the recipe, and resume reliably?
- Does the new configuration improve quality or cost once monitoring and recovery overhead are included?

QoreRL aims to make this process reproducible and manageable while building evidence for increasingly automated decisions.

## First-milestone workflow (planned)

```text
Choose a supported recipe and validate the configuration
                  |
Train and collect numerical / learning diagnostics
                  |
On a configured pause condition: preserve evidence and stop safely
                  |
Review symptoms, possible causes, and supported adjustment options
                  |
User edits the configuration
                  |
Validate compatibility and restart from a recovery checkpoint
                  |
Continue training and measure the adjustment's actual outcome
```

Matching training and rollout configuration does not guarantee matching numerics. Recipes must be checked using actual engine execution, including relevant batching and prefill/decode behavior. Trajectories must retain the weight and recipe versions that produced them.

Changing a recipe affects future execution; it does not undo previous optimizer updates. Restoring an earlier checkpoint is a separate recovery action. A restart under a different recipe is a new continuation of the experiment, not a promise of identical training behavior.

## Run the current research prototype

```bash
uv sync --group dev
uv run pytest -q
uv run qorerl --steps 60 --seeds 0,1,2 --out results/first-study
```

Python 3.12+ and a CPU are sufficient. This command runs three arms from a shared tiny-MoE warm-start checkpoint: BF16, fixed MXFP8 emulation, and adaptive Wgrad activation precision. Calibration occurs before updates 20, 40, and 60, on the same frozen weights and recorded trajectories. Probe runs do not advance the live optimizer. Output directories must be new.

```text
MXFP8 forward remains unchanged
                  |
Every 20 updates: compare each Linear's weight gradient with full BF16 reference
                  |
Negative gradient cosine, with non-negligible norms?
                  |
Use saved pre-quantization BF16 activation ONLY in that Linear's Wgrad
                  |
Keep Dgrad and G's existing numerical values; measure the intervention's outcome
```

The saved activation comes from the **current low-precision forward**, not the all-BF16 reference run. Restored modules remain active in subsequent updates. This research arm applies the proposed rule and reports results even when a candidate fails to improve alignment or loss; it is not the planned user-approved recovery workflow.

- [Detailed experiment protocol (中文)](docs/wgrad-experiment.zh-CN.md)
- [First CPU experiment results (中文)](docs/experiments/2026-11-08/README.md)
- [Linear precision semantics](qorerl/precision.py)
- [Paired calibration and private trial updates](qorerl/calibration.py)
- [Tiny MoE and GRPO experiment runner](qorerl/experiment.py)

Results include `report.md`, per-step JSONL, per-layer calibration JSONL, and model/optimizer checkpoints. Reported runtime is emulator runtime, not an MXFP8 acceleration benchmark.

## First research question

> When a module's MXFP8 weight gradient points against its BF16 reference, does restoring only the Wgrad activation operand improve alignment, local updates, and ultimately RL learning?

The implemented study compares:

| Experiment | Question |
| --- | --- |
| BF16 reference training | How does the higher-precision baseline learn? |
| Fixed MXFP8 emulation | What changes when linear Fprop, Dgrad and Wgrad operands are quantized? |
| Periodic gradient-triggered intervention | What happens when only selected Wgrad activations retain BF16 values? |

It records gradient alignment separately from policy loss and reward. A direction recovery is not assumed to guarantee better training. It also verifies that the intervention leaves forward logits and unselected parameter gradients unchanged.

Selective channel scaling, rotations, refresh schedules, and other repairs remain follow-up research directions. This first prototype does not implement them.

## Evaluation principles

- **Measure more than tensor error.** Include operator outputs, gradient behavior, and end-to-end training quality.
- **Separate evidence from hypotheses.** A warning is not a root-cause diagnosis, and a closer BF16 gradient is not proof of better learning.
- **Check train-rollout mismatch directly.** Compare log probabilities on matched inputs and model snapshots, with numerical finiteness checks and attention to tail errors.
- **Validate combined changes.** Individually acceptable repairs can interact when enabled together.
- **Use real execution paths.** Simulated quantization is useful for screening; final validation must cover the intended kernels and precision semantics.
- **Count the full cost.** Include observation, candidate evaluation, transformations, synchronization, checkpoint/recovery overhead, memory, and wall-clock time to a stated quality target.
- **Test recovery itself.** Verify that a supported configuration change resumes from a compatible checkpoint with the intended numerical behavior.
- **Keep a baseline and a fallback.** No transformation is assumed beneficial merely because it produces smaller outliers.

Multiple observation batches improve coverage but do not guarantee an unbiased view of future workloads. Candidate selection and acceptance evaluation should use separate observations. Paired numerical comparisons use the same model snapshot and token sequences to distinguish execution differences from policy changes.

## Initial scope and roadmap

The prototype uses standalone PyTorch and a tiny MoE. The first native integration target is one pinned **Megatron Core / Transformer Engine + vLLM** stack, with a small set of reproducible recipes. **SGLang** and broader model/backend coverage are later targets. These are plans, not current compatibility claims.

QoreRL will reuse existing training and inference infrastructure. Its initial engineering focus is precision configuration, evidence collection, and recovery across those engines rather than rebuilding their execution or distributed scheduling systems.

### Available research foundation

- [x] Build a standalone tiny-MoE group-relative RL experiment with paired gradient probes.
- [x] Implement Wgrad-only BF16 activation restoration and verify Fprop/Dgrad isolation.
- [x] Record private AdamW trial updates, training rewards, and reproducible experiment artifacts.

### First milestone: usable, observable, recoverable training

- [ ] Establish one native MXFP8 training/rollout baseline and measure quality and cost.
- [ ] Package selected research methods as documented options and tested recipes.
- [ ] Add recipe versioning, compatibility validation, and paired train-rollout checks.
- [ ] Add numerical diagnostics, configurable warning/pause rules, and incident reports.
- [ ] Implement complete recovery checkpoints and restart after user-selected configuration changes.
- [ ] Publish reproducible comparisons and recovery examples, including unsuccessful adjustments.

### Ongoing research and later automation

- [ ] Validate the Wgrad activation intervention on native kernels, including its memory and execution costs.
- [ ] Study selective scaling, rotations, precision changes, and their combinations.
- [ ] Evaluate sensitivity probes and candidate comparisons on representative RL workloads.
- [ ] Recommend adjustments from measured evidence, with user approval before activation.
- [ ] Automate bounded, validated adjustments with post-change monitoring and recovery.

Research continues alongside the first milestone. A general automatic-repair solution is not a release requirement.

## Development

Development has two parallel tracks: building the practical recipe/diagnostic/recovery workflow, and continuing controlled numerical experiments. Use `uv run ruff check qorerl tests` and `uv run pytest -q` for validation. The numerical emulator includes explicit backward formulas; it is not a claim that quantization has an exact differentiable gradient.

Design discussions and experiment proposals are welcome through [GitHub Issues](https://github.com/wengeezhang/QoreRL/issues).
