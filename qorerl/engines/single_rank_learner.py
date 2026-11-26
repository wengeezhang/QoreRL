"""Single-rank GRPO reference update loop shared by the initial model adapters.

Native distributed learners should implement LearnerEngine using their backend's
optimizer and schedules, rather than inherit this single-rank implementation.
"""

import io
from contextlib import ExitStack
from dataclasses import asdict

import torch

from qorerl.training.batching import collate_microbatch
from qorerl.training.contracts import PolicyVersion
from qorerl.training.objective import grpo_loss


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def _calibration_prefixes(trajectories, max_rows):
    """Replay a bounded set of distinct causal contexts in rollout order.

    Trie nodes correspond to complete token prefixes. Repeated prompts and
    shared response prefixes therefore cannot appear on both sides of the
    fit/held-out split. Replaying one sequence at a time also excludes padding.
    """
    trie = {}
    remaining = max_rows
    for trajectory in trajectories:
        tokens = trajectory.prompt_ids + trajectory.response_ids[:-1]
        node, shared = trie, 0
        for token in tokens:
            if token not in node:
                break
            node = node[token]
            shared += 1
        stop = min(len(tokens), shared + remaining)
        if stop == shared:
            continue
        for token in tokens[shared:stop]:
            node[token] = {}
            node = node[token]
        yield tokens[:stop], shared
        remaining -= stop - shared
        if remaining == 0:
            break


class _CalibrationInputs:
    def __init__(self, targets):
        self.rows = {name: [] for name in targets}
        self.seen = set()

    def begin_sequence(self, length, shared):
        self.length, self.shared = length, shared
        self.seen.clear()

    def capture(self, name, x):
        if name not in self.rows:
            return
        if name in self.seen or x.numel() != self.length * x.shape[-1]:
            raise ValueError(f"Calibration expected one unpadded token row per position for {name}")
        self.seen.add(name)
        # Single-sequence replay works for both [S,1,K] and [1,S,K] adapters.
        rows = x.detach().reshape(self.length, -1)[self.shared :]
        self.rows[name].append(rows.clone())


class SingleRankGRPOLearner:
    """Explicit FP32 master weights, FP32 accumulation and AdamW states.

    Model parameters may be BF16; operator precision belongs to the model adapter.
    There is no distributed reduction. One optimizer step consumes the whole
    rollout batch per epoch.
    """

    def setup_optimizer(self, cfg, model, pad_id, device):
        self.cfg, self.model, self.pad_id, self.device = cfg, model, pad_id, device
        recipe_id = (
            self.recipe_state.active_identity if hasattr(self, "recipe_state") else cfg.policy_recipe_id
        )
        self.policy_version = PolicyVersion(0, recipe_id)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.master_parameters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.master_parameters, lr=cfg.learning_rate, weight_decay=0.0)
        self.optimizer_steps = 0
        # These adapters disable dropout; eval() retains autograd and removes an
        # avoidable difference between tokenwise generation and teacher forcing.
        self.model.eval()

    def logits(self, input_ids):
        raise NotImplementedError

    def _discovery_projections(self):
        raise ValueError("This learner has no scoped projections for outlier discovery")

    def _sample_calibration_inputs(self, trajectories, max_rows):
        installation = self.recipe_installation
        with ExitStack() as cleanup:
            if installation is None:
                projections = self._discovery_projections()
                capture = _CalibrationInputs(projections)
                for name, module in projections.items():

                    def hook(_module, args, target=name):
                        capture.capture(target, args[0])

                    cleanup.callback(module.register_forward_pre_hook(hook).remove)
            else:
                observed = self.recipe_state.observation_targets
                if not observed:
                    return {}
                capture = _CalibrationInputs(observed)
                cleanup.callback(setattr, installation, "calibrator", installation.calibrator)
                installation.calibrator = capture
            for tokens, shared in _calibration_prefixes(trajectories, max_rows):
                capture.begin_sequence(len(tokens), shared)
                self.logits(torch.tensor([tokens], device=self.device))
                if capture.seen != capture.rows.keys():
                    raise ValueError("Calibration replay did not execute every scoped target")
        return {name: torch.cat(parts) for name, parts in capture.rows.items()}

    @torch.no_grad()
    def prepare_candidate(self, batch):
        """Replay a bounded activation window and let the recipe evaluate a possible update."""
        from qorerl.recipes.calibration import outlier_report

        config = self.cfg.calibration
        handler = self.cfg.recipe.candidate_handler
        if not config.discovery_only and handler is None:
            raise ValueError("This recipe cannot prepare candidates")
        if batch.policy_version.recipe_id != self.policy_version.recipe_id or (
            self.policy_version.weight_version - batch.policy_version.weight_version not in (0, 1)
        ):
            raise ValueError("Calibration input did not come from the current or preceding rollout policy")
        if not batch.trajectories or any(
            t.policy_version != batch.policy_version for t in batch.trajectories
        ):
            raise ValueError("Calibration requires a nonempty rollout batch with consistent policy identity")
        samples = self._sample_calibration_inputs(batch.trajectories, config.max_rows)
        if not samples:
            return None, []
        if config.discovery_only:
            return None, outlier_report(samples, config)
        return handler.prepare_candidate(self.cfg, self.recipe_state, self.recipe_installation, samples)

    def apply_candidate(self, artifact, *, initial=False):
        """Install the recipe payload and expose its new identity for weight publication."""
        handler = self.cfg.recipe.candidate_handler
        if handler is None:
            raise ValueError("This recipe cannot apply candidates")
        handler.apply_candidate(self.cfg, self.recipe_state, artifact, initial=initial)
        self.policy_version = PolicyVersion(
            self.policy_version.weight_version, self.recipe_state.active_identity
        )
        return self.policy_version

    def update(self, batch):
        if batch.policy_version != self.policy_version:
            raise ValueError(
                f"Learner policy {self.policy_version} cannot consume batch {batch.policy_version}"
            )
        count = len(batch.trajectories)
        if not count or not torch.isfinite(batch.advantages).all():
            raise ValueError("Invalid advantages or empty batch")
        sums = dict(loss=0.0, ratio_sum=0.0, abs_log_ratio_sum=0.0, clipped_tokens=0, tokens=0)
        last_norm = 0.0
        active = bool(batch.advantages.count_nonzero())
        pre_update = dict(ratio_sum=0.0, abs_log_ratio_sum=0.0, clipped_tokens=0, tokens=0)
        pre_update_max = 0.0
        probe = getattr(self, "recipe_probe", None)
        observations, probe_seconds = [], 0.0
        if probe is not None:
            probe.begin(
                step=self.policy_version.weight_version + 1,
                weight_version=self.policy_version.weight_version,
                optimizer_steps=self.optimizer_steps,
            )
        for epoch in range(self.cfg.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, count, self.cfg.micro_batch_size):
                indices = list(range(start, min(start + self.cfg.micro_batch_size, count)))
                microbatch = collate_microbatch(batch, indices, self.pad_id, self.device)
                self.model.zero_grad(set_to_none=True)
                loss, stats = grpo_loss(self.logits(microbatch.input_ids), microbatch, self.cfg.clip_epsilon)
                if epoch == 0:
                    # No optimizer step has happened yet. Same snapshot and
                    # recorded tokens isolate engine mismatch from policy learning.
                    for key in pre_update:
                        pre_update[key] += stats[key]
                    pre_update_max = max(pre_update_max, stats["abs_log_ratio_max"])
                fraction = len(indices) / count
                # Do not average microbatch means equally: the last may be smaller.
                if active:
                    (loss * fraction).backward()
                    for param, master in zip(self.parameters, self.master_parameters, strict=True):
                        if param.grad is not None:
                            grad = param.grad.detach().float()
                            if master.grad is None:
                                master.grad = grad.clone()
                            else:
                                master.grad.add_(grad)
                # The zero-advantage path has no backward to free its graph.
                # Drop its last root before allocating any replay intermediates.
                del loss
                if probe is not None and epoch == 0 and start == 0:
                    # Canonical snapshots were captured before rotation. Replay
                    # after backward releases training activations first; no
                    # optimizer update has changed either the weights or version.
                    observations, probe_seconds = probe.finish()
                sums["loss"] += stats["loss"] * fraction / self.cfg.update_epochs
                for key in ("ratio_sum", "abs_log_ratio_sum", "clipped_tokens", "tokens"):
                    sums[key] += stats[key]
            if active:
                last_norm = torch.nn.utils.clip_grad_norm_(
                    self.master_parameters, self.cfg.max_grad_norm, error_if_nonfinite=True
                ).item()
                self.optimizer.step()
                with torch.no_grad():
                    for param, master in zip(self.parameters, self.master_parameters, strict=True):
                        if not torch.isfinite(master).all():
                            raise FloatingPointError("Optimizer produced non-finite parameters")
                        param.copy_(master)
                self.optimizer_steps += 1
        self.model.zero_grad(set_to_none=True)
        self.policy_version = PolicyVersion(
            self.policy_version.weight_version + 1, self.policy_version.recipe_id
        )
        tokens = sums["tokens"]
        metrics = {
            "loss": sums["loss"],
            "ratio_mean": sums["ratio_sum"] / tokens,
            "abs_log_ratio_mean": sums["abs_log_ratio_sum"] / tokens,
            "clip_fraction": sums["clipped_tokens"] / tokens,
            "pre_update_ratio_mean": pre_update["ratio_sum"] / pre_update["tokens"],
            "pre_update_abs_log_ratio_mean": pre_update["abs_log_ratio_sum"] / pre_update["tokens"],
            "pre_update_abs_log_ratio_max": pre_update_max,
            "pre_update_clip_fraction": pre_update["clipped_tokens"] / pre_update["tokens"],
            "gradient_norm": last_norm,
            "optimizer_steps": self.optimizer_steps,
            "skipped_zero_advantage": not active,
            "weight_version": self.policy_version.weight_version,
        }
        if probe is not None:
            # Host capture + replay wall time may include queued training work;
            # this is not an isolated measurement of instrumentation overhead.
            metrics.update(recipe_observations=observations, recipe_probe_seconds=probe_seconds)
        return metrics

    def checkpoint(self):
        stream = io.BytesIO()
        torch.save(
            {
                "schema": 2,
                "model": cpu_tree(self.model.state_dict()),
                "master_parameters": cpu_tree(self.master_parameters),
                "optimizer": cpu_tree(self.optimizer.state_dict()),
                "policy_version": asdict(self.policy_version),
                "optimizer_steps": self.optimizer_steps,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if self.device == "cuda" else None,
                "recipe_state": self.recipe_state.to_dict() if hasattr(self, "recipe_state") else None,
            },
            stream,
        )
        return stream.getvalue()

    def restore(self, payload):
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if state.get("schema") == 1:
            # Checkpoints written before the role/naming refactor.
            policy_version = PolicyVersion.from_dict({"weights": state["version"], "recipe": state["recipe"]})
            saved_parameters = state["masters"]
        elif state.get("schema") == 2:
            policy_version = PolicyVersion.from_dict(state["policy_version"])
            saved_parameters = state["master_parameters"]
        else:
            raise ValueError("Checkpoint schema is incompatible")
        expected_id = self.cfg.policy_recipe_id
        if hasattr(self, "recipe_state"):
            expected_id = self.recipe_state.validate(state.get("recipe_state"))
        elif state.get("recipe_state") is not None:
            raise ValueError("Checkpoint contains unsupported recipe state")
        if policy_version.recipe_id != expected_id:
            raise ValueError("Checkpoint recipe is incompatible")
        self.model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            for master, saved in zip(self.master_parameters, saved_parameters, strict=True):
                master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "recipe_state"):
            self.recipe_state.load(state["recipe_state"])
        self.policy_version = policy_version
        self.optimizer_steps = state["optimizer_steps"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])

    def close(self):
        pass
