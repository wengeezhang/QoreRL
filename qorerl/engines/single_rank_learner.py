"""Single-rank GRPO reference update loop shared by the initial model adapters.

Native distributed learners should implement LearnerEngine using their backend's
optimizer and schedules, rather than inherit this single-rank implementation.
"""

import io
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


class SingleRankGRPOLearner:
    """Explicit FP32 master weights, FP32 accumulation and AdamW states.

    Model parameters may be BF16. No native MXFP8 or distributed gradient reduction
    is claimed here. One optimizer step consumes the whole rollout batch per epoch.
    """

    def setup_optimizer(self, cfg, model, pad_id, device):
        self.cfg, self.model, self.pad_id, self.device = cfg, model, pad_id, device
        self.policy_version = PolicyVersion(0, cfg.recipe_id)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.master_parameters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.master_parameters, lr=cfg.learning_rate, weight_decay=0.0)
        self.optimizer_steps = 0
        # These adapters disable dropout; eval() retains autograd and removes an
        # avoidable difference between tokenwise generation and teacher forcing.
        self.model.eval()

    def logits(self, input_ids):
        raise NotImplementedError

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
        for _ in range(self.cfg.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, count, self.cfg.micro_batch_size):
                indices = list(range(start, min(start + self.cfg.micro_batch_size, count)))
                microbatch = collate_microbatch(batch, indices, self.pad_id, self.device)
                self.model.zero_grad(set_to_none=True)
                loss, stats = grpo_loss(self.logits(microbatch.input_ids), microbatch, self.cfg.clip_epsilon)
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
        return {
            "loss": sums["loss"],
            "ratio_mean": sums["ratio_sum"] / tokens,
            "abs_log_ratio_mean": sums["abs_log_ratio_sum"] / tokens,
            "clip_fraction": sums["clipped_tokens"] / tokens,
            "gradient_norm": last_norm,
            "optimizer_steps": self.optimizer_steps,
            "skipped_zero_advantage": not active,
            "weight_version": self.policy_version.weight_version,
        }

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
        if policy_version.recipe_id != self.cfg.recipe_id:
            raise ValueError("Checkpoint recipe is incompatible")
        self.model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            for master, saved in zip(self.master_parameters, saved_parameters, strict=True):
                master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        self.policy_version = policy_version
        self.optimizer_steps = state["optimizer_steps"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])

    def close(self):
        pass
