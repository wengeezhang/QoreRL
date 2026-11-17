"""CPU-only wire objects. Engine names/layouts never leak into the algorithm."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyVersion:
    weights: int
    recipe: str


@dataclass(frozen=True)
class Prompt:
    group_id: int
    token_ids: tuple[int, ...]
    answer: str


@dataclass(frozen=True)
class Trajectory:
    group_id: int
    sample_id: int
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    text: str
    finish_reason: str
    policy: PolicyVersion

    def validate(self):
        if not self.prompt_ids or not self.response_ids:
            raise ValueError("Empty prompt/response cannot enter training")
        if len(self.response_ids) != len(self.behavior_log_probs):
            raise ValueError("Every generated token, including EOS, needs its behavior log probability")
        if any(not math.isfinite(p) or p > 1e-5 for p in self.behavior_log_probs):
            raise ValueError("Invalid behavior log probability")


@dataclass
class WeightSnapshot:
    policy: PolicyVersion
    model_id: str
    tensors: dict[str, torch.Tensor]

    def validate(self):
        if not self.tensors:
            raise ValueError("Cannot publish an empty weight snapshot")
        for name, value in self.tensors.items():
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError(f"Snapshot tensor must be finite and on CPU: {name}")


@dataclass
class TrainingBatch:
    trajectories: list[Trajectory]
    rewards: torch.Tensor
    advantages: torch.Tensor
    policy: PolicyVersion


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    behavior_log_probs: torch.Tensor
    advantages: torch.Tensor


def assemble_batch(prompts, trajectories, group_size, policy):
    """Group by identity, never by arrival order; reject missing/duplicate/stale samples."""
    expected = {p.group_id: p for p in prompts}
    if len(expected) != len(prompts):
        raise ValueError("Duplicate prompt group IDs")
    received = {}
    for item in trajectories:
        item.validate()
        if item.policy != policy:
            raise ValueError("Stale weights or precision recipe in rollout")
        prompt = expected.get(item.group_id)
        if prompt is None or prompt.token_ids != item.prompt_ids:
            raise ValueError("Unknown or altered prompt")
        key = (item.group_id, item.sample_id)
        if key in received or not 0 <= item.sample_id < group_size:
            raise ValueError("Duplicate/invalid rollout sample ID")
        received[key] = item
    keys = [(p.group_id, i) for p in prompts for i in range(group_size)]
    if set(received) != set(keys):
        raise ValueError("Incomplete GRPO group; never silently train on partial groups")
    ordered = [received[key] for key in keys]
    rewards = torch.tensor([float(t.text.strip() == expected[t.group_id].answer) for t in ordered])
    grouped = rewards.reshape(-1, group_size)
    advantages = (
        (grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, unbiased=False, keepdim=True) + 1e-6)
    ).flatten()
    return TrainingBatch(ordered, rewards, advantages, policy)


def collate(batch, indices, pad_id, device):
    """Right padding, causal shift, and response-only mask (EOS included)."""
    items = [batch.trajectories[i] for i in indices]
    width = max(len(t.prompt_ids) + len(t.response_ids) - 1 for t in items)
    tokens = torch.full((len(items), width + 1), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(items), width), dtype=torch.bool, device=device)
    old = torch.zeros((len(items), width), dtype=torch.float32, device=device)
    for row, item in enumerate(items):
        seq = item.prompt_ids + item.response_ids
        tokens[row, : len(seq)] = torch.tensor(seq, device=device)
        start = len(item.prompt_ids) - 1
        stop = start + len(item.response_ids)
        mask[row, start:stop] = True
        old[row, start:stop] = torch.tensor(item.behavior_log_probs, device=device)
    return PackedBatch(tokens[:, :-1], tokens[:, 1:], mask, old, batch.advantages[indices].to(device))
