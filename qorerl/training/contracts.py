"""Engine-independent data contracts and rollout identity validation.

Transported tensors currently live on CPU. Device-side padding and masking belong
to batching.py; rewards and advantages belong to the task and algorithm modules.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyVersion:
    """Published policy identity, not an optimizer-step count or a content hash."""

    weight_version: int
    recipe_id: str

    def __post_init__(self):
        if type(self.weight_version) is not int or self.weight_version < 0:
            raise ValueError("weight_version must be a nonnegative integer")
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("recipe_id must be a nonempty string")

    @classmethod
    def from_dict(cls, value: Mapping):
        """Read current metadata and the original {weights, recipe} checkpoint fields."""
        if set(value) == {"weight_version", "recipe_id"}:
            return cls(value["weight_version"], value["recipe_id"])
        if set(value) == {"weights", "recipe"}:
            return cls(value["weights"], value["recipe"])
        raise ValueError("Invalid or ambiguous policy-version fields")


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
    policy_version: PolicyVersion

    def validate(self):
        if not self.prompt_ids or not self.response_ids:
            raise ValueError("Empty prompt/response cannot enter training")
        if len(self.response_ids) != len(self.behavior_log_probs):
            raise ValueError("Every generated token, including EOS, needs its behavior log probability")
        if any(not math.isfinite(p) or p > 1e-5 for p in self.behavior_log_probs):
            raise ValueError("Invalid behavior log probability")


@dataclass
class WeightSnapshot:
    policy_version: PolicyVersion
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
    policy_version: PolicyVersion


def validate_and_order_trajectories(prompts, trajectories, group_size, policy_version):
    """Group by identity, never by arrival order; reject missing/duplicate/stale samples."""
    if not prompts or type(group_size) is not int or group_size < 1:
        raise ValueError("Expected nonempty prompts and a positive group_size")
    expected = {p.group_id: p for p in prompts}
    if len(expected) != len(prompts):
        raise ValueError("Duplicate prompt group IDs")
    received = {}
    for item in trajectories:
        item.validate()
        if item.policy_version != policy_version:
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
        raise ValueError("Incomplete rollout group; never silently train on partial groups")
    return [received[key] for key in keys]
