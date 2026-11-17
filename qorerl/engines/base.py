"""Role interfaces and shared publication checks, independent of optimizer choice.

The coordinator uses these methods without inspecting engine-specific models.
The initial weight/checkpoint payloads are CPU snapshots and bytes; distributed
transports will require extending those payload contracts.
"""

from typing import Protocol

from qorerl.training.contracts import PolicyVersion, Prompt, TrainingBatch, Trajectory, WeightSnapshot


class LearnerEngine(Protocol):
    def update(self, batch: TrainingBatch) -> dict: ...
    def export_weights(self) -> WeightSnapshot: ...
    def checkpoint(self) -> bytes: ...
    def restore(self, payload: bytes) -> None: ...
    def close(self) -> None: ...


class RolloutEngine(Protocol):
    def install_weights(self, snapshot: WeightSnapshot) -> PolicyVersion: ...
    def generate(
        self, prompts: list[Prompt], policy_version: PolicyVersion, seed: int
    ) -> list[Trajectory]: ...
    def close(self) -> None: ...


def validate_install(snapshot: WeightSnapshot, current: PolicyVersion | None, cfg):
    snapshot.validate()
    if snapshot.model_id != cfg.model or snapshot.policy_version.recipe_id != cfg.recipe_id:
        raise ValueError("Model/recipe mismatch during weight publication")
    if current is not None and snapshot.policy_version.weight_version <= current.weight_version:
        raise ValueError("Weight versions must increase on an initialized rollout engine")
