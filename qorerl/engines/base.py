"""Role interfaces and shared publication checks, independent of optimizer choice.

The coordinator uses these methods without inspecting engine-specific models.
The initial weight/checkpoint payloads are CPU snapshots and bytes; distributed
transports will require extending those payload contracts.
"""

from typing import Protocol

from qorerl.recipes.calibration import CandidatePreparation
from qorerl.training.contracts import PolicyVersion, Prompt, TrainingBatch, Trajectory, WeightSnapshot


class LearnerEngine(Protocol):
    policy_version: PolicyVersion
    optimizer_steps: int

    policy_version: PolicyVersion
    optimizer_steps: int

    def update(self, batch: TrainingBatch) -> dict: ...
    def prepare_candidate(self, batch: TrainingBatch) -> CandidatePreparation: ...
    def apply_candidate(self, artifact: dict, *, initial: bool = False) -> PolicyVersion: ...
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
    expected_id = cfg.policy_recipe_id
    smooth = cfg.recipe.uses_smoothquant
    if smooth:
        from qorerl.recipes.state import smoothing_factor_digest

        state = snapshot.recipe_state
        if not isinstance(state, dict) or state.get("identity") != expected_id:
            raise ValueError("SmoothQuant publication requires the configured recipe state")
        digest = smoothing_factor_digest(state.get("scales"))
        if digest != state.get("scale_digest"):
            raise ValueError("SmoothQuant publication smoothing factor digest mismatch")
        expected_id = f"{expected_id}:{digest}"
    if snapshot.model_id != cfg.model or snapshot.policy_version.recipe_id != expected_id:
        raise ValueError("Model/recipe mismatch during weight publication")
    if current is not None:
        version = snapshot.policy_version
        transform_changed = smooth and version.recipe_id != current.recipe_id
        if version.weight_version < current.weight_version or (
            version.weight_version == current.weight_version and not transform_changed
        ):
            raise ValueError("Weight versions must increase unless publishing a new SmoothQuant transform")
