"""Atomic checkpoint publication and same-configuration recovery validation."""

import json
from dataclasses import asdict
from pathlib import Path

import ray

from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(directory, learner, cfg, step, policy_version):
    """Expose the recovery directory only after its payload and metadata exist."""
    root = directory / "checkpoints"
    root.mkdir(exist_ok=True)
    pending = root / f".step-{step:06d}.pending"
    target = root / f"step-{step:06d}"
    pending.mkdir()
    (pending / "learner.pt").write_bytes(ray.get(learner.checkpoint.remote(), timeout=cfg.rpc_timeout))
    write_json(
        pending / "metadata.json",
        {
            "schema": 2,
            "step": step,
            "policy_version": asdict(policy_version),
            "config": cfg.to_dict(),
            "boundary": "all_rollouts_acknowledged_no_inflight_samples",
        },
    )
    pending.rename(target)
    return str(target.resolve())


def resume_metadata(path, cfg):
    """Read schema 1/2 metadata; normalize names without changing training semantics."""
    metadata = json.loads((Path(path) / "metadata.json").read_text())
    schema = metadata.get("schema")
    if schema not in (1, 2):
        raise ValueError("Unsupported checkpoint schema")
    previous = TrainConfig(**metadata["config"])
    allowed = {"steps", "checkpoint_every", "rpc_timeout", "learner_runtime_env", "rollout_runtime_env"}
    changed = [k for k, v in cfg.to_dict().items() if k not in allowed and previous.to_dict()[k] != v]
    if changed:
        raise ValueError(f"Resume currently requires identical training semantics; changed: {changed}")
    key = "policy" if schema == 1 else "policy_version"
    policy_version = PolicyVersion.from_dict(metadata[key])
    step = metadata["step"]
    if (
        type(step) is not int
        or step < 0
        or cfg.steps <= step
        or policy_version.weight_version != step
        or policy_version.recipe_id != cfg.recipe_id
    ):
        raise ValueError("Resume requires a consistent checkpoint and a larger total steps value")
    return {**metadata, "policy_version": asdict(policy_version)}
