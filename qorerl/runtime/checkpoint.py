"""Atomic checkpoint publication and same-configuration recovery validation."""

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import ray

from qorerl.runtime.calibration import review_id
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(directory, learner, cfg, step, policy_version, *, review=None):
    """Expose the recovery directory only after its payload and metadata exist."""
    root = directory / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step-{step:06d}" + ("-before-calibration" if review is not None else "")
    pending = root / f".{name}.pending"
    target = root / name
    pending.mkdir()
    payload = ray.get(learner.checkpoint.remote(), timeout=cfg.rpc_timeout)
    (pending / "learner.pt").write_bytes(payload)
    pending_metadata = {}
    if review is not None:
        review["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
        write_json(pending / "candidate.json", review)
        pending_metadata["pending_candidate"] = review_id(review)
    write_json(
        pending / "metadata.json",
        {
            **pending_metadata,
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
    allowed = {
        "steps",
        "checkpoint_every",
        "rpc_timeout",
        "learner_runtime_env",
        "rollout_runtime_env",
        "recipe_observation",  # Diagnostics neither change the recipe nor its learned state.
    }
    changed = [k for k, v in cfg.to_dict().items() if k not in allowed and previous.to_dict()[k] != v]
    if changed:
        raise ValueError(f"Resume currently requires identical training semantics; changed: {changed}")
    key = "policy" if schema == 1 else "policy_version"
    policy_version = PolicyVersion.from_dict(metadata[key])
    valid_recipe = policy_version.recipe_id == cfg.policy_recipe_id
    if cfg.recipe.uses_smoothquant:
        valid_recipe = (
            re.fullmatch(re.escape(cfg.policy_recipe_id) + r":[0-9a-f]{64}", policy_version.recipe_id)
            is not None
        )
    step = metadata["step"]
    if (
        type(step) is not int
        or step < 0
        or cfg.steps < step
        or (cfg.steps == step and not metadata.get("pending_candidate"))
        or policy_version.weight_version != step
        or not valid_recipe
    ):
        raise ValueError(
            "Resume requires a consistent checkpoint and remaining steps or a final pending review"
        )
    return {**metadata, "policy_version": asdict(policy_version)}
