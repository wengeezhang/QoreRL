"""Review artifacts for a proposed change at a synchronous training boundary."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from qorerl.recipes.calibration import make_scale_artifact, parse_scale_artifact


def artifact_context(cfg):
    return dict(model=cfg.model, model_revision=cfg.model_revision, recipe_id=cfg.policy_recipe_id)


def prepare_review(cfg, policy, optimizer_steps, overrides, evidence):
    return {
        "schema": 1,
        "source_policy": asdict(policy),
        "optimizer_steps": optimizer_steps,
        "scale_artifact": make_scale_artifact(overrides, **artifact_context(cfg)),
        "evidence": evidence,
    }


def review_id(review):
    data = json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def read_review(checkpoint, metadata, cfg, decision, candidate_id):
    """Bind confirmation to the saved proposal, weights and optimizer state."""
    expected = metadata.get("pending_candidate")
    directory = Path(checkpoint)
    if (directory / "candidate.json").exists() != (expected is not None):
        raise ValueError("Incomplete checkpoint candidate metadata")
    if expected is None:
        if decision is not None:
            raise ValueError("Calibration decisions require a checkpoint with a pending candidate")
        return None
    review = json.loads((directory / "candidate.json").read_text())
    if review_id(review) != expected or review["source_policy"] != metadata["policy_version"]:
        raise ValueError("Calibration candidate differs from its saved checkpoint review")
    if decision is None and cfg.calibration.approval == "manual":
        raise ValueError(
            f"Review {directory / 'candidate.json'} and pass --accept-candidate {expected}"
            f" or --reject-candidate {expected}"
        )
    if decision is not None and candidate_id != expected:
        raise ValueError("Decision refers to a different calibration candidate")
    return review


def read_payload(checkpoint, review):
    payload = (Path(checkpoint) / "learner.pt").read_bytes()
    if review is not None and hashlib.sha256(payload).hexdigest() != review["checkpoint_sha256"]:
        raise ValueError("Calibration checkpoint payload does not match the reviewed candidate")
    return payload


def load_scales(cfg, artifact):
    return parse_scale_artifact(artifact, **artifact_context(cfg))


def crosses_interval(before, after, interval):
    return interval > 0 and before // interval < after // interval
