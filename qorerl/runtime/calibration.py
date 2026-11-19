"""Review artifacts for a proposed change at a synchronous training boundary."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from qorerl.recipes.calibration import make_scale_artifact, parse_scale_artifact
from qorerl.recipes.state import down_proj_name


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


def load_initial_scales(cfg, artifact):
    context = artifact_context(cfg)
    recipe = cfg.recipe
    if recipe.allow_layer_expansion and isinstance(artifact, dict):
        # An offline fit for the same initial layers also fits the first batch
        # of an expanding run. Saved proposals still require an exact policy.
        fixed_recipe_id = replace(recipe, allow_layer_expansion=False).identity
        if artifact.get("recipe_id") == fixed_recipe_id:
            context["recipe_id"] = fixed_recipe_id
    scales = parse_scale_artifact(artifact, **context)
    if recipe.layers is not None and scales.keys() - {down_proj_name(i) for i in recipe.layers}:
        raise ValueError("Startup scale artifact names a layer outside the initial SmoothQuant targets")
    return scales


def crosses_interval(before, after, interval):
    return interval > 0 and before // interval < after // interval
