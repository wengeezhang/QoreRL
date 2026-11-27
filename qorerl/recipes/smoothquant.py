"""Fit, apply, and serialize SmoothQuant channel smoothing factors.

A factor S is a positive FP32 vector acting on the canonical projection as
X/S and W*S. Fitting and evaluation produce a candidate without changing the
installed factors. Only application mutates the recipe state.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch.nn import functional as F

from qorerl.recipes.calibration import activation_windows, outlier_report
from qorerl.recipes.state import down_proj_name, smooth_input, smooth_weight, smoothing_factor_digest

_TARGET = re.compile(r"model\.layers\.(0|[1-9][0-9]*)\.mlp\.down_proj\.weight")


def _check_names(names):
    if any(not isinstance(name, str) or not _TARGET.fullmatch(name) for name in names):
        raise ValueError("Calibration requires canonical down projection weight names")


def _artifact_context(model, model_revision, recipe_id):
    context = {"model": model, "model_revision": model_revision, "recipe_id": recipe_id}
    if any(not isinstance(value, str) or not value.strip() for value in context.values()):
        raise ValueError("Smoothing artifact requires model, model_revision, and base recipe_id")
    return context


def make_smoothing_artifact(smoothing_factors, *, model, model_revision, recipe_id):
    """Serialize the final per-channel factors and their digest in the portable v1 format."""
    context = _artifact_context(model, model_revision, recipe_id)
    if not smoothing_factors:
        raise ValueError("Smoothing artifact must contain at least one named smoothing factor vector")
    digest = smoothing_factor_digest(smoothing_factors)
    _check_names(smoothing_factors)
    return {
        "schema": 1,
        "kind": "smoothquant_scales",
        **context,
        "scales": {name: factor.tolist() for name, factor in sorted(smoothing_factors.items())},
        "scale_digest": digest,
    }


def parse_smoothing_artifact(artifact, *, model, model_revision, recipe_id, targets=None):
    """Check provenance, finite positive FP32 vectors and digest, then installed scope/width if supplied."""
    context = _artifact_context(model, model_revision, recipe_id)
    expected = {"schema", "kind", "scales", "scale_digest", *context}
    if not isinstance(artifact, dict) or set(artifact) != expected:
        raise ValueError("Invalid SmoothQuant smoothing artifact fields")
    if (
        type(artifact["schema"]) is not int
        or artifact["schema"] != 1
        or artifact["kind"] != "smoothquant_scales"
    ):
        raise ValueError("Unsupported SmoothQuant smoothing artifact schema or kind")
    for field, value in context.items():
        if artifact[field] != value:
            raise ValueError(f"Smoothing artifact {field} mismatch")
    vectors = artifact["scales"]
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError("Smoothing artifact must contain at least one named smoothing factor vector")
    _check_names(vectors)
    smoothing_factors = {}
    for name, vector in vectors.items():
        if (
            not isinstance(vector, list)
            or not vector
            or any(type(value) not in (int, float) for value in vector)
        ):
            raise ValueError(f"Expected a nonempty JSON smoothing factor vector for {name}")
        if targets is not None:
            if name not in targets:
                raise ValueError(f"Smoothing artifact target outside installed recipe: {name}")
            if len(vector) != targets[name][1]:
                raise ValueError(f"Smoothing artifact width mismatch: {name}")
        try:
            smoothing_factors[name] = torch.tensor(vector, dtype=torch.float32, device="cpu")
        except (OverflowError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Unrepresentable smoothing factor vector for {name}") from exc
    if smoothing_factor_digest(smoothing_factors) != artifact["scale_digest"]:
        raise ValueError("Smoothing artifact digest mismatch")
    return smoothing_factors


def _fit_smoothing_factor(maxima, weight, current_factor, alpha):
    weight_maxima = weight.abs().amax(0).double()
    active = (maxima > 0) & (weight_maxima > 0)
    # Channels unseen in the fit window retain their current smoothing factor.
    factor = current_factor.to(device=weight.device).clone()
    # Evaluate in FP64 before rounding the final S to checkpoint precision.
    factor[active] = (maxima[active].pow(alpha) / weight_maxima[active].pow(1 - alpha)).float()
    return factor.cpu()


class _NonfiniteProjection(ValueError):
    """An unusable transform can be rejected without masking backend exceptions."""


def _measure_output_relative_l2(fprop, x, weight, smoothing_factor, reference, reference_norm):
    """Run one projection configuration and measure relative L2 against its BF16 reference."""
    activation, transformed_weight = (
        (x, weight)
        if smoothing_factor is None
        else (smooth_input(x, smoothing_factor), smooth_weight(weight, smoothing_factor))
    )
    if not torch.isfinite(activation).all().item() or not torch.isfinite(transformed_weight).all().item():
        raise _NonfiniteProjection("SmoothQuant transformed operands exceed the BF16 range")
    output = fprop(activation, transformed_weight)
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != reference.shape
        or output.dtype != torch.bfloat16
        or output.device != reference.device
    ):
        raise ValueError("Quantized FPROP must return the paired BF16 output matrix")
    relative_l2 = torch.linalg.vector_norm(output.double() - reference.double()).item() / reference_norm
    if not math.isfinite(relative_l2):
        raise _NonfiniteProjection("Nonfinite held-out quantized FPROP error")
    return relative_l2


@torch.no_grad()
def prepare_smoothquant_candidate(
    weights, samples, current_smoothing_factors, config, *, fprop=None, allow_layer_expansion=False
):
    """Measure a new overlay against direct low precision, or a refresh against the active S.

    Both comparisons use the same BF16 reference and require a strict held-out
    gain. First activation also requires an outlier in the authorized base scope.
    """
    if config.discovery_only:
        return {}, outlier_report(samples, config)
    if (
        not isinstance(weights, Mapping)
        or not weights
        or not isinstance(samples, Mapping)
        or (set(weights) != set(samples))
    ):
        raise ValueError("Calibration samples and weights must cover the same nonempty scope")
    _check_names(weights)
    if type(allow_layer_expansion) is not bool:
        raise ValueError("allow_layer_expansion must be boolean")
    if (
        not isinstance(current_smoothing_factors, dict)
        or not current_smoothing_factors.keys() <= weights.keys()
    ):
        raise ValueError("Active smoothing factors must name a subset of the supplied calibration scope")
    if not allow_layer_expansion and current_smoothing_factors.keys() != weights.keys():
        raise ValueError("Active smoothing factors must cover the entire supplied calibration scope")
    smoothing_factor_digest(current_smoothing_factors)
    if fprop is None:
        from qorerl.recipes.te_mxfp8 import native_mxfp8_linear

        fprop = native_mxfp8_linear
    smoothing_factors, evidence = {}, []
    for name, maxima, heldout, event in activation_windows(
        samples, config, explicit_targets=current_smoothing_factors.keys()
    ):
        activating = name not in current_smoothing_factors
        event["action"] = "activate" if activating else "refresh"
        if activating:
            event.update(
                direct_low_precision_relative_l2=None,
                max_activation_relative_l2=config.max_activation_relative_l2,
            )
        evidence.append(event)
        if event["status"] != "discovered":
            continue
        weight = weights[name]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected canonical BF16 weight matrix for {name}")
        weight = weight.detach()
        if not weight.numel() or heldout.shape[1] != weight.shape[1] or heldout.device != weight.device:
            raise ValueError(f"Calibration requires matching input width and device for {name}")
        current_factor = (
            torch.ones(weight.shape[1], dtype=torch.float32, device="cpu")
            if activating
            else current_smoothing_factors[name]
        )
        if current_factor.shape != (weight.shape[1],):
            raise ValueError(f"Active smoothing factor width mismatch for {name}")
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"Nonfinite calibration weight for {name}")
        with torch.autocast(heldout.device.type, enabled=False):
            reference = F.linear(heldout, weight)
            norm = torch.linalg.vector_norm(reference.double()).item()
            if not math.isfinite(norm):
                raise ValueError(f"Nonfinite held-out BF16 reference on {name}")
            if norm == 0:
                event.update(status="insufficient_signal", reason="zero_heldout_reference")
                continue
            if activating:
                try:
                    before = _measure_output_relative_l2(fprop, heldout, weight, None, reference, norm)
                except _NonfiniteProjection as exc:
                    event.update(status="invalid_candidate", reason=f"Direct low precision: {exc}")
                    continue
                event["direct_low_precision_relative_l2"] = before
            else:
                before = _measure_output_relative_l2(fprop, heldout, weight, current_factor, reference, norm)
            event["current_relative_l2"] = before
            candidate_factor = _fit_smoothing_factor(maxima, weight, current_factor, config.alpha)
            if not torch.isfinite(candidate_factor).all().item() or not (candidate_factor > 0).all().item():
                event.update(status="invalid_candidate", reason="scale_outside_positive_fp32_range")
                continue
            try:
                after = _measure_output_relative_l2(fprop, heldout, weight, candidate_factor, reference, norm)
            except _NonfiniteProjection as exc:
                event.update(status="invalid_candidate", reason=str(exc))
                continue
        event["candidate_relative_l2"] = after
        improvement = (before - after) / before if before > 0 else None
        event["relative_improvement"] = improvement
        if (
            after >= before
            or improvement < config.min_relative_improvement
            or torch.equal(candidate_factor, current_factor)
        ):
            event["status"] = "no_gain"
            continue
        if (
            activating
            and config.max_activation_relative_l2 is not None
            and after > config.max_activation_relative_l2
        ):
            event.update(status="activation_cost_exceeded", reason="above_max_activation_relative_l2")
            continue
        event["status"] = "proposed"
        smoothing_factors[name] = candidate_factor
    return smoothing_factors, evidence


def apply_smoothquant_candidate(state, smoothing_factors):
    """Merge approved factors into the active mapping after validating the complete transform."""
    if not state.spec.uses_smoothquant:
        raise ValueError("Installing smoothing factors requires the SmoothQuant recipe")
    if (
        not isinstance(smoothing_factors, dict)
        or not smoothing_factors
        or not smoothing_factors.keys() <= state.targets.keys()
    ):
        raise ValueError("A SmoothQuant candidate must name at least one eligible installed target")
    state.install_smoothing_factors(state.smoothing_factors | smoothing_factors)


def prepare_candidate(cfg, state, installation, samples):
    """Return a portable proposal and evidence for the live learner's projection kernel."""
    smoothing_factors, evidence = prepare_smoothquant_candidate(
        {name: installation.weights[name] for name in samples},
        samples,
        state.smoothing_factors,
        cfg.calibration,
        fprop=installation.linear,
        allow_layer_expansion=state.spec.allow_layer_expansion,
    )
    if not smoothing_factors:
        return None, evidence
    artifact = make_smoothing_artifact(
        smoothing_factors, model=cfg.model, model_revision=cfg.model_revision, recipe_id=state.spec.identity
    )
    return artifact, evidence


def apply_candidate(cfg, state, artifact, *, initial=False):
    """Check the artifact against the installed recipe and apply its per-target changes."""
    recipe = state.spec
    recipe_id = recipe.identity
    if initial and recipe.allow_layer_expansion and isinstance(artifact, dict):
        # A fixed offline fit may seed an expanding run with the same initial layers.
        fixed_recipe_id = replace(recipe, allow_layer_expansion=False).identity
        if artifact.get("recipe_id") == fixed_recipe_id:
            recipe_id = fixed_recipe_id
    smoothing_factors = parse_smoothing_artifact(
        artifact,
        model=cfg.model,
        model_revision=cfg.model_revision,
        recipe_id=recipe_id,
        targets=state.targets,
    )
    if initial:
        initial_layers = recipe.layers if recipe.smoothquant_layers is None else recipe.smoothquant_layers
        if initial_layers is not None and smoothing_factors.keys() - {
            down_proj_name(i) for i in initial_layers
        }:
            raise ValueError("Startup artifact names a layer outside the initial SmoothQuant targets")
    apply_smoothquant_candidate(state, smoothing_factors)
