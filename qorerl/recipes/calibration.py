"""Fit and evaluate SmoothQuant scale proposals on canonical, scoped learner operands.

The first half of bounded token rows supplies channel maxima for discovery and
fitting. The disjoint second half measures both quantized FPROPs against the
same untransformed BF16 projection. A proposal is data, never a state mutation.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from qorerl.recipes.state import scale_digest, smooth_input, smooth_weight

_TARGET = re.compile(r"model\.layers\.(0|[1-9][0-9]*)\.mlp\.down_proj\.weight")


@dataclass(frozen=True)
class CalibrationConfig:
    refresh_every_optimizer_steps: int = 20
    approval: str = "manual"
    discover_outliers: bool = True
    discovery_only: bool = False
    outlier_threshold: float = 4.0
    alpha: float = 0.5
    max_rows: int = 128
    min_relative_improvement: float = 0.0
    scale_artifact: str | None = None
    max_activation_relative_l2: float | None = None

    def __post_init__(self):
        if type(self.refresh_every_optimizer_steps) is not int or self.refresh_every_optimizer_steps < 0:
            raise ValueError("refresh_every_optimizer_steps must be a nonnegative integer")
        if self.scale_artifact is not None and (
            not isinstance(self.scale_artifact, str) or not self.scale_artifact.strip()
        ):
            raise ValueError("scale_artifact must be a nonempty JSON file path")
        if not self.refresh_every_optimizer_steps and self.scale_artifact is None:
            raise ValueError("A zero refresh interval requires an external scale_artifact")
        if self.approval not in ("manual", "auto"):
            raise ValueError("approval must be manual or auto")
        if type(self.discover_outliers) is not bool or type(self.discovery_only) is not bool:
            raise ValueError("discover_outliers and discovery_only must be boolean")
        if self.discovery_only and (not self.discover_outliers or self.scale_artifact is not None):
            raise ValueError("discovery_only requires outlier discovery without an external artifact")
        if type(self.max_rows) is not int or self.max_rows < (1 if self.discovery_only else 2):
            raise ValueError("max_rows must allow one discovery row or two disjoint fitting/evaluation rows")
        for field in ("alpha", "min_relative_improvement", "outlier_threshold"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number")
        if not 0 <= self.alpha <= 1 or not 0 <= self.min_relative_improvement <= 1:
            raise ValueError("alpha and min_relative_improvement must be in [0, 1]")
        if self.outlier_threshold < 1:
            raise ValueError("outlier_threshold must be >= 1 for the channel peak-to-mean ratio")
        bound = self.max_activation_relative_l2
        if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound) or bound < 0):
            raise ValueError("max_activation_relative_l2 must be a finite nonnegative number or None")

    @classmethod
    def from_dict(cls, options):
        if not isinstance(options, dict) or set(options) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported calibration configuration or fields")
        return cls(**options)


def _check_names(names):
    if any(not isinstance(name, str) or not _TARGET.fullmatch(name) for name in names):
        raise ValueError("Calibration requires canonical down projection weight names")


def _artifact_context(model, model_revision, recipe_id):
    context = {"model": model, "model_revision": model_revision, "recipe_id": recipe_id}
    if any(not isinstance(value, str) or not value.strip() for value in context.values()):
        raise ValueError("Scale artifact requires model, model_revision, and base recipe_id")
    return context


def make_scale_artifact(scales, *, model, model_revision, recipe_id):
    """Serialize final multiplicative scales S (X/S, W*S), including their content digest."""
    context = _artifact_context(model, model_revision, recipe_id)
    digest = scale_digest(scales)
    _check_names(scales)
    return {
        "schema": 1,
        "kind": "smoothquant_scales",
        **context,
        "scales": {name: scale.tolist() for name, scale in sorted(scales.items())},
        "scale_digest": digest,
    }


def parse_scale_artifact(artifact, *, model, model_revision, recipe_id, targets=None):
    """Check provenance, finite positive FP32 vectors and digest, then installed scope/width if supplied."""
    context = _artifact_context(model, model_revision, recipe_id)
    expected = {"schema", "kind", "scales", "scale_digest", *context}
    if not isinstance(artifact, dict) or set(artifact) != expected:
        raise ValueError("Invalid SmoothQuant scale artifact fields")
    if (
        type(artifact["schema"]) is not int
        or artifact["schema"] != 1
        or artifact["kind"] != "smoothquant_scales"
    ):
        raise ValueError("Unsupported SmoothQuant scale artifact schema or kind")
    for field, value in context.items():
        if artifact[field] != value:
            raise ValueError(f"Scale artifact {field} mismatch")
    vectors = artifact["scales"]
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError("Scale artifact must contain at least one named scale vector")
    _check_names(vectors)
    scales = {}
    for name, vector in vectors.items():
        if (
            not isinstance(vector, list)
            or not vector
            or any(type(value) not in (int, float) for value in vector)
        ):
            raise ValueError(f"Expected a nonempty JSON scale vector for {name}")
        if targets is not None:
            if name not in targets:
                raise ValueError(f"Scale artifact target outside installed recipe: {name}")
            if len(vector) != targets[name][1]:
                raise ValueError(f"Scale artifact width mismatch: {name}")
        try:
            scales[name] = torch.tensor(vector, dtype=torch.float32, device="cpu")
        except (OverflowError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Unrepresentable scale vector for {name}") from exc
    if scale_digest(scales) != artifact["scale_digest"]:
        raise ValueError("Scale artifact digest mismatch")
    return scales


def _activation_windows(samples, config, *, active_scales=None):
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("Outlier reporting requires named activation samples")
    if any(not isinstance(name, str) or not name.strip() for name in samples):
        raise ValueError("Activation samples must have nonempty module names")
    for name in sorted(samples):
        x = samples[name]
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] == 0:
            raise ValueError(f"Expected BF16 activation matrix with nonempty channel dimension for {name}")
        rows = x[: config.max_rows].detach()
        if not torch.isfinite(rows).all().item():
            raise ValueError(f"Nonfinite calibration activation rows for {name}")
        fit_rows = len(rows) if config.discovery_only else len(rows) // 2
        fit, heldout = rows[:fit_rows], rows[fit_rows:]
        maxima = fit.abs().amax(0).double() if len(fit) else None
        require_outlier = config.discover_outliers or (
            active_scales is not None and name not in active_scales
        )
        score, reason = None, None
        if maxima is None:
            status, reason = "insufficient_rows", "empty_fit_window"
        elif maxima.max().item() == 0:
            status, reason = "insufficient_signal", "all_zero_fit_window"
        else:
            score = (maxima.max() / maxima.mean()).item()
            selected = not require_outlier or score >= config.outlier_threshold
            status = "discovered" if selected else "not_selected"
        yield (
            name,
            maxima,
            heldout,
            {
                "target": name,
                "selection": "outlier" if require_outlier else "explicit",
                "outlier_score": score,
                "threshold": config.outlier_threshold,
                "alpha": config.alpha,
                "fit_rows": fit_rows,
                "heldout_rows": len(heldout),
                "status": status,
                "reason": reason,
                "current_relative_l2": None,
                "candidate_relative_l2": None,
                "relative_improvement": None,
            },
        )


@torch.no_grad()
def outlier_report(samples, config):
    """Score named BF16 [token, channel] activations without requiring model weights or a recipe.

    Discovery-only windows use all bounded rows. When a caller also plans to
    fit scales, the scorer reserves the same held-out suffix used by prepare.
    """
    return [event for _, _, _, event in _activation_windows(samples, config)]


def discover(samples, config):
    """Return the selected module names and scores for an activation window."""
    return {
        event["target"]: event["outlier_score"]
        for event in outlier_report(samples, config)
        if event["status"] == "discovered"
    }


def _fit_scale(maxima, weight, current, alpha):
    weight_maxima = weight.abs().amax(0).double()
    active = (maxima > 0) & (weight_maxima > 0)
    # Channels unseen in the fit window retain their existing scale.
    scale = current.to(device=weight.device).clone()
    # Evaluate in FP64 before rounding the final S to checkpoint precision.
    scale[active] = (maxima[active].pow(alpha) / weight_maxima[active].pow(1 - alpha)).float()
    return scale.cpu()


class _NonfiniteProjection(ValueError):
    """An unusable transform can be rejected without masking backend exceptions."""


def _error(fprop, x, weight, scale, reference, reference_norm):
    activation, transformed_weight = smooth_input(x, scale), smooth_weight(weight, scale)
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
    error = torch.linalg.vector_norm(output.double() - reference.double()).item() / reference_norm
    if not math.isfinite(error):
        raise _NonfiniteProjection("Nonfinite held-out quantized FPROP error")
    return error


@torch.no_grad()
def prepare(weights, samples, current_scales, config, *, fprop=None, allow_layer_expansion=False):
    """Evaluate active scale refreshes and, when authorized, first activations on eligible BF16 layers.

    Refreshes require a strict held-out gain. Activation requires a detected outlier
    and reports both direct MXFP8 and fitted SmoothQuant error against BF16.
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
        not isinstance(current_scales, dict)
        or not current_scales
        or not current_scales.keys() <= weights.keys()
    ):
        raise ValueError("Active scales must name a nonempty subset of the supplied calibration scope")
    if not allow_layer_expansion and current_scales.keys() != weights.keys():
        raise ValueError("Active scales must cover the entire supplied calibration scope")
    scale_digest(current_scales)
    if fprop is None:
        from qorerl.recipes.te_mxfp8 import native_mxfp8_linear

        fprop = native_mxfp8_linear
    overrides, evidence = {}, []
    for name, maxima, heldout, event in _activation_windows(samples, config, active_scales=current_scales):
        activating = name not in current_scales
        event["action"] = "activate" if activating else "refresh"
        if activating:
            event.update(
                direct_mxfp8_relative_l2=None, max_activation_relative_l2=config.max_activation_relative_l2
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
        old = (
            torch.ones(weight.shape[1], dtype=torch.float32, device="cpu")
            if activating
            else current_scales[name]
        )
        if old.shape != (weight.shape[1],):
            raise ValueError(f"Active scale width mismatch for {name}")
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
                event["current_relative_l2"] = 0.0
                try:
                    event["direct_mxfp8_relative_l2"] = _error(fprop, heldout, weight, old, reference, norm)
                except _NonfiniteProjection as exc:
                    event.update(status="invalid_candidate", reason=f"Direct MXFP8: {exc}")
                    continue
            else:
                before = _error(fprop, heldout, weight, old, reference, norm)
                event["current_relative_l2"] = before
            new = _fit_scale(maxima, weight, old, config.alpha)
            if not torch.isfinite(new).all().item() or not (new > 0).all().item():
                event.update(status="invalid_candidate", reason="scale_outside_positive_fp32_range")
                continue
            try:
                after = _error(fprop, heldout, weight, new, reference, norm)
            except _NonfiniteProjection as exc:
                event.update(status="invalid_candidate", reason=str(exc))
                continue
        event["candidate_relative_l2"] = after
        if activating:
            if config.max_activation_relative_l2 is not None and after > config.max_activation_relative_l2:
                event.update(status="activation_cost_exceeded", reason="above_max_activation_relative_l2")
                continue
        else:
            improvement = (before - after) / before if before > 0 else None
            event["relative_improvement"] = improvement
            if after >= before or improvement < config.min_relative_improvement or torch.equal(new, old):
                event["status"] = "no_gain"
                continue
        event["status"] = "proposed"
        overrides[name] = new
    return overrides, evidence
