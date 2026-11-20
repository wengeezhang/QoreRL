"""Configuration and activation observation for recipe calibration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


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
    initial_artifact: str | None = None
    max_activation_relative_l2: float | None = None

    def __post_init__(self):
        if type(self.refresh_every_optimizer_steps) is not int or self.refresh_every_optimizer_steps < 0:
            raise ValueError("refresh_every_optimizer_steps must be a nonnegative integer")
        if self.initial_artifact is not None and (
            not isinstance(self.initial_artifact, str) or not self.initial_artifact.strip()
        ):
            raise ValueError("initial_artifact must be a nonempty JSON file path")
        if not self.refresh_every_optimizer_steps and self.initial_artifact is None:
            raise ValueError("A zero refresh interval requires an external initial_artifact")
        if self.approval not in ("manual", "auto"):
            raise ValueError("approval must be manual or auto")
        if type(self.discover_outliers) is not bool or type(self.discovery_only) is not bool:
            raise ValueError("discover_outliers and discovery_only must be boolean")
        if self.discovery_only and (not self.discover_outliers or self.initial_artifact is not None):
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
        if isinstance(options, dict) and "scale_artifact" in options:
            if "initial_artifact" in options:
                raise ValueError("Specify initial_artifact only once")
            options = options.copy()
            options["initial_artifact"] = options.pop("scale_artifact")
        if not isinstance(options, dict) or set(options) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported calibration configuration or fields")
        return cls(**options)


def activation_windows(samples, config, *, explicit_targets=None):
    """Yield per-channel maxima, held-out rows, and discovery evidence for each observed module.

    With outlier screening disabled, explicit targets bypass the threshold.
    Other targets must still qualify through the observed fit rows.
    """
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
            explicit_targets is not None and name not in explicit_targets
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
    evaluate a candidate, the scorer reserves a disjoint held-out suffix.
    """
    return [event for _, _, _, event in activation_windows(samples, config)]


def discover(samples, config):
    """Return the selected module names and scores for an activation window."""
    return {
        event["target"]: event["outlier_score"]
        for event in outlier_report(samples, config)
        if event["status"] == "discovered"
    }


def make_scale_artifact(scales, *, model, model_revision, recipe_id):
    """Compatibility entry point for offline tools using the original documented import."""
    from qorerl.recipes.smoothquant import make_smoothing_artifact

    return make_smoothing_artifact(scales, model=model, model_revision=model_revision, recipe_id=recipe_id)
