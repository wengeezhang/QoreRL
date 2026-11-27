"""SmoothQuant fitting from full-model BF16 activations and joint logits selection."""

import math
from copy import deepcopy

import torch

from qorerl.recipes.calibration import CandidatePreparation, NonfiniteLogits, activation_windows
from qorerl.recipes.smoothquant import (
    _fit_smoothing_factor,
    apply_smoothquant_candidate,
    make_smoothing_artifact,
)


def _fit_joint_factors(samples, weights, current_factors, config):
    """Fit all selected layers; no local-error filter can qualify a joint proposal."""
    factors, evidence = {}, []
    for name, maxima, _, event in activation_windows(samples, config, explicit_targets=current_factors):
        event["action"] = "refresh" if name in current_factors else "activate"
        evidence.append(event)
        if event["status"] != "discovered":
            continue
        weight = weights[name].detach()
        if (
            weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or weight.shape[1] != maxima.numel()
            or weight.device != maxima.device
            or not torch.isfinite(weight).all().item()
        ):
            raise ValueError(f"Expected matching finite canonical BF16 weights for {name}")
        current_factor = current_factors.get(name, torch.ones(weight.shape[1], dtype=torch.float32))
        factor = _fit_smoothing_factor(maxima, weight, current_factor, config.alpha)
        if not torch.isfinite(factor).all().item() or not (factor > 0).all().item():
            event.update(status="invalid_candidate", reason="smoothing_factor_outside_positive_fp32_range")
        elif torch.equal(factor, current_factor):
            event.update(status="unchanged", reason="unchanged_smoothing_factor")
        else:
            factors[name] = factor
            event["status"] = "fitted"
    return factors, evidence


def _logits_relative_l2(logits, reference, reference_norm):
    if logits.shape != reference.shape:
        raise ValueError("Full-model logits must share token positions and vocabulary shape")
    value = (torch.linalg.vector_norm(logits.double() - reference.double()) / reference_norm).item()
    if not math.isfinite(value):
        raise NonfiniteLogits("Nonfinite full-model relative L2")
    return value


@torch.no_grad()
def prepare_candidate(cfg, state, installation, replay):
    config = cfg.calibration
    comparison = {
        "scope": "model_logits",
        "reference": "full_model_bf16",
        "activation_source": "full_model_bf16",
        "reference_execution": replay.reference_execution,
        "fit_rows": replay.fit_rows,
        "heldout_rows": replay.heldout_rows,
        "current_relative_l2": None,
        "candidate_relative_l2": None,
        "relative_improvement": None,
        "min_relative_improvement": config.min_relative_improvement,
        "max_logits_relative_l2": config.max_logits_relative_l2,
        "targets": [],
        "status": "insufficient_rows",
        "reason": None,
    }
    if not replay.fit_rows or not replay.heldout_rows or not state.observation_targets:
        comparison["reason"] = "need_fitting_rows_heldout_rows_and_targets"
        return CandidatePreparation(None, [], comparison)

    # All low-precision projections are bypassed, including those outside the
    # editable overlay. Only authorized targets are captured and fitted.
    samples, reference = replay.run(reference=True, collect_inputs=True)
    reference_norm = torch.linalg.vector_norm(reference.double()).item()
    if not math.isfinite(reference_norm) or reference_norm == 0:
        comparison.update(status="insufficient_signal", reason="zero_or_nonfinite_reference_norm")
        return CandidatePreparation(None, [], comparison)
    factors, evidence = _fit_joint_factors(samples, installation.weights, state.smoothing_factors, config)
    del samples
    comparison["targets"] = sorted(factors)
    _, current = replay.run()
    comparison["current_relative_l2"] = _logits_relative_l2(current, reference, reference_norm)
    del current
    invalid_fit = any(row["status"] == "invalid_candidate" for row in evidence)
    if invalid_fit or not factors:
        comparison.update(
            status="invalid_candidate" if invalid_fit else "no_candidate",
            reason="invalid_fitted_factor" if invalid_fit else "no_changed_factors",
        )
    else:
        # Validate/merge into a private state; the live policy and its caches are untouched.
        provisional_state = deepcopy(state)
        apply_smoothquant_candidate(provisional_state, factors)
        try:
            _, candidate = replay.run(state=provisional_state)
            comparison["candidate_relative_l2"] = _logits_relative_l2(candidate, reference, reference_norm)
            del candidate
        except NonfiniteLogits as exc:
            comparison.update(status="invalid_candidate", reason=str(exc))
        else:
            current_relative_l2 = comparison["current_relative_l2"]
            candidate_relative_l2 = comparison["candidate_relative_l2"]
            improvement = (
                (current_relative_l2 - candidate_relative_l2) / current_relative_l2
                if current_relative_l2 > 0
                else None
            )
            comparison["relative_improvement"] = improvement
            if candidate_relative_l2 >= current_relative_l2 or improvement < config.min_relative_improvement:
                comparison.update(status="no_gain", reason="joint_logits_error_did_not_improve_enough")
            elif (
                config.max_logits_relative_l2 is not None
                and candidate_relative_l2 > config.max_logits_relative_l2
            ):
                comparison.update(status="error_cap_exceeded", reason="above_max_logits_relative_l2")
            else:
                comparison["status"] = "proposed"

    accepted = comparison["status"] == "proposed"
    for row in evidence:
        if row["status"] == "fitted":
            row.update(
                status="proposed" if accepted else "model_rejected",
                reason="joint_logits_gate_passed" if accepted else comparison["reason"],
            )
    artifact = (
        make_smoothing_artifact(
            factors, model=cfg.model, model_revision=cfg.model_revision, recipe_id=state.spec.identity
        )
        if accepted
        else None
    )
    return CandidatePreparation(artifact, evidence, comparison)
