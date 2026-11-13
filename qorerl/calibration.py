"""Paired BF16/MXFP8 gradient probes and Wgrad-activation-only interventions."""

import copy
from dataclasses import dataclass

import torch

from qorerl.precision import precision, quantized_layers
from qorerl.rl import policy_loss


@dataclass
class Probe:
    gradients: dict
    logits: torch.Tensor
    metrics: dict
    expert_counts: dict


def capture(model, batch, mode, overrides=()):
    """Same parameter snapshot and recorded trajectories; never optimizer.step()."""
    model.zero_grad(set_to_none=True)
    try:
        with precision(model, mode, overrides):
            loss, metrics, logits = policy_loss(model, batch)
            loss.backward()
        gradients = {
            name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()
        }
        for name, grad in gradients.items():
            if grad is not None and not torch.isfinite(grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
        return Probe(gradients, logits.detach(), metrics, copy.deepcopy(model.expert_counts()))
    finally:
        model.zero_grad(set_to_none=True)


def gradient_comparison(reference, candidate, min_norm=1e-8):
    if reference is None or candidate is None:
        return {
            "status": "inactive_both" if reference is candidate else "unmatched_activity",
            "cosine": None,
            "reference_norm": None,
            "candidate_norm": None,
        }
    a, b = reference.detach().double().reshape(-1), candidate.detach().double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("Non-finite gradient comparison")
    na, nb = a.norm().item(), b.norm().item()
    if min(na, nb) <= min_norm:
        return {"status": "too_small", "cosine": None, "reference_norm": na, "candidate_norm": nb}
    return {
        "status": "ok",
        "cosine": max(-1.0, min(1.0, (a @ b).item() / (na * nb))),
        "reference_norm": na,
        "candidate_norm": nb,
    }


def combined_gradient(model, gradients):
    return torch.cat(
        [
            (torch.zeros_like(p) if gradients[name] is None else gradients[name]).reshape(-1)
            for name, p in model.named_parameters()
        ]
    )


def assign_gradients(model, gradients):
    model.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        param.grad = None if gradients[name] is None else gradients[name].clone()


def shadow_update(model, optimizer, batch, gradients, reference, max_grad_norm):
    """Try one REAL AdamW step on a private model AND private optimizer state.

    Evaluates both precision objectives with frozen advantages/old logprobs.
    This is a local surrogate-loss diagnostic, NOT a reward improvement proof.
    """
    shadow = copy.deepcopy(model)
    shadow_opt = torch.optim.AdamW(shadow.parameters())
    shadow_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assign_gradients(shadow, gradients)
    grad_norm = torch.nn.utils.clip_grad_norm_(shadow.parameters(), max_grad_norm).item()
    shadow_opt.step()
    delta = {
        name: p.detach() - dict(model.named_parameters())[name].detach()
        for name, p in shadow.named_parameters()
    }
    update = combined_gradient(model, delta).double()
    ref_grad = combined_gradient(model, reference).double()
    losses = {}
    with torch.no_grad():
        for mode in ("mxfp8", "bf16"):
            with precision(shadow, mode):
                losses[f"{mode}_loss_after"] = policy_loss(shadow, batch)[0].item()
    return {
        **losses,
        "unclipped_grad_norm": grad_norm,
        "update_norm": update.norm().item(),
        "reference_gradient_dot_update": (ref_grad @ update).item(),
    }


def calibrate(
    model,
    optimizer,
    batch,
    step,
    previous_overrides=(),
    cosine_threshold=0.0,
    min_norm=1e-8,
    max_grad_norm=1.0,
    trial_updates=True,
):
    if not -1 <= cosine_threshold <= 0:
        raise ValueError("Only opposite directions may trigger repair: threshold must be in [-1,0]")
    reference = capture(model, batch, "bf16")
    baseline = capture(model, batch, "mxfp8")  # Pure MXFP8, including on later calibration rounds.
    records, triggered = [], set()
    for layer_name in quantized_layers(model):
        name = layer_name + ".weight"
        comparison = gradient_comparison(reference.gradients[name], baseline.gradients[name], min_norm)
        selected = comparison["status"] == "ok" and comparison["cosine"] < cosine_threshold
        if selected:
            triggered.add(layer_name)
        records.append({"layer": layer_name, "before": comparison, "triggered": selected})
    # Conservative experiment policy: retain previously activated overrides.
    # No claim of beneficial recovery is used to accept them; measure it instead.
    overrides = set(previous_overrides) | triggered
    candidate = capture(model, batch, "mxfp8", overrides) if overrides else baseline
    if not torch.equal(candidate.logits, baseline.logits):
        raise AssertionError("Wgrad-only intervention changed forward logits")
    for name in baseline.gradients:
        if name.removesuffix(".weight") in overrides:
            continue
        before, after = baseline.gradients[name], candidate.gradients[name]
        equal = before is after if before is None or after is None else torch.equal(before, after)
        if not equal:
            raise AssertionError(f"Wgrad-only intervention changed an unselected gradient: {name}")
    for record in records:
        name = record["layer"] + ".weight"
        after = gradient_comparison(reference.gradients[name], candidate.gradients[name], min_norm)
        record.update(after=after, active=record["layer"] in overrides)
    report = {
        "step": step,
        "triggered": sorted(triggered),
        "new_overrides": sorted(overrides - set(previous_overrides)),
        "active_overrides": sorted(overrides),
        "forward_identical": True,
        "nonselected_gradients_identical": True,
        "reference_loss_before": reference.metrics["loss"],
        "mxfp8_loss_before": baseline.metrics["loss"],
        "global_before": gradient_comparison(
            combined_gradient(model, reference.gradients),
            combined_gradient(model, baseline.gradients),
            min_norm,
        ),
        "global_after": gradient_comparison(
            combined_gradient(model, reference.gradients),
            combined_gradient(model, candidate.gradients),
            min_norm,
        ),
        "layers": records,
        "reference_expert_counts": reference.expert_counts,
        "mxfp8_expert_counts": baseline.expert_counts,
    }
    if trial_updates:
        report["baseline_trial"] = shadow_update(
            model, optimizer, batch, baseline.gradients, reference.gradients, max_grad_norm
        )
        report["candidate_trial"] = shadow_update(
            model, optimizer, batch, candidate.gradients, reference.gradients, max_grad_norm
        )
    return overrides, candidate, report
