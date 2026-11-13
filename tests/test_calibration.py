import copy

import pytest
import torch

import qorerl.calibration as calibration
from qorerl.model import ModelConfig, TinyMoE
from qorerl.precision import precision, quantized_layers
from qorerl.rl import RolloutBatch, group_advantages, policy_loss, prompts, rollout


def fixture_batch():
    torch.manual_seed(21)
    model = TinyMoE(ModelConfig(layers=1))
    generator = torch.Generator().manual_seed(41)
    prefix, target = prompts(2, generator)
    with precision(model, "mxfp8"):
        batch = rollout(model, prefix, target, 4, generator)
    # Deterministic nonzero advantages to test plumbing independently of sparse reward luck.
    batch.advantages = group_advantages(torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]), 4)
    return model, batch


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def test_log_probs_replay_and_group_advantages():
    model, batch = fixture_batch()
    with precision(model, "mxfp8"):
        _, metrics, _ = policy_loss(model, batch)
    assert abs(metrics["ratio_mean"] - 1) < 1e-6
    assert metrics["clip_fraction"] == 0
    assert torch.equal(group_advantages(torch.ones(8), 4), torch.zeros(8))


def test_calibration_does_not_mutate_live_state_and_trials_use_adam_state():
    model, batch = fixture_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial = calibration.capture(model, batch, "mxfp8")
    calibration.assign_gradients(model, initial.gradients)
    optimizer.step()  # Populate momentum; the trial must not silently use a fresh optimizer.
    before_model = copy.deepcopy(model.state_dict())
    before_opt = copy.deepcopy(optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    overrides, candidate, report = calibration.calibrate(model, optimizer, batch, 20)
    assert equal_nested(before_model, model.state_dict())
    assert equal_nested(before_opt, optimizer.state_dict())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert report["forward_identical"] and report["nonselected_gradients_identical"]
    assert report["candidate_trial"]["update_norm"] > 0
    assert set(overrides).issubset(quantized_layers(model))
    assert candidate.logits.shape[-1] == 16


def test_explicit_fault_exercises_trigger_and_sticky_repair(monkeypatch):
    """FAULT INJECTION test only: never used in research training runs."""
    model, batch = fixture_batch()
    optimizer = torch.optim.AdamW(model.parameters())
    target = "blocks.0.attn.qkv"
    real_capture = calibration.capture
    baseline = real_capture(model, batch, "mxfp8")

    def controlled_reference(model, batch, mode, overrides=()):
        probe = real_capture(model, batch, mode, overrides)
        if mode == "bf16":
            probe.gradients[target + ".weight"] = -baseline.gradients[target + ".weight"]
        return probe

    monkeypatch.setattr(calibration, "capture", controlled_reference)
    overrides, _, report = calibration.calibrate(model, optimizer, batch, 20, trial_updates=False)
    assert target in report["triggered"] and target in overrides
    selected = next(r for r in report["layers"] if r["layer"] == target)
    assert selected["before"]["cosine"] < -0.999
    assert report["forward_identical"] and report["nonselected_gradients_identical"]
    monkeypatch.setattr(calibration, "capture", real_capture)
    later, _, _ = calibration.calibrate(model, optimizer, batch, 40, overrides, trial_updates=False)
    assert target in later


def test_missing_zero_and_nonfinite_gradients_are_not_silent_repairs():
    a = torch.ones(2, 3)
    assert calibration.gradient_comparison(None, None)["status"] == "inactive_both"
    assert calibration.gradient_comparison(None, a)["status"] == "unmatched_activity"
    assert calibration.gradient_comparison(torch.zeros_like(a), a)["cosine"] is None
    assert calibration.gradient_comparison(a, -a)["cosine"] < -0.999
    with pytest.raises(FloatingPointError):
        calibration.gradient_comparison(a, torch.full_like(a, float("nan")))


def test_all_zero_advantage_batch_does_not_trigger():
    model, batch = fixture_batch()
    empty_signal = RolloutBatch(
        batch.tokens, batch.old_log_probs, torch.zeros_like(batch.advantages), batch.rewards
    )
    optimizer = torch.optim.AdamW(model.parameters())
    overrides, _, report = calibration.calibrate(model, optimizer, empty_signal, 20, trial_updates=False)
    assert not overrides
    assert all(r["before"]["status"] in {"too_small", "inactive_both"} for r in report["layers"])
