"""Algorithm semantics independent of GPU libraries and Ray scheduling."""

import math
from dataclasses import replace

import pytest
import torch

from qorerl.engines.torch_test import TestLearner as CPULearner
from qorerl.engines.torch_test import TestRollout as CPURollout
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion, Prompt, Trajectory, assemble_batch, collate
from qorerl.training.objective import grpo_loss


def cpu_config(**kwargs):
    return TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id="fp32-test-v1",
        model="cpu-fixture",
        **kwargs,
    )


def sample(group, index, response, text, prompt=(1, 3), policy=PolicyVersion(0, "fp32-test-v1")):
    return Trajectory(group, index, prompt, response, (-math.log(14),) * len(response), text, "stop", policy)


def example_batch():
    prompts = [Prompt(7, (1, 3), "1"), Prompt(9, (1, 5, 3), "2")]
    rows = [
        sample(7, 0, (5, 2), "1"),
        sample(7, 1, (6,), "2"),
        sample(9, 0, (6, 2), "2", prompt=(1, 5, 3)),
        sample(9, 1, (5, 6, 2), "1 2", prompt=(1, 5, 3)),
    ]
    return assemble_batch(prompts, list(reversed(rows)), 2, rows[0].policy)


def test_groups_shift_mask_and_response_normalization():
    batch = example_batch()
    assert [(t.group_id, t.sample_id) for t in batch.trajectories] == [(7, 0), (7, 1), (9, 0), (9, 1)]
    torch.testing.assert_close(batch.advantages, torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=3e-6, rtol=0)
    packed = collate(batch, [0, 1, 2, 3], 0, "cpu")
    assert packed.labels[0, packed.response_mask[0]].tolist() == [5, 2]
    assert packed.response_mask.sum(-1).tolist() == [2, 1, 2, 3]
    assert packed.response_mask[:, 0].sum() == 0  # prompt prediction is excluded

    torch.manual_seed(12)
    logits = torch.randn(4, packed.labels.shape[1], 14, requires_grad=True)
    full, _ = grpo_loss(logits, packed, 0.2)
    expected = []
    for i in range(4):
        mask = packed.response_mask[i]
        logp = logits[i].log_softmax(-1).gather(-1, packed.labels[i, :, None]).squeeze(-1)[mask]
        ratio = (logp - packed.behavior_log_probs[i, mask]).exp()
        adv = batch.advantages[i]
        expected.append(-torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean())
    torch.testing.assert_close(full, torch.stack(expected).mean())
    full.backward()
    assert torch.count_nonzero(logits.grad[~packed.response_mask]) == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "stale", "recipe", "prompt", "nonfinite"])
def test_reject_invalid_rollouts(fault):
    policy = PolicyVersion(0, "fp32-test-v1")
    rows = [sample(7, 0, (5,), "1"), sample(7, 1, (6,), "2")]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "stale":
        rows[0] = replace(rows[0], policy=replace(policy, weights=1))
    elif fault == "recipe":
        rows[0] = replace(rows[0], policy=replace(policy, recipe="unknown"))
    elif fault == "prompt":
        rows[0] = replace(rows[0], prompt_ids=(1, 4))
    else:
        rows[0] = replace(rows[0], behavior_log_probs=(-math.inf,))
    with pytest.raises(ValueError):
        assemble_batch([Prompt(7, (1, 3), "1")], rows, 2, policy)


def test_uneven_microbatches_preserve_update_and_checkpoint():
    batch = example_batch()
    full = CPULearner(cpu_config(micro_batch_size=4))
    split = CPULearner(cpu_config(micro_batch_size=3))
    full.update(batch)
    split.update(batch)
    for a, b in zip(full.masters, split.masters, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    restored = CPULearner(cpu_config(micro_batch_size=4))
    restored.restore(full.checkpoint())
    assert restored.policy == full.policy
    next_batch = replace(batch, policy=full.policy)
    full.update(next_batch)
    restored.update(next_batch)
    for a, b in zip(full.masters, restored.masters, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_advantage_skips_adam_momentum_update():
    learner = CPULearner(cpu_config())
    batch = example_batch()
    learner.update(batch)  # establish nonzero optimizer momentum first
    before = [p.clone() for p in learner.masters]
    metrics = learner.update(replace(batch, policy=learner.policy, advantages=torch.zeros(4)))
    assert metrics["skipped_zero_advantage"] and metrics["optimizer_steps"] == 1
    for a, b in zip(before, learner.masters, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_sampler_records_exact_chosen_probabilities_and_requires_publication():
    cfg = cpu_config(group_size=4, max_new_tokens=3)
    learner, rollout = CPULearner(cfg), CPURollout(cfg)
    prompt = Prompt(7, (1, 5, 3), "1")
    with pytest.raises(ValueError, match="acknowledge"):
        rollout.generate([prompt], learner.policy, 19)
    rollout.install_weights(learner.export_weights())
    rows = rollout.generate([prompt], learner.policy, 19)
    batch = assemble_batch([prompt], rows, 4, learner.policy)
    packed = collate(batch, list(range(4)), 0, "cpu")
    lp = learner.logits(packed.input_ids).log_softmax(-1).gather(-1, packed.labels[..., None]).squeeze(-1)
    torch.testing.assert_close(lp[packed.response_mask], packed.behavior_log_probs[packed.response_mask])
    with pytest.raises(ValueError, match="increase"):
        rollout.install_weights(learner.export_weights())


def test_ppo_only_clips_the_advantage_improving_side():
    batch = example_batch()
    packed = collate(batch, [0, 1, 2, 3], 0, "cpu")
    logits = torch.zeros(4, packed.labels.shape[1], 14, requires_grad=True)
    # A>0 & ratio>1.2 and A<0 & ratio<0.8 should be clipped.
    # The opposite two cases must still have a gradient.
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
    packed.behavior_log_probs = torch.full_like(packed.behavior_log_probs, -math.log(14))
    packed.behavior_log_probs -= ratios.log()[:, None]
    loss, stats = grpo_loss(logits, packed, 0.2)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:2]) == 0
    assert logits.grad[2:].abs().sum() > 0
    assert stats["clipped_tokens"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"group_size": 1}, {"update_epochs": 0}, {"recipe_id": "mxfp8"}, {"learning_rate": float("nan")}],
)
def test_reject_unsupported_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
