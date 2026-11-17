"""Algorithm semantics independent of GPU libraries and Ray scheduling."""

import math
from dataclasses import replace

import pytest
import torch

from qorerl.engines.torch_test import TestLearner as CPULearner
from qorerl.engines.torch_test import TestRollout as CPURollout
from qorerl.training.batching import collate_microbatch
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import (
    PolicyVersion,
    Prompt,
    TrainingBatch,
    Trajectory,
    validate_and_order_trajectories,
)
from qorerl.training.objective import compute_group_advantages, grpo_loss
from qorerl.training.tasks import exact_match_rewards


def cpu_config(**kwargs):
    return TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id="fp32-test-v1",
        model="cpu-fixture",
        **kwargs,
    )


def sample(group, index, response, text, prompt=(1, 3), policy_version=PolicyVersion(0, "fp32-test-v1")):
    return Trajectory(
        group, index, prompt, response, (-math.log(14),) * len(response), text, "stop", policy_version
    )


def make_training_batch(prompts, trajectories, group_size, policy_version):
    ordered = validate_and_order_trajectories(prompts, trajectories, group_size, policy_version)
    rewards = exact_match_rewards(prompts, ordered)
    advantages = compute_group_advantages(rewards, group_size)
    return TrainingBatch(ordered, rewards, advantages, policy_version)


def example_batch():
    prompts = [Prompt(7, (1, 3), "1"), Prompt(9, (1, 5, 3), "2")]
    rows = [
        sample(7, 0, (5, 2), "1"),
        sample(7, 1, (6,), "2"),
        sample(9, 0, (6, 2), "2", prompt=(1, 5, 3)),
        sample(9, 1, (5, 6, 2), "1 2", prompt=(1, 5, 3)),
    ]
    return make_training_batch(prompts, list(reversed(rows)), 2, rows[0].policy_version)


def test_groups_shift_mask_and_response_normalization():
    batch = example_batch()
    assert [(t.group_id, t.sample_id) for t in batch.trajectories] == [(7, 0), (7, 1), (9, 0), (9, 1)]
    torch.testing.assert_close(batch.rewards, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=3e-6, rtol=0)
    microbatch = collate_microbatch(batch, [0, 1, 2, 3], 0, "cpu")
    assert microbatch.labels[0, microbatch.response_mask[0]].tolist() == [5, 2]
    assert microbatch.response_mask.sum(-1).tolist() == [2, 1, 2, 3]
    assert microbatch.response_mask[:, 0].sum() == 0  # prompt prediction is excluded

    torch.manual_seed(12)
    logits = torch.randn(4, microbatch.labels.shape[1], 14, requires_grad=True)
    full, _ = grpo_loss(logits, microbatch, 0.2)
    expected = []
    for i in range(4):
        mask = microbatch.response_mask[i]
        logp = logits[i].log_softmax(-1).gather(-1, microbatch.labels[i, :, None]).squeeze(-1)[mask]
        ratio = (logp - microbatch.behavior_log_probs[i, mask]).exp()
        adv = batch.advantages[i]
        expected.append(-torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean())
    torch.testing.assert_close(full, torch.stack(expected).mean())
    full.backward()
    assert torch.count_nonzero(logits.grad[~microbatch.response_mask]) == 0


def test_group_advantages_keep_groups_independent_and_equal_rewards_zero():
    rewards = torch.tensor([2.0, 4.0, 100.0, 100.0])
    expected = torch.tensor([-1.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(compute_group_advantages(rewards, 2), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize(
    "rewards",
    [torch.tensor([]), torch.ones(3), torch.ones(2, 2), torch.tensor([1.0, float("nan")])],
)
def test_group_advantages_reject_incomplete_or_invalid_rewards(rewards):
    with pytest.raises(ValueError):
        compute_group_advantages(rewards, 2)


def test_policy_version_reads_legacy_metadata_without_ambiguous_fields():
    expected = PolicyVersion(7, "bf16-baseline-v1")
    assert PolicyVersion.from_dict({"weight_version": 7, "recipe_id": "bf16-baseline-v1"}) == expected
    assert PolicyVersion.from_dict({"weights": 7, "recipe": "bf16-baseline-v1"}) == expected
    with pytest.raises(ValueError, match="ambiguous"):
        PolicyVersion.from_dict({"weights": 7, "recipe": "old", "weight_version": 8, "recipe_id": "new"})


@pytest.mark.parametrize("fault", ["missing", "duplicate", "stale", "recipe", "prompt", "nonfinite"])
def test_reject_invalid_rollouts(fault):
    policy = PolicyVersion(0, "fp32-test-v1")
    rows = [sample(7, 0, (5,), "1"), sample(7, 1, (6,), "2")]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "stale":
        rows[0] = replace(rows[0], policy_version=replace(policy, weight_version=1))
    elif fault == "recipe":
        rows[0] = replace(rows[0], policy_version=replace(policy, recipe_id="unknown"))
    elif fault == "prompt":
        rows[0] = replace(rows[0], prompt_ids=(1, 4))
    else:
        rows[0] = replace(rows[0], behavior_log_probs=(-math.inf,))
    with pytest.raises(ValueError):
        make_training_batch([Prompt(7, (1, 3), "1")], rows, 2, policy)


def test_uneven_microbatches_preserve_update_and_checkpoint():
    batch = example_batch()
    full = CPULearner(cpu_config(micro_batch_size=4))
    split = CPULearner(cpu_config(micro_batch_size=3))
    full.update(batch)
    split.update(batch)
    for a, b in zip(full.master_parameters, split.master_parameters, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    restored = CPULearner(cpu_config(micro_batch_size=4))
    restored.restore(full.checkpoint())
    assert restored.policy_version == full.policy_version
    next_batch = replace(batch, policy_version=full.policy_version)
    full.update(next_batch)
    restored.update(next_batch)
    for a, b in zip(full.master_parameters, restored.master_parameters, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_advantage_skips_adam_momentum_update():
    learner = CPULearner(cpu_config())
    batch = example_batch()
    learner.update(batch)  # establish nonzero optimizer momentum first
    before = [p.clone() for p in learner.master_parameters]
    metrics = learner.update(replace(batch, policy_version=learner.policy_version, advantages=torch.zeros(4)))
    assert metrics["skipped_zero_advantage"] and metrics["optimizer_steps"] == 1
    for a, b in zip(before, learner.master_parameters, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_pre_update_diagnostics_do_not_include_later_policy_updates():
    batch = example_batch()
    single = CPULearner(cpu_config(update_epochs=1, learning_rate=0.01))
    multiple = CPULearner(cpu_config(update_epochs=2, learning_rate=0.01))
    first, both = single.update(batch), multiple.update(batch)
    for key in ("ratio_mean", "abs_log_ratio_mean", "clip_fraction"):
        assert both[f"pre_update_{key}"] == first[key]
    assert both["optimizer_steps"] == 2
    assert both["ratio_mean"] != both["pre_update_ratio_mean"]


def test_sampler_records_exact_chosen_probabilities_and_requires_publication():
    cfg = cpu_config(group_size=4, max_new_tokens=3)
    learner, rollout = CPULearner(cfg), CPURollout(cfg)
    prompt = Prompt(7, (1, 5, 3), "1")
    with pytest.raises(ValueError, match="acknowledge"):
        rollout.generate([prompt], learner.policy_version, 19)
    rollout.install_weights(learner.export_weights())
    rows = rollout.generate([prompt], learner.policy_version, 19)
    batch = make_training_batch([prompt], rows, 4, learner.policy_version)
    microbatch = collate_microbatch(batch, list(range(4)), 0, "cpu")
    lp = (
        learner.logits(microbatch.input_ids)
        .log_softmax(-1)
        .gather(-1, microbatch.labels[..., None])
        .squeeze(-1)
    )
    torch.testing.assert_close(
        lp[microbatch.response_mask], microbatch.behavior_log_probs[microbatch.response_mask]
    )
    with pytest.raises(ValueError, match="increase"):
        rollout.install_weights(learner.export_weights())


def test_ppo_only_clips_the_advantage_improving_side():
    batch = example_batch()
    microbatch = collate_microbatch(batch, [0, 1, 2, 3], 0, "cpu")
    logits = torch.zeros(4, microbatch.labels.shape[1], 14, requires_grad=True)
    # A>0 & ratio>1.2 and A<0 & ratio<0.8 should be clipped.
    # The opposite two cases must still have a gradient.
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
    microbatch.behavior_log_probs = torch.full_like(microbatch.behavior_log_probs, -math.log(14))
    microbatch.behavior_log_probs -= ratios.log()[:, None]
    loss, stats = grpo_loss(logits, microbatch, 0.2)
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
