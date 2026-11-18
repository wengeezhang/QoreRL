"""Exercise dynamic scale publication and recovery using the explicit CPU fixture."""

import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from qorerl.engines.factory import make_learner_engine, make_rollout_engine
from qorerl.engines.single_rank_learner import _calibration_prefixes
from qorerl.recipes.spec import MXFP8_SMOOTHQUANT
from qorerl.recipes.state import down_proj_name
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import TrainingBatch
from qorerl.training.objective import compute_group_advantages
from qorerl.training.tasks import CopyTask, exact_match_rewards


def fixture_config():
    return TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        model="cpu-fixture",
        recipe_id=MXFP8_SMOOTHQUANT,
        group_size=8,
        max_new_tokens=1,
        recipe_calibration={"refresh_every_optimizer_steps": 1},
    )


def sample(cfg, rollout, step):
    prompts = CopyTask().batch(step, cfg.prompts_per_step, cfg.seed)
    policy = rollout.policy_version
    trajectories = rollout.generate(prompts, policy, cfg.seed + step * 1000003)
    rewards = exact_match_rewards(prompts, trajectories)
    return TrainingBatch(trajectories, rewards, compute_group_advantages(rewards, cfg.group_size), policy)


def checkpoint_state(learner):
    return torch.load(io.BytesIO(learner.checkpoint()), weights_only=True)


def assert_same_training_state(a, b):
    for field in ("model", "master_parameters", "optimizer", "optimizer_steps", "torch_rng"):
        torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)


def assert_same_checkpoint(a, b):
    assert_same_training_state(a, b)
    assert a["policy_version"] == b["policy_version"]
    assert a["recipe_state"]["scale_digest"] == b["recipe_state"]["scale_digest"]
    torch.testing.assert_close(a["recipe_state"]["scales"], b["recipe_state"]["scales"], rtol=0, atol=0)


def test_live_smoothquant_candidate_preserves_canonical_state_and_resumes_training():
    cfg = fixture_config()
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    rollout.install_weights(learner.export_weights())
    batch = sample(cfg, rollout, 1)
    update = learner.update(batch)
    assert update["optimizer_steps"] == 1 and update["pre_update_abs_log_ratio_mean"] < 1e-5
    version_before = rollout.install_weights(learner.export_weights())
    before = checkpoint_state(learner)
    proposals, evidence = learner.prepare_calibration(batch)
    assert down_proj_name(0) in proposals and down_proj_name(2) not in proposals
    assert {event["target"] for event in evidence} == set(learner.recipe_state.targets)
    for event in evidence:
        if event["target"] in proposals:
            assert event["candidate_relative_l2"] < event["current_relative_l2"]
    assert_same_checkpoint(checkpoint_state(learner), before)

    ids = torch.tensor([[1, 4, 3]])
    logits_before = rollout.model(ids)
    approved = learner.apply_calibration(proposals)
    assert approved.weight_version == version_before.weight_version
    assert approved.recipe_id != version_before.recipe_id
    after = checkpoint_state(learner)
    assert_same_training_state(after, before)
    for name, values in after["recipe_state"]["scales"].items():
        expected = proposals.get(name, before["recipe_state"]["scales"][name])
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert rollout.install_weights(learner.export_weights()) == approved
    assert not torch.equal(rollout.model(ids), logits_before)
    torch.testing.assert_close(rollout.model(ids), learner.logits(ids), rtol=0, atol=0)
    with pytest.raises(ValueError, match="increase"):
        rollout.install_weights(learner.export_weights())

    restored = make_learner_engine(cfg)
    restored.restore(learner.checkpoint())
    restored_rollout = make_rollout_engine(cfg)
    assert restored_rollout.install_weights(restored.export_weights()) == approved
    next_batch = sample(cfg, rollout, 2)
    assert next_batch.trajectories == sample(cfg, restored_rollout, 2).trajectories
    assert learner.update(next_batch) == restored.update(next_batch)
    assert learner.policy_version.recipe_id == approved.recipe_id
    assert_same_checkpoint(checkpoint_state(learner), checkpoint_state(restored))


def test_prefix_capture_excludes_duplicate_contexts_and_obeys_bound():
    trajectories = [
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 2)),
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 4)),
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 8, 2)),
        SimpleNamespace(prompt_ids=(1, 4), response_ids=(7, 6, 2)),
    ]
    prefixes = list(_calibration_prefixes(trajectories, max_rows=5))
    assert prefixes == [((1, 3, 7), 0), ((1, 3, 7, 8), 3), ((1, 4), 1)]


def test_scoped_scale_updates_and_tampering_fail_before_policy_changes():
    cfg = replace(fixture_config(), recipe_options={"layers": [0, 1]})
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    old = rollout.install_weights(learner.export_weights())
    valid = {down_proj_name(0): torch.full((32,), 2.0)}
    for invalid in (
        {down_proj_name(1): torch.zeros(32)},
        {down_proj_name(1): torch.ones(64)},
        {down_proj_name(2): torch.ones(32)},
        {down_proj_name(1): torch.ones(32, dtype=torch.bool)},
    ):
        with pytest.raises(ValueError):
            learner.apply_calibration({**valid, **invalid})
        assert learner.policy_version == old and learner.recipe_state.active_identity == old.recipe_id
    learner.apply_calibration(valid)
    snapshot = learner.export_weights()
    snapshot.recipe_state["scales"][down_proj_name(0)].add_(1)
    with pytest.raises(ValueError, match="digest"):
        rollout.install_weights(snapshot)
    assert rollout.policy_version == old

    # A valid scale payload must agree with the policy identity in both transports.
    snapshot = learner.export_weights()
    snapshot.policy_version = old
    with pytest.raises(ValueError, match="recipe"):
        rollout.install_weights(snapshot)
    state = checkpoint_state(learner)
    state["policy_version"]["recipe_id"] = old.recipe_id
    stream = io.BytesIO()
    torch.save(state, stream)
    target = make_learner_engine(cfg)
    before = checkpoint_state(target)
    with pytest.raises(ValueError, match="recipe"):
        target.restore(stream.getvalue())
    assert_same_checkpoint(checkpoint_state(target), before)
