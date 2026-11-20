"""Exercise candidate publication and recovery using the explicit CPU fixture."""

import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

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


@pytest.mark.parametrize("expand", [False, True], ids=["refresh", "first-activation"])
def test_live_smoothquant_candidate_preserves_canonical_state_and_resumes_training(
    expand, smoothing_artifact
):
    options = {"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True} if expand else {}
    cfg = replace(fixture_config(), recipe_options=options)
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    rollout.install_weights(learner.export_weights())
    batch = sample(cfg, rollout, 1)
    update = learner.update(batch)
    assert update["optimizer_steps"] == 1 and update["pre_update_abs_log_ratio_mean"] < 1e-5
    version_before = rollout.install_weights(learner.export_weights())
    before = checkpoint_state(learner)
    name = down_proj_name(0)
    x = torch.linspace(-1.0, 1.0, 64).reshape(2, 32).bfloat16()
    if expand:
        assert set(before["recipe_state"]["scales"]) == {down_proj_name(1)}
        for model in (learner.model, rollout.model):
            projection = model.model.layers[0].mlp
            bf16 = F.linear(x * projection.gain, projection.down_proj.weight)
            direct = learner.recipe_installation.linear(x * projection.gain, projection.down_proj.weight)
            torch.testing.assert_close(projection(x, name, model.installation), direct, rtol=0, atol=0)
            assert not torch.equal(direct, bf16)
            protected = model.model.layers[2].mlp
            torch.testing.assert_close(
                protected(x, down_proj_name(2), model.installation),
                F.linear(x * protected.gain, protected.down_proj.weight),
                rtol=0,
                atol=0,
            )
        with pytest.raises(ValueError, match="outside installed|eligible"):
            learner.apply_candidate(smoothing_artifact(cfg, {down_proj_name(2): torch.ones(32)}))
    candidate, evidence = learner.prepare_candidate(batch)
    proposals = candidate["scales"]
    assert name in proposals and down_proj_name(2) not in proposals
    assert {event["target"] for event in evidence} == set(learner.recipe_state.targets)
    for event in evidence:
        if event["target"] in proposals and event["target"] in before["recipe_state"]["scales"]:
            assert event["candidate_relative_l2"] < event["current_relative_l2"]
    if expand:
        activation = next(event for event in evidence if event["target"] == name)
        assert activation["current_relative_l2"] == activation["direct_low_precision_relative_l2"]
        assert 0 < activation["candidate_relative_l2"] < activation["current_relative_l2"]
        assert down_proj_name(2) not in {event["target"] for event in evidence}
    assert_same_checkpoint(checkpoint_state(learner), before)

    ids = torch.tensor([[1, 4, 3]])
    logits_before = rollout.model(ids)
    approved = learner.apply_candidate(candidate)
    assert approved.weight_version == version_before.weight_version
    assert approved.recipe_id != version_before.recipe_id
    after = checkpoint_state(learner)
    assert_same_training_state(after, before)
    expected = before["recipe_state"]["scales"] | {
        name: torch.tensor(vector) for name, vector in proposals.items()
    }
    torch.testing.assert_close(after["recipe_state"]["scales"], expected, rtol=0, atol=0)
    assert rollout.install_weights(learner.export_weights()) == approved
    if expand:
        for model in (learner.model, rollout.model):
            projection = model.model.layers[0].mlp
            state = model.installation.state
            activation = x * projection.gain
            expected_projection = learner.recipe_installation.linear(
                state.transform_input(activation, name),
                state.transform_weight(projection.down_proj.weight, name),
            )
            torch.testing.assert_close(
                projection(x, name, model.installation), expected_projection, rtol=0, atol=0
            )
            assert not torch.equal(
                expected_projection,
                learner.recipe_installation.linear(activation, projection.down_proj.weight),
            )
            protected = model.model.layers[2].mlp
            torch.testing.assert_close(
                protected(x, down_proj_name(2), model.installation),
                F.linear(x * protected.gain, protected.down_proj.weight),
                rtol=0,
                atol=0,
            )
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


@pytest.mark.parametrize("discovery_only", [False, True], ids=["proposals", "discovery"])
def test_fixed_overlay_observes_and_modifies_only_its_initial_subset(discovery_only, smoothing_artifact):
    cfg = replace(
        fixture_config(),
        recipe_options={"layers": [0, 1], "smoothquant_layers": [1]},
        recipe_calibration={"refresh_every_optimizer_steps": 1, "discovery_only": discovery_only},
    )
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    original = rollout.install_weights(learner.export_weights())
    _, evidence = learner.prepare_candidate(sample(cfg, rollout, 1))
    assert [event["target"] for event in evidence] == [down_proj_name(1)]
    for index in (0, 2):
        with pytest.raises(ValueError):
            learner.apply_candidate(smoothing_artifact(cfg, {down_proj_name(index): torch.ones(32)}))
    assert learner.policy_version == original


@pytest.mark.parametrize("expand", [False, True])
def test_empty_initial_overlay_is_direct_until_a_candidate_is_approved(expand):
    cfg = replace(
        fixture_config(),
        recipe_options={"layers": [0, 1], "smoothquant_layers": [], "allow_layer_expansion": expand},
    )
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    rollout.install_weights(learner.export_weights())
    batch = sample(cfg, rollout, 1)
    if not expand:
        learner.logits = lambda *_: pytest.fail("Empty observation scope triggered a forward pass")
    candidate, evidence = learner.prepare_candidate(batch)
    assert not learner.recipe_state.smoothing_factors
    if expand:
        assert all(event["action"] == "activate" for event in evidence)
        assert down_proj_name(0) in candidate["scales"]
        learner.apply_candidate(candidate)
        rollout.install_weights(learner.export_weights())
        assert set(rollout.model.installation.state.smoothing_factors) == set(candidate["scales"])
    else:
        assert candidate is None and evidence == []


def test_prefix_capture_excludes_duplicate_contexts_and_obeys_bound():
    trajectories = [
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 2)),
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 4)),
        SimpleNamespace(prompt_ids=(1, 3), response_ids=(7, 8, 2)),
        SimpleNamespace(prompt_ids=(1, 4), response_ids=(7, 6, 2)),
    ]
    prefixes = list(_calibration_prefixes(trajectories, max_rows=5))
    assert prefixes == [((1, 3, 7), 0), ((1, 3, 7, 8), 3), ((1, 4), 1)]


def test_scoped_scale_updates_and_tampering_fail_before_policy_changes(smoothing_artifact):
    cfg = replace(fixture_config(), recipe_options={"layers": [0, 1]})
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    old = rollout.install_weights(learner.export_weights())
    valid = smoothing_artifact(cfg, {down_proj_name(0): torch.full((32,), 2.0)})
    for name, vector in (
        (down_proj_name(1), [0] * 32),
        (down_proj_name(1), [1] * 64),
        (down_proj_name(2), [1] * 32),
        (down_proj_name(1), [True] * 32),
    ):
        damaged = valid | {"scales": {**valid["scales"], name: vector}}
        with pytest.raises(ValueError):
            learner.apply_candidate(damaged)
        assert learner.policy_version == old and learner.recipe_state.active_identity == old.recipe_id
    learner.apply_candidate(valid)
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
