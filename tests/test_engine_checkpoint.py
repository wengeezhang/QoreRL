"""Checkpoint naming migrations must preserve optimizer state and continuation."""

import io
import math

import pytest
import torch

from qorerl.engines.factory import make_learner_engine, make_rollout_engine
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import TrainingBatch, Trajectory


def cpu_config():
    return TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id="fp32-test-v1",
        model="cpu-fixture",
        group_size=2,
    )


def encode(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


def batch_for(learner):
    trajectories = [
        Trajectory(0, i, (1, 3), (5 + i,), (-math.log(14),), str(1 + i), "length", learner.policy_version)
        for i in range(2)
    ]
    return TrainingBatch(
        trajectories, torch.tensor([1.0, 0.0]), torch.tensor([1.0, -1.0]), learner.policy_version
    )


def legacy_state(state):
    """The on-disk schema emitted before the refactor, including flat version keys."""
    policy_version = state["policy_version"]
    return {
        "schema": 1,
        "model": state["model"],
        "masters": state["master_parameters"],
        "optimizer": state["optimizer"],
        "version": policy_version["weight_version"],
        "recipe": policy_version["recipe_id"],
        "optimizer_steps": state["optimizer_steps"],
        "torch_rng": state["torch_rng"],
        "cuda_rng": state["cuda_rng"],
    }


@pytest.mark.parametrize("schema", [1, 2])
def test_checkpoint_schema_preserves_adam_continuation(schema):
    cfg = cpu_config()
    original = make_learner_engine(cfg)
    original.update(batch_for(original))
    state = torch.load(io.BytesIO(original.checkpoint()), weights_only=True)
    assert state["schema"] == 2 and state["optimizer_steps"] == 1
    if schema == 1:
        state = legacy_state(state)
    restored = make_learner_engine(cfg)
    restored.restore(encode(state))
    assert restored.policy_version == original.policy_version
    assert restored.optimizer_steps == original.optimizer_steps

    original.update(batch_for(original))
    restored.update(batch_for(restored))
    for a, b in zip(original.master_parameters, restored.master_parameters, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for name, a in original.model.state_dict().items():
        torch.testing.assert_close(a, restored.model.state_dict()[name], atol=0, rtol=0)
    for index, a in original.optimizer.state_dict()["state"].items():
        for key, value in a.items():
            torch.testing.assert_close(
                value, restored.optimizer.state_dict()["state"][index][key], atol=0, rtol=0
            )

    # Restored publication identity is usable by a newly constructed rollout engine.
    rollout = make_rollout_engine(cfg)
    assert rollout.install_weights(restored.export_weights()) == restored.policy_version


@pytest.mark.parametrize("schema", [1, 2])
def test_checkpoint_rejects_changed_recipe_before_loading_weights(schema):
    cfg = cpu_config()
    source = make_learner_engine(cfg)
    source.update(batch_for(source))
    state = torch.load(io.BytesIO(source.checkpoint()), weights_only=True)
    if schema == 1:
        state = legacy_state(state)
        state["recipe"] = "unsupported-recipe"
    else:
        state["policy_version"]["recipe_id"] = "unsupported-recipe"
    target = make_learner_engine(cfg)
    before = {name: value.clone() for name, value in target.model.state_dict().items()}
    with pytest.raises(ValueError, match="recipe"):
        target.restore(encode(state))
    for name, value in before.items():
        torch.testing.assert_close(value, target.model.state_dict()[name], atol=0, rtol=0)


def test_checkpoint_rejects_unknown_schema():
    learner = make_learner_engine(cpu_config())
    with pytest.raises(ValueError, match="schema"):
        learner.restore(encode({"schema": 999}))
