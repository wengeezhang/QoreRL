"""Recipe identity, canonical state and recovery are enforced before execution."""

import io
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qorerl.engines.base import validate_install
from qorerl.engines.single_rank_learner import SingleRankGRPOLearner
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, resolve_recipe
from qorerl.recipes.state import RecipeState
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion, WeightSnapshot


def test_recipe_identity_covers_numerical_options_but_not_layer_order():
    spec = resolve_recipe(MXFP8_HADAMARD, {"layers": [1, 0], "seed": 7})
    assert spec.identity == resolve_recipe(MXFP8_HADAMARD, {"layers": [0, 1], "seed": 7}).identity
    assert spec.identity != resolve_recipe(MXFP8_HADAMARD, {"layers": [0, 1], "seed": 8}).identity
    assert spec.identity != resolve_recipe(MXFP8_HADAMARD, {"layers": [0, 1], "block_size": 32}).identity
    assert spec.identity != resolve_recipe(MXFP8_HADAMARD, {"layers": [0]}).identity
    assert resolve_recipe("bf16-baseline-v1").identity == "bf16-baseline-v1"
    assert spec.select_layers(2) == (0, 1)
    with pytest.raises(ValueError, match="exceed"):
        spec.select_layers(1)


@pytest.mark.parametrize(
    "options",
    [
        {"layers": []},
        {"layers": [1, 1]},
        {"layers": [-1]},
        {"layers": [True]},
        {"block_size": 33},
        {"block_size": True},
        {"block_size": None},
        {"seed": -1},
        {"seed": 0.1},
        {"anything": 1},
    ],
)
def test_invalid_recipe_options_fail_before_starting_engines(options):
    with pytest.raises(ValueError):
        TrainConfig(recipe_id=MXFP8_HADAMARD, recipe_options=options)


def test_no_silent_recipe_fallback_or_ignored_options():
    with pytest.raises(ValueError, match="Unsupported options"):
        TrainConfig(recipe_options={"seed": 0})
    with pytest.raises(ValueError, match="Unsupported options"):
        TrainConfig(recipe_id=MXFP8_MLP, recipe_options={"block_size": 128})
    with pytest.raises(ValueError, match="no native quantization fallback"):
        TrainConfig(learner_backend="torch_test", rollout_backend="torch_test", recipe_id=MXFP8_MLP)


def make_state(spec):
    state = RecipeState(spec)
    state.add_target(0, torch.zeros(64, 128))
    return state


def test_transform_state_is_independent_and_detects_wrong_signs_and_targets():
    spec = resolve_recipe(MXFP8_HADAMARD)
    source, destination = make_state(spec), make_state(spec)
    payload = source.to_dict()
    destination.validate(payload)
    name = next(iter(source.signs))
    payload["signs"][name][0] *= -1
    with pytest.raises(ValueError, match="fixed signs"):
        destination.validate(payload)
    torch.testing.assert_close(source.signs[name], destination.signs[name])
    payload = source.to_dict()
    payload["targets"][name] = [128, 128]
    with pytest.raises(ValueError, match="targets"):
        destination.validate(payload)
    # Public inspection must not detach the serialized signs from live caches.
    x = torch.randn(3, 128)
    before = source.rotate(x, name)
    source.signs[name].neg_()
    torch.testing.assert_close(source.rotate(x, name), before, rtol=0, atol=0)
    destination.validate(source.to_dict())


def test_three_native_configs_change_only_the_recipe():
    directory = Path(__file__).resolve().parents[1] / "configs"
    baseline = TrainConfig.load(directory / "grpo_megatron_vllm.json")
    for filename in ("grpo_mxfp8_mlp_down.json", "grpo_mxfp8_hadamard_mlp_down.json"):
        cfg = TrainConfig.load(directory / filename)
        assert replace(cfg, recipe_id=baseline.recipe_id, recipe_options={}) == baseline
        assert cfg.recipe.is_mxfp8 and cfg.recipe.layers is None


def test_publication_rejects_same_preset_with_different_rotation():
    cfg = TrainConfig(recipe_id=MXFP8_HADAMARD, recipe_options={"seed": 5})
    other = replace(cfg, recipe_options={"seed": 6})
    snapshot = WeightSnapshot(PolicyVersion(1, cfg.policy_recipe_id), cfg.model, {"w": torch.ones(1)})
    with pytest.raises(ValueError, match="Model/recipe"):
        validate_install(snapshot, None, other)


def test_checkpoint_keeps_fixed_transform_without_changing_parameter_basis():
    # Exercise the actual single-rank checkpoint implementation on CPU. This is
    # a state test; the model here does not claim native recipe execution.
    cfg = TrainConfig(recipe_id=MXFP8_HADAMARD)

    def learner():
        instance = SingleRankGRPOLearner()
        instance.recipe_state = make_state(cfg.recipe)
        instance.setup_optimizer(cfg, torch.nn.Linear(128, 64, bias=False), 0, "cpu")
        return instance

    source, restored = learner(), learner()
    restored.restore(source.checkpoint())
    torch.testing.assert_close(source.model.weight, restored.model.weight, rtol=0, atol=0)
    assert set(source.model.state_dict()) == {"weight"}
    state = torch.load(io.BytesIO(source.checkpoint()), weights_only=True)
    name = next(iter(state["recipe_state"]["signs"]))
    state["recipe_state"]["signs"][name][0] *= -1
    stream = io.BytesIO()
    torch.save(state, stream)
    before = restored.model.weight.detach().clone()
    with pytest.raises(ValueError, match="fixed signs"):
        restored.restore(stream.getvalue())
    torch.testing.assert_close(before, restored.model.weight, rtol=0, atol=0)
