"""Recipe identity, canonical state and recovery are enforced before execution."""

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qorerl.engines.base import validate_install
from qorerl.engines.single_rank_learner import SingleRankGRPOLearner
from qorerl.recipes.calibration import make_scale_artifact
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, MXFP8_SMOOTHQUANT, resolve_recipe
from qorerl.recipes.state import RecipeState, down_proj_name
from qorerl.runtime.calibration import artifact_context, load_initial_scales, load_scales
from qorerl.runtime.ray_runtime import run_sync_grpo
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


def test_expansion_is_an_opt_in_permission_of_smoothquant():
    fixed = resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [1]})
    assert "allow_layer_expansion" not in fixed.to_dict()
    assert (
        fixed.identity
        == resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [1], "allow_layer_expansion": False}).identity
    )
    expand = resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [1], "allow_layer_expansion": True})
    assert expand.identity != fixed.identity
    assert expand.select_layers(3) == fixed.eligible_layers(3) == (1,)
    assert expand.eligible_layers(3) == (1,)
    with pytest.raises(ValueError, match="exceed"):
        expand.eligible_layers(1)
    with pytest.raises(ValueError, match="boolean"):
        resolve_recipe(MXFP8_SMOOTHQUANT, {"allow_layer_expansion": 1})
    with pytest.raises(ValueError, match="Unsupported options"):
        resolve_recipe(MXFP8_HADAMARD, {"allow_layer_expansion": True})


def test_first_activation_with_identity_scale_is_a_new_recoverable_policy():
    spec = resolve_recipe(
        MXFP8_SMOOTHQUANT, {"layers": [0, 1], "smoothquant_layers": [0], "allow_layer_expansion": True}
    )
    state = make_state(spec)
    initial = down_proj_name(0)
    extra = state.add_target(1, torch.zeros(32, 64), active=False)
    before = state.to_dict()
    assert set(state.report()["targets"]) == {initial, extra}
    assert state.report()["smoothquant_targets"] == [initial]
    assert set(state.report()["eligible_targets"]) == {initial, extra}
    state.install_scales({**state.scales, extra: torch.ones(64)})
    promoted = state.active_identity
    assert state.is_active(extra) and promoted != state.validate(before)
    for invalid in ({extra: torch.ones(64)}, {**state.scales, "unknown": torch.ones(64)}):
        with pytest.raises(ValueError, match="scales"):
            state.install_scales(invalid)
        assert state.active_identity == promoted
    state.load(before)
    assert not state.is_active(extra)
    assert state.is_active(initial)


def test_startup_artifact_cannot_activate_layers_outside_initial_selection(tmp_path):
    artifact = tmp_path / "scales.json"
    cfg = TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id=MXFP8_SMOOTHQUANT,
        recipe_options={"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True},
        recipe_calibration={"scale_artifact": str(artifact)},
    )
    artifact.write_text(
        json.dumps(make_scale_artifact({down_proj_name(0): torch.ones(32)}, **artifact_context(cfg)))
    )
    with pytest.raises(ValueError, match="outside the initial"):
        run_sync_grpo(cfg, tmp_path / "output")


def test_offline_fit_for_same_initial_layers_can_seed_expansion_but_not_a_live_review():
    fixed = TrainConfig(recipe_id=MXFP8_SMOOTHQUANT, recipe_options={"layers": [1]})
    cfg = replace(fixed, recipe_options={"layers": [1], "allow_layer_expansion": True})
    name = down_proj_name(1)
    scales = {name: torch.ones(32)}
    artifact = make_scale_artifact(scales, **artifact_context(fixed))
    torch.testing.assert_close(load_initial_scales(cfg, artifact)[name], scales[name])
    with pytest.raises(ValueError, match="recipe_id"):
        load_scales(cfg, artifact)
    with pytest.raises(ValueError, match="recipe_id"):
        load_initial_scales(
            replace(cfg, recipe_options={"layers": [2], "allow_layer_expansion": True}), artifact
        )


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


@pytest.mark.parametrize("bad_scale", ["missing", "width", "dtype", "negative", "zero", "nan", "inf", "grad"])
def test_smoothquant_validates_all_scale_vectors_before_install(bad_scale):
    state = make_state(resolve_recipe(MXFP8_SMOOTHQUANT))
    first = next(iter(state.targets))
    second = state.add_target(1, torch.zeros(32, 64))
    previous = state.active_identity
    scales = state.scales
    scales[first].fill_(2)
    if bad_scale == "missing":
        scales.pop(second)
    elif bad_scale == "width":
        scales[second] = torch.ones(32)
    elif bad_scale == "dtype":
        scales[second] = scales[second].bfloat16()
    elif bad_scale == "grad":
        scales[second].requires_grad_()
    else:
        scales[second][0] = {"negative": -1, "zero": 0, "nan": float("nan"), "inf": float("inf")}[bad_scale]
    with pytest.raises(ValueError, match="scale"):
        state.install_scales(scales)
    assert state.active_identity == previous
    assert state.scales[first].eq(1).all()


def test_smoothquant_scale_ownership_and_checkpoint_round_trip():
    spec = resolve_recipe(MXFP8_SMOOTHQUANT)
    state, other = make_state(spec), make_state(spec)
    name = next(iter(state.targets))
    scale = torch.linspace(0.25, 4, 128)
    state.install_scales({name: scale})
    identity = state.active_identity
    assert identity != other.active_identity
    scale.zero_()
    state.scales[name].neg_()
    payload = state.to_dict()
    assert other.validate(payload) == identity
    assert identity != other.active_identity
    other.load(payload)
    assert other.active_identity == identity == f"{spec.identity}:{payload['scale_digest']}"
    torch.testing.assert_close(state.scales[name], other.scales[name], rtol=0, atol=0)
    payload["scales"][name][0] *= 2
    with pytest.raises(ValueError, match="digest"):
        other.load(payload)
    assert other.active_identity == identity


@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize("expand", [True, False])
def test_smoothing_is_bounded_by_the_fixed_low_precision_scope(precision, expand):
    base = (7, 15, 20, 25)
    initial = base[:-1]
    options = {
        "precision": precision,
        "layers": base,
        "smoothquant_layers": initial,
        "allow_layer_expansion": expand,
    }
    spec = resolve_recipe("smoothquant-mlp-down-v1", options)
    assert spec.identity == resolve_recipe(f"{precision}-smoothquant-mlp-down-v1", options).identity
    assert spec.select_layers(32) == base
    assert spec.initial_smoothquant_layers(32) == initial
    assert spec.eligible_layers(32) == (base if expand else initial)
    state = RecipeState(spec)
    w = torch.ones(32, 64)
    for index in base:
        state.add_target(index, w)
    with pytest.raises(ValueError, match="outside configured"):
        state.add_target(23, w)
    direct = down_proj_name(25)
    assert state.transform_weight(w, direct) is w
    proposed = {**state.scales, direct: torch.full((64,), 2.0)}
    with pytest.raises(ValueError, match="eligible"):
        state.install_scales({**proposed, down_proj_name(23): torch.ones(64)})
    if expand:
        state.install_scales(proposed)
        torch.testing.assert_close(state.transform_weight(w, direct), w * 2)
    else:
        with pytest.raises(ValueError, match="eligible"):
            state.install_scales(proposed)
    assert set(state.report()["targets"]) == {down_proj_name(i) for i in base}


def test_format_selection_and_invalid_bounds_are_checked_before_execution():
    assert TrainConfig().recipe.name == MXFP8_MLP
    for name in ("mlp-down-v1", "hadamard-mlp-down-v1", "smoothquant-mlp-down-v1"):
        assert resolve_recipe(name).precision == "mxfp8"
        assert resolve_recipe(name, {"precision": "nvfp4"}).precision == "nvfp4"
        assert resolve_recipe(name).identity != resolve_recipe(name, {"precision": "nvfp4"}).identity
    with pytest.raises(ValueError, match="conflicts"):
        resolve_recipe(MXFP8_SMOOTHQUANT, {"precision": "nvfp4"})
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_recipe("mlp-down-v1", {"precision": "int7"})
    with pytest.raises(ValueError, match="subset"):
        resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [7], "smoothquant_layers": [8]})
    with pytest.raises(ValueError, match="exceed"):
        resolve_recipe(MXFP8_SMOOTHQUANT, {"smoothquant_layers": [8]}).select_layers(8)
    with pytest.raises(ValueError, match="CPU test engines"):
        TrainConfig(
            learner_backend="torch_test",
            rollout_backend="torch_test",
            recipe_id="smoothquant-mlp-down-v1",
            recipe_options={"precision": "nvfp4"},
        )
