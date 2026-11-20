"""CPU installation tests using explicit Megatron and native-kernel stubs.

These verify wrapper handling, canonical state, and rotation autograd. They do
not validate Megatron CUDA execution or MXFP8 quantization accuracy.
"""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from qorerl.engines.megatron import MegatronLearner
from qorerl.recipes import megatron_mxfp8 as adapter
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, MXFP8_SMOOTHQUANT, resolve_recipe
from qorerl.recipes.state import down_proj_name
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import TrainingBatch, Trajectory


class TERowParallelLinear(torch.nn.Linear):
    def __init__(self, dtype):
        super().__init__(64, 64, bias=False, dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


class MLP(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(64, 64, bias=False, dtype=dtype)
        self.linear_fc2 = TERowParallelLinear(dtype)

    def forward(self, x):
        return self.linear_fc2(self.linear_fc1(x))[0]


class GPTModel(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.config = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            num_moe_experts=None,
        )
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = MLP(dtype)
            self.decoder.layers.append(layer)

    def forward(self, x):
        for layer in self.decoder.layers:
            x = layer.mlp(x)
        return x


class Float16Module(torch.nn.Module):
    """Mirror the pinned wrapper's structure and state_dict delegation."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.config = module.config

    def forward(self, x):
        return self.module(x)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)


@pytest.fixture
def megatron_stubs(monkeypatch):
    names = [
        "megatron",
        "megatron.core",
        "megatron.core.extensions",
        "megatron.core.extensions.transformer_engine",
        "megatron.core.transformer",
        "megatron.core.transformer.module",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules["megatron.core.extensions.transformer_engine"].TERowParallelLinear = TERowParallelLinear
    modules["megatron.core.transformer.module"].Float16Module = Float16Module

    # Keep real CPU tensors and explicitly replace the two native boundaries.
    # Spoofing Tensor.device would hide problems in rotation/device handling.
    original_weight_validator = adapter._validate_native_weight
    monkeypatch.setattr(adapter, "validate_native_mxfp8", lambda: {"backend": "cpu_test_stub"})
    monkeypatch.setattr(adapter, "_validate_native_weight", lambda weight: None)
    calls = []

    def linear(x, weight):
        calls.append((x.detach().clone(), weight.detach().clone()))
        return F.linear(x, weight)

    monkeypatch.setattr(adapter, "native_mxfp8_linear", linear)
    return SimpleNamespace(calls=calls, original_weight_validator=original_weight_validator)


@pytest.mark.parametrize("wrapped", [False, True])
def test_install_discovers_selected_projection_and_preserves_canonical_state(megatron_stubs, wrapped):
    raw = GPTModel()
    model = Float16Module(raw) if wrapped else raw
    before_parameters = dict(model.named_parameters())
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    recipe = resolve_recipe(MXFP8_HADAMARD, {"layers": [1], "block_size": 32, "seed": 9})
    installation = adapter.install(model, recipe, num_layers=2)

    assert model is raw if not wrapped else model.module is raw
    assert set(installation.state.targets) == {down_proj_name(1)}
    assert raw.decoder.layers[0].mlp.linear_fc2.forward.__func__ is TERowParallelLinear.forward
    assert "_qorerl_recipe" not in vars(raw.decoder.layers[0].mlp.linear_fc2)
    assert raw.decoder.layers[1].mlp.linear_fc2._qorerl_recipe == recipe.identity
    after_parameters = dict(model.named_parameters())
    assert before_parameters.keys() == after_parameters.keys()
    for name, parameter in before_parameters.items():
        assert parameter is after_parameters[name]

    x = torch.randn(3, 64).bfloat16().requires_grad_()
    model(x).float().square().sum().backward()
    assert len(megatron_stubs.calls) == 1
    assert installation.calls == {down_proj_name(1): 1}
    assert model.state_dict().keys() == before_state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert model.state_dict().keys() == raw.state_dict().keys()


@pytest.mark.parametrize("recipe_name", [MXFP8_MLP, MXFP8_HADAMARD])
def test_wrapped_projection_returns_gradients_to_canonical_parameters(megatron_stubs, recipe_name):
    # Float64 isolates installer/rotation autograd algebra from BF16 rounding.
    # The fixture explicitly bypasses the CUDA/BF16 guard and native GEMM only.
    torch.manual_seed(721)
    model = Float16Module(GPTModel(dtype=torch.float64))
    reference = copy.deepcopy(model)
    options = {"layers": [1]}
    if recipe_name == MXFP8_HADAMARD:
        options.update(block_size=32, seed=71)
    recipe = resolve_recipe(recipe_name, options)
    installation = adapter.install(model, recipe, num_layers=2)
    x = torch.randn(2, 3, 64, dtype=torch.float64, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    actual, expected = model(x), reference(x_reference)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=1e-12, atol=1e-12)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-12, atol=1e-12)
    canonical = model.module.decoder.layers[1].mlp.linear_fc2.weight
    effective = megatron_stubs.calls[0][1]
    if recipe.uses_hadamard:
        assert not torch.equal(effective, canonical)
    else:
        torch.testing.assert_close(effective, canonical, rtol=0, atol=0)
    assert installation.report()["calls"] == {down_proj_name(1): 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_model_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("sequence_parallel", True),
        ("num_moe_experts", 4),
    ],
)
def test_unsupported_parallelism_or_moe_fails_before_patching(megatron_stubs, field, value):
    raw = GPTModel()
    setattr(raw.config, field, value)
    with pytest.raises(ValueError, match="requires|dense"):
        adapter.install(Float16Module(raw), resolve_recipe(MXFP8_MLP), num_layers=2)
    assert all("forward" not in vars(layer.mlp.linear_fc2) for layer in raw.decoder.layers)


@pytest.mark.parametrize("failure", ["bias", "linear_type", "mlp_type", "dimensions"])
def test_all_targets_are_validated_before_any_projection_is_patched(megatron_stubs, failure):
    raw = GPTModel()
    second = raw.decoder.layers[1].mlp
    if failure == "bias":
        second.linear_fc2.bias = torch.nn.Parameter(torch.zeros(64, dtype=torch.bfloat16))
    elif failure == "linear_type":
        second.linear_fc2 = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    elif failure == "mlp_type":

        class UnsupportedMLP(MLP):
            pass

        raw.decoder.layers[1].mlp = UnsupportedMLP(torch.bfloat16)
    else:
        second.linear_fc2.weight = torch.nn.Parameter(torch.zeros(64, 48, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias-free|standard MCore|multiples of 32"):
        adapter.install(Float16Module(raw), resolve_recipe(MXFP8_MLP), num_layers=2)
    assert all("forward" not in vars(layer.mlp.linear_fc2) for layer in raw.decoder.layers)
    assert all("_qorerl_recipe" not in vars(layer.mlp.linear_fc2) for layer in raw.decoder.layers)


def test_rejects_incomplete_model_and_repeated_installation(megatron_stubs):
    model = Float16Module(GPTModel())
    recipe = resolve_recipe(MXFP8_MLP)
    with pytest.raises(ValueError, match="complete single-rank decoder"):
        adapter.install(model, recipe, num_layers=3)
    adapter.install(model, recipe, num_layers=2)
    with pytest.raises(ValueError, match="already installed"):
        adapter.install(model, recipe, num_layers=2)


def test_unwrap_is_restricted_to_the_supported_precision_wrapper(megatron_stubs):
    unknown_wrapper = torch.nn.Module()
    unknown_wrapper.module = GPTModel()
    unknown_wrapper.config = unknown_wrapper.module.config
    with pytest.raises(ValueError, match="optionally wrapped in Float16Module"):
        adapter.install(unknown_wrapper, resolve_recipe(MXFP8_MLP), num_layers=2)


def test_cpu_weights_are_rejected_by_production_guard(megatron_stubs, monkeypatch):
    monkeypatch.setattr(adapter, "_validate_native_weight", megatron_stubs.original_weight_validator)
    model = Float16Module(GPTModel())
    with pytest.raises(ValueError, match="canonical CUDA BF16 weights"):
        adapter.install(model, resolve_recipe(MXFP8_MLP), num_layers=2)
    assert all("forward" not in vars(layer.mlp.linear_fc2) for layer in model.module.decoder.layers)


def test_smoothquant_routes_unscaled_capture_and_maps_both_gradients(megatron_stubs):
    model = GPTModel(dtype=torch.float64)
    target = model.decoder.layers[1].mlp.linear_fc2
    reference = copy.deepcopy(target)
    name = down_proj_name(1)
    recipe = resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [1]})
    installed = adapter.install(model, recipe, num_layers=2)
    assert installed.weights == {name: target.weight}
    before = target.weight.detach().clone()
    factor = torch.linspace(0.25, 4, 64, dtype=torch.float32)
    installed.state.install_smoothing_factors({name: factor})
    captured = []
    installed.calibrator = SimpleNamespace(capture=lambda n, x: captured.append((n, x.detach().clone())))
    x = torch.randn(5, 2, 64, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gradient = torch.randn_like(x)
    y = target(x)[0]
    y_ref = reference(x_ref)[0]
    y.backward(gradient)
    y_ref.backward(gradient)

    assert captured[0][0] == name
    torch.testing.assert_close(captured[0][1], x, rtol=0, atol=0)
    effective_x, effective_w = megatron_stubs.calls[0]
    torch.testing.assert_close(effective_x, x / factor, rtol=0, atol=0)
    torch.testing.assert_close(effective_w, before * factor, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(target.weight.grad, reference.weight.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_smoothquant_expansion_and_resume_keep_base_quantized(megatron_stubs, smoothing_artifact):
    cfg = TrainConfig(
        recipe_id=MXFP8_SMOOTHQUANT,
        recipe_options={"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True},
    )
    model = GPTModel()
    model.decoder.layers.append(copy.deepcopy(model.decoder.layers[0]))
    layer = model.decoder.layers[0].mlp.linear_fc2
    outside = model.decoder.layers[2].mlp.linear_fc2
    learner = MegatronLearner.__new__(MegatronLearner)
    installation = learner.recipe_installation = adapter.install(model, cfg.recipe, num_layers=3)
    learner.recipe_state = installation.state
    learner.setup_optimizer(cfg, model, 0, "cpu")
    learner.logits = lambda ids: model(F.one_hot(ids, 64).bfloat16())
    initial_policy = learner.policy_version
    initial_checkpoint = learner.checkpoint()
    promoted, initial = down_proj_name(0), down_proj_name(1)
    before = copy.deepcopy(model.state_dict())
    trajectory = Trajectory(0, 0, (1, 3), (2,), (-1.0,), "", "stop", initial_policy)

    samples = learner._sample_calibration_inputs([trajectory], 2)
    assert set(samples) == {initial, promoted} and len(megatron_stubs.calls) == 2
    raw_x, raw_w = megatron_stubs.calls[0]
    torch.testing.assert_close(samples[promoted], raw_x.reshape(2, 64), rtol=0, atol=0)
    torch.testing.assert_close(raw_w, layer.weight, rtol=0, atol=0)
    assert installation.report()["calls"] == {promoted: 1, initial: 1}
    assert outside.forward.__func__ is TERowParallelLinear.forward
    factor = torch.linspace(0.5, 2, 64)
    promoted_policy = learner.apply_candidate(smoothing_artifact(cfg, {promoted: factor}))
    assert promoted_policy.weight_version == initial_policy.weight_version
    assert promoted_policy.recipe_id != initial_policy.recipe_id
    promoted_checkpoint = learner.checkpoint()
    x = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    for checkpoint, active in [(initial_checkpoint, False), (promoted_checkpoint, True)]:
        learner.restore(checkpoint)
        megatron_stubs.calls.clear()
        layer.zero_grad()
        layer(x)[0].float().sum().backward()
        assert layer.weight.grad is not None
        assert learner.recipe_state.is_active(promoted) == active
        assert len(megatron_stubs.calls) == 1
        effective_x, effective_w = megatron_stubs.calls[0]
        torch.testing.assert_close(effective_x, (x / factor).bfloat16() if active else x, rtol=0, atol=0)
        torch.testing.assert_close(
            effective_w, (layer.weight * factor).bfloat16() if active else layer.weight, rtol=0, atol=0
        )
        assert outside.forward.__func__ is TERowParallelLinear.forward
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_nvfp4_dispatch_and_calibration_share_the_selected_kernel(megatron_stubs, monkeypatch):
    from qorerl.recipes import te_nvfp4

    cfg = TrainConfig(
        recipe_id="smoothquant-mlp-down-v1",
        recipe_options={"precision": "nvfp4", "layers": [0, 1], "smoothquant_layers": [1]},
    )
    monkeypatch.setattr(
        adapter, "validate_native_mxfp8", lambda: pytest.fail("NVFP4 loaded the MXFP8 backend")
    )
    monkeypatch.setattr(te_nvfp4, "validate_native_nvfp4", lambda: {"backend": "te_nvfp4_stub"})
    calls = []

    def nvfp4_linear(x, weight):
        calls.append((x.detach().clone(), weight.detach().clone()))
        return F.linear(x, weight)

    monkeypatch.setattr(te_nvfp4, "native_nvfp4_linear", nvfp4_linear)
    model = GPTModel()
    installed = adapter.install(model, cfg.recipe, num_layers=2)
    assert installed.linear is nvfp4_linear  # Native calibration probes reuse the actual FPROP.
    smooth = down_proj_name(1)
    factor = torch.linspace(0.5, 2, 64)
    installed.state.install_smoothing_factors({smooth: factor})
    direct_layer, smooth_layer = [layer.mlp.linear_fc2 for layer in model.decoder.layers]
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    direct_layer(x)[0].sum().backward()
    smooth_layer(x)[0].sum().backward()
    assert not megatron_stubs.calls
    assert len(calls) == 2 and direct_layer.weight.grad is not None and smooth_layer.weight.grad is not None
    torch.testing.assert_close(calls[0][0], x, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], direct_layer.weight, rtol=0, atol=0)
    torch.testing.assert_close(calls[1][0], (x / factor).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(calls[1][1], (smooth_layer.weight * factor).bfloat16(), rtol=0, atol=0)
    assert installed.report()["fprop"] == installed.report()["wgrad"] == "nvfp4_e2m1"


@pytest.mark.parametrize(
    "recipe,options",
    [
        ("bf16-baseline-v1", {}),
        (MXFP8_MLP, {"layers": [1]}),
        (MXFP8_HADAMARD, {"layers": [1], "block_size": 32}),
    ],
)
def test_native_discovery_observes_existing_operator_and_removes_capture(megatron_stubs, recipe, options):
    cfg = TrainConfig(
        recipe_id=recipe,
        recipe_options=options,
        recipe_calibration={"discovery_only": True, "max_rows": 2, "outlier_threshold": 1},
    )
    model = Float16Module(GPTModel())
    learner = MegatronLearner.__new__(MegatronLearner)
    learner.recipe_installation = None
    if cfg.recipe.is_mxfp8:
        learner.recipe_installation = adapter.install(model, cfg.recipe, num_layers=2)
        learner.recipe_state = learner.recipe_installation.state
    learner.setup_optimizer(cfg, model, 0, "cpu")
    learner.logits = lambda ids: model(F.one_hot(ids, 64).bfloat16())
    ids = torch.tensor([[1, 3, 5]])
    expected = learner.logits(ids)
    policy = learner.policy_version
    trajectory = Trajectory(0, 0, (1, 3, 5), (2,), (-1.0,), "", "stop", policy)
    batch = TrainingBatch([trajectory], torch.zeros(1), torch.zeros(1), policy)
    before = copy.deepcopy(model.state_dict())
    fc2s = [layer.mlp.linear_fc2 for layer in model.module.decoder.layers]
    forwards = [module.forward for module in fc2s]
    candidate, evidence = learner.prepare_candidate(batch)
    assert candidate is None and learner.policy_version == policy
    assert {e["target"] for e in evidence} == (
        {down_proj_name(1)} if cfg.recipe.is_mxfp8 else {down_proj_name(0), down_proj_name(1)}
    )
    assert all(e["status"] == "discovered" and e["fit_rows"] == 2 for e in evidence)
    torch.testing.assert_close(learner.logits(ids), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def fail(_):
        if not cfg.recipe.is_mxfp8:
            assert all(module._forward_pre_hooks for module in fc2s)
        raise RuntimeError("injected replay error")

    learner.logits = fail
    with pytest.raises(RuntimeError, match="injected replay error"):
        learner.prepare_candidate(batch)
    assert [module.forward for module in fc2s] == forwards
    assert all(not module._forward_pre_hooks for module in fc2s)
    if learner.recipe_installation is None:
        assert not hasattr(learner, "recipe_state")
    else:
        assert learner.recipe_installation.calibrator is None
