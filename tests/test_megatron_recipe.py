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

from qorerl.recipes import megatron_mxfp8 as adapter
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, resolve_recipe
from qorerl.recipes.state import down_proj_name


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
