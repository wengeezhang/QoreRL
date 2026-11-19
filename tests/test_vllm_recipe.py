"""CPU contract tests with stub kernels; these do not validate native GPU MXFP8."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from qorerl.engines.vllm import VLLMRollout, WeightLoaderExtension
from qorerl.recipes import vllm_mxfp8 as adapter
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, MXFP8_SMOOTHQUANT, resolve_recipe
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion, WeightSnapshot


class _Unquantized:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16))
        self.quant_method = _Unquantized()
        self.tp_size = 1


class _Kernel:
    def __init__(self, config):
        self.config = config
        self.last_input = None

    @staticmethod
    def can_implement(config):
        return True, None

    def process_weights_after_loading(self, carrier):
        carrier.weight_scale = torch.nn.Parameter(carrier.weight_scale.flatten(), requires_grad=False)

    def apply_weights(self, carrier, x, bias):
        self.last_input = x.clone()
        # Deliberate stub: verifies routing only, not native numerical accuracy.
        return torch.nn.functional.linear(x, carrier.weight.to(x.dtype), bias)


@pytest.fixture
def stub_runtime(monkeypatch):
    quantized_inputs = []

    def quantize(value, *, is_sf_swizzled_layout):
        assert not is_sf_swizzled_layout
        quantized_inputs.append(value.clone())
        scales = torch.ones(value.shape[0], value.shape[1] // 32, dtype=torch.uint8)
        return value.to(torch.float8_e4m3fn), scales

    backend = adapter._MXFP8Backend(_Kernel, SimpleNamespace, quantize, _Linear, _Unquantized, {})
    monkeypatch.setattr(adapter, "_load_mxfp8_backend", lambda: backend)
    monkeypatch.setattr(adapter, "_validate_weight", lambda weight, name: None)
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=2)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = _Linear()
        model.model.layers.append(layer)
    return model, quantized_inputs


def test_selected_cache_keeps_canonical_parameters_and_rotates_both_operands(stub_runtime):
    model, quantized_inputs = stub_runtime
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    recipe = resolve_recipe(MXFP8_HADAMARD, {"layers": [1], "block_size": 32, "seed": 7})
    installed = adapter.install(model, recipe)
    name = "model.layers.1.mlp.down_proj.weight"
    assert set(dict(model.named_parameters())) == set(original)
    for parameter_name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[parameter_name], rtol=0, atol=0)
    assert isinstance(model.model.layers[0].mlp.down_proj.quant_method, _Unquantized)
    torch.testing.assert_close(quantized_inputs[0], installed.state.rotate(original[name], name))
    target = model.model.layers[1].mlp.down_proj
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    output = target.quant_method.apply(target, x)
    assert output.shape == (3, 128)
    torch.testing.assert_close(target.quant_method.kernel.last_input, installed.state.rotate(x, name))
    report = installed.report()
    assert report["all_targets_executed"]
    assert report["runtime_targets"][0]["canonical_weight"] == name
    assert report["runtime_targets"][0]["runtime_module"] == name.removesuffix(".weight")
    installed.refresh()
    assert not installed.report()["all_targets_executed"]


def test_refit_uses_new_canonical_weights_without_composing_rotation(stub_runtime):
    model, quantized_inputs = stub_runtime
    installed = adapter.install(model, resolve_recipe(MXFP8_HADAMARD, {"layers": [1]}))
    target = model.model.layers[1].mlp.down_proj
    name = "model.layers.1.mlp.down_proj.weight"
    new_weight = torch.randn_like(target.weight)
    with torch.no_grad():
        target.weight.copy_(new_weight)
    installed.refresh()
    installed.refresh()
    expected = installed.state.rotate(new_weight, name)
    torch.testing.assert_close(quantized_inputs[-1], expected, rtol=0, atol=0)
    torch.testing.assert_close(quantized_inputs[-2], expected, rtol=0, atol=0)
    torch.testing.assert_close(target.weight, new_weight, rtol=0, atol=0)
    assert installed.refresh_count == 3


def test_failed_refresh_blocks_execution_and_does_not_publish_partial_caches(stub_runtime):
    model, _ = stub_runtime
    installed = adapter.install(model, resolve_recipe(MXFP8_MLP))
    old_carriers = [method.carrier for method in installed.methods]
    target = model.model.layers[1].mlp.down_proj
    target.weight = torch.nn.Parameter(torch.zeros(128, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="shape changed"):
        installed.refresh()
    assert not installed.ready
    assert [method.carrier for method in installed.methods] == old_carriers
    first = model.model.layers[0].mlp.down_proj
    with pytest.raises(RuntimeError, match="cache is invalid"):
        first.quant_method.apply(first, torch.zeros(1, 128, dtype=torch.bfloat16))


def test_rejects_partial_install_and_repeated_install(stub_runtime):
    model, _ = stub_runtime
    model.model.layers[1].mlp.down_proj.tp_size = 2
    with pytest.raises(ValueError, match="single-rank"):
        adapter.install(model, resolve_recipe(MXFP8_MLP))
    assert isinstance(model.model.layers[0].mlp.down_proj.quant_method, _Unquantized)
    model.model.layers[1].mlp.down_proj.tp_size = 1
    adapter.install(model, resolve_recipe(MXFP8_MLP))
    with pytest.raises(ValueError, match="one-time"):
        adapter.install(model, resolve_recipe(MXFP8_MLP))


def test_native_backend_rejects_unsupported_version_and_cpu(monkeypatch):
    monkeypatch.setattr(adapter, "version", lambda name: "0.29.0")
    with pytest.raises(RuntimeError, match="targets vLLM 0.32.0"):
        adapter._load_mxfp8_backend()
    monkeypatch.setattr(adapter, "version", lambda name: "0.32.0" if name == "vllm" else "0.7.0.post1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU/ROCm fallback"):
        adapter._load_mxfp8_backend()


@pytest.mark.parametrize("stage", ["load", "empty", "coverage", "refresh", "sync"])
def test_worker_refit_failure_invalidates_execution(stub_runtime, monkeypatch, stage):
    model, _ = stub_runtime
    installed = adapter.install(model, resolve_recipe(MXFP8_MLP))
    extension = WeightLoaderExtension()
    extension.model_runner = SimpleNamespace(model=model)
    extension.qorerl_recipe_installation = installed
    tensors = {name: value.detach().clone() for name, value in model.named_parameters()}
    expected = set(tensors)
    failure = RuntimeError(f"injected {stage} failure")

    def load_weights(weights):
        assert not installed.ready  # Invalidated before any canonical write.
        if stage == "load":
            raise failure
        if stage == "empty":
            return set()
        if stage == "coverage":
            return {sorted(expected)[0]}
        return expected

    original_refresh = installed.refresh

    def refresh_failure():
        # Even a refresh failing after publishing must leave execution disabled.
        original_refresh()
        raise failure

    def synchronize():
        assert installed.ready
        if stage == "sync":
            raise failure

    model.load_weights = load_weights
    if stage == "refresh":
        monkeypatch.setattr(installed, "refresh", refresh_failure)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with pytest.raises(RuntimeError) as caught:
        extension.qorerl_load_weights(tensors, installed.state.to_dict())
    if stage in {"load", "refresh", "sync"}:
        assert caught.value is failure  # Preserve the failure; never return an acknowledgement.
    assert not installed.ready
    target = model.model.layers[0].mlp.down_proj
    with pytest.raises(RuntimeError, match="cache is invalid"):
        target.quant_method.apply(target, torch.zeros(1, 128, dtype=torch.bfloat16))


def test_worker_refit_acknowledges_only_after_cuda_sync(stub_runtime, monkeypatch):
    model, _ = stub_runtime
    installed = adapter.install(model, resolve_recipe(MXFP8_MLP))
    extension = WeightLoaderExtension()
    extension.model_runner = SimpleNamespace(model=model)
    extension.qorerl_recipe_installation = installed
    tensors = {name: value.detach().clone() for name, value in model.named_parameters()}
    expected = set(tensors)
    events = []

    def load_weights(weights):
        assert not installed.ready
        events.append("load")
        return expected

    def synchronize():
        assert installed.ready and installed.refresh_count == 2
        events.append("sync")

    model.load_weights = load_weights
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert extension.qorerl_load_weights(tensors, installed.state.to_dict()) == len(expected)
    assert events == ["load", "sync"]
    assert installed.ready


@pytest.mark.parametrize("malformation", ["missing", "shape", "dtype"])
def test_worker_rejects_malformed_recipe_weights_before_mutation(stub_runtime, malformation):
    model, _ = stub_runtime
    installed = adapter.install(model, resolve_recipe(MXFP8_HADAMARD))
    extension = WeightLoaderExtension()
    extension.model_runner = SimpleNamespace(model=model)
    extension.qorerl_recipe_installation = installed
    tensors = {name: value.detach().clone() for name, value in model.named_parameters()}
    name = "model.layers.0.mlp.down_proj.weight"
    if malformation == "missing":
        tensors.pop(name)
    elif malformation == "shape":
        tensors[name] = tensors[name][:32]
    else:
        tensors[name] = tensors[name].float()

    def unexpected_load(weights):
        pytest.fail("Malformed canonical weights reached the vLLM loader")

    model.load_weights = unexpected_load
    with pytest.raises(ValueError, match="[Cc]anonical recipe weight"):
        extension.qorerl_load_weights(tensors, installed.state.to_dict())
    # Nothing was mutated; the previously installed weights remain usable.
    assert installed.ready and installed.refresh_count == 1


def test_smoothquant_worker_refit_verifies_identity_and_rebuilds_effective_weights(stub_runtime, monkeypatch):
    model, quantized_inputs = stub_runtime
    recipe = resolve_recipe(MXFP8_SMOOTHQUANT, {"layers": [1]})
    installed = adapter.install(model, recipe)
    extension = WeightLoaderExtension()
    extension.model_runner = SimpleNamespace(model=model)
    extension.qorerl_recipe_installation = installed
    target = model.model.layers[1].mlp.down_proj
    name = installed.methods[0].name
    tensors = {n: p.detach().clone() for n, p in model.named_parameters()}
    tensors[name] *= 0.5
    unit_id = installed.state.active_identity
    new_scale = torch.linspace(0.25, 2, 128)
    source = adapter.RecipeState(recipe)
    source.add_target(1, target.weight)
    source.install_scales({name: new_scale})
    assert unit_id != source.active_identity
    loads = []

    def load_weights(_):
        assert not installed.ready
        loads.append(name)
        with torch.no_grad():
            target.weight.copy_(tensors[name])
        return set(tensors)

    model.load_weights = load_weights
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(ValueError, match="identity"):
        extension.qorerl_load_weights(tensors, source.to_dict(), unit_id)
    assert installed.ready and not loads
    payload = source.to_dict()
    payload["scales"][name][0] *= 2
    with pytest.raises(ValueError, match="digest"):
        extension.qorerl_load_weights(tensors, payload, source.active_identity)
    assert installed.ready and not loads

    # Repeating a worker refit always starts from canonical weights.
    for _ in range(2):
        assert extension.qorerl_load_weights(tensors, source.to_dict(), source.active_identity) == (
            source.active_identity
        )
        torch.testing.assert_close(quantized_inputs[-1], (tensors[name] * new_scale).bfloat16())
        torch.testing.assert_close(target.weight, tensors[name], rtol=0, atol=0)
        assert installed.report()["identity"] == source.active_identity
        x = torch.randn(3, 128, dtype=torch.bfloat16)
        target.quant_method.apply(target, x)
        torch.testing.assert_close(target.quant_method.kernel.last_input, (x / new_scale).bfloat16())
    # The active activation scales cannot be used with a carrier from another transform.
    installed.state.install_scales({name: torch.ones(128)})
    assert not installed.ready
    with pytest.raises(RuntimeError, match="cache is invalid"):
        target.quant_method.apply(target, x)


def test_worker_preserves_quantized_base_across_smoothquant_expansion(stub_runtime, monkeypatch):
    model, quantized = stub_runtime
    model.model.layers.append(copy.deepcopy(model.model.layers[0]))
    model.config.num_hidden_layers = len(model.model.layers)
    recipe = resolve_recipe(
        MXFP8_SMOOTHQUANT,
        {"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True},
    )
    target = model.model.layers[0].mlp.down_proj
    outside = model.model.layers[2].mlp.down_proj
    bf16_method = outside.quant_method
    installed = adapter.install(model, recipe)
    candidate_method, initial_method = installed.methods
    promoted, initial = candidate_method.name, initial_method.name
    extension = WeightLoaderExtension()
    extension.model_runner = SimpleNamespace(model=model)
    extension.qorerl_recipe_installation = installed
    tensors = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.load_weights = lambda _: set(tensors)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    source = adapter.RecipeState(recipe)
    source.add_target(0, target.weight, active=False)
    source.add_target(1, initial_method.layer.weight)
    x = torch.randn(2, 128, dtype=torch.bfloat16)

    assert len(quantized) == 2 and candidate_method.carrier is not None
    torch.testing.assert_close(quantized[0], target.weight, rtol=0, atol=0)
    assert outside.quant_method is bf16_method
    assert set(installed.state.scales) == {initial}
    assert extension.qorerl_load_weights(tensors, source.to_dict(), source.active_identity) == (
        source.active_identity
    )
    candidate_method.apply(target, x)
    torch.testing.assert_close(candidate_method.kernel.last_input, x, rtol=0, atol=0)
    initial_method.apply(initial_method.layer, x)
    report = extension.qorerl_execution_report()
    assert report["all_targets_executed"] and set(report["targets"]) == {initial, promoted}
    assert [entry["canonical_weight"] for entry in report["runtime_targets"]] == [promoted, initial]

    new_scale = torch.linspace(0.25, 2, 128)
    source.install_scales({**source.scales, promoted: new_scale})
    assert extension.qorerl_load_weights(tensors, source.to_dict(), source.active_identity) == (
        source.active_identity
    )
    assert installed.state.active_identity == source.active_identity and len(quantized) == 6
    assert outside.quant_method is bf16_method
    torch.testing.assert_close(quantized[-2], (target.weight * new_scale).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(target.weight, tensors[promoted], rtol=0, atol=0)
    assert not installed.report()["all_targets_executed"]
    candidate_method.apply(target, x)
    initial_method.apply(initial_method.layer, x)
    torch.testing.assert_close(candidate_method.kernel.last_input, (x / new_scale).bfloat16(), rtol=0, atol=0)
    assert [entry["apply_calls"] for entry in installed.report()["runtime_targets"]] == [1, 1]


def test_nvfp4_refit_packs_weights_once_and_serves_nvfp4_without_flashinfer(stub_runtime, monkeypatch):
    from qorerl.recipes import te_nvfp4

    model, mxfp8_calls = stub_runtime
    monkeypatch.setattr(adapter, "_load_mxfp8_backend", lambda: pytest.fail("NVFP4 selected MXFP8"))
    monkeypatch.setattr(adapter, "version", {"vllm": "0.32.0"}.__getitem__)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.linear",
        SimpleNamespace(RowParallelLinear=_Linear, UnquantizedLinearMethod=_Unquantized),
    )
    monkeypatch.setattr(te_nvfp4, "validate_native_nvfp4", lambda: {"backend": "te_nvfp4_stub"})
    packed_weights, consumed = [], []

    def pack(weight):
        packed = SimpleNamespace(weight=weight.detach().clone())
        packed_weights.append(packed)
        return packed

    def linear(x, packed, bias):
        consumed.append((x.detach().clone(), packed))
        output = torch.nn.functional.linear(x, packed.weight)
        return output if bias is None else output + bias

    monkeypatch.setattr(te_nvfp4, "pack_nvfp4_weight", pack)
    monkeypatch.setattr(te_nvfp4, "nvfp4_linear_from_packed", linear)
    monkeypatch.setattr(te_nvfp4, "nvfp4_packed_nbytes", lambda _: 8224)
    recipe = resolve_recipe(
        "smoothquant-mlp-down-v1",
        {"precision": "nvfp4", "layers": [1], "smoothquant_layers": [], "allow_layer_expansion": True},
    )
    installed = adapter.install(model, recipe)
    (method,) = installed.methods
    target = method.layer
    outside = model.model.layers[0].mlp.down_proj
    assert isinstance(outside.quant_method, _Unquantized)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)
    for _ in range(2):
        actual = method.apply(target, x, bias)
        torch.testing.assert_close(actual, torch.nn.functional.linear(x, target.weight) + bias)
    assert len(packed_weights) == 1 and [c[1] for c in consumed] == [method.carrier, method.carrier]
    torch.testing.assert_close(packed_weights[0].weight, target.weight)
    torch.testing.assert_close(consumed[0][0], x)

    scale = torch.linspace(0.5, 2, 128)
    installed.state.install_scales({method.name: scale})
    assert not installed.ready
    installed.refresh()
    method.apply(target, x)
    assert len(packed_weights) == 2 and consumed[-1][1] is packed_weights[1]
    torch.testing.assert_close(packed_weights[1].weight, (target.weight * scale).bfloat16())
    torch.testing.assert_close(consumed[-1][0], (x / scale).bfloat16())
    report = installed.report()
    assert report["backend"] == "vllm_transformer_engine_native_nvfp4"
    assert report["runtime_targets"][0]["cache_bytes"] == 8224 and report["all_targets_executed"]
    assert not mxfp8_calls


def test_smoothquant_rollout_requires_worker_ack_of_exact_scale_identity(stub_runtime):
    model, _ = stub_runtime
    cfg = TrainConfig(recipe_id=MXFP8_SMOOTHQUANT)
    installed = adapter.install(model, cfg.recipe)
    rollout = VLLMRollout.__new__(VLLMRollout)
    rollout.cfg = cfg
    rollout.policy_version = PolicyVersion(0, installed.state.active_identity)
    scales = installed.state.scales
    next(iter(scales.values()))[0] *= 2
    installed.state.install_scales(scales)
    snapshot = WeightSnapshot(
        PolicyVersion(0, installed.state.active_identity),
        cfg.model,
        {name: tensor.detach().clone() for name, tensor in model.named_parameters()},
        installed.state.to_dict(),
    )
    rollout.llm = SimpleNamespace(collective_rpc=lambda *args, **kwargs: [cfg.policy_recipe_id])
    with pytest.raises(RuntimeError, match="acknowledge"):
        rollout.install_weights(snapshot)
    assert rollout.policy_version is None
    rollout.llm = SimpleNamespace(collective_rpc=lambda *args, **kwargs: [installed.state.active_identity])
    assert rollout.install_weights(snapshot) == snapshot.policy_version
