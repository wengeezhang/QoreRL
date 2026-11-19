"""Scoped low-precision dense GEMMs for a vLLM 0.32.0 eager worker.

Install after the BF16 model loads and before vLLM profiles memory. Canonical
weights and their HF loaders remain unchanged. The adapter holds disposable
low-precision carriers outside the model's parameters and rebuilds them after each refit.
This intentionally retains BF16 weights, so it is not a weight-memory saving path.

The reference CUDA stack is vLLM 0.32.0 / FlashInfer 0.7.0.post1 / PyTorch 2.13.0.
The explicit FlashInfer CUTLASS GEMM and CuTe-DSL quantizer have no BF16 fallback.
NVFP4 uses the learner's Transformer Engine packer and native forward GEMM.
"""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from qorerl.recipes.state import RecipeState


@dataclass(frozen=True)
class _MXFP8Backend:
    kernel: Any
    config: Any
    quantize: Any
    linear: Any
    unquantized: Any
    metadata: dict

    def make_kernel(self, shape, name):
        config = self.config(weight_shape=shape)
        supported, reason = self.kernel.can_implement(config)
        if not supported:
            raise ValueError(f"Native MXFP8 cannot implement {name}: {reason}")
        return self.kernel(config)

    def pack(self, weight, kernel):
        values, scales = self.quantize(weight, is_sf_swizzled_layout=False)
        if values.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
            raise RuntimeError("Unexpected native MXFP8 payload or E8M0 scale dtype")
        carrier = torch.nn.Module()
        carrier.register_parameter("weight", torch.nn.Parameter(values, requires_grad=False))
        carrier.register_parameter("weight_scale", torch.nn.Parameter(scales, requires_grad=False))
        kernel.process_weights_after_loading(carrier)
        return carrier

    @staticmethod
    def packed_nbytes(carrier):
        return sum(p.numel() * p.element_size() for p in carrier.parameters())


class _TransformerEngineNVFP4Kernel:
    @staticmethod
    def apply_weights(carrier, x, bias):
        from qorerl.recipes.te_nvfp4 import nvfp4_linear_from_packed

        return nvfp4_linear_from_packed(x, carrier, bias)


@dataclass(frozen=True)
class _NVFP4Backend:
    linear: Any
    unquantized: Any
    metadata: dict

    @staticmethod
    def make_kernel(shape, name):
        # RecipeState validates both axes before kernels are installed.
        return _TransformerEngineNVFP4Kernel()

    @staticmethod
    def pack(weight, kernel):
        from qorerl.recipes.te_nvfp4 import pack_nvfp4_weight

        return pack_nvfp4_weight(weight)

    @staticmethod
    def packed_nbytes(carrier):
        from qorerl.recipes.te_nvfp4 import nvfp4_packed_nbytes

        return nvfp4_packed_nbytes(carrier)


def _vllm_version():
    try:
        result = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("The native rollout adapter requires vLLM 0.32.0") from exc
    if result.split("+")[0] != "0.32.0":
        raise RuntimeError(f"The native rollout adapter targets vLLM 0.32.0; found {result}")
    return result


def _load_nvfp4_backend():
    from qorerl.recipes.te_nvfp4 import validate_native_nvfp4

    vllm_version = _vllm_version()
    from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod

    metadata = {
        **validate_native_nvfp4(),
        "backend": "vllm_transformer_engine_native_nvfp4",
        "vllm_version": vllm_version,
        "train_rollout_numerical_agreement": "unvalidated",
    }
    return _NVFP4Backend(RowParallelLinear, UnquantizedLinearMethod, metadata)


def _load_mxfp8_backend():
    vllm_version = _vllm_version()
    try:
        flashinfer_version = version("flashinfer-python")
    except PackageNotFoundError as exc:
        raise RuntimeError("Native rollout MXFP8 requires FlashInfer with CuTe-DSL") from exc
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Native rollout MXFP8 requires NVIDIA CUDA; no CPU/ROCm fallback")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(
            f"The native rollout MXFP8 adapter currently supports SM100-family GPUs; got {capability}"
        )
    try:
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
        from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.utils.flashinfer import has_flashinfer_cutedsl
    except (ImportError, OSError) as exc:
        raise RuntimeError("The vLLM native MXFP8/FlashInfer CUDA extension could not be loaded") from exc
    supported, reason = FlashInferCutlassMxfp8LinearKernel.is_supported()
    if not supported or not has_flashinfer_cutedsl():
        raise RuntimeError(f"Native rollout MXFP8 requires FlashInfer CUTLASS and CuTe-DSL: {reason}")
    return _MXFP8Backend(
        FlashInferCutlassMxfp8LinearKernel,
        Mxfp8LinearLayerConfig,
        mxfp8_e4m3_quantize,
        RowParallelLinear,
        UnquantizedLinearMethod,
        {
            "backend": "vllm_flashinfer_cutlass_native_mxfp8",
            "vllm_version": vllm_version,
            "flashinfer_version": flashinfer_version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "quantizer": "vllm.mxfp8_e4m3_quantize/flashinfer_cute_dsl",
            "train_rollout_numerical_agreement": "unvalidated",
        },
    )


def _validate_weight(weight, name):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
        raise ValueError(f"Native rollout requires a CUDA BF16 matrix: {name}")
    if not torch.isfinite(weight).all().item():
        raise ValueError(f"Nonfinite canonical rollout weight: {name}")


class _CachedLinearMethod:
    """Per-instance vLLM ``quant_method.apply`` adapter, without global patches."""

    uses_meta_device = False

    def __init__(self, installation, name, runtime_name, layer, kernel):
        self.installation = installation
        self.name = name
        self.runtime_name = runtime_name
        self.layer = layer
        self.kernel = kernel
        self.carrier = None
        self.apply_calls = 0

    def apply(self, layer, x, bias=None):
        if not self.installation.ready:
            raise RuntimeError("Low-precision rollout cache is invalid; a complete weight refit is required")
        if layer is not self.layer:
            raise RuntimeError("Low-precision runtime method was attached to the wrong layer")
        if self.carrier is None:
            raise RuntimeError(f"Low-precision rollout target is missing its carrier: {self.name}")
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.device != layer.weight.device:
            raise ValueError("Native rollout requires BF16 activations on the weight device")
        effective_input = self.installation.state.transform_input(x, self.name).contiguous()
        with torch.autograd.profiler.record_function(
            f"qorerl.vllm.{self.installation.state.spec.precision}.fprop"
        ):
            output = self.kernel.apply_weights(self.carrier, effective_input, bias)
        self.apply_calls += 1
        return output


class VLLMRecipeInstallation:
    """Runtime cache owner. Call ``refresh`` after every canonical BF16 refit.

    A failed refresh leaves execution disabled. The worker must synchronize CUDA
    before acknowledging publication. Refits need space for old and new quantized
    carriers temporarily; the canonical BF16 weights are always retained.
    """

    def __init__(self, state, backend, methods):
        self.state = state
        self.backend = backend
        self.methods = methods
        self._ready = False
        self._cache_scale_digest = None
        self.refresh_count = 0

    @property
    def ready(self):
        return self._ready and self._cache_scale_digest == self.state.scale_digest

    def invalidate(self):
        self._ready = False

    @torch.no_grad()
    def refresh(self):
        self.invalidate()
        carriers = []
        for method in self.methods:
            weight = method.layer.weight
            _validate_weight(weight, method.name)
            if tuple(weight.shape) != self.state.targets[method.name]:
                raise ValueError(f"Canonical rollout weight shape changed: {method.name}")
            transformed = self.state.transform_weight(weight.detach(), method.name).contiguous()
            if self.state.spec.uses_smoothquant and self.state.is_active(method.name):
                if not torch.isfinite(transformed).all().item():
                    raise ValueError(f"Nonfinite scaled SmoothQuant rollout weight: {method.name}")
            carriers.append(self.backend.pack(transformed, method.kernel))
        for method, carrier in zip(self.methods, carriers, strict=True):
            method.carrier = carrier
            method.apply_calls = 0
        self.refresh_count += 1
        self._cache_scale_digest = self.state.scale_digest
        self._ready = True

    def report(self):
        entries = []
        for method in self.methods:
            carrier = method.carrier
            entries.append(
                {
                    "canonical_weight": method.name,
                    "runtime_module": method.runtime_name,
                    "shape": list(self.state.targets[method.name]),
                    "kernel": type(method.kernel).__name__,
                    "apply_calls": method.apply_calls,
                    "cache_bytes": self.backend.packed_nbytes(carrier) if carrier is not None else 0,
                }
            )
        return {
            **self.state.report(),
            **self.backend.metadata,
            "runtime_targets": entries,
            "ready": self.ready,
            "refresh_count": self.refresh_count,
            "execution_counter_scope": "since_latest_refresh",
            "canonical_weights_retained": True,
            "refit_cache_strategy": "build_new_carriers_then_publish",
            "all_targets_executed": bool(entries) and all(entry["apply_calls"] > 0 for entry in entries),
        }


def install(model, recipe):
    """Install the configured low-precision format on selected single-rank dense Qwen3 MLP outputs.

    This requires an already loaded, unquantized BF16 model. Call from a custom
    vLLM worker's ``load_model`` before KV-cache sizing, with eager mode enabled.
    ``installation.state`` supplies the shared snapshot/checkpoint contract.
    """
    if not recipe.is_low_precision:
        raise ValueError("The vLLM recipe adapter requires a low-precision recipe")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("The native rollout recipe adapter supports dense Qwen3 only")
    engine_config = getattr(model, "vllm_config", None)
    if not getattr(getattr(engine_config, "model_config", None), "enforce_eager", False):
        raise ValueError("The native rollout cache adapter requires vLLM enforce_eager=True")
    num_layers = config.num_hidden_layers
    selected = recipe.select_layers(num_layers)
    backend = _load_nvfp4_backend() if recipe.precision == "nvfp4" else _load_mxfp8_backend()
    state = RecipeState(recipe)
    installation = VLLMRecipeInstallation(state, backend, [])
    modules = dict(model.named_modules())
    for index in selected:
        runtime_name = f"model.layers.{index}.mlp.down_proj"
        layer = modules.get(runtime_name)
        if not isinstance(layer, backend.linear) or getattr(layer, "tp_size", None) != 1:
            raise ValueError(f"Expected a single-rank vLLM RowParallelLinear: {runtime_name}")
        if not isinstance(layer.quant_method, backend.unquantized):
            raise ValueError(f"Expected a BF16 target; recipe installation is one-time: {runtime_name}")
        _validate_weight(layer.weight, runtime_name)
        name = state.add_target(index, layer.weight)
        kernel = backend.make_kernel(tuple(layer.weight.shape), name)
        installation.methods.append(_CachedLinearMethod(installation, name, runtime_name, layer, kernel))
    installation.refresh()
    # Attach only after every target has been validated and its cache built.
    for method in installation.methods:
        method.layer.quant_method = method
        method.layer.requires_unquantized_input = True
    return installation
