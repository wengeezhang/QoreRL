"""Selected Qwen3 projections using vLLM 0.32.0's native MXFP8 kernel.

Install after the BF16 model loads and before vLLM profiles memory. Canonical
weights and their HF loaders remain unchanged. The adapter holds disposable
MXFP8 carriers outside the model's parameters and rebuilds them after each refit.
This intentionally retains BF16 weights, so it is not a weight-memory saving path.

The reference CUDA stack is vLLM 0.32.0 / FlashInfer 0.7.0.post1 / PyTorch 2.13.0.
The explicit FlashInfer CUTLASS GEMM and CuTe-DSL quantizer have no BF16 fallback.
Matching MXFP8 formats does not establish numerical agreement with TE training.
"""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from qorerl.recipes.state import RecipeState


@dataclass(frozen=True)
class _Backend:
    kernel: Any
    config: Any
    quantize: Any
    linear: Any
    unquantized: Any
    metadata: dict


def _load_backend():
    try:
        vllm_version = version("vllm")
        flashinfer_version = version("flashinfer-python")
    except PackageNotFoundError as exc:
        raise RuntimeError("Native rollout MXFP8 requires vLLM 0.32.0 and FlashInfer with CuTe-DSL") from exc
    if vllm_version.split("+")[0] != "0.32.0":
        raise RuntimeError(f"The native rollout MXFP8 adapter targets vLLM 0.32.0; found {vllm_version}")
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
    return _Backend(
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
        raise ValueError(f"Native rollout MXFP8 requires a CUDA BF16 matrix: {name}")
    if not torch.isfinite(weight).all().item():
        raise ValueError(f"Nonfinite canonical rollout weight: {name}")


class _CachedMXFP8Method:
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
        if not self.installation.ready or self.carrier is None:
            raise RuntimeError("MXFP8 rollout cache is invalid; a complete weight refit is required")
        if layer is not self.layer:
            raise RuntimeError("MXFP8 runtime method was attached to the wrong layer")
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.device != layer.weight.device:
            raise ValueError("Native rollout MXFP8 requires BF16 activations on the weight device")
        effective_input = self.installation.state.transform_input(x, self.name).contiguous()
        with torch.autograd.profiler.record_function("qorerl.vllm.mxfp8.fprop"):
            output = self.kernel.apply_weights(self.carrier, effective_input, bias)
        self.apply_calls += 1
        return output


class VLLMMXFP8Installation:
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
            if self.state.spec.uses_smoothquant and not torch.isfinite(transformed).all().item():
                raise ValueError(f"Nonfinite scaled SmoothQuant rollout weight: {method.name}")
            values, scales = self.backend.quantize(transformed, is_sf_swizzled_layout=False)
            if values.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
                raise RuntimeError("Unexpected native MXFP8 payload or E8M0 scale dtype")
            carrier = torch.nn.Module()
            carrier.register_parameter("weight", torch.nn.Parameter(values, requires_grad=False))
            carrier.register_parameter("weight_scale", torch.nn.Parameter(scales, requires_grad=False))
            method.kernel.process_weights_after_loading(carrier)
            carriers.append(carrier)
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
                    "cache_bytes": sum(p.numel() * p.element_size() for p in carrier.parameters())
                    if carrier is not None
                    else 0,
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
    """Install native MXFP8 on selected single-rank dense Qwen3 MLP outputs.

    This requires an already loaded, unquantized BF16 model. Call from a custom
    vLLM worker's ``load_model`` before KV-cache sizing, with eager mode enabled.
    ``installation.state`` supplies the shared snapshot/checkpoint contract.
    """
    if not recipe.is_mxfp8:
        raise ValueError("The vLLM MXFP8 adapter requires an MXFP8 recipe")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("The native rollout MXFP8 adapter supports dense Qwen3 only")
    engine_config = getattr(model, "vllm_config", None)
    if not getattr(getattr(engine_config, "model_config", None), "enforce_eager", False):
        raise ValueError("The native rollout MXFP8 cache adapter requires vLLM enforce_eager=True")
    selected = recipe.select_layers(config.num_hidden_layers)
    backend = _load_backend()
    state = RecipeState(recipe)
    installation = VLLMMXFP8Installation(state, backend, [])
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
        kernel_config = backend.config(weight_shape=tuple(layer.weight.shape))
        supported, reason = backend.kernel.can_implement(kernel_config)
        if not supported:
            raise ValueError(f"Native MXFP8 cannot implement {name}: {reason}")
        installation.methods.append(
            _CachedMXFP8Method(installation, name, runtime_name, layer, backend.kernel(kernel_config))
        )
    installation.refresh()
    # Attach only after every target has been validated and its cache built.
    for method in installation.methods:
        method.layer.quant_method = method
        method.layer.requires_unquantized_input = True
    return installation
