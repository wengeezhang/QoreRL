"""Native TE 2.18 block-scaled linear operations for Megatron and rollout."""

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import Any

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


@dataclass(frozen=True)
class _Backend:
    version: str
    quantizer: Any
    operand_dtype: Any
    gemm: Any
    available: Any
    precision: str = "mxfp8"

    def pack(self, tensor, *, weight=False, columnwise=True):
        options = {}
        if self.precision == "nvfp4":
            # Deterministic cast, 16x16 weight blocks and 1x16 activation blocks.
            # A recipe-level rotation, when selected, is applied before this call.
            options = {"with_2d_quantization": weight, "with_random_sign_mask": False}
        quantizer = self.quantizer(self.operand_dtype, rowwise=True, columnwise=columnwise, **options)
        return quantizer(tensor.contiguous())


@lru_cache(maxsize=2)
def _load_backend(precision="mxfp8"):
    """Load CUDA extensions only when a native recipe is actually requested."""
    if precision not in {"mxfp8", "nvfp4"}:
        raise ValueError(f"Unsupported native linear precision: {precision}")
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Native {precision.upper()} requires Transformer Engine 2.18.x with its CUDA extension"
        ) from exc
    release = re.match(r"^(\d+)\.(\d+)(?:\.|$)", te_version)
    if release is None or tuple(map(int, release.groups())) != (2, 18):
        raise RuntimeError(
            f"Native {precision.upper()} adapter targets Transformer Engine 2.18.x; found {te_version}. "
            "Other TE versions need explicit API and native-kernel validation."
        )
    try:
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.cpp_extensions import general_gemm

        if precision == "nvfp4":
            from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer as quantizer
        else:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer as quantizer
    except (ImportError, OSError) as exc:
        raise RuntimeError("Transformer Engine's PyTorch/CUDA extension could not be loaded") from exc

    parameters = tuple(signature(general_gemm).parameters)
    if parameters[:3] != ("A", "B", "out_dtype") or "workspace" in parameters:
        raise RuntimeError("Unsupported Transformer Engine general_gemm ABI; expected the TE 2.18 API")
    if precision == "nvfp4":
        return _Backend(
            te_version, quantizer, tex.DType.kFloat4E2M1, general_gemm, te.is_nvfp4_available, precision
        )
    return _Backend(te_version, quantizer, tex.DType.kFloat8E4M3, general_gemm, te.is_mxfp8_available)


def _cuda_device(device, precision="mxfp8"):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Native {precision.upper()} requires a supported NVIDIA CUDA GPU; no CPU fallback is available"
        )
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise RuntimeError(f"Native {precision.upper()} requires a CUDA device, got {device}")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


@lru_cache(maxsize=None)
def _validate_device(index, precision="mxfp8"):
    backend = _load_backend(precision)
    with torch.cuda.device(index):
        capability = torch.cuda.get_device_capability(index)
        # TE 2.18's all-layout MXFP8 support excludes SM12, even though those
        # devices are also marketed as Blackwell. Do not use a name-only check.
        if precision == "mxfp8" and not (10 <= capability[0] < 12):
            raise RuntimeError(
                f"TE 2.18 native MXFP8 Fprop/Dgrad/Wgrad requires SM10 or SM11; got SM{capability}"
            )
        if precision == "nvfp4" and capability[0] != 10:
            raise RuntimeError(f"QoreRL NVFP4 adapter targets Blackwell SM10.x; got SM{capability}")
        supported, reason = backend.available(return_reason=True)
        if not supported:
            raise RuntimeError(f"Native {precision.upper()} is unavailable: {reason}")
        return backend.version, torch.cuda.get_device_name(index), capability


def validate_native_format(precision, device=None):
    """Fail closed on unsupported TE/CUDA configurations and return metadata.

    This validates capability and the Python ABI. A successful GPU forward and
    backward smoke test is still required to validate the installed CUDA stack.
    """
    device = _cuda_device(device, precision)
    te_version, gpu_name, capability = _validate_device(device.index, precision)
    metadata = {
        "backend": f"transformer_engine_native_{precision}",
        "transformer_engine_version": te_version,
        "device": str(device),
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "operand_format": "e2m1" if precision == "nvfp4" else "e4m3",
        "scale_format": "e4m3" if precision == "nvfp4" else "e8m0",
        "block_size": 16 if precision == "nvfp4" else 32,
        "output_and_gradient_dtype": "bfloat16",
    }
    if precision == "nvfp4":
        metadata.update(
            global_scale_format="fp32_per_tensor",
            weight_block_shape=[16, 16],
            activation_block_shape=[1, 16],
            quantizer_rht=False,
            stochastic_rounding=False,
        )
    return metadata


def _pad_rows(tensor):
    rows = tensor.shape[0]
    padding = (-rows) % 32
    return F.pad(tensor, (0, 0, 0, padding)) if padding else tensor


class _NativeBlockLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, backend):
        ctx.backend = backend
        flattened = input.reshape(-1, input.shape[-1]).contiguous()
        x_quantized = backend.pack(_pad_rows(flattened))
        w_quantized = backend.pack(weight, weight=True)
        with torch.autograd.profiler.record_function(f"qorerl.{backend.precision}.fprop"):
            output = backend.gemm(
                w_quantized,
                x_quantized,
                out_dtype=torch.bfloat16,
                layout="TN",
                use_split_accumulator=False,
            )[0]
        ctx.save_for_backward(x_quantized, w_quantized)
        ctx.input_shape = input.shape
        ctx.rows = flattened.shape[0]
        ctx.out_features = weight.shape[0]
        return output[: ctx.rows].reshape(*input.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        backend = ctx.backend
        x_quantized, w_quantized = ctx.saved_tensors
        dy = grad_output.reshape(-1, ctx.out_features).contiguous()
        g_quantized = backend.pack(_pad_rows(dy))
        dinput = dweight = None
        if ctx.needs_input_grad[0]:
            with torch.autograd.profiler.record_function(f"qorerl.{backend.precision}.dgrad"):
                dinput = backend.gemm(
                    w_quantized,
                    g_quantized,
                    out_dtype=torch.bfloat16,
                    layout="NN",
                    grad=True,
                    use_split_accumulator=True,
                )[0]
            dinput = dinput[: ctx.rows].reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            with torch.autograd.profiler.record_function(f"qorerl.{backend.precision}.wgrad"):
                dweight = backend.gemm(
                    x_quantized,
                    g_quantized,
                    out_dtype=torch.bfloat16,
                    layout="NT",
                    grad=True,
                    use_split_accumulator=True,
                )[0]
        return dinput, dweight, None


def check_operands(input, weight, bias, precision):
    """The contraction dimensions of all three GEMMs are padded or aligned to 32."""
    if input.ndim < 2 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"{precision.upper()} linear requires input [..., K] and weight [N, K], with input rank >= 2"
        )
    if input.numel() == 0 or weight.numel() == 0:
        raise ValueError(f"{precision.upper()} linear does not support empty inputs or weights")
    if any(size % 32 for size in weight.shape):
        raise ValueError(
            f"{precision.upper()} linear weight dimensions must both be divisible by 32; got {weight.shape}"
        )
    tensors = (input, weight) if bias is None else (input, weight, bias)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError(f"{precision.upper()} linear requires BF16 inputs, weights, and bias")
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError(
            f"{precision.upper()} linear inputs, weights, and bias must be on the same CUDA device"
        )
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError(f"{precision.upper()} linear bias must have shape [N]")


def native_block_linear(input, weight, bias, precision):
    check_operands(input, weight, bias, precision)
    device = _cuda_device(input.device, precision)
    _validate_device(device.index, precision)
    # cuBLAS workspace allocation is device-sensitive in the pinned TE ABI.
    with torch.cuda.device(device):
        output = _NativeBlockLinear.apply(input, weight, _load_backend(precision))
    return output if bias is None else output + bias
