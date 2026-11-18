"""Native Transformer Engine MXFP8 GEMMs for a single-rank linear projection.

Targets the TE 2.18 ABI used by Megatron Bridge 0.6.2 (TE source commit
15288857644368b8b7bdddd9bead7d6d3d04de80). In this ABI ``general_gemm`` owns
its workspace; older releases have a different positional signature.

Only the operands are quantized. Inputs, outputs, parameters, and returned
gradients are BF16. Fprop, Dgrad, and Wgrad all use native MXFP8 GEMMs.
There is no emulation or high-precision GEMM fallback in this module.
"""

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
    fp8_dtype: Any
    gemm: Any
    available: Any


@lru_cache(maxsize=1)
def _load_backend():
    """Load CUDA extensions only when a native recipe is actually requested."""
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError as exc:
        raise RuntimeError("Native MXFP8 requires Transformer Engine 2.18.x with its CUDA extension") from exc
    release = re.match(r"^(\d+)\.(\d+)(?:\.|$)", te_version)
    if release is None or tuple(map(int, release.groups())) != (2, 18):
        raise RuntimeError(
            f"Native MXFP8 adapter targets Transformer Engine 2.18.x; found {te_version}. "
            "Other TE versions need explicit API and native-kernel validation."
        )
    try:
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.cpp_extensions import general_gemm
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    except (ImportError, OSError) as exc:
        raise RuntimeError("Transformer Engine's PyTorch/CUDA extension could not be loaded") from exc

    parameters = tuple(signature(general_gemm).parameters)
    if parameters[:3] != ("A", "B", "out_dtype") or "workspace" in parameters:
        raise RuntimeError("Unsupported Transformer Engine general_gemm ABI; expected the TE 2.18 API")
    return _Backend(te_version, MXFP8Quantizer, tex.DType.kFloat8E4M3, general_gemm, te.is_mxfp8_available)


def _cuda_device(device):
    if not torch.cuda.is_available():
        raise RuntimeError("Native MXFP8 requires a supported NVIDIA CUDA GPU; no CPU fallback is available")
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise RuntimeError(f"Native MXFP8 requires a CUDA device, got {device}")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


@lru_cache(maxsize=None)
def _validate_device(index):
    backend = _load_backend()
    with torch.cuda.device(index):
        capability = torch.cuda.get_device_capability(index)
        # TE 2.18's all-layout MXFP8 support excludes SM12, even though those
        # devices are also marketed as Blackwell. Do not use a name-only check.
        if not (10 <= capability[0] < 12):
            raise RuntimeError(
                f"TE 2.18 native MXFP8 Fprop/Dgrad/Wgrad requires SM10 or SM11; got SM{capability}"
            )
        supported, reason = backend.available(return_reason=True)
        if not supported:
            raise RuntimeError(f"Native MXFP8 is unavailable: {reason}")
        return backend.version, torch.cuda.get_device_name(index), capability


def validate_native_mxfp8(device=None):
    """Fail closed on unsupported TE/CUDA configurations and return metadata.

    This validates capability and the Python ABI. A successful GPU forward and
    backward smoke test is still required to validate the installed CUDA stack.
    """
    device = _cuda_device(device)
    te_version, gpu_name, capability = _validate_device(device.index)
    return {
        "backend": "transformer_engine_native_mxfp8",
        "transformer_engine_version": te_version,
        "device": str(device),
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "operand_format": "e4m3",
        "scale_format": "e8m0",
        "block_size": 32,
        "output_and_gradient_dtype": "bfloat16",
    }


def _pad_rows(tensor):
    rows = tensor.shape[0]
    padding = (-rows) % 32
    return F.pad(tensor, (0, 0, 0, padding)) if padding else tensor


def _quantize(backend, tensor):
    # The columnwise values must be independently quantized from BF16. A
    # transpose of the rowwise quantization has different block scaling.
    quantizer = backend.quantizer(backend.fp8_dtype, rowwise=True, columnwise=True)
    # Keep internal=False, the default, so save_for_backward can save the full
    # MXFP8Tensor directly, including both representations and their scales.
    return quantizer(tensor)


class _NativeMXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        backend = _load_backend()
        flattened = input.reshape(-1, input.shape[-1]).contiguous()
        x_quantized = _quantize(backend, _pad_rows(flattened))
        w_quantized = _quantize(backend, weight.contiguous())
        with torch.autograd.profiler.record_function("qorerl.mxfp8.fprop"):
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
        backend = _load_backend()
        x_quantized, w_quantized = ctx.saved_tensors
        dy = grad_output.reshape(-1, ctx.out_features).contiguous()
        g_quantized = _quantize(backend, _pad_rows(dy))
        dinput = dweight = None
        if ctx.needs_input_grad[0]:
            with torch.autograd.profiler.record_function("qorerl.mxfp8.dgrad"):
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
            with torch.autograd.profiler.record_function("qorerl.mxfp8.wgrad"):
                dweight = backend.gemm(
                    x_quantized,
                    g_quantized,
                    out_dtype=torch.bfloat16,
                    layout="NT",
                    grad=True,
                    use_split_accumulator=True,
                )[0]
        return dinput, dweight


def native_mxfp8_linear(input, weight, bias=None):
    """Compute ``input @ weight.T`` with native MXFP8 in all three GEMMs.

    ``input`` may have arbitrary leading dimensions, whose flattened token
    count is padded internally to a multiple of 32 and then unpadded. Both
    dimensions of ``weight`` must already be multiples of 32. All tensors
    must be BF16 on the same supported CUDA device. Bias, when present, is
    added in BF16 outside the GEMM and uses ordinary PyTorch autograd.

    Pass effective operands after the recipe's Hadamard rotation or SmoothQuant
    input/weight scaling. Autograd maps both returned gradients through those
    transforms to canonical activations and trainable parameters.
    No quantized weight cache or new trainable parameter is created.
    """
    if input.ndim < 2 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError("MXFP8 linear requires input [..., K] and weight [N, K], with input rank >= 2")
    if input.numel() == 0 or weight.numel() == 0:
        raise ValueError("MXFP8 linear does not support empty inputs or weights")
    if any(size % 32 for size in weight.shape):
        raise ValueError(f"MXFP8 linear weight dimensions must both be divisible by 32; got {weight.shape}")
    tensors = (input, weight) if bias is None else (input, weight, bias)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("MXFP8 linear requires BF16 inputs, weights, and bias")
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("MXFP8 linear inputs, weights, and bias must be on the same CUDA device")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("MXFP8 linear bias must have shape [N]")
    device = _cuda_device(input.device)
    _validate_device(device.index)
    # cuBLAS workspace allocation is device-sensitive in the pinned TE ABI.
    with torch.cuda.device(device):
        output = _NativeMXFP8Linear.apply(input, weight)
    return output if bias is None else output + bias
