"""Explicit linear Fprop/Dgrad/Wgrad numerics, not native MXFP8 kernels.

All GEMMs accumulate in FP32. BF16 operands and output activations are emulated
by round trips through torch.bfloat16. MXFP8 uses independently quantized
32-element blocks along EACH GEMM's contraction axis. FP32 master parameters
and gradients are retained for AdamW. No STE through the quantizer is implied:
the custom backward specifies the training recipe explicitly.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def bf16_values(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Finite E4M3FN, ties-to-even, including subnormals; portable to CPU.

    A finite saturation policy is used. Callers reject NaN/Inf, rather than
    allowing an invalid scale or saturation to hide a training failure.
    """
    a = x.abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(a.clamp(min=2.0**-6)))
    spacing = torch.exp2(exponent - 3)
    return torch.copysign((torch.round(a / spacing) * spacing).clamp(max=448.0), x)


def mx_round(x: torch.Tensor, axis: int, block_size: int = 32) -> torch.Tensor:
    """Quantize/dequantize with ceil(log2(amax/448)) E8M0 block scales.

    Incomplete blocks are zero padded ONLY inside quantization, then trimmed.
    Padding never creates tokens, routing decisions, rewards, or loss entries.
    Scale exponent range is [-127, 127]. This implementation does not claim
    bitwise equivalence to TE, including extreme/subnormal scaling behavior.
    """
    if block_size != 32:
        raise ValueError("MXFP8 requires blocks of 32")
    if not torch.isfinite(x).all():
        raise FloatingPointError("Non-finite MXFP8 operand")
    if x.numel() == 0:
        return x.float().clone()
    moved = x.float().movedim(axis, -1)
    size = moved.shape[-1]
    padded = F.pad(moved, (0, (-size) % block_size))
    groups = padded.reshape(*padded.shape[:-1], -1, block_size)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    # Compute the exponent without underflowing amax/448 in FP32.
    exponent = torch.ceil(torch.log2(amax.double().clamp(min=2.0**-149) / 448.0))
    scale = torch.exp2(exponent.clamp(-127, 127)).float()
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    result = e4m3_round(groups / scale) * scale
    return result.reshape(padded.shape)[..., :size].movedim(-1, axis).contiguous()


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, mode, restore_activation):
        x_bf = bf16_values(x.reshape(-1, x.shape[-1]))
        w_bf = bf16_values(weight)
        ctx.save_for_backward(x_bf, w_bf)
        ctx.input_shape = x.shape
        ctx.mode = mode
        ctx.restore_activation = restore_activation
        if mode == "mxfp8":
            # [M,K] @ [N,K].T: contraction over K.
            output = mx_round(x_bf, 1) @ mx_round(w_bf, 1).T
        else:
            output = x_bf @ w_bf.T
        return bf16_values(output).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x_bf, w_bf = ctx.saved_tensors
        g_bf = bf16_values(grad_output.reshape(-1, w_bf.shape[0]))
        if ctx.mode == "mxfp8":
            # Dgrad: [M,N] @ [N,K], contraction over N. Unchanged by repair.
            dx = mx_round(g_bf, 1) @ mx_round(w_bf, 0)
            # Wgrad: [N,M] @ [M,K], contraction over M.
            # Independently quantize ORIGINAL BF16 operands columnwise.
            g_wgrad = mx_round(g_bf, 0)
            x_wgrad = x_bf if ctx.restore_activation else mx_round(x_bf, 0)
            dw = g_wgrad.T @ x_wgrad
        else:
            dx = g_bf @ w_bf
            dw = g_bf.T @ x_bf
        return bf16_values(dx).reshape(ctx.input_shape), dw, None, None


class PrecisionLinear(nn.Module):
    """Bias-free linear; Wgrad-only intervention never changes Fprop/Dgrad."""

    def __init__(self, in_features: int, out_features: int, quantizable: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=in_features**-0.5)
        self.quantizable = quantizable
        self.mode = "bf16"
        self.restore_activation = False

    def forward(self, x):
        return _Linear.apply(x, self.weight, self.mode, self.restore_activation)


def quantized_layers(model: nn.Module) -> dict[str, PrecisionLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, PrecisionLinear) and module.quantizable
    }


@contextmanager
def precision(model: nn.Module, mode: str, overrides=()):
    if mode not in {"bf16", "mxfp8"}:
        raise ValueError(f"Unknown precision: {mode}")
    overrides = set(overrides)
    unknown = overrides - quantized_layers(model).keys()
    if unknown:
        raise ValueError(f"Unknown/non-quantizable layer overrides: {sorted(unknown)}")
    modules = [m for m in model.modules() if isinstance(m, PrecisionLinear)]
    previous = [(m.mode, m.restore_activation) for m in modules]
    try:
        names = {id(m): name for name, m in model.named_modules()}
        for module in modules:
            module.mode = mode if module.quantizable else "bf16"
            module.restore_activation = names[id(module)] in overrides and mode == "mxfp8"
        yield
    finally:
        for module, (old_mode, old_restore) in zip(modules, previous):
            module.mode, module.restore_activation = old_mode, old_restore
