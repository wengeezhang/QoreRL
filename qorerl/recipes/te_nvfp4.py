"""Native NVFP4 operations shared by the learner and the vLLM worker.

The rollout worker retains a rowwise FP4 weight and its two levels of scales.
Training retains both orientations to support Dgrad and Wgrad.
"""

import torch

from qorerl.recipes.te_block_linear import (
    _cuda_device,
    _load_backend,
    _pad_rows,
    _validate_device,
    check_operands,
    native_block_linear,
    validate_native_format,
)


def validate_native_nvfp4(device=None):
    return validate_native_format("nvfp4", device)


def native_nvfp4_linear(input, weight, bias=None):
    return native_block_linear(input, weight, bias, "nvfp4")


@torch.no_grad()
def pack_nvfp4_weight(weight):
    if weight.ndim != 2 or min(weight.shape) <= 0 or any(d % 32 for d in weight.shape):
        raise ValueError("NVFP4 rollout weight must be [N, K] with both dimensions divisible by 32")
    if weight.dtype != torch.bfloat16:
        raise ValueError("NVFP4 rollout requires a canonical BF16 weight")
    device = _cuda_device(weight.device, "nvfp4")
    _validate_device(device.index, "nvfp4")
    with torch.cuda.device(device):
        return _load_backend("nvfp4").pack(weight, weight=True, columnwise=False)


def nvfp4_packed_nbytes(packed_weight):
    return sum(t.nbytes for t in packed_weight.get_metadata().values() if isinstance(t, torch.Tensor))


@torch.no_grad()
def nvfp4_linear_from_packed(input, packed_weight, bias=None):
    # The TE tensor exposes its logical BF16 shape/dtype. Its device storage holds FP4 data.
    check_operands(input, packed_weight, bias, "nvfp4")
    backend = _load_backend("nvfp4")
    with torch.cuda.device(input.device):
        rows = input.reshape(-1, input.shape[-1])
        xq = backend.pack(_pad_rows(rows), columnwise=False)
        output = backend.gemm(packed_weight, xq, out_dtype=torch.bfloat16, layout="TN")[0]
        output = output[: rows.shape[0]].reshape(*input.shape[:-1], packed_weight.shape[0])
    return output if bias is None else output + bias
