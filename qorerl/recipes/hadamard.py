"""Fixed signed block Hadamard rotations with gradients in the original basis."""

import torch

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def make_signs(width: int, seed: int) -> torch.Tensor:
    """Return reproducible CPU float32 signs without changing global RNG state."""
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (width,), generator=generator, device="cpu").float().mul_(2).sub_(1)


def _transform(x: torch.Tensor, signs: torch.Tensor, block_size: int, inverse: bool) -> torch.Tensor:
    """Apply D H or H D using butterfly sums, with no dense Hadamard matrix."""
    dtype = x.dtype
    work = x.to(torch.float64 if dtype == torch.float64 else torch.float32)
    if not inverse:
        work = work * signs

    shape = work.shape
    blocks = work.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    stride = 1
    while stride < block_size:
        pairs = blocks.reshape(*blocks.shape[:-1], block_size // (2 * stride), 2, stride)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        blocks = torch.stack((left + right, left - right), dim=-2).reshape(blocks.shape)
        stride *= 2
    work = blocks.reshape(shape) * (block_size**-0.5)
    if inverse:
        work = work * signs
    return work.to(dtype)


class _Rotate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, signs, block_size, inverse):
        ctx.save_for_backward(signs)
        ctx.block_size = block_size
        ctx.inverse = inverse
        return _transform(x, signs, block_size, inverse)

    @staticmethod
    def backward(ctx, grad_output):
        (signs,) = ctx.saved_tensors
        # R = D H is generally not symmetric: its transpose is H D.
        grad_input = _transform(grad_output, signs, ctx.block_size, not ctx.inverse)
        return grad_input, None, None, None


def _validate_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or x.dtype not in _DTYPES:
        raise TypeError("x must be a float16, bfloat16, float32, or float64 tensor")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty final channel axis")


def _validate_block_size(block_size: int) -> None:
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a positive power of two")


def _validate_signs(signs: torch.Tensor, width: int | None = None) -> None:
    if not isinstance(signs, torch.Tensor) or signs.is_complex() or signs.dtype == torch.bool:
        raise TypeError("signs must be a real numeric tensor")
    if signs.ndim != 1 or signs.shape[0] == 0 or (width is not None and signs.shape[0] != width):
        raise ValueError("signs must have shape [channel width]")
    if signs.requires_grad:
        raise ValueError("signs must be fixed, with requires_grad=False")
    if not torch.all((signs == 1) | (signs == -1)).item():
        raise ValueError("signs must contain only -1 and +1")


class PreparedHadamard:
    """A reusable fixed transform with owned signs and lazy device copies.

    Construction validates signs once and owns a CPU float32 clone, so later
    mutations of the caller's signs cannot change the rotation. Device copies
    are reused per device and accumulation dtype. ``apply`` performs metadata
    checks without synchronizing to inspect sign values. This plain Python
    object adds no model parameters or state-dict entries.
    """

    def __init__(self, signs: torch.Tensor, block_size: int):
        _validate_block_size(block_size)
        _validate_signs(signs)
        if signs.shape[0] % block_size:
            raise ValueError("channel width must be divisible by block_size; padding is not supported")
        self._signs_cpu = signs.detach().to(device="cpu", dtype=torch.float32).clone()
        self._block_size = block_size
        self._width = signs.shape[0]
        self._device_signs = {(torch.device("cpu"), torch.float32): self._signs_cpu}

    def apply(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply the prepared rotation, or its inverse, to the final axis."""
        _validate_input(x)
        if x.shape[-1] != self._width:
            raise ValueError("x channel width must match the prepared signs")
        if not isinstance(inverse, bool):
            raise TypeError("inverse must be a bool")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        key = (x.device, dtype)
        signs = self._device_signs.get(key)
        if signs is None:
            signs = self._signs_cpu.to(device=x.device, dtype=dtype)
            self._device_signs[key] = signs
        return _Rotate.apply(x, signs, self._block_size, inverse)


def rotate(x: torch.Tensor, signs: torch.Tensor, block_size: int, inverse: bool = False) -> torch.Tensor:
    """Rotate the final axis of ``x`` by a fixed orthogonal block transform.

    Forward applies ``(x * signs) @ H``; inverse applies ``(x @ H) * signs``,
    where H contains normalized Sylvester Hadamard blocks. Applying the same
    forward rotation to a linear layer's input and weight preserves its output.
    Autograd maps both gradients back to their original, unrotated basis.

    The positive channel width must divide exactly into power-of-two blocks;
    channels are never padded. Signs are fixed, one-dimensional +/-1 tensors
    and are moved to the input's device automatically. The result preserves
    shape, dtype, and device. FP16/BF16/FP32 use FP32 butterfly accumulation;
    FP64 uses FP64 accumulation. For repeated execution, ``PreparedHadamard``
    avoids repeated sign validation and device copies.
    """
    _validate_input(x)
    _validate_block_size(block_size)
    if x.shape[-1] % block_size:
        raise ValueError("channel width must be divisible by block_size; padding is not supported")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")
    _validate_signs(signs, x.shape[-1])
    signs = signs.to(device=x.device, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
    return _Rotate.apply(x, signs, block_size, inverse)
