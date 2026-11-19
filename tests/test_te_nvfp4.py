"""CPU tests cover TE dispatch. CUDA cases exercise real E2M1 operands on SM10.x."""

from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from torch.nn import functional as F

from qorerl.recipes import te_block_linear as native
from qorerl.recipes import te_nvfp4 as nvfp4


def test_nvfp4_quantizer_roles_layouts_and_rollout_weight_reuse(monkeypatch):
    packs, gemms = [], []

    class IdentityFP4Quantizer:
        def __init__(self, dtype, *, rowwise, columnwise, with_2d_quantization, with_random_sign_mask):
            assert dtype == "E2M1" and rowwise and not with_random_sign_mask
            self.weight = with_2d_quantization
            self.columnwise = columnwise

        def __call__(self, tensor):
            assert tensor.is_contiguous()
            packs.append((self.weight, self.columnwise, tuple(tensor.shape)))
            return tensor.clone()

    def gemm(A, B, out_dtype, *, layout, grad=False, use_split_accumulator=False):
        gemms.append((layout, grad, use_split_accumulator))
        left = B.T if layout[1] == "T" else B
        right = A.T if layout[0] == "T" else A
        return ((left.float() @ right.float()).to(out_dtype),)

    backend = native._Backend("mock", IdentityFP4Quantizer, "E2M1", gemm, None, "nvfp4")

    def load_backend(precision):
        assert precision == "nvfp4"
        return backend

    for module in (native, nvfp4):
        monkeypatch.setattr(module, "_load_backend", load_backend)
        monkeypatch.setattr(module, "_cuda_device", lambda device, precision: torch.device("cpu"))
        monkeypatch.setattr(module, "_validate_device", lambda index, precision: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    generator = torch.Generator().manual_seed(413)
    x = torch.randn(3, 64, generator=generator, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 64, generator=generator, dtype=torch.bfloat16, requires_grad=True)
    expected = F.linear(x, weight)
    actual = nvfp4.native_nvfp4_linear(x, weight)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.bfloat16)
    dx, dw = torch.autograd.grad(actual, (x, weight), upstream)
    ref_dx, ref_dw = torch.autograd.grad(expected, (x, weight), upstream)
    for result, reference in ((actual, expected), (dx, ref_dx), (dw, ref_dw)):
        torch.testing.assert_close(result, reference)

    packed_weight = nvfp4.pack_nvfp4_weight(weight)
    for rows in (1, 3):
        result = nvfp4.nvfp4_linear_from_packed(x[:rows], packed_weight)
        torch.testing.assert_close(result, expected[:rows])

    assert packs == [
        (False, True, (32, 64)),  # learner activation, rowwise + columnwise
        (True, True, (32, 64)),  # learner weight, 2D scales
        (False, True, (32, 32)),  # upstream gradient, padded for Wgrad
        (True, False, (32, 64)),  # cached rollout weight, rowwise only
        (False, False, (32, 64)),  # rollout activation per forward
        (False, False, (32, 64)),
    ]
    assert gemms == [
        ("TN", False, False),
        ("NN", True, True),
        ("NT", True, True),
        ("TN", False, False),
        ("TN", False, False),
    ]


@pytest.mark.parametrize("rows", [1, 33])
def test_native_cuda_nvfp4_cached_rollout_and_training_gradients(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native NVFP4 smoke requires an SM10.x CUDA GPU")
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError:
        pytest.skip("Native NVFP4 smoke requires Transformer Engine 2.18.x")
    if te_version.split(".")[:2] != ["2", "18"]:
        pytest.skip(f"Native NVFP4 smoke requires Transformer Engine 2.18.x, found {te_version}")
    assert nvfp4.validate_native_nvfp4()["operand_format"] == "e2m1"

    generator = torch.Generator(device="cuda").manual_seed(721 + rows)
    x = torch.randn(rows, 64, generator=generator, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(96, 64, generator=generator, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = F.linear(x, weight)
    training = nvfp4.native_nvfp4_linear(x, weight)
    for saved in training.grad_fn.saved_tensors:
        metadata = saved.get_metadata()
        assert metadata["rowwise_data"] is not None and metadata["columnwise_data"] is not None

    packed = nvfp4.pack_nvfp4_weight(weight)
    packed_metadata = packed.get_metadata()
    assert packed_metadata["rowwise_data"].nbytes == weight.numel() // 2
    assert packed_metadata["rowwise_scale_inv"] is not None and packed_metadata["columnwise_data"] is None
    assert nvfp4.nvfp4_packed_nbytes(packed) > packed_metadata["rowwise_data"].nbytes
    inference = nvfp4.nvfp4_linear_from_packed(x, packed)
    upstream = torch.randn(training.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    dx, dw = torch.autograd.grad(training, (x, weight), upstream)
    ref_dx, ref_dw = torch.autograd.grad(reference, (x, weight), upstream)

    for actual, expected in ((training, reference), (inference, reference), (dx, ref_dx), (dw, ref_dw)):
        assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
        assert torch.isfinite(actual).all()
        # A relative norm check tolerates FP4 rounding near zero while catching
        # missing or transposed contributions in each GEMM independently.
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.45
    # Both engines use deterministic rowwise activation and 2D weight casts for FPROP.
    assert (inference.float() - training.float()).norm() / training.float().norm() < 0.02
