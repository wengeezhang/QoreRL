"""CPU tests check autograd plumbing with identity quantization, never MXFP8 quality.

The explicitly named native CUDA test exercises real TE quantization/GEMMs and
is skipped when the required hardware or TE release is not installed.
"""

import re
import sys
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from qorerl.recipes import te_mxfp8 as native


@pytest.fixture(autouse=True)
def clear_native_caches():
    native._load_backend.cache_clear()
    native._validate_device.cache_clear()
    yield
    native._load_backend.cache_clear()
    native._validate_device.cache_clear()


@pytest.fixture
def identity_backend(monkeypatch):
    """Inject lossless CPU operands, solely to check the custom autograd graph."""
    quantized, calls = [], []

    class IdentityQuantizer:
        def __init__(self, dtype, *, rowwise, columnwise):
            assert dtype == "mock_e4m3" and rowwise and columnwise

        def __call__(self, tensor):
            assert tensor.is_contiguous()
            value = tensor.clone()
            quantized.append(value)
            return value

    def gemm(A, B, out_dtype, *, layout, use_split_accumulator, grad=False):
        calls.append((layout, A, B, grad, use_split_accumulator))
        # TE describes transpose flags in column-major order. This generic
        # matrix operation is deliberately independent of F.linear's autograd.
        left = B.T if layout[1] == "T" else B
        right = A.T if layout[0] == "T" else A
        return (left.float() @ right.float()).to(out_dtype), None, None, None

    backend = native._Backend("mock", IdentityQuantizer, "mock_e4m3", gemm, None)
    monkeypatch.setattr(native, "_load_backend", lambda: backend)
    return SimpleNamespace(quantized=quantized, calls=calls)


@pytest.mark.parametrize("shape", [(1, 64), (31, 64), (32, 64), (33, 64), (2, 3, 64)])
def test_identity_backend_padding_forward_and_gradients_match_linear(identity_backend, shape):
    generator = torch.Generator().manual_seed(318)
    # Exercise noncontiguous input and weight without hiding them behind a copy.
    x = torch.randn(*shape[:-1], shape[-1] * 2, generator=generator).bfloat16()[..., ::2]
    x.requires_grad_()
    weight = torch.randn(64, 32, generator=generator).bfloat16().T.requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected = F.linear(expected_x, expected_weight)
    actual = native._NativeMXFP8Linear.apply(x, weight)
    upstream = torch.randn(actual.shape, generator=generator).bfloat16()

    rows = x.numel() // x.shape[-1]
    padded_rows = (rows + 31) // 32 * 32
    saved_x, saved_weight = actual.grad_fn.saved_tensors
    assert saved_x.shape == (padded_rows, 64)
    assert saved_weight.shape == (32, 64)
    torch.testing.assert_close(saved_x[:rows], x.reshape(-1, 64), rtol=0, atol=0)
    assert torch.count_nonzero(saved_x[rows:]) == 0
    torch.testing.assert_close(actual, expected)

    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(weight.grad, expected_weight.grad)
    assert x.grad.dtype == weight.grad.dtype == actual.dtype == torch.bfloat16

    # Padded forward rows and padded upstream gradients must contribute nothing
    # to the unpadded F.linear objective or its canonical weight gradient.
    xq, wq, gq = identity_backend.quantized
    assert len(identity_backend.calls) == 3
    assert gq.shape == (padded_rows, 32)
    assert torch.count_nonzero(gq[rows:]) == 0
    torch.testing.assert_close(gq[:rows], upstream.reshape(-1, 32), rtol=0, atol=0)
    expected_calls = [("TN", wq, xq, False, False), ("NN", wq, gq, True, True), ("NT", xq, gq, True, True)]
    for call, expected_call in zip(identity_backend.calls, expected_calls, strict=True):
        assert call[0] == expected_call[0] and call[3:] == expected_call[3:]
        for operand, expected_operand in zip(call[1:3], expected_call[1:3], strict=True):
            torch.testing.assert_close(operand, expected_operand, rtol=0, atol=0)


@pytest.mark.parametrize("input_grad,weight_grad", [(True, False), (False, True)])
def test_identity_backend_only_computes_requested_gradients(identity_backend, input_grad, weight_grad):
    x = torch.ones(3, 64, dtype=torch.bfloat16, requires_grad=input_grad)
    weight = torch.ones(32, 64, dtype=torch.bfloat16, requires_grad=weight_grad)
    native._NativeMXFP8Linear.apply(x, weight).sum().backward()
    assert [call[0] for call in identity_backend.calls] == ["TN", "NN" if input_grad else "NT"]
    if input_grad:
        torch.testing.assert_close(x.grad, torch.full_like(x, 32))
        assert weight.grad is None
    else:
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 3))
        assert x.grad is None


def test_identity_backend_operands_use_autograd_saved_tensor_hooks(identity_backend):
    packed, unpacked = [], []

    def pack(tensor):
        packed.append(tensor.shape)
        return tensor.clone()

    def unpack(tensor):
        unpacked.append(tensor.shape)
        return tensor

    x = torch.ones(3, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(32, 64, dtype=torch.bfloat16, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = native._NativeMXFP8Linear.apply(x, weight)
    output.sum().backward()
    assert packed == unpacked == [torch.Size([32, 64]), torch.Size([32, 64])]
    torch.testing.assert_close(x.grad, torch.full_like(x, 32))
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 3))


def test_cpu_request_fails_without_attempting_te_import(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_import():
        raise AssertionError("A CPU request must not try to import Transformer Engine")

    monkeypatch.setattr(native, "_load_backend", unexpected_import)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        native.validate_native_mxfp8()


def test_explicit_cpu_device_rejected_even_on_a_cuda_host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        native.validate_native_mxfp8("cpu")


@pytest.mark.parametrize("te_version", ["2.7.0", "2.17.0", "2.19.0", "3.0.0", "unknown"])
def test_unsupported_te_version_fails_before_import(monkeypatch, te_version):
    monkeypatch.setattr(native, "version", lambda name: te_version)
    with pytest.raises(RuntimeError, match="targets Transformer Engine 2.18.x"):
        native._load_backend()


def test_old_workspace_gemm_abi_is_rejected(monkeypatch):
    monkeypatch.setattr(native, "version", lambda name: "2.18.0+test")
    names = [
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.tensor",
        "transformer_engine.pytorch.tensor.mxfp8_tensor",
        "transformer_engine_torch",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    def old_gemm(A, B, workspace, out_dtype=None):
        raise AssertionError("ABI validation must not execute a GEMM")

    modules["transformer_engine.pytorch.cpp_extensions"].general_gemm = old_gemm
    modules["transformer_engine.pytorch.tensor.mxfp8_tensor"].MXFP8Quantizer = object
    with pytest.raises(RuntimeError, match="Unsupported Transformer Engine general_gemm ABI"):
        native._load_backend()


@pytest.mark.parametrize("capability", [(9, 0), (12, 0), (12, 1)])
def test_all_layout_capability_guard_rejects_unsupported_architectures(monkeypatch, capability):
    monkeypatch.setattr(native, "_load_backend", lambda: SimpleNamespace(version="2.18.0"))
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    with pytest.raises(RuntimeError, match="requires SM10 or SM11"):
        native._validate_device(0)


def test_te_availability_reason_is_preserved(monkeypatch):
    backend = SimpleNamespace(version="2.18.0", available=lambda **kwargs: (False, "test unavailable"))
    monkeypatch.setattr(native, "_load_backend", lambda: backend)
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (10, 0))
    with pytest.raises(RuntimeError, match="Native MXFP8 is unavailable: test unavailable"):
        native._validate_device(0)


@pytest.mark.parametrize("rows", [1, 31, 33])
def test_native_cuda_saved_mxfp8_operands_and_all_three_gemms(rows):
    if not torch.cuda.is_available():
        pytest.skip("Native MXFP8 test requires a CUDA GPU")
    if not 10 <= torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("Native MXFP8 training test requires SM10 or SM11")
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError:
        pytest.skip("Native MXFP8 test requires Transformer Engine 2.18.x")
    if not re.match(r"^2\.18(?:\.|$)", te_version):
        pytest.skip(f"Native MXFP8 test requires TE 2.18.x, found {te_version}")
    # Once the declared GPU/package prerequisites are present, extension/ABI
    # failures are real failures, not silently converted into test skips.
    native.validate_native_mxfp8()
    generator = torch.Generator(device="cuda").manual_seed(610 + rows)
    x = (torch.randn(rows, 64, generator=generator, device="cuda") * 0.25).bfloat16().requires_grad_()
    weight = (torch.randn(64, 64, generator=generator, device="cuda") * 0.25).bfloat16()
    weight.requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = F.linear(x_ref, weight_ref)
    actual = native.native_mxfp8_linear(x, weight)

    # Check the actual saved TE tensor objects, and independently evaluate the
    # forward GEMM from their dequantized operands. This is stronger than only
    # comparing with BF16, which also includes quantization error.
    xq, wq = actual.grad_fn.saved_tensors
    from transformer_engine.pytorch import MXFP8Tensor

    assert isinstance(xq, MXFP8Tensor) and isinstance(wq, MXFP8Tensor)
    assert xq._columnwise_data is not None and wq._columnwise_data is not None
    with torch.no_grad():
        x_dequant = xq.dequantize(dtype=torch.float32)
        w_dequant = wq.dequantize(dtype=torch.float32)
        from_saved = F.linear(x_dequant, w_dequant)[:rows].bfloat16()
    torch.testing.assert_close(actual, from_saved, rtol=0.01, atol=0.003)
    torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.1)

    upstream = (torch.randn(actual.shape, generator=generator, device="cuda") * 0.25).bfloat16()
    actual.backward(upstream)
    expected.backward(upstream)
    torch.cuda.synchronize()
    for value in (actual, x.grad, weight.grad):
        assert value.dtype == torch.bfloat16 and torch.isfinite(value).all()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0.15, atol=0.1)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=0.15, atol=0.1)
    # A norm bound prevents the elementwise absolute tolerance from masking a
    # missing/zero gradient in the small-magnitude M=1 case.
    for value, reference in [(actual, expected), (x.grad, x_ref.grad), (weight.grad, weight_ref.grad)]:
        relative_error = (value.float() - reference.float()).norm() / reference.float().norm()
        assert relative_error < 0.15
