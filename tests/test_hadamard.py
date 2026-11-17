import pytest
import torch
import torch.nn.functional as F

from qorerl.recipes import hadamard
from qorerl.recipes.hadamard import PreparedHadamard, make_signs, rotate


def _dense_hadamard(width, block_size, dtype=torch.float64):
    block = torch.ones(1, 1, dtype=dtype)
    butterfly = torch.tensor([[1, 1], [1, -1]], dtype=dtype)
    while block.shape[0] < block_size:
        block = torch.kron(block, butterfly)
    return torch.block_diag(*([block / block_size**0.5] * (width // block_size)))


def test_signs_are_reproducible_and_do_not_advance_global_rng():
    state = torch.random.get_rng_state().clone()
    signs = make_signs(128, seed=42)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert signs.device.type == "cpu" and signs.dtype == torch.float32
    assert set(signs.tolist()) == {-1, 1}
    assert torch.equal(signs, make_signs(128, seed=42))
    assert not torch.equal(signs, make_signs(128, seed=43))


def test_signs_are_on_cpu_even_with_a_different_default_device():
    with torch.device("meta"):
        signs = make_signs(8, seed=42)
    assert signs.device.type == "cpu"


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
def test_rotation_is_orthogonal_and_invertible(block_size):
    width = 32
    signs = make_signs(width, seed=5)
    identity = torch.eye(width, dtype=torch.float64)
    rotation = rotate(identity, signs, block_size)
    torch.testing.assert_close(rotation @ rotation.T, identity, atol=1e-14, rtol=1e-14)
    x = torch.randn(2, 3, width, dtype=torch.float64)
    torch.testing.assert_close(rotate(rotate(x, signs, block_size), signs, block_size, inverse=True), x)
    torch.testing.assert_close(rotate(rotate(x, signs, block_size, inverse=True), signs, block_size), x)


def test_signed_forward_and_inverse_have_correct_orientation():
    signs = torch.tensor([1, -1, 1, 1, -1, 1, -1, -1], dtype=torch.float64)
    hadamard = _dense_hadamard(8, 4)
    rotation = torch.diag(signs) @ hadamard
    assert not torch.equal(rotation, rotation.T)
    x = torch.arange(24, dtype=torch.float64).reshape(3, 8)
    torch.testing.assert_close(rotate(x, signs, 4), x @ rotation)
    torch.testing.assert_close(rotate(x, signs, 4, inverse=True), x @ rotation.T)


@pytest.mark.parametrize("inverse", [False, True])
def test_rotating_input_and_weight_preserves_linear_output_and_canonical_gradients(inverse):
    signs = make_signs(16, seed=12)
    x = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    reference = F.linear(x, weight, bias)
    actual = F.linear(rotate(x, signs, 4, inverse), rotate(weight, signs, 4, inverse), bias)
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(reference, (x, weight, bias), grad_outputs=upstream)
    actual_grads = torch.autograd.grad(actual, (x, weight, bias), grad_outputs=upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("inverse", [False, True])
def test_double_precision_gradcheck(inverse):
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    signs = make_signs(8, seed=9)

    def transform(value):
        return rotate(value, signs, 4, inverse)

    assert torch.autograd.gradcheck(transform, (x,))
    assert torch.autograd.gradgradcheck(transform, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("inverse", [False, True])
def test_preserves_metadata_and_uses_full_precision_accumulation(dtype, inverse):
    # A noncontiguous input also exercises the reshape path.
    x = torch.randn(2, 32, 3).to(dtype).transpose(1, 2)
    signs = make_signs(32, seed=3)
    actual = rotate(x, signs, 16, inverse)
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    full = x.to(accumulation_dtype)
    hadamard = _dense_hadamard(32, 16, accumulation_dtype)
    expected = ((full @ hadamard) * signs) if inverse else ((full * signs) @ hadamard)
    assert actual.shape == x.shape and actual.dtype == dtype and actual.device == x.device
    torch.testing.assert_close(actual, expected.to(dtype))


def test_vector_and_empty_batch_shapes():
    signs = make_signs(8, seed=3)
    for shape in [(8,), (0, 8), (2, 0, 8)]:
        x = torch.empty(shape)
        assert rotate(x, signs, 4).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cpu_signs_are_accepted_for_cuda_input():
    x = torch.randn(2, 16, device="cuda", requires_grad=True)
    signs = make_signs(16, seed=3)
    output = rotate(x, signs, 4)
    torch.testing.assert_close(output.cpu(), rotate(x.cpu(), signs, 4))
    output.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)


def test_prepared_transform_owns_signs_and_does_not_register_model_state():
    signs = make_signs(16, seed=4)
    original = signs.clone()
    prepared = PreparedHadamard(signs, 4)
    torch.testing.assert_close(signs, original)
    x = torch.randn(3, 16)
    expected = rotate(x, signs, 4)
    signs.zero_()
    torch.testing.assert_close(prepared.apply(x), expected)
    torch.testing.assert_close(prepared.apply(expected, inverse=True), x)
    model = torch.nn.Linear(16, 8)
    parameter_names = set(dict(model.named_parameters()))
    state_keys = set(model.state_dict())
    model.rotation = prepared
    assert set(dict(model.named_parameters())) == parameter_names
    assert set(model.state_dict()) == state_keys


@pytest.mark.parametrize("inverse", [False, True])
def test_prepared_transform_preserves_canonical_gradients(inverse):
    prepared = PreparedHadamard(make_signs(16, seed=4), 4)
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    expected = F.linear(x, weight)
    actual = F.linear(prepared.apply(x, inverse), prepared.apply(weight, inverse))
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (x, weight), upstream)
    actual_grads = torch.autograd.grad(actual, (x, weight), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"))],
)
def test_prepared_transform_reuses_device_signs_across_calls_and_low_precision_dtypes(monkeypatch, device):
    prepared = PreparedHadamard(make_signs(16, seed=4), 4)
    recorded_signs = []
    original_apply = hadamard._Rotate.apply

    def record_apply(x, signs, block_size, inverse):
        recorded_signs.append(signs)
        return original_apply(x, signs, block_size, inverse)

    monkeypatch.setattr(hadamard._Rotate, "apply", record_apply)
    for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.float32, torch.float64]:
        prepared.apply(torch.ones(2, 16, device=device, dtype=dtype))
    assert recorded_signs[0] is recorded_signs[1] is recorded_signs[2] is recorded_signs[4]
    assert recorded_signs[3] is recorded_signs[5]
    assert recorded_signs[0] is not recorded_signs[3]
    assert recorded_signs[0].device.type == recorded_signs[3].device.type == device
    assert recorded_signs[0].dtype == torch.float32
    assert recorded_signs[3].dtype == torch.float64


def test_prepared_execution_does_not_read_sign_values(monkeypatch):
    prepared = PreparedHadamard(make_signs(16, seed=4), 4)
    x = torch.randn(3, 16, requires_grad=True)

    def forbid_item(*args, **kwargs):
        raise AssertionError("Prepared execution must not read tensor values on the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbid_item)
        reconstructed = prepared.apply(prepared.apply(x), inverse=True)
        reconstructed.sum().backward()
    torch.testing.assert_close(reconstructed, x)
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize(
    "signs, block_size, message",
    [
        (torch.empty(0), 4, "shape"),
        (torch.ones(8, 1), 4, "shape"),
        (torch.ones(8), 3, "positive power of two"),
        (torch.ones(7), 4, "divisible"),
        (torch.zeros(8), 4, "only -1"),
        (torch.ones(8, requires_grad=True), 4, "fixed"),
    ],
)
def test_prepared_transform_validates_construction(signs, block_size, message):
    with pytest.raises(ValueError, match=message):
        PreparedHadamard(signs, block_size)


def test_prepared_transform_validates_input_metadata():
    prepared = PreparedHadamard(make_signs(16, seed=4), 4)
    with pytest.raises(ValueError, match="match the prepared signs"):
        prepared.apply(torch.ones(8))
    with pytest.raises(ValueError, match="channel axis"):
        prepared.apply(torch.tensor(1.0))
    with pytest.raises(TypeError, match="x must be"):
        prepared.apply(torch.ones(16, dtype=torch.int64))
    with pytest.raises(TypeError, match="inverse must be a bool"):
        prepared.apply(torch.ones(16), inverse=1)


@pytest.mark.parametrize("block_size", [0, -1, 3, 6, 2.0, True])
def test_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="positive power of two"):
        rotate(torch.ones(8), make_signs(8, 0), block_size)


def test_rejects_channels_that_need_padding():
    with pytest.raises(ValueError, match="divisible"):
        rotate(torch.ones(10), make_signs(10, 0), 4)


@pytest.mark.parametrize("x", [torch.tensor(1.0), torch.empty(2, 0)])
def test_rejects_missing_channel_axis(x):
    with pytest.raises(ValueError, match="channel axis"):
        rotate(x, make_signs(8, 0), 4)


@pytest.mark.parametrize("x", [torch.ones(8, dtype=torch.int64), torch.ones(8, dtype=torch.complex64), []])
def test_rejects_unsupported_input_type(x):
    with pytest.raises(TypeError, match="x must be"):
        rotate(x, make_signs(8, 0), 4)


@pytest.mark.parametrize("signs", [torch.ones(7), torch.ones(1, 8)])
def test_rejects_invalid_sign_shape(signs):
    with pytest.raises(ValueError, match="shape"):
        rotate(torch.ones(8), signs, 4)


@pytest.mark.parametrize("bad_sign", [0, 0.5, float("nan"), float("inf")])
def test_rejects_non_sign_values(bad_sign):
    signs = torch.ones(8)
    signs[0] = bad_sign
    with pytest.raises(ValueError, match=r"only -1 and \+1"):
        rotate(torch.ones(8), signs, 4)


@pytest.mark.parametrize(
    "signs", [[1] * 8, torch.ones(8, dtype=torch.bool), torch.ones(8, dtype=torch.complex64)]
)
def test_rejects_unsupported_sign_type(signs):
    with pytest.raises(TypeError, match="real numeric"):
        rotate(torch.ones(8), signs, 4)


def test_rejects_trainable_signs_and_non_boolean_inverse():
    with pytest.raises(ValueError, match="fixed"):
        rotate(torch.ones(8), torch.ones(8, requires_grad=True), 4)
    with pytest.raises(TypeError, match="inverse must be a bool"):
        rotate(torch.ones(8), torch.ones(8), 4, inverse=1)


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_rejects_invalid_sign_width(width):
    with pytest.raises(ValueError, match="positive integer"):
        make_signs(width, 0)


@pytest.mark.parametrize("seed", [1.5, "0", True])
def test_rejects_invalid_seed_type(seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        make_signs(8, seed)
