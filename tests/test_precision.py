import pytest
import torch

from qorerl.precision import PrecisionLinear, bf16_values, e4m3_round, mx_round, precision


def test_e4m3_known_values_and_ties():
    x = torch.tensor([0, 2**-10, 3 * 2**-10, 2**-6, 1.0625, 1.1875, 448, 480, -1.0625])
    expected = torch.tensor([0, 0, 2**-8, 2**-6, 1, 1.25, 448, 448, -1])
    torch.testing.assert_close(e4m3_round(x), expected, rtol=0, atol=0)


def test_e4m3_matches_pytorch_cast_for_finite_range():
    x = torch.linspace(-448, 448, 20001)
    torch.testing.assert_close(e4m3_round(x), x.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)


def test_block_scale_padding_and_axes():
    torch.manual_seed(7)
    x = torch.randn(35, 33)
    x[0, 0] = 1000
    row = mx_round(x, 1)
    column = mx_round(x, 0)
    assert row.shape == column.shape == x.shape
    assert not torch.equal(row, column)
    torch.testing.assert_close(column, mx_round(x.T, 1).T, rtol=0, atol=0)
    # A separately quantized row is invariant to other rows for Fprop.
    torch.testing.assert_close(row[0:1], mx_round(x[0:1], 1), rtol=0, atol=0)
    assert torch.equal(mx_round(torch.zeros(3, 5), 1), torch.zeros(3, 5))
    assert mx_round(torch.empty(0, 32), 0).shape == (0, 32)
    with pytest.raises(FloatingPointError):
        mx_round(torch.tensor([[float("inf")]]), 1)


def test_repair_only_changes_wgrad_not_forward_or_dgrad():
    torch.manual_seed(19)
    layer = PrecisionLinear(35, 33)
    x = torch.randn(37, 35, requires_grad=True)
    g = torch.randn(37, 33)
    with precision(layer, "mxfp8"):
        baseline = layer(x)
        baseline.backward(g)
    dx_before, dw_before = x.grad.clone(), layer.weight.grad.clone()
    x.grad = None
    layer.zero_grad()
    with precision(layer, "mxfp8", {""}):
        repaired = layer(x)
        repaired.backward(g)
    assert torch.equal(baseline, repaired)
    assert torch.equal(dx_before, x.grad)
    assert not torch.equal(dw_before, layer.weight.grad)
    expected = mx_round(bf16_values(g), 0).T @ bf16_values(x)
    torch.testing.assert_close(layer.weight.grad, expected, rtol=0, atol=0)
    # Baseline Wgrad uses its OWN columnwise quantization, not saved Fprop Q(X).
    expected_baseline = mx_round(bf16_values(g), 0).T @ mx_round(bf16_values(x), 0)
    torch.testing.assert_close(dw_before, expected_baseline, rtol=0, atol=0)


def test_backward_recipe_is_captured_at_forward_and_context_restores_on_error():
    layer = PrecisionLinear(5, 10)
    x = torch.randn(3, 5, requires_grad=True)
    with precision(layer, "mxfp8", {""}):
        y = layer(x)
    assert layer.mode == "bf16" and not layer.restore_activation
    y.sum().backward()  # The backward still uses the captured MXFP8 recipe.
    torch.testing.assert_close(layer.weight.grad, torch.ones(3, 10).T @ bf16_values(x), rtol=0, atol=0)
    with pytest.raises(RuntimeError), precision(layer, "mxfp8"):
        raise RuntimeError("test")
    assert layer.mode == "bf16"
