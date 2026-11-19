"""Megatron MXFP8 Fprop, Dgrad and Wgrad with native TE GEMMs and BF16 outputs."""

from qorerl.recipes.te_block_linear import native_block_linear, validate_native_format


def validate_native_mxfp8(device=None):
    return validate_native_format("mxfp8", device)


def native_mxfp8_linear(input, weight, bias=None):
    return native_block_linear(input, weight, bias, "mxfp8")
