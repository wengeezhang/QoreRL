"""Selected dense Qwen3 MLP projections, preserving Bridge's canonical weights.

This adapter intentionally supports one model rank. It changes per-instance
execution, not the upstream class or state_dict layout. Canonical parameters
remain the optimizer/Bridge parameters. Autograd maps the native kernel's
operand gradients back through the configured transform.
"""

from types import MethodType

import torch

from qorerl.recipes.state import RecipeState
from qorerl.recipes.te_mxfp8 import native_mxfp8_linear, validate_native_mxfp8


class MegatronRecipeInstallation:
    def __init__(self, state, backend, linear, targets=()):
        self.state, self.backend = state, backend
        self.linear = linear
        self.calls = {name: 0 for name in state.targets}
        self.observer = None
        self.calibrator = None
        self.weights = {name: module.weight for name, module in targets}

    def forward(self, module, x, name):
        if self.calibrator is not None:
            self.calibrator.capture(name, x)
        if self.observer is not None:
            self.observer.capture(name, x, module.weight)
        self.calls[name] += 1
        x = self.state.transform_input(x, name)
        weight = self.state.transform_weight(module.weight, name)
        # SmoothQuant maps dX_eff to dX_eff/S and dW_eff to dW_eff*S.
        output = self.linear(x, weight)
        return output, None  # MCore's bias-free TERowParallelLinear contract.

    def report(self):
        operand_format = "nvfp4_e2m1" if self.state.spec.precision == "nvfp4" else "mxfp8_e4m3"
        return {
            **self.state.report(),
            **self.backend,
            "fprop": operand_format,
            "dgrad": operand_format,
            "wgrad": operand_format,
            "parameter_export": "canonical_bf16",
            "gradient_return": "bf16_in_canonical_basis",
            "calls": dict(self.calls),
        }


def _validate_native_weight(weight):
    if weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
        raise ValueError("Native recipe requires canonical CUDA BF16 weights")


def install(model, recipe, num_layers):
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.transformer.module import Float16Module

    if not recipe.is_low_precision:
        raise ValueError("Expected a scoped low-precision recipe")
    if recipe.precision == "nvfp4":
        from qorerl.recipes.te_nvfp4 import native_nvfp4_linear, validate_native_nvfp4

        backend, linear = validate_native_nvfp4(), native_nvfp4_linear
    else:
        backend, linear = validate_native_mxfp8(), native_mxfp8_linear
    # Bridge applies this BF16 wrapper independently of wrap_with_ddp=False.
    # Unwrap only the known mixed-precision wrapper for structural inspection;
    # the caller retains it for forward execution and checkpoint handling.
    if isinstance(model, Float16Module):
        model = model.module
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "layers"):
        raise ValueError("Recipe requires a GPT decoder, optionally wrapped in Float16Module")
    config = model.config
    for field in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if getattr(config, field, None) != 1:
            raise ValueError(f"Low-precision MLP recipe requires {field}=1")
    if getattr(config, "sequence_parallel", False) or getattr(config, "num_moe_experts", None):
        raise ValueError("First MLP recipe supports dense, non-sequence-parallel execution only")
    layers = model.decoder.layers
    if len(layers) != num_layers:
        raise ValueError("Expected the complete single-rank decoder layer list")
    state, targets = RecipeState(recipe), []
    selected = recipe.select_layers(num_layers)
    for index in selected:
        mlp = layers[index].mlp
        module = mlp.linear_fc2
        if type(mlp).__name__ != "MLP" or not isinstance(module, TERowParallelLinear):
            raise ValueError("Recipe requires standard MCore MLP with TERowParallelLinear fc2")
        if getattr(module, "_qorerl_recipe", None) is not None:
            raise ValueError("Recipe is already installed on this Megatron projection")
        if getattr(module, "bias", None) is not None:
            raise ValueError("First MLP recipe requires bias-free down projections")
        _validate_native_weight(module.weight)
        targets.append((state.add_target(index, module.weight), module))
    installation = MegatronRecipeInstallation(state, backend, linear, targets)
    # Validate every target before changing any execution method.
    for name, module in targets:

        def forward(this, x, _name=name):
            return installation.forward(this, x, _name)

        module.forward = MethodType(forward, module)
        module._qorerl_recipe = recipe.identity
    return installation
