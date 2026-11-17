"""Canonical target names and fixed transforms shared by both engine adapters.

Only canonical BF16 model weights are published. Quantized/rotated weights are
disposable engine caches, rebuilt after every refit. Fixed signs are included in
checkpoints and snapshots, so recovery checks the actual transform, not only a seed.
"""

import torch

from qorerl.recipes.hadamard import PreparedHadamard, make_signs


def down_proj_name(index):
    return f"model.layers.{index}.mlp.down_proj.weight"


class RecipeState:
    def __init__(self, spec):
        self.spec = spec
        self.targets = {}
        self._signs = {}
        self._transforms = {}

    @property
    def signs(self):
        # Callers may inspect or serialize signs, but cannot change a live
        # rotation independently of its recorded identity/state.
        return {name: value.clone() for name, value in self._signs.items()}

    def add_target(self, index, weight):
        name = down_proj_name(index)
        if name in self.targets:
            raise ValueError(f"Duplicate recipe target: {name}")
        if weight.ndim != 2 or any(dim % 32 for dim in weight.shape):
            raise ValueError(f"MXFP8 target dimensions must be multiples of 32: {name} {weight.shape}")
        if self.spec.uses_hadamard:
            if weight.shape[1] % self.spec.hadamard_block_size:
                raise ValueError(f"Hadamard block_size does not divide input channels: {name}")
            self._signs[name] = make_signs(weight.shape[1], self.spec.seed + index)
            self._transforms[name] = PreparedHadamard(self._signs[name], self.spec.hadamard_block_size)
        self.targets[name] = tuple(weight.shape)
        return name

    def rotate(self, value, name):
        if not self.spec.uses_hadamard:
            return value
        return self._transforms[name].apply(value)

    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "identity": self.spec.identity,
            "targets": {name: list(shape) for name, shape in sorted(self.targets.items())},
            "signs": {name: value.cpu().clone() for name, value in sorted(self._signs.items())},
        }

    def validate(self, saved):
        current = self.to_dict()
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError("Missing or invalid recipe state")
        for key in ("identity", "spec", "targets"):
            if saved[key] != current[key]:
                raise ValueError(f"Recipe state mismatch: {key}")
        if not isinstance(saved["signs"], dict) or set(saved["signs"]) != set(self._signs):
            raise ValueError("Recipe state mismatch: rotation targets")
        for name, signs in self._signs.items():
            value = saved["signs"][name]
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), signs):
                raise ValueError(f"Recipe state mismatch: fixed signs for {name}")

    def validate_weights(self, tensors):
        for name, shape in self.targets.items():
            weight = tensors.get(name)
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != shape:
                raise ValueError(f"Missing or incorrectly shaped canonical recipe weight: {name}")
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"Canonical recipe weight must be BF16: {name}")

    def report(self):
        return {
            **self.spec.to_dict(),
            "identity": self.spec.identity,
            "targets": {name: list(shape) for name, shape in sorted(self.targets.items())},
            "validation": "experimental_gpu_validation_pending",
            "other_operators": "bf16_baseline",
        }
