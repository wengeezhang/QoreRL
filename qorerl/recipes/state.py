"""Canonical transform state shared by learner, checkpoint and rollout.

Model parameters stay in the original basis. Hadamard signs are fixed by the
recipe. SmoothQuant scales start at one, and a calibration commit replaces the
entire mapping. Its content digest identifies the transform used by a policy.
"""

import hashlib

import torch

from qorerl.recipes.hadamard import PreparedHadamard, make_signs


def down_proj_name(index):
    return f"model.layers.{index}.mlp.down_proj.weight"


def scale_digest(scales):
    """SHA-256 of sorted target names and their length-framed little-endian FP32 vectors."""
    if not isinstance(scales, dict) or not scales or any(not isinstance(name, str) for name in scales):
        raise ValueError("SmoothQuant requires named scale vectors")
    digest = hashlib.sha256(b"qorerl.smoothquant.scales.v1\x00")
    for name, value in sorted(scales.items()):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.ndim != 1
            or not value.numel()
            or value.requires_grad
            or not torch.isfinite(value).all().item()
            or not (value > 0).all().item()
        ):
            raise ValueError(f"SmoothQuant scale must be a finite positive fixed CPU float32 vector: {name}")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(value.numel().to_bytes(8, "little"))
        digest.update(value.contiguous().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def smooth_input(value, scale):
    """Apply the fixed 1/S diagonal on the input channels before quantization."""
    return (value / scale.to(value.device)).to(value.dtype)


def smooth_weight(value, scale):
    """Apply the fixed S diagonal on canonical weight columns before quantization."""
    return (value * scale.to(value.device)).to(value.dtype)


class RecipeState:
    def __init__(self, spec):
        self.spec = spec
        self.targets = {}
        self._signs = {}
        self._transforms = {}
        self._initial_scale_targets = set()
        self._scales = {}
        self._scale_digest = None
        self._device_scales = {}

    @property
    def signs(self):
        return {name: value.clone() for name, value in self._signs.items()}

    @property
    def scales(self):
        return {name: value.clone() for name, value in self._scales.items()}

    @property
    def scale_digest(self):
        if self.spec.uses_smoothquant and self._scale_digest is None:
            self._scale_digest = scale_digest(self._scales)
        return self._scale_digest

    @property
    def active_identity(self):
        if self.spec.uses_smoothquant:
            return f"{self.spec.identity}:{self.scale_digest}"
        return self.spec.identity

    def is_active(self, name):
        return name in (self._scales if self.spec.uses_smoothquant else self.targets)

    def add_target(self, index, weight, *, active=True):
        name = down_proj_name(index)
        if name in self.targets:
            raise ValueError(f"Duplicate recipe target: {name}")
        if not active and not self.spec.allow_layer_expansion:
            raise ValueError("Inactive targets require SmoothQuant layer expansion")
        if weight.ndim != 2 or any(dim <= 0 or dim % 32 for dim in weight.shape):
            raise ValueError(f"MXFP8 target dimensions must be multiples of 32: {name} {weight.shape}")
        if self.spec.uses_hadamard:
            if weight.shape[1] % self.spec.hadamard_block_size:
                raise ValueError(f"Hadamard block_size does not divide input channels: {name}")
            self._signs[name] = make_signs(weight.shape[1], self.spec.seed + index)
            self._transforms[name] = PreparedHadamard(self._signs[name], self.spec.hadamard_block_size)
        if self.spec.uses_smoothquant and active:
            self._initial_scale_targets.add(name)
            self._scales[name] = torch.ones(weight.shape[1], dtype=torch.float32, device="cpu")
            self._scale_digest = None
        self.targets[name] = tuple(weight.shape)
        return name

    def _check_scales(self, scales):
        if (
            not self.spec.uses_smoothquant
            or not isinstance(scales, dict)
            or not self._initial_scale_targets <= scales.keys() <= self.targets.keys()
        ):
            raise ValueError(
                "SmoothQuant scales must cover every initial target and stay within eligible targets"
            )
        digest = scale_digest(scales)
        for name, value in scales.items():
            if value.shape != (self.targets[name][1],):
                raise ValueError(f"SmoothQuant scale width mismatch for {name}")
        return digest

    def install_scales(self, scales):
        digest = self._check_scales(scales)
        self._scales = {name: value.contiguous().clone() for name, value in scales.items()}
        self._scale_digest = digest
        self._device_scales.clear()

    def _scale_for(self, name, device):
        key = (name, device)
        if key not in self._device_scales:
            self._device_scales[key] = self._scales[name].to(device)
        return self._device_scales[key]

    def rotate(self, value, name):
        if not self.spec.uses_hadamard:
            return value
        return self._transforms[name].apply(value)

    def transform_input(self, value, name):
        if self.spec.uses_smoothquant:
            return smooth_input(value, self._scale_for(name, value.device))
        return self.rotate(value, name)

    def transform_weight(self, value, name):
        if self.spec.uses_smoothquant:
            return smooth_weight(value, self._scale_for(name, value.device))
        return self.rotate(value, name)

    def _metadata(self):
        return {
            "spec": self.spec.to_dict(),
            "identity": self.spec.identity,
            "targets": {name: list(shape) for name, shape in sorted(self.targets.items())},
        }

    def to_dict(self):
        result = {**self._metadata(), "signs": self.signs}
        if self.spec.uses_smoothquant:
            result.update(scales=self.scales, scale_digest=self.scale_digest)
        return result

    def validate(self, saved):
        """Validate a candidate payload without changing the live transform; return its policy recipe ID."""
        metadata = self._metadata()
        fields = set(metadata) | {"signs"}
        if self.spec.uses_smoothquant:
            fields.update(("scale_digest", "scales"))
        if not isinstance(saved, dict) or set(saved) != fields:
            raise ValueError("Missing or invalid recipe state")
        for key, value in metadata.items():
            if saved[key] != value:
                raise ValueError(f"Recipe state mismatch: {key}")
        if not isinstance(saved["signs"], dict) or set(saved["signs"]) != set(self._signs):
            raise ValueError("Recipe state mismatch: rotation targets")
        for name, signs in self._signs.items():
            value = saved["signs"][name]
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), signs):
                raise ValueError(f"Recipe state mismatch: fixed signs for {name}")
        if self.spec.uses_smoothquant:
            digest = self._check_scales(saved["scales"])
            if saved["scale_digest"] != digest:
                raise ValueError("SmoothQuant scale digest mismatch")
            return f"{self.spec.identity}:{digest}"
        return self.spec.identity

    def load(self, saved):
        self.validate(saved)
        if self.spec.uses_smoothquant:
            self.install_scales(saved["scales"])

    def validate_weights(self, tensors):
        for name, shape in self.targets.items():
            weight = tensors.get(name)
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != shape:
                raise ValueError(f"Missing or incorrectly shaped canonical recipe weight: {name}")
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"Canonical recipe weight must be BF16: {name}")

    def report(self):
        report = {
            **self.spec.to_dict(),
            "identity": self.active_identity,
            "targets": {
                name: list(shape) for name, shape in sorted(self.targets.items()) if self.is_active(name)
            },
            "validation": "experimental_gpu_validation_pending",
            "other_operators": "bf16_baseline",
        }
        if self.spec.uses_smoothquant:
            report.update(scale_digest=self.scale_digest, preset_identity=self.spec.identity)
        if self.spec.allow_layer_expansion:
            report["eligible_targets"] = sorted(self.targets)
        return report
