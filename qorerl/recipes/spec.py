"""Quantization format, fixed low-precision scope and optional transforms."""

import hashlib
import json
from dataclasses import dataclass

BASELINES = {"bf16-baseline-v1", "fp32-test-v1"}
MLP = "mlp-down-v1"
HADAMARD = "hadamard-mlp-down-v1"
SMOOTHQUANT = "smoothquant-mlp-down-v1"
PRESETS = {MLP, HADAMARD, SMOOTHQUANT}
FORMATS = {"mxfp8": "e4m3-e8m0-block32", "nvfp4": "e2m1-e4m3-block16"}
MXFP8_MLP = f"mxfp8-{MLP}"
MXFP8_HADAMARD = f"mxfp8-{HADAMARD}"
MXFP8_SMOOTHQUANT = f"mxfp8-{SMOOTHQUANT}"


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    layers: tuple[int, ...] | None = None
    hadamard_block_size: int | None = None
    seed: int = 0
    allow_layer_expansion: bool = False
    smoothquant_layers: tuple[int, ...] | None = None

    @property
    def precision(self):
        return self.name.partition("-")[0]

    @property
    def is_low_precision(self):
        return self.precision in FORMATS

    @property
    def is_mxfp8(self):
        return self.precision == "mxfp8"

    @property
    def uses_hadamard(self):
        return self.hadamard_block_size is not None

    @property
    def uses_smoothquant(self):
        return self.name.endswith("-" + SMOOTHQUANT)

    def to_dict(self):
        transform = "block-rht-dh-v1" if self.uses_hadamard else None
        if self.uses_smoothquant:
            transform = "smoothquant-input-channel-v1"
        return {
            # Schema 1 expansion could promote BF16 layers. Those checkpoints cannot be resumed
            # under the fixed low-precision scope. Legacy fixed recipes retain their identities.
            "schema": 2 if self.allow_layer_expansion or self.smoothquant_layers is not None else 1,
            "name": self.name,
            "scope": "mlp.down_proj" if self.is_low_precision else "baseline",
            "layers": list(self.layers) if self.layers is not None else None,
            "hadamard_block_size": self.hadamard_block_size,
            "seed": self.seed,
            "transform": transform,
            "quantization": FORMATS.get(self.precision),
            **({"allow_layer_expansion": True} if self.allow_layer_expansion else {}),
            **(
                {"smoothquant_layers": list(self.smoothquant_layers)}
                if self.smoothquant_layers is not None
                else {}
            ),
        }

    @property
    def identity(self):
        if self.name in BASELINES:
            return self.name
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"{self.name}:{hashlib.sha256(encoded).hexdigest()[:16]}"

    def select_layers(self, num_layers):
        """Immutable scope of the low-precision projection kernels."""
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("Recipe requires a model with a positive layer count")
        selected = tuple(range(num_layers)) if self.layers is None else self.layers
        if any(index >= num_layers for index in (*selected, *(self.smoothquant_layers or ()))):
            raise ValueError(f"Recipe layer indices exceed model's {num_layers} layers")
        return selected

    def initial_smoothquant_layers(self, num_layers):
        selected = self.select_layers(num_layers)
        if not self.uses_smoothquant:
            return ()
        return selected if self.smoothquant_layers is None else self.smoothquant_layers

    def eligible_layers(self, num_layers):
        if self.uses_smoothquant and not self.allow_layer_expansion:
            return self.initial_smoothquant_layers(num_layers)
        return self.select_layers(num_layers)


def _indices(options, key, *, allow_empty=False):
    values = options.get(key)
    if values is None:
        return None
    if (
        not isinstance(values, (list, tuple))
        or (not values and not allow_empty)
        or any(type(i) is not int or i < 0 for i in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{key} must contain distinct nonnegative layer indices")
    return tuple(sorted(values))


def resolve_recipe(name, options=None):
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("recipe_options must be an object")
    if not isinstance(name, str):
        raise ValueError(f"Unsupported precision recipe: {name!r}")
    if name in BASELINES:
        if options:
            raise ValueError(f"Unsupported options for {name}: {sorted(options)}")
        return RecipeSpec(name)

    if name in PRESETS:
        preset = name
        precision = options.get("precision", "mxfp8")
    else:
        precision, _, preset = name.partition("-")
        if precision not in FORMATS or preset not in PRESETS:
            raise ValueError(f"Unsupported precision recipe: {name!r}")
        if options.get("precision", precision) != precision:
            raise ValueError(f"precision option conflicts with recipe {name}")
    if not isinstance(precision, str) or precision not in FORMATS:
        raise ValueError(f"Unsupported precision {precision!r}; choose from {sorted(FORMATS)}")
    allowed = {"layers", "precision"}
    if preset == HADAMARD:
        allowed.update(("block_size", "seed"))
    if preset == SMOOTHQUANT:
        allowed.update(("allow_layer_expansion", "smoothquant_layers"))
    if set(options) - allowed:
        raise ValueError(f"Unsupported options for {name}: {sorted(set(options) - allowed)}")
    layers = _indices(options, "layers")
    initial = _indices(options, "smoothquant_layers", allow_empty=True)
    if initial is not None and layers is not None and not set(initial) <= set(layers):
        raise ValueError("smoothquant_layers must be a subset of low-precision layers")
    if initial == layers:
        initial = None
    block_size = options.get("block_size", 128) if preset == HADAMARD else None
    if preset == HADAMARD and (
        type(block_size) is not int or block_size < 2 or block_size & (block_size - 1)
    ):
        raise ValueError("Hadamard block_size must be a power of two >= 2")
    seed = options.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Hadamard seed must be an integer in [0, 2**31)")
    expand = options.get("allow_layer_expansion", False)
    if type(expand) is not bool:
        raise ValueError("allow_layer_expansion must be a boolean")
    return RecipeSpec(f"{precision}-{preset}", layers, block_size, seed, expand, initial)
