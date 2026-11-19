"""Small, versioned recipe catalogue. IDs describe scope, not claimed quality.

The first MXFP8 recipes quantize selected MLP down projections only. All other
operators retain the BF16 baseline. A recipe's identity includes its options;
publishing the same weights under a different transform is a different policy.
"""

import hashlib
import json
from dataclasses import dataclass

BASELINES = {"bf16-baseline-v1", "fp32-test-v1"}
MXFP8_MLP = "mxfp8-mlp-down-v1"
MXFP8_HADAMARD = "mxfp8-hadamard-mlp-down-v1"
MXFP8_SMOOTHQUANT = "mxfp8-smoothquant-mlp-down-v1"


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    layers: tuple[int, ...] | None = None
    hadamard_block_size: int | None = None
    seed: int = 0
    allow_layer_expansion: bool = False

    @property
    def is_mxfp8(self):
        return self.name in {MXFP8_MLP, MXFP8_HADAMARD, MXFP8_SMOOTHQUANT}

    @property
    def uses_hadamard(self):
        return self.hadamard_block_size is not None

    @property
    def uses_smoothquant(self):
        return self.name == MXFP8_SMOOTHQUANT

    def to_dict(self):
        transform = "block-rht-dh-v1" if self.uses_hadamard else None
        if self.uses_smoothquant:
            transform = "smoothquant-input-channel-v1"
        return {
            "schema": 1,
            "name": self.name,
            "scope": "mlp.down_proj" if self.is_mxfp8 else "baseline",
            "layers": list(self.layers) if self.layers is not None else None,
            "hadamard_block_size": self.hadamard_block_size,
            "seed": self.seed,
            "transform": transform,
            "quantization": "e4m3-e8m0-block32" if self.is_mxfp8 else None,
            # Older fixed-scope checkpoints use this exact spec to derive their ID.
            **({"allow_layer_expansion": True} if self.allow_layer_expansion else {}),
        }

    @property
    def identity(self):
        if self.name in BASELINES:
            return self.name  # Preserve existing baseline checkpoint identities.
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"{self.name}:{hashlib.sha256(encoded).hexdigest()[:16]}"

    def select_layers(self, num_layers):
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("Recipe requires a model with a positive layer count")
        selected = tuple(range(num_layers)) if self.layers is None else self.layers
        if any(index >= num_layers for index in selected):
            raise ValueError(f"Recipe layer indices exceed model's {num_layers} layers")
        return selected

    def eligible_layers(self, num_layers):
        initial = self.select_layers(num_layers)
        return tuple(range(num_layers)) if self.allow_layer_expansion else initial


def resolve_recipe(name, options=None):
    if name not in BASELINES | {MXFP8_MLP, MXFP8_HADAMARD, MXFP8_SMOOTHQUANT}:
        raise ValueError(f"Unsupported precision recipe: {name!r}")
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("recipe_options must be an object")
    allowed = set() if name in BASELINES else {"layers"}
    if name == MXFP8_HADAMARD:
        allowed |= {"block_size", "seed"}
    if name == MXFP8_SMOOTHQUANT:
        allowed.add("allow_layer_expansion")
    if set(options) - allowed:
        raise ValueError(f"Unsupported options for {name}: {sorted(set(options) - allowed)}")
    layers = options.get("layers")
    if layers is not None:
        if (
            not isinstance(layers, (list, tuple))
            or not layers
            or any(type(index) is not int or index < 0 for index in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("recipe layers must be distinct nonnegative indices; omit to select all")
        layers = tuple(sorted(layers))
    block_size = options.get("block_size", 128) if name == MXFP8_HADAMARD else None
    if name == MXFP8_HADAMARD and (
        type(block_size) is not int or block_size < 2 or block_size & (block_size - 1)
    ):
        raise ValueError("Hadamard block_size must be a power of two >= 2")
    seed = options.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Hadamard seed must be an integer in [0, 2**31)")
    expand = options.get("allow_layer_expansion", False)
    if type(expand) is not bool:
        raise ValueError("allow_layer_expansion must be a boolean")
    return RecipeSpec(name, layers, block_size, seed, expand)
