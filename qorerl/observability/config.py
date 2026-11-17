"""Explicit, opt-in bounds for training-time local recipe replays."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecipeObservationConfig:
    enabled: bool = False
    every_steps: int = 20
    layers_per_window: int = 2
    max_rows: int = 128
    repeats: int = 5
    max_snapshot_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("recipe_observation.enabled must be a boolean")
        for name in ("every_steps", "layers_per_window", "max_rows", "repeats", "max_snapshot_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"recipe_observation.{name} must be a positive integer")

    @classmethod
    def from_dict(cls, options):
        if not isinstance(options, dict):
            raise ValueError("recipe_observation must be an object")
        unknown = set(options) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unsupported recipe_observation options: {sorted(unknown)}")
        return cls(**options)
