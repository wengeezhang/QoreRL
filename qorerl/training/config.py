"""Small, explicit configuration surface; unsupported execution is rejected."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qorerl.observability.config import RecipeObservationConfig
from qorerl.recipes.calibration import CalibrationConfig
from qorerl.recipes.spec import resolve_recipe


@dataclass(frozen=True)
class TrainConfig:
    learner_backend: str = "megatron"
    rollout_backend: str = "vllm"
    model: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "main"
    seed: int = 42
    steps: int = 10
    prompts_per_step: int = 4
    group_size: int = 4
    micro_batch_size: int = 2
    update_epochs: int = 1
    learning_rate: float = 1e-6
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    max_new_tokens: int = 64
    max_model_len: int = 512
    rollout_workers: int = 1
    worker_placement: str = "separate"
    gpu_memory_utilization: float | None = None
    checkpoint_every: int = 5
    rpc_timeout: float = 600.0
    recipe_id: str = "mlp-down-v1"
    recipe_options: dict = field(default_factory=dict)
    recipe_observation: dict = field(default_factory=dict)
    recipe_calibration: dict = field(default_factory=dict)
    learner_runtime_env: dict = field(default_factory=dict)
    rollout_runtime_env: dict = field(default_factory=dict)

    @property
    def is_cpu_test(self):
        return self.learner_backend == self.rollout_backend == "torch_test"

    @property
    def recipe(self):
        return resolve_recipe(self.recipe_id, self.recipe_options)

    @property
    def policy_recipe_id(self):
        return self.recipe.identity

    @property
    def observation(self):
        return RecipeObservationConfig.from_dict(self.recipe_observation)

    @property
    def calibration(self):
        return CalibrationConfig.from_dict(self.recipe_calibration)

    @property
    def active_calibration(self):
        if self.recipe_calibration or self.recipe.candidate_handler is not None:
            return self.calibration
        return None

    def __post_init__(self):
        if self.worker_placement not in ("separate", "collocate"):
            raise ValueError("worker_placement must be separate or collocate")
        if self.worker_placement == "collocate" and self.rollout_workers != 1:
            raise ValueError("collocate currently requires exactly one rollout worker")
        if self.gpu_memory_utilization is None:
            # Reserve room for learner parameters, optimizer state and activations
            # when both engines remain resident on one GPU. This is a budget,
            # not a guarantee that a particular model or batch will fit.
            object.__setattr__(
                self, "gpu_memory_utilization", 0.2 if self.worker_placement == "collocate" else 0.7
            )
        pair = (self.learner_backend, self.rollout_backend)
        if pair not in {("megatron", "vllm"), ("torch_test", "torch_test")}:
            raise ValueError("Supported pairs: megatron/vllm or explicit torch_test/torch_test")
        recipe = self.recipe
        if not isinstance(self.recipe_calibration, dict):
            raise ValueError("recipe_calibration must be an object")
        calibration = self.active_calibration
        if (
            calibration
            and recipe.candidate_handler is None
            and not (calibration.discovery_only and not self.is_cpu_test)
        ):
            raise ValueError("This recipe cannot prepare candidates; native recipes support discovery_only")
        if self.is_cpu_test and not (
            recipe.name == "fp32-test-v1" or (recipe.is_mxfp8 and recipe.uses_smoothquant)
        ):
            raise ValueError(
                "CPU test engines only support fp32-test-v1 and explicit SmoothQuant emulation; "
                "no native quantization fallback"
            )
        if not self.is_cpu_test and recipe.name == "fp32-test-v1":
            raise ValueError("fp32-test-v1 requires the explicit CPU test engines")
        observation = self.observation
        if observation.enabled and (self.is_cpu_test or not recipe.is_mxfp8):
            raise ValueError("recipe_observation requires a native Megatron MXFP8 recipe; no CPU fallback")
        if observation.enabled and recipe.uses_smoothquant:
            raise ValueError(
                "Use recipe_calibration for SmoothQuant evidence; Recipe Insights compares the fixed recipes"
            )
        for name in (
            "steps",
            "prompts_per_step",
            "micro_batch_size",
            "update_epochs",
            "max_new_tokens",
            "max_model_len",
            "rollout_workers",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.rollout_workers > self.prompts_per_step:
            raise ValueError("Each rollout worker must receive at least one prompt")
        for name in ("learning_rate", "max_grad_norm", "rpc_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            type(self.gpu_memory_utilization) not in (int, float)
            or not 0 < self.clip_epsilon < 1
            or not 0 < self.gpu_memory_utilization < 1
        ):
            raise ValueError("clip_epsilon and gpu_memory_utilization must be between 0 and 1")
        if self.max_new_tokens >= self.max_model_len:
            raise ValueError("max_new_tokens must leave room for the prompt")

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text())
        calibration = data.get("recipe_calibration")
        if isinstance(calibration, dict):
            for key in ("initial_artifact", "scale_artifact"):
                value = calibration.get(key)
                if isinstance(value, str) and value.strip():
                    calibration[key] = str((path.parent / value).resolve())
        return cls(**data)

    def to_dict(self):
        result = asdict(self)
        options = result["recipe_calibration"]
        if "scale_artifact" in options:
            options["initial_artifact"] = options.pop("scale_artifact")
        return result
