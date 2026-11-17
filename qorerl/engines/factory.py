"""Construct role engines without importing native GPU libraries in the driver."""

from qorerl.engines.base import LearnerEngine, RolloutEngine
from qorerl.training.config import TrainConfig


def make_learner_engine(cfg: TrainConfig) -> LearnerEngine:
    if cfg.learner_backend == "torch_test":
        from qorerl.engines.torch_test import TestLearner

        return TestLearner(cfg)
    if cfg.learner_backend == "megatron":
        from qorerl.engines.megatron import MegatronLearner

        return MegatronLearner(cfg)
    raise ValueError(f"Unsupported learner backend: {cfg.learner_backend!r}")


def make_rollout_engine(cfg: TrainConfig) -> RolloutEngine:
    if cfg.rollout_backend == "torch_test":
        from qorerl.engines.torch_test import TestRollout

        return TestRollout(cfg)
    if cfg.rollout_backend == "vllm":
        from qorerl.engines.vllm import VLLMRollout

        return VLLMRollout(cfg)
    raise ValueError(f"Unsupported rollout backend: {cfg.rollout_backend!r}")
